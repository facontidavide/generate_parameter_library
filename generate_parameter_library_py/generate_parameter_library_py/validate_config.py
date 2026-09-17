#!/usr/bin/env python3

# Copyright 2026 PickNik Inc.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#    * Neither the name of the PickNik Inc. nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Validate a ROS 2 parameter configuration file against parameter definitions.

This runs without a ROS 2 installation, a colcon workspace or a running node, so
it can be used as a CI step or by anyone who configures a robot from a laptop.
"""

import argparse
import difflib
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

from generate_parameter_library_py.parse_yaml import GenerateCode
from generate_parameter_library_py.python_validators import ParameterValidators

ROS_PARAMETERS_KEY = 'ros__parameters'
WILDCARD_NODE = '/**'

ERROR = 'ERROR'
WARNING = 'WARNING'

# Most unknown parameters for which a 'did you mean' suggestion is computed.
MAX_SUGGESTIONS = 10


@dataclass
class Diagnostic:
    """One finding, printed as a single line."""

    severity: str
    location: str
    message: str

    def __str__(self):
        return f'{self.severity}: {self.location}: {self.message}'


class _ValidatorParam:
    """The duck type the validators in python_validators.py expect."""

    def __init__(self, name: str, value: Any):
        self.name = name
        self.value = value


class DeclaredParameter:
    """A parameter of a definition file, flattened to a dotted name."""

    def __init__(self, declaration, name: Optional[str] = None):
        variable = declaration.code_gen_variable
        self.name = name if name is not None else variable.param_name
        self.scalar_type = variable.defined_base_type
        self.is_array = variable.array_type
        self.fixed_size = variable.fixed_size
        self.default_value = variable.default_value
        self.validations = declaration.parameter_validations

    @property
    def has_default(self) -> bool:
        return self.default_value is not None


def yaml_type_name(value: Any) -> str:
    """Name the type of a value the way a definition file would spell it."""
    if isinstance(value, bool):
        return 'bool'
    if isinstance(value, int):
        return 'int'
    if isinstance(value, float):
        return 'double'
    if isinstance(value, str):
        return 'string'
    if isinstance(value, list):
        return 'array'
    if value is None:
        return 'null'
    return type(value).__name__


def check_scalar_type(expected: str, value: Any) -> Optional[str]:
    actual = yaml_type_name(value)
    if actual == expected:
        return None
    if expected == 'double' and actual == 'int':
        return (
            f"expected type 'double', got 'int' ({value}); ROS 2 does not convert "
            f'integers to doubles, write {value}.0'
        )
    return f"expected type '{expected}', got '{actual}'"


def check_type(parameter: DeclaredParameter, value: Any) -> Optional[str]:
    """Return an error message when a value does not match the declared type."""
    scalar = parameter.scalar_type
    size = parameter.fixed_size

    if not parameter.is_array:
        problem = check_scalar_type(scalar, value)
        if problem is not None:
            return problem
        if size is not None and scalar == 'string' and len(value) > size:
            return f'string is longer than the fixed size of {size}'
        return None

    if not isinstance(value, list):
        return f"expected type '{scalar}_array', got '{yaml_type_name(value)}'"
    if size is not None and len(value) > size:
        return f'array has {len(value)} elements, more than the fixed size of {size}'
    for index, element in enumerate(value):
        problem = check_scalar_type(scalar, element)
        if problem is not None:
            return f'element {index} of the array: {problem}'
    return None


def run_validations(parameter: DeclaredParameter, value: Any) -> List[Tuple[str, str]]:
    """Replay the definition's validators and return (severity, message) pairs.

    A validator that rejects the value is an error. A validator that cannot run
    here, because it is written in C++ or because it raised, is a warning: the
    value may well be correct and only the check is missing.
    """
    results = []
    for validation in parameter.validations:
        name = validation.function_base_name
        function = getattr(ParameterValidators, name, None)
        if function is None:
            results.append(
                (
                    WARNING,
                    f'custom validator {validation.function_name} is C++ and '
                    f'cannot be evaluated here, not checked',
                )
            )
            continue
        try:
            result = function(
                _ValidatorParam(parameter.name, value), *validation.arguments
            )
        except Exception as error:  # noqa: BLE001 - a validator must not stop the run
            results.append(
                (
                    WARNING,
                    f'validator {name} raised {type(error).__name__}: {error}, '
                    f'not checked',
                )
            )
            continue
        if result:
            results.append((ERROR, result))
    return results


def join_name(prefix: str, part: Any) -> str:
    """Join a dotted parameter name, tolerating an empty prefix."""
    return f'{prefix}.{part}' if prefix else str(part)


def flatten(tree: Dict, prefix: str = '') -> Dict[str, Any]:
    """Flatten a nested mapping into dotted names."""
    flat = {}
    for key, value in tree.items():
        name = join_name(prefix, key)
        if isinstance(value, dict):
            flat.update(flatten(value, name))
        else:
            flat[name] = value
    return flat


def expand_mapped_name(
    name: str, mapped_params: List[str], values: Dict[str, Any]
) -> Optional[List[str]]:
    """Resolve the __map_ segments of a name using the keys found in the config.

    Returns None when a map's key list is missing from the configuration, since
    the set of parameters to expect cannot be known in that case.
    """
    names = ['']
    index = 0
    for segment in name.split('.'):
        if segment.startswith('__map_'):
            if index >= len(mapped_params):
                return None
            keys = values.get(mapped_params[index])
            index += 1
            if not isinstance(keys, list):
                return None
            names = [join_name(base, key) for base in names for key in keys]
        else:
            names = [join_name(base, segment) for base in names]
    return names


def load_definition(path: str):
    """Parse a parameter definition file and return the generator holding it."""
    generator = GenerateCode('markdown')
    generator.parse(path, '')
    return generator


def definition_namespace(path: str) -> Optional[str]:
    """Read the root element of a definition without parsing its parameters.

    Returns None when the file does not have exactly one root element, so that
    the caller parses it and reports the error the parser raises.
    """
    document = load_yaml(path)
    keys = list(document)
    return keys[0] if len(keys) == 1 else None


def load_definitions(
    paths: List[str], used: Set[str]
) -> Tuple[Dict[str, Any], List[str]]:
    """Parse the definitions whose namespace a configuration actually uses.

    Parsing a definition runs the same per parameter work as code generation, so
    with many definitions on the command line only the ones a configuration
    refers to are parsed. A single definition is always parsed, because it is
    applied to every section. Returns the parsed definitions and the paths that
    were left out, which the caller reports so that skipping stays visible.
    """
    if len(paths) == 1:
        generator = load_definition(paths[0])
        return {generator.namespace: generator}, []
    generators = {}
    unused = []
    for path in paths:
        namespace = definition_namespace(path)
        if namespace is None or namespace in used:
            generator = load_definition(path)
            generators[generator.namespace] = generator
        else:
            unused.append(path)
    return generators, unused


def declared_parameters(
    generator, values: Dict[str, Any]
) -> Dict[str, DeclaredParameter]:
    """Flatten a definition into dotted names, expanding mapped parameters."""
    parameters = {}
    for declaration in generator.declare_parameters:
        parameter = DeclaredParameter(declaration)
        parameters[parameter.name] = parameter
    for declaration in generator.declare_dynamic_parameters:
        template = DeclaredParameter(declaration)
        names = expand_mapped_name(template.name, declaration.mapped_params, values)
        if names is None:
            continue
        for name in names:
            parameters[name] = DeclaredParameter(declaration, name=name)
    return parameters


def node_sections(document: Dict) -> Dict[str, Dict[str, Any]]:
    """Return the parameter tree of every node in a ROS parameter file.

    Supports both `node: ros__parameters:` and a namespace level above it. A file
    without any `ros__parameters` key is treated as a single unnamed section, so
    that plain parameter trees can be validated too.
    """
    sections = {}
    for key, value in document.items():
        if not isinstance(value, dict):
            continue
        if ROS_PARAMETERS_KEY in value:
            sections[str(key)] = value[ROS_PARAMETERS_KEY] or {}
            continue
        for nested_key, nested_value in value.items():
            if isinstance(nested_value, dict) and ROS_PARAMETERS_KEY in nested_value:
                sections[f'{key}/{nested_key}'] = nested_value[ROS_PARAMETERS_KEY] or {}
    if not sections:
        return {'': document}
    return sections


def validate_section(
    location: str,
    generator,
    values: Dict[str, Any],
    strict: bool,
) -> List[Diagnostic]:
    """Validate one node's parameters against one definition."""
    diagnostics = []
    parameters = declared_parameters(generator, values)

    for name, parameter in sorted(parameters.items()):
        where = join_name(location, name)
        if name not in values:
            if not parameter.has_default:
                diagnostics.append(
                    Diagnostic(
                        ERROR,
                        where,
                        'missing from config and the definition sets no '
                        'default_value, the node will fail to start',
                    )
                )
            elif strict:
                diagnostics.append(
                    Diagnostic(
                        WARNING,
                        where,
                        f'missing from config, will use default_value '
                        f'{parameter.default_value}',
                    )
                )
            continue

        value = values[name]
        problem = check_type(parameter, value)
        if problem is not None:
            diagnostics.append(Diagnostic(ERROR, where, problem))
            continue
        for severity, message in run_validations(parameter, value):
            diagnostics.append(Diagnostic(severity, where, message))

    if strict:
        unknown = [name for name in sorted(values) if name not in parameters]
        # Every suggestion compares the name against every declared parameter, so
        # they are only worth computing while there are few names to suggest for.
        # A configuration that has drifted wholesale is not helped by a list of
        # guesses anyway.
        suggest = len(unknown) <= MAX_SUGGESTIONS
        for name in unknown:
            where = join_name(location, name)
            close = (
                difflib.get_close_matches(name, parameters.keys(), n=1)
                if suggest
                else []
            )
            hint = f" (did you mean '{close[0]}'?)" if close else ''
            diagnostics.append(Diagnostic(ERROR, where, f'unknown parameter{hint}'))

    return diagnostics


def load_yaml(path: str) -> Dict:
    with open(path) as handle:
        document = yaml.safe_load(handle)
    return document if document is not None else {}


def validate(
    definition_paths: List[str],
    config_paths: List[str],
    strict: bool = False,
) -> List[Diagnostic]:
    """Validate every config against the definitions whose namespace matches."""
    diagnostics = []
    configs = [(path, node_sections(load_yaml(path))) for path in config_paths]
    used = {name for _, sections in configs for name in sections}
    definitions, unused = load_definitions(definition_paths, used)
    single_definition = len(definition_paths) == 1
    if unused:
        shown = ', '.join(os.path.basename(path) for path in unused[:5])
        if len(unused) > 5:
            shown += f' and {len(unused) - 5} more'
        diagnostics.append(
            Diagnostic(
                WARNING,
                'parameter definitions',
                f'{len(unused)} matched no configuration section and were not '
                f'checked: {shown}',
            )
        )

    for config_path, sections in configs:
        wildcard = flatten(sections.pop(WILDCARD_NODE, {}))
        for node_name, tree in sections.items():
            values = dict(wildcard)
            values.update(flatten(tree))
            if node_name in definitions:
                generator = definitions[node_name]
            elif single_definition:
                generator = next(iter(definitions.values()))
            else:
                diagnostics.append(
                    Diagnostic(
                        WARNING,
                        f'{config_path}:{node_name}',
                        'no parameter definition has this namespace, not checked',
                    )
                )
                continue
            diagnostics.extend(validate_section(node_name, generator, values, strict))
    return diagnostics


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog='generate_parameter_library_validate',
        description='Validate ROS 2 parameter configuration files against '
        'generate_parameter_library parameter definitions, without a ROS 2 '
        'installation or a built workspace.',
    )
    parser.add_argument(
        '--param-definition',
        action='append',
        required=True,
        metavar='FILE',
        help='a parameter definition YAML, may be given more than once',
    )
    parser.add_argument(
        '--config',
        action='append',
        required=True,
        metavar='FILE',
        help='a parameter configuration YAML to validate, may be given more '
        'than once',
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help='also report parameters that no definition declares and '
        'parameters missing from the config that will take their default',
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    diagnostics = validate(args.param_definition, args.config, args.strict)
    for diagnostic in diagnostics:
        stream = sys.stderr if diagnostic.severity == ERROR else sys.stdout
        print(diagnostic, file=stream)
    errors = sum(1 for d in diagnostics if d.severity == ERROR)
    if errors:
        print(f'{errors} error(s) found', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
