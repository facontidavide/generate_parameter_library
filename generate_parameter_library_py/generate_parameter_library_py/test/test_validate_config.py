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

import os
import tempfile

import pytest

from generate_parameter_library_py.validate_config import main, validate

DEFINITION = """my_node:
  rate:
    type: double
    default_value: 10.0
    validation:
      gt<>: [0.0]
  name:
    type: string
    default_value: "robot"
  mandatory_gain:
    type: double
  mode:
    type: string
    default_value: "fast"
    validation:
      one_of<>: [["fast", "slow"]]
  joints:
    type: string_array
    default_value: ["a", "b"]
  offsets:
    type: double_array_fixed_2
    default_value: [0.0, 0.0]
  checked_by_cpp:
    type: int
    default_value: 3
    validation:
      "my_project::integer_equal_value": [3]
  background:
    r:
      type: int
      default_value: 0
      validation:
        bounds<>: [0, 255]
"""

MAPPED_DEFINITION = """mapped_node:
  joints:
    type: string_array
    default_value: ["shoulder", "elbow"]
  pid:
    __map_joints:
      p:
        type: double
        default_value: 1.0
        validation:
          gt<>: [0.0]
"""


def write(directory, name, text):
    path = os.path.join(directory, name)
    with open(path, 'w') as handle:
        handle.write(text)
    return path


@pytest.fixture
def workspace():
    with tempfile.TemporaryDirectory() as directory:
        yield directory


@pytest.fixture
def definition(workspace):
    return write(workspace, 'definition.yaml', DEFINITION)


def messages(diagnostics):
    return [str(diagnostic) for diagnostic in diagnostics]


def errors(diagnostics):
    return [d for d in diagnostics if d.severity == 'ERROR']


def test_valid_config_reports_nothing(workspace, definition):
    config = write(
        workspace,
        'config.yaml',
        """my_node:
  ros__parameters:
    rate: 20.0
    mandatory_gain: 1.5
    mode: "slow"
    background:
      r: 128
""",
    )
    assert validate([definition], [config]) == []


def test_missing_mandatory_parameter_is_an_error(workspace, definition):
    config = write(
        workspace,
        'config.yaml',
        """my_node:
  ros__parameters:
    rate: 20.0
""",
    )
    found = errors(validate([definition], [config]))
    assert len(found) == 1
    assert 'mandatory_gain' in found[0].location
    assert 'default_value' in found[0].message


def test_integer_for_a_double_is_reported(workspace, definition):
    config = write(
        workspace,
        'config.yaml',
        """my_node:
  ros__parameters:
    rate: 20
    mandatory_gain: 1.5
""",
    )
    found = errors(validate([definition], [config]))
    assert len(found) == 1
    assert 'my_node.rate' == found[0].location
    assert '20.0' in found[0].message


def test_wrong_scalar_type_is_reported(workspace, definition):
    config = write(
        workspace,
        'config.yaml',
        """my_node:
  ros__parameters:
    rate: "fast"
    mandatory_gain: 1.5
""",
    )
    found = errors(validate([definition], [config]))
    assert "expected type 'double', got 'string'" in found[0].message


def test_value_out_of_bounds_is_reported(workspace, definition):
    config = write(
        workspace,
        'config.yaml',
        """my_node:
  ros__parameters:
    mandatory_gain: 1.5
    background:
      r: 300
""",
    )
    found = errors(validate([definition], [config]))
    assert len(found) == 1
    assert found[0].location == 'my_node.background.r'


def test_value_outside_one_of_is_reported(workspace, definition):
    config = write(
        workspace,
        'config.yaml',
        """my_node:
  ros__parameters:
    mandatory_gain: 1.5
    mode: "medium"
""",
    )
    found = errors(validate([definition], [config]))
    assert len(found) == 1
    assert found[0].location == 'my_node.mode'


def test_array_element_type_is_checked(workspace, definition):
    config = write(
        workspace,
        'config.yaml',
        """my_node:
  ros__parameters:
    mandatory_gain: 1.5
    joints: ["a", 2]
""",
    )
    found = errors(validate([definition], [config]))
    assert 'element 1 of the array' in found[0].message


def test_fixed_size_array_is_checked(workspace, definition):
    config = write(
        workspace,
        'config.yaml',
        """my_node:
  ros__parameters:
    mandatory_gain: 1.5
    offsets: [0.0, 1.0, 2.0]
""",
    )
    found = errors(validate([definition], [config]))
    assert 'fixed size of 2' in found[0].message


def test_custom_cpp_validator_is_skipped_with_a_warning(workspace, definition):
    config = write(
        workspace,
        'config.yaml',
        """my_node:
  ros__parameters:
    mandatory_gain: 1.5
    checked_by_cpp: 7
""",
    )
    diagnostics = validate([definition], [config])
    assert errors(diagnostics) == []
    assert [d.severity for d in diagnostics] == ['WARNING']
    assert 'my_project::integer_equal_value' in diagnostics[0].message


def test_unknown_parameter_is_reported_in_strict_mode_only(workspace, definition):
    config = write(
        workspace,
        'config.yaml',
        """my_node:
  ros__parameters:
    mandatory_gain: 1.5
    background:
      colour: 1
""",
    )
    assert errors(validate([definition], [config])) == []
    found = errors(validate([definition], [config], strict=True))
    assert len(found) == 1
    assert found[0].location == 'my_node.background.colour'
    assert "did you mean 'background.r'" in found[0].message


def test_defaults_are_reported_in_strict_mode(workspace, definition):
    config = write(
        workspace,
        'config.yaml',
        """my_node:
  ros__parameters:
    mandatory_gain: 1.5
""",
    )
    diagnostics = validate([definition], [config], strict=True)
    assert errors(diagnostics) == []
    assert any('will use default_value' in str(d) for d in diagnostics)


def test_wildcard_node_supplies_values(workspace, definition):
    config = write(
        workspace,
        'config.yaml',
        """/**:
  ros__parameters:
    mandatory_gain: 1.5
my_node:
  ros__parameters:
    rate: 5.0
""",
    )
    assert validate([definition], [config]) == []


def test_plain_parameter_tree_without_ros_parameters(workspace, definition):
    config = write(
        workspace,
        'config.yaml',
        """rate: 20.0
mandatory_gain: 1.5
""",
    )
    assert validate([definition], [config]) == []


def test_mapped_parameters_expand_from_the_config(workspace):
    definition = write(workspace, 'mapped.yaml', MAPPED_DEFINITION)
    config = write(
        workspace,
        'config.yaml',
        """mapped_node:
  ros__parameters:
    joints: ["shoulder", "wrist"]
    pid:
      shoulder:
        p: 2.0
      wrist:
        p: -1.0
""",
    )
    found = errors(validate([definition], [config]))
    assert len(found) == 1
    assert found[0].location == 'mapped_node.pid.wrist.p'


def test_several_definitions_match_by_namespace(workspace):
    first = write(workspace, 'first.yaml', DEFINITION)
    second = write(workspace, 'second.yaml', MAPPED_DEFINITION)
    config = write(
        workspace,
        'config.yaml',
        """my_node:
  ros__parameters:
    mandatory_gain: 1.5
mapped_node:
  ros__parameters:
    joints: ["shoulder"]
    pid:
      shoulder:
        p: 2.0
""",
    )
    assert validate([first, second], [config]) == []


def test_validator_that_raises_is_a_warning(workspace, monkeypatch):
    """A validator we cannot evaluate must not be reported as a bad value."""
    from generate_parameter_library_py import validate_config

    def explode(param, *arguments):
        raise RuntimeError('boom')

    monkeypatch.setattr(
        validate_config.ParameterValidators, 'gt', explode, raising=True
    )
    definition = write(workspace, 'definition.yaml', DEFINITION)
    config = write(
        workspace,
        'config.yaml',
        """my_node:
  ros__parameters:
    rate: 20.0
    mandatory_gain: 1.5
""",
    )
    diagnostics = validate([definition], [config])
    assert errors(diagnostics) == []
    assert [d.severity for d in diagnostics] == ['WARNING']
    assert 'RuntimeError' in diagnostics[0].message


def test_unused_definitions_are_reported_and_not_parsed(workspace, definition):
    """A definition no section refers to is skipped, and the skip is visible."""
    broken = write(
        workspace,
        'broken.yaml',
        """other_node:
  bad:
    type: not_a_real_type
""",
    )
    config = write(
        workspace,
        'config.yaml',
        """my_node:
  ros__parameters:
    mandatory_gain: 1.5
""",
    )
    diagnostics = validate([definition, broken], [config])
    assert errors(diagnostics) == []
    assert [d.severity for d in diagnostics] == ['WARNING']
    assert 'broken.yaml' in diagnostics[0].message


def test_main_returns_non_zero_on_error(workspace, definition):
    config = write(
        workspace,
        'config.yaml',
        """my_node:
  ros__parameters:
    rate: "fast"
    mandatory_gain: 1.5
""",
    )
    code = main(['--param-definition', definition, '--config', config])
    assert code == 1


def test_main_returns_zero_on_success(workspace, definition):
    config = write(
        workspace,
        'config.yaml',
        """my_node:
  ros__parameters:
    mandatory_gain: 1.5
""",
    )
    assert main(['--param-definition', definition, '--config', config]) == 0
