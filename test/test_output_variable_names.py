import pytest
import torch

from hypatorch.utils import get_output_variable_names, validate_io_keys


class _Name(torch.nn.Module):
    def forward(self, x):
        return x


class _Tuple(torch.nn.Module):
    def forward(self, x, y):
        return x, y


class _Call(torch.nn.Module):
    """Returns a single value built by a call whose args contain a comma."""

    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.cat(x, dim=self.dim)


class _Expr(torch.nn.Module):
    def forward(self, x):
        return x.detach()


def test_bare_name_return():
    assert get_output_variable_names(_Name().forward) == ["x"]


def test_tuple_return_counts_each_element():
    assert get_output_variable_names(_Tuple().forward) == ["x", "y"]


def test_call_return_is_single_output():
    # `torch.cat(x, dim=self.dim)` contains a comma but is ONE returned value;
    # the old comma-splitting parser wrongly reported two.
    assert get_output_variable_names(_Call().forward) == ["_output_0"]


def test_expression_return_is_single_output():
    assert get_output_variable_names(_Expr().forward) == ["_output_0"]


def test_validate_accepts_configured_key_for_expression_return():
    # An op may return an expression while the config labels the single output
    # (e.g. {"x": "s_out"}). Validation must accept this (mapped positionally).
    expected_outputs = get_output_variable_names(_Call().forward)
    validate_io_keys(
        module_name="concat",
        module_object_name="_Call",
        input_key_map={"x": "s_in"},
        output_key_map={"x": "s_out"},
        expected_inputs=(["x"], []),
        expected_outputs=expected_outputs,
    )


class _MultiReturn(torch.nn.Module):
    """Several returns, positionally consistent."""

    def forward(self, x):
        if x.sum() > 0:
            wide = torch.cat([x, x], dim=-1)
            return wide, x
        narrow = x.detach()
        return narrow, x


class _InconsistentReturn(torch.nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return x
        return x, x


class _NestedFunction(torch.nn.Module):
    def forward(self, x):
        def scale(value):
            return value * 2

        scaled = scale(x)
        return scaled


class _Mixed(torch.nn.Module):
    def forward(self, x):
        return torch.cat([x, x], dim=-1), x


def test_several_returns_are_named_by_the_last_one():
    # Names describe the outputs; values come from whichever branch ran, which
    # holds because every return is positionally consistent.
    assert get_output_variable_names(_MultiReturn().forward) == ["narrow", "x"]


def test_returns_of_different_arity_are_rejected():
    with pytest.raises(ValueError, match="same number of values"):
        get_output_variable_names(_InconsistentReturn().forward)


def test_a_nested_function_return_is_ignored():
    assert get_output_variable_names(_NestedFunction().forward) == ["scaled"]


def test_validate_rejects_half_named_output_keys():
    # `torch.cat(...)` has no name while `x` does, so a config naming one and
    # positioning the other would silently assign the wrong values.
    expected_outputs = get_output_variable_names(_Mixed().forward)
    assert expected_outputs == ["_output_0", "x"]

    with pytest.raises(ValueError, match="Name every returned value or none"):
        validate_io_keys(
            module_name="mixed",
            module_object_name="_Mixed",
            input_key_map={"x": "image"},
            output_key_map={"x": "passthrough", "y": "wide"},
            expected_inputs=(["x"], []),
            expected_outputs=expected_outputs,
        )
