from functools import partial
import pytest
import pandas as pd
from pandalytics.general_utils import safe_partial, replace_none


@pytest.mark.parametrize(
    "func,args,kws,expected",
    [
        (pd.to_numeric, tuple(), dict(hi="!"), partial(pd.to_numeric)),  # bad kwarg
        (
            pd.to_numeric,
            [
                pd.Series(),
            ],
            dict(errors="ignore", downcast="float"),
            partial(
                pd.to_numeric, errors="ignore", downcast="float"
            ),  # only correct arg and kwargs
        ),
        (
            pd.to_numeric,
            [
                pd.Series(),
            ],
            dict(errors="ignore", downcast="float", hi="!"),
            partial(
                pd.to_numeric, errors="ignore", downcast="float"
            ),  # both bad and correct kwargs
        ),
    ],
)
def test_safe_partial(func, args, kws, expected):
    test = safe_partial(func, **kws)
    assert test.func == expected.func, "functions are mismatched."
    assert test.args == expected.args, "args are mismatched."
    assert test.keywords == expected.keywords, "args are mismatched."


@pytest.mark.parametrize(
    "variable,replacement_value,expected",
    [
        (None, 1, 1),
        (None, dict(a=1), dict(a=1)),
        (None, "b", "b"),
        ("b", {}, "b"),
        ([], {}, []),
    ],
)
def test_replace_none(variable, replacement_value, expected):
    assert (
        replace_none(variable, replacement_value) == expected
    ), f"replace_none({variable}, {replacement_value}) did NOT return {expected}"
