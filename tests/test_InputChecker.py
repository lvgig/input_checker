import pandas as pd
import numpy as np
import pytest
import re
import tubular
import tubular.testing.helpers as h
import tubular.testing.test_data as data_generators_p

import input_checker
from input_checker._version import __version__
from input_checker.checker import InputChecker
from input_checker.exceptions import InputCheckerError


class TestInit(object):
    """Tests for InputChecker.init()."""

    def test_super_init_called(self, mocker):
        """Test that init calls BaseTransformer.init."""

        expected_call_args = {0: {"args": (), "kwargs": {"columns": ["a", "b"]}}}

        with h.assert_function_call(
            mocker, tubular.base.BaseTransformer, "__init__", expected_call_args
        ):
            InputChecker(columns=["a", "b"])

    def test_inheritance(self):
        """Test that InputChecker inherits from tubular.base.BaseTransformer."""

        x = InputChecker()

        h.assert_inheritance(x, tubular.base.BaseTransformer)

    def test_arguments(self):
        """Test that InputChecker init has expected arguments."""

        h.test_function_arguments(
            func=InputChecker.__init__,
            expected_arguments=[
                "self",
                "columns",
                "categorical_columns",
                "numerical_columns",
                "datetime_columns",
                "skip_infer_columns",
            ],
            expected_default_values=(None, None, None, None, None),
        )

    def test_version_attribute(self):
        """Test that __version__ attribute takes expected value."""

        x = InputChecker(columns=["a"])

        h.assert_equal_dispatch(
            expected=__version__,
            actual=x.version_,
            msg="__version__ attribute",
        )

    def test_columns_attributes_generated(self):
        """Test all columns attributes are saved with InputChecker init"""

        x = InputChecker(
            columns=["a", "b", "c", "d"],
            numerical_columns=["a"],
            categorical_columns=["b"],
            datetime_columns=["d"],
            skip_infer_columns=["c"],
        )

        df = data_generators_p.create_df_2()

        df["d"] = pd.to_datetime(
            [
                "01/02/2020",
                "01/02/2021",
                "08/04/2019",
                "01/03/2020",
                "29/03/2019",
                "15/10/2018",
                "24/07/2020",
            ]
        )

        x.fit(df)

        assert hasattr(x, "columns") is True, "columns attribute not present after init"

        assert (
            hasattr(x, "numerical_columns") is True
        ), "numerical_columns attribute not present after init"

        assert (
            hasattr(x, "categorical_columns") is True
        ), "categorical_columns attribute not present after init"

        assert (
            hasattr(x, "datetime_columns") is True
        ), "datetime_columns attribute not present after init"

        assert (
            hasattr(x, "skip_infer_columns") is True
        ), "skip_infer_columns attribute not present after init"

    def test_check_type_called(self, mocker):
        """Test all check type is called by the init method."""
        spy = mocker.spy(input_checker.checker.InputChecker, "_check_type")

        x = InputChecker(
            columns=["a", "b", "c", "d"],
            numerical_columns=["a"],
            categorical_columns=["b"],
            datetime_columns=["d"],
            skip_infer_columns=["c"],
        )

        assert (
            spy.call_count == 5
        ), "unexpected number of calls to InputChecker._check_type with init"

        call_0_args = spy.call_args_list[0]
        call_0_pos_args = call_0_args[0]

        call_1_args = spy.call_args_list[1]
        call_1_pos_args = call_1_args[0]

        call_2_args = spy.call_args_list[2]
        call_2_pos_args = call_2_args[0]

        call_3_args = spy.call_args_list[3]
        call_3_pos_args = call_3_args[0]

        call_4_args = spy.call_args_list[4]
        call_4_pos_args = call_4_args[0]

        expected_pos_args_0 = (
            x,
            ["a", "b", "c", "d"],
            "input columns",
            [list, type(None), str],
        )
        expected_pos_args_1 = (
            x,
            ["b"],
            "categorical columns",
            [list, str, type(None)],
        )
        expected_pos_args_2 = (
            x,
            ["a"],
            "numerical columns",
            [list, dict, str, type(None)],
        )
        expected_pos_args_3 = (
            x,
            ["d"],
            "datetime columns",
            [list, dict, str, type(None)],
        )
        expected_pos_args_4 = (
            x,
            ["c"],
            "skip infer columns",
            [list, type(None)],
        )

        assert (
            expected_pos_args_0 == call_0_pos_args
        ), "positional args unexpected in _check_type call for columns argument"

        assert (
            expected_pos_args_1 == call_1_pos_args
        ), "positional args unexpected in _check_type call for categorical columns argument"

        assert (
            expected_pos_args_2 == call_2_pos_args
        ), "positional args unexpected in _check_type call for numerical columns argument"

        assert (
            expected_pos_args_3 == call_3_pos_args
        ), "positional args unexpected in _check_type call for datetime columns argument"

        assert (
            expected_pos_args_4 == call_4_pos_args
        ), "positional args unexpected in _check_type call for skip infer columns argument"

    def test_check_is_string_value_called(self, mocker):
        """Test all check string is called by the init method when option set to infer."""
        spy = mocker.spy(input_checker.checker.InputChecker, "_is_string_value")

        x = InputChecker(
            numerical_columns="infer",
            categorical_columns="infer",
            datetime_columns="infer",
        )

        assert (
            spy.call_count == 3
        ), "unexpected number of calls to InputChecker._is_string_value with init"

        call_0_args = spy.call_args_list[0]
        call_0_pos_args = call_0_args[0]

        call_1_args = spy.call_args_list[1]
        call_1_pos_args = call_1_args[0]

        call_2_args = spy.call_args_list[2]
        call_2_pos_args = call_2_args[0]

        expected_pos_args_0 = (x, x.categorical_columns, "categorical columns", "infer")
        expected_pos_args_1 = (x, x.numerical_columns, "numerical columns", "infer")
        expected_pos_args_2 = (x, x.datetime_columns, "datetime columns", "infer")

        assert (
            expected_pos_args_0 == call_0_pos_args
        ), "positional args unexpected in _is_string_value call for numerical columns argument"

        assert (
            expected_pos_args_1 == call_1_pos_args
        ), "positional args unexpected in _is_string_value call for categorical columns argument"

        assert (
            expected_pos_args_2 == call_2_pos_args
        ), "positional args unexpected in _is_string_value call for categorical columns argument"

    def test_check_is_empty_called(self, mocker):
        """Test all check is empty is called by the init method."""
        spy = mocker.spy(input_checker.checker.InputChecker, "_is_empty")

        x = InputChecker(
            columns=["a", "b", "c", "d"],
            numerical_columns=["a"],
            categorical_columns=["b", "c"],
            datetime_columns=["d"],
        )

        assert (
            spy.call_count == 4
        ), "unexpected number of calls to InputChecker._is_empty with init"

        call_0_args = spy.call_args_list[0]
        call_0_pos_args = call_0_args[0]

        call_1_args = spy.call_args_list[1]
        call_1_pos_args = call_1_args[0]

        call_2_args = spy.call_args_list[2]
        call_2_pos_args = call_2_args[0]

        call_3_args = spy.call_args_list[3]
        call_3_pos_args = call_3_args[0]

        expected_pos_args_0 = (x, "input columns", ["a", "b", "c", "d"])
        expected_pos_args_1 = (x, "categorical columns", ["b", "c"])
        expected_pos_args_2 = (x, "numerical columns", ["a"])
        expected_pos_args_3 = (x, "datetime columns", ["d"])

        assert (
            expected_pos_args_0 == call_0_pos_args
        ), "positional args unexpected in _is_empty call for categorical columns argument"

        assert (
            expected_pos_args_1 == call_1_pos_args
        ), "positional args unexpected in _is_empty call for numerical columns argument"

        assert (
            expected_pos_args_2 == call_2_pos_args
        ), "positional args unexpected in _is_empty call for numerical columns argument"

        assert (
            expected_pos_args_3 == call_3_pos_args
        ), "positional args unexpected in _is_empty call for numerical columns argument"

    def test_check_is_listed_in_columns_called(self, mocker):
        spy = mocker.spy(input_checker.checker.InputChecker, "_is_listed_in_columns")

        InputChecker(
            columns=["a", "b", "c", "d"],
            numerical_columns=["a"],
            categorical_columns=["b", "c"],
            datetime_columns=["d"],
        )

        assert (
            spy.call_count == 1
        ), "unexpected number of calls to InputChecker._is_listed_in_columns with init"


class TestConsolidateInputs(object):
    def test_arguments(self):
        """Test that _consolidate_inputs has expected arguments."""
        h.test_function_arguments(
            func=InputChecker._consolidate_inputs,
            expected_arguments=["self", "X"],
            expected_default_values=None,
        )

    def test_infer_datetime_columns(self):
        """Test that _consolidate_inputs infers the correct datetime columns"""
        x = InputChecker(datetime_columns="infer")

        df = data_generators_p.create_df_2()

        df["d"] = pd.to_datetime(
            [
                "01/02/2020",
                "01/02/2021",
                "08/04/2019",
                "01/03/2020",
                "29/03/2019",
                "15/10/2018",
                "24/07/2020",
            ]
        )
        df["e"] = pd.to_datetime(
            [
                "01/02/2020",
                "01/02/2021",
                "08-04-2019",
                "01/03/2020",
                "29/03/2019",
                "15/10/2018",
                "24/07/2020",
            ]
        )

        x.fit(df)

        assert x.datetime_columns == [
            "d",
            "e",
        ], "infer datetime not finding correct columns"

    def test_infer_datetime_dict(self):
        """Test that _consolidate_inputs infers the correct datetime dict"""

        x = InputChecker(datetime_columns="infer")

        df = data_generators_p.create_df_2()

        df["d"] = pd.to_datetime(
            [
                "01/02/2020",
                "01/02/2021",
                "08/04/2019",
                "01/03/2020",
                "29/03/2019",
                "15/10/2018",
                "24/07/2020",
            ]
        )

        x.fit(df)

        assert (
            x.datetime_dict["d"]["maximum"] is False
        ), "infer numerical not specifying maximum value check as true"

        assert (
            x.datetime_dict["d"]["minimum"] is True
        ), "infer numerical not specifying maximum value check as true"

    def test_infer_categorical_columns(self):
        """Test that _consolidate_inputs infers the correct categorical columns"""
        x = InputChecker(categorical_columns="infer")

        df = data_generators_p.create_df_2()

        df["d"] = [True, True, False, True, True, False, np.nan]

        df["d"] = df["d"].astype("bool")

        x.fit(df)

        assert x.categorical_columns == [
            "b",
            "c",
            "d",
        ], "infer categorical not finding correct columns"

    def test_infer_numerical_columns(self):
        """Test that _consolidate_inputs infers the correct numerical columns"""

        x = InputChecker(numerical_columns="infer")

        df = data_generators_p.create_df_2()

        x.fit(df)

        assert x.numerical_columns == [
            "a"
        ], "infer numerical not finding correct columns"

    def test_infer_numerical_skips_infer_columns(self):
        """Test that _consolidate_inputs skips right columns when inferring numerical"""

        x = InputChecker(numerical_columns="infer", skip_infer_columns=["a"])

        df = data_generators_p.create_df_2()

        df["d"] = df["a"]

        x.fit(df)

        assert x.numerical_columns == [
            "d"
        ], "infer numerical not finding correct columns when skipping infer columns"

    def test_infer_categorical_skips_infer_columns(self):
        """Test that _consolidate_inputs skips right columns when inferring categorical"""

        x = InputChecker(categorical_columns="infer", skip_infer_columns=["b"])

        df = data_generators_p.create_df_2()

        x.fit(df)

        assert x.categorical_columns == [
            "c"
        ], "infer categorical not finding correct columns when skipping infer columns"

    def test_infer_datetime_skips_infer_columns(self):
        """Test that _consolidate_inputs skips right columns when inferring datetime"""

        x = InputChecker(datetime_columns="infer", skip_infer_columns=["d"])

        df = data_generators_p.create_df_2()

        df["d"] = pd.to_datetime(
            [
                "01/02/2020",
                "01/02/2021",
                "08/04/2019",
                "01/03/2020",
                "29/03/2019",
                "15/10/2018",
                "24/07/2020",
            ]
        )

        df["a"] = df["d"]

        x.fit(df)

        assert x.datetime_columns == [
            "a"
        ], "infer datetime not finding correct columns when skipping infer columns"

    def test_infer_numerical_dict(self):
        """Test that _consolidate_inputs infers the correct numerical dict"""

        x = InputChecker(numerical_columns="infer")

        df = data_generators_p.create_df_2()

        x.fit(df)

        assert (
            x.numerical_dict["a"]["maximum"] is True
        ), "infer numerical not specifying maximum value check as true"

        assert (
            x.numerical_dict["a"]["minimum"] is True
        ), "infer numerical not specifying minimum value check as true"

    def test_datetime_type(self):
        """Test that datetime columns is a list after calling _consolidate_inputs"""

        x = InputChecker(datetime_columns="infer")

        df = data_generators_p.create_df_2()

        df["d"] = pd.to_datetime(
            [
                "01/02/2020",
                "01/02/2021",
                "08/04/2019",
                "01/03/2020",
                "29/03/2019",
                "15/10/2018",
                "24/07/2020",
            ]
        )

        x.fit(df)

        assert (
            type(x.datetime_columns) is list
        ), f"incorrect datetime_columns type returned from _consolidate_inputs - expected: list but got: {type(x.datetime_columns)} "

    def test_categorical_type(self):
        """Test that categorical columns is a list after calling _consolidate_inputs"""

        x = InputChecker(categorical_columns="infer")

        df = data_generators_p.create_df_2()

        x.fit(df)

        assert (
            type(x.categorical_columns) is list
        ), f"incorrect categorical_columns type returned from _consolidate_inputs - expected: list but got: {type(x.categorical_columns)} "

    def test_numerical_type(self):
        """Test that numerical columns and dict are a list and dict after calling _consolidate_inputs"""

        x = InputChecker(numerical_columns="infer")

        df = data_generators_p.create_df_2()

        x.fit(df)

        assert (
            type(x.numerical_columns) is list
        ), f"incorrect numerical_columns type returned from _consolidate_inputs - expected: list but got: {type(x.numerical_columns)} "

        assert (
            type(x.numerical_dict) is dict
        ), f"incorrect numerical_dict type returned from _consolidate_inputs - expected: dict but got: {type(x.numerical_dict)} "

    def test_check_is_subset_called(self, mocker):
        """Test all check _is_subset is called by the _consolidate_inputs method."""

        x = InputChecker(
            columns=["a", "b", "c", "d"],
            numerical_columns=["a"],
            categorical_columns=["c"],
            datetime_columns=["d"],
            skip_infer_columns=["b"],
        )

        df = data_generators_p.create_df_2()

        df["d"] = pd.to_datetime(
            [
                "01/02/2020",
                "01/02/2021",
                "08/04/2019",
                "01/03/2020",
                "29/03/2019",
                "15/10/2018",
                "24/07/2020",
            ]
        )

        spy = mocker.spy(input_checker.checker.InputChecker, "_is_subset")

        x.fit(df)

        assert (
            spy.call_count == 5
        ), "unexpected number of calls to InputChecker._is_subset with _consolidate_inputs"

        call_0_args = spy.call_args_list[0]
        call_0_pos_args = call_0_args[0]

        call_1_args = spy.call_args_list[1]
        call_1_pos_args = call_1_args[0]

        call_2_args = spy.call_args_list[2]
        call_2_pos_args = call_2_args[0]

        call_3_args = spy.call_args_list[3]
        call_3_pos_args = call_3_args[0]

        call_4_args = spy.call_args_list[4]
        call_4_pos_args = call_4_args[0]

        expected_pos_args_0 = (x, "skip infer columns", ["b"], df)
        expected_pos_args_1 = (x, "input columns", ["a", "b", "c", "d"], df)
        expected_pos_args_2 = (x, "categorical columns", ["c"], df)
        expected_pos_args_3 = (x, "numerical columns", ["a"], df)
        expected_pos_args_4 = (x, "datetime columns", ["d"], df)

        assert (
            expected_pos_args_0 == call_0_pos_args
        ), "positional args unexpected in _is_subset call for skip_infer_columns columns argument"

        assert (
            expected_pos_args_1 == call_1_pos_args
        ), "positional args unexpected in _is_subset call for input columns argument"

        assert (
            expected_pos_args_2 == call_2_pos_args
        ), "positional args unexpected in _is_subset call for categorical columns argument"

        assert (
            expected_pos_args_3 == call_3_pos_args
        ), "positional args unexpected in _is_subset call for numerical columns argument"

        assert (
            expected_pos_args_4 == call_4_pos_args
        ), "positional args unexpected in _is_subset call for datetime columns argument"


class TestFitTypeChecker(object):
    """Tests for InputChecker._fit_type_checker()."""

    def test_arguments(self):
        """Test that InputChecker _fit_type_checker has expected arguments."""

        h.test_function_arguments(
            func=InputChecker._fit_type_checker, expected_arguments=["self", "X"]
        )

    def test_no_column_classes_before_fit(self):
        """Test column_classes is not present before fit called"""

        x = InputChecker()

        assert (
            hasattr(x, "column_classes") is False
        ), "column_classes attribute present before fit"

    def test_column_classes_after_fit(self):
        """Test column_classes is present after fit called"""

        df = data_generators_p.create_df_2()

        x = InputChecker()

        x.fit(df)

        assert hasattr(
            x, "column_classes"
        ), "column_classes attribute not present after fit"

    def test_correct_columns_classes(self):
        """Test fit type checker saves types for correct columns after fit called"""

        df = data_generators_p.create_df_2()

        x = InputChecker(columns=["a"])

        x.fit(df)

        assert list(x.column_classes.keys()) == [
            "a"
        ], f"incorrect values returned from _fit_value_checker - expected: ['a'] but got: {list(x.column_classes.keys())}"

    def test_correct_classes_identified(self):
        """Test fit type checker identifies correct classes is present after fit called"""

        df = data_generators_p.create_df_2()

        x = InputChecker()

        df["d"] = pd.to_datetime(
            [
                "01/02/2020",
                "01/02/2021",
                "08/04/2019",
                "01/03/2020",
                "29/03/2019",
                "15/10/2018",
                "24/07/2020",
            ]
        )

        x.fit(df)

        assert (
            x.column_classes["a"] == "float64"
        ), f"incorrect type returned from _fit_type_checker for column 'a' - expected: float64 but got: {x.column_classes['a']}"

        assert (
            x.column_classes["b"] == "object"
        ), f"incorrect type returned from _fit_type_checker for column 'b' - expected: object but got: {x.column_classes['b']}"

        assert (
            x.column_classes["c"] == "category"
        ), f"incorrect type returned from _fit_type_checker for column 'c' - expected: category but got: {x.column_classes['c']}"

        assert (
            x.column_classes["d"] == "datetime64[ns]"
        ), f"incorrect type returned from _fit_type_checker for column 'd' - expected: datetime64[ns] but got: {x.column_classes['d']}"


class TestFitNullChecker(object):
    """Tests for InputChecker._fit_null_checker()."""

    def test_arguments(self):
        """Test that InputChecker _fit_null_checker has expected arguments."""

        h.test_function_arguments(
            func=InputChecker._fit_null_checker, expected_arguments=["self", "X"]
        )

    def test_no_expected_values_before_fit(self):
        """Test null_map is not present before fit called"""

        x = InputChecker()

        assert hasattr(x, "null_map") is False, "null_map attribute present before fit"

    def test_expected_values_after_fit(self):
        """Test null_map is present after fit called"""

        df = data_generators_p.create_df_2()

        x = InputChecker()

        x.fit(df)

        assert hasattr(x, "null_map"), "null_map attribute not present after fit"

    def test_correct_columns_nulls(self):
        """Test fit nulls checker saves map for correct columns after fit called"""

        df = data_generators_p.create_df_2()

        x = InputChecker(columns=["a"])

        x.fit(df)

        assert list(x.null_map.keys()) == [
            "a"
        ], f"incorrect values returned from _fit_null_checker - expected: ['a'] but got: {list(x.null_map.keys())}"

    def test_correct_classes_identified(self):
        """Test fit null checker identifies correct columns with nulls after fit called"""

        df = data_generators_p.create_df_2()

        x = InputChecker()

        df["b"] = df["b"].fillna("a")

        x.fit(df)

        assert (
            x.null_map["a"] == 1
        ), f"incorrect values returned from _fit_null_checker - expected: 1 but got: {x.null_map['a']}"

        assert (
            x.null_map["b"] == 0
        ), f"incorrect values returned from _fit_null_checker - expected: 0 but got: {x.null_map['b']}"

        assert (
            x.null_map["c"] == 1
        ), f"incorrect values returned from _fit_null_checker - expected: 1 but got: {x.null_map['c']}"


class TestFitValueChecker(object):
    """Tests for InputChecker._fit_value_checker()."""

    def test_arguments(self):
        """Test that InputChecker _fit_value_checker has expected arguments."""

        h.test_function_arguments(
            func=InputChecker._fit_value_checker, expected_arguments=["self", "X"]
        )

    def test_no_expected_values_before_fit(self):
        """Test expected_values is not present before fit called"""

        x = InputChecker(categorical_columns=["b", "c"])

        assert (
            hasattr(x, "expected_values") is False
        ), "expected_values attribute present before fit"

    def test_expected_values_after_fit(self):
        """Test expected_values is present after fit called"""

        df = data_generators_p.create_df_2()

        x = InputChecker(categorical_columns=["b", "c"])

        x.fit(df)

        assert hasattr(
            x, "expected_values"
        ), "expected_values attribute not present after fit"

    def test_correct_columns_map(self):
        """Test fit value checker saves levels for correct columns after fit called"""

        df = data_generators_p.create_df_2()

        x = InputChecker(categorical_columns=["b", "c"])

        x.fit(df)

        assert list(x.expected_values.keys()) == [
            "b",
            "c",
        ], f"incorrect values returned from _fit_value_checker - expected: ['b', 'c'] but got: {list(x.expected_values.keys())}"

    def test_correct_values_identified(self):
        """Test fit value checker identifies corrcet levels after fit called"""

        df = data_generators_p.create_df_2()

        df["d"] = [True, True, False, True, True, False, np.nan]

        df["d"] = df["d"].astype("bool")

        x = InputChecker(categorical_columns=["b", "c", "d"])

        x.fit(df)

        assert x.expected_values["b"] == [
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
            np.nan,
        ], f"incorrect values returned from _fit_value_checker - expected: ['a', 'b', 'c', 'd', 'e', 'f', np.nan] but got: {x.expected_values['b']}"

        assert x.expected_values["c"] == [
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
            np.nan,
        ], f"incorrect values returned from _fit_value_checker - expected: ['a', 'b', 'c', 'd', 'e', 'f', np.nan] but got: {x.expected_values['c']}"

        assert x.expected_values["d"] == [
            True,
            False,
        ], f"incorrect values returned from _fit_value_checker - expected: [True, False, np.nan] but got: {x.expected_values['d']}"


class TestFitNumericalChecker(object):
    """Tests for InputChecker._fit_numerical_checker()."""

    def test_arguments(self):
        """Test that InputChecker _fit_numerical_checker has expected arguments."""

        h.test_function_arguments(
            func=InputChecker._fit_numerical_checker, expected_arguments=["self", "X"]
        )

    def test_no_expected_values_before_fit(self):
        """Test numerical_values is not present before fit called"""

        x = InputChecker()

        assert (
            hasattr(x, "numerical_values") is False
        ), "numerical_values attribute present before fit"

    def test_expected_values_after_fit(self):
        """Test numerical_values is present after fit called"""

        df = data_generators_p.create_df_2()

        x = InputChecker(numerical_columns=["a"])

        x.fit(df)

        assert hasattr(
            x, "numerical_values"
        ), "numerical_values attribute not present after fit"

    def test_correct_columns_num_values(self):
        """Test fit numerical checker saves values for correct columns after fit called"""

        df = data_generators_p.create_df_2()

        x = InputChecker(numerical_columns=["a"])

        x.fit(df)

        assert list(x.numerical_values.keys()) == [
            "a"
        ], f"incorrect values returned from numerical_values - expected: ['a'] but got: {list(x.numerical_values.keys())}"

    def test_correct_numerical_values_identified(self):
        """Test fit numerical checker identifies correct range values after fit called"""

        df = data_generators_p.create_df_2()

        x = InputChecker(numerical_columns=["a"])

        x.fit(df)

        assert (
            x.numerical_values["a"]["maximum"] == 6
        ), f"incorrect values returned from _fit_numerical_checker - expected: 1 but got: {x.numerical_values['a']['maximum']}"

        assert (
            x.numerical_values["a"]["minimum"] == 1
        ), f"incorrect values returned from _fit_numerical_checker - expected: 0 but got: {x.numerical_values['a']['minimum']}"

    def test_correct_numerical_values_identified_dict(self):
        """Test fit numerical checker identifies correct range values after fit called when inputting a dictionary"""

        df = data_generators_p.create_df_2()

        numerical_dict = {}
        numerical_dict["a"] = {}
        numerical_dict["a"]["maximum"] = True
        numerical_dict["a"]["minimum"] = False

        x = InputChecker(numerical_columns=numerical_dict)

        x.fit(df)

        assert (
            x.numerical_values["a"]["maximum"] == 6
        ), f"incorrect values returned from _fit_numerical_checker - expected: 1 but got: {x.numerical_values['a']['maximum']}"

        assert (
            x.numerical_values["a"]["minimum"] is None
        ), f"incorrect values returned from _fit_numerical_checker - expected: None but got: {x.numerical_values['a']['minimum']}"


class TestFitDatetimeChecker(object):
    """Tests for InputChecker._fit_datetime_checker()."""

    def test_arguments(self):
        """Test that InputChecker _fit_value_checker has expected arguments."""

        h.test_function_arguments(
            func=InputChecker._fit_datetime_checker, expected_arguments=["self", "X"]
        )

    def test_no_datetime_values_before_fit(self):
        """Test expected_values is not present before fit called"""

        x = InputChecker(datetime_columns=["b", "c"])

        assert (
            hasattr(x, "datetime_values") is False
        ), "datetime_values attribute present before fit"

    def test_datetime_values_after_fit(self):
        """Test datetime_values is present after fit called"""

        df = data_generators_p.create_df_2()

        df["d"] = pd.to_datetime(
            [
                "01/02/2020",
                "01/02/2021",
                "08/04/2019",
                "01/03/2020",
                "29/03/2019",
                "15/10/2018",
                "24/07/2020",
            ]
        )
        df["e"] = pd.to_datetime(
            [
                "01/02/2020",
                "01/02/2021",
                "08-04-2019",
                "01/03/2020",
                "29/03/2019",
                "15/10/2018",
                "24/07/2020",
            ]
        )

        x = InputChecker(datetime_columns=["d", "e"])

        x.fit(df)

        assert hasattr(
            x, "datetime_values"
        ), "datetime_values attribute not present after fit"

    def test_correct_columns_map(self):
        """Test fit datetime checker saves minimum dates for correct columns after fit called"""

        df = data_generators_p.create_df_2()

        df["d"] = pd.to_datetime(
            [
                "01/02/2020",
                "01/02/2021",
                "08/04/2019",
                "01/03/2020",
                "29/03/2019",
                "15/10/2018",
                "24/07/2020",
            ]
        )
        df["e"] = pd.to_datetime(
            [
                "01/02/2020",
                "01/02/2021",
                "08-04-2019",
                "01/03/2020",
                "29/03/2019",
                "15/10/2018",
                "24/07/2020",
            ]
        )

        x = InputChecker(datetime_columns=["d", "e"])

        x.fit(df)

        assert list(x.datetime_values.keys()) == [
            "d",
            "e",
        ], f"incorrect values returned from _fit_datetime_checker - expected: ['d', 'e'] but got: {list(x.datetime_values.keys())} "

    def test_correct_datetime_values_identified(self):
        """Test fit datetime checker identifies correct minimum bound after fit called"""

        df = data_generators_p.create_df_2()

        df["d"] = pd.to_datetime(
            [
                "01/02/2020",
                "01/02/2021",
                "08/04/2019",
                "01/03/2020",
                "29/03/2019",
                "15/10/2018",
                "24/07/2020",
            ]
        )

        x = InputChecker(datetime_columns=["d"])

        x.fit(df)

        expected_min_d = pd.to_datetime("15/10/2018").date()

        actual_min_d = x.datetime_values["d"]["minimum"]
        actual_max_d = x.datetime_values["d"]["maximum"]

        assert (
            actual_min_d == expected_min_d
        ), f"incorrect values returned from _fit_datetime_checker - expected: {expected_min_d}, but got: {actual_min_d}"

        assert (
            actual_max_d is None
        ), f"incorrect values returned from _fit_datetime_checker - expected: None, but got: {actual_max_d}"

    def test_correct_datetime_values_identified_dict(self):
        """Test fit datetime checker identifies correct range values after fit called when inputting a dictionary"""

        df = data_generators_p.create_df_2()

        df["d"] = pd.to_datetime(
            [
                "01/02/2020",
                "01/02/2021",
                "08/04/2019",
                "01/03/2020",
                "29/03/2019",
                "15/10/2018",
                "24/07/2020",
            ]
        )

        datetime_dict = {"d": {"maximum": True, "minimum": True}}

        x = InputChecker(datetime_columns=datetime_dict)

        x.fit(df)

        expected_min_d = pd.to_datetime("15/10/2018").date()
        expected_max_d = pd.to_datetime("01/02/2021").date()

        actual_min_d = x.datetime_values["d"]["minimum"]
        actual_max_d = x.datetime_values["d"]["maximum"]

        assert (
            actual_min_d == expected_min_d
        ), f"incorrect values returned from _fit_datetime_checker - expected: {expected_min_d}, but got: {actual_min_d}"

        assert (
            actual_max_d == expected_max_d
        ), f"incorrect values returned from _fit_datetime_checker - expected: {expected_max_d}, but got: {actual_max_d}"


class TestFit(object):
    """Tests for InputChecker.fit()."""

    def test_arguments(self):
        """Test that InputChecker fit has expected arguments."""

        h.test_function_arguments(
            func=InputChecker.fit,
            expected_arguments=["self", "X", "y"],
            expected_default_values=(None,),
        )

    def test_super_fit_called(self, mocker):
        """Test that BaseTransformer fit called."""

        expected_call_args = {
            0: {"args": (data_generators_p.create_df_2(), None), "kwargs": {}}
        }

        df = data_generators_p.create_df_2()

        x = InputChecker(columns=["a"])

        with h.assert_function_call(
            mocker, tubular.base.BaseTransformer, "fit", expected_call_args
        ):
            x.fit(df)

    def test_all_columns_selected(self):
        """Test fit selects all columns when columns parameter set to None"""

        df = data_generators_p.create_df_2()

        x = InputChecker(columns=None)

        assert (
            x.columns is None
        ), f"incorrect columns attribute before fit when columns parameter set to None - expected: None but got: {x.columns}"

        x.fit(df)

        assert x.columns == [
            "a",
            "b",
            "c",
        ], f"incorrect columns identified when columns parameter set to None - expected: ['a', 'b', 'c'] but got: {x.columns}"

    def test_fit_returns_self(self):
        """Test fit returns self?"""

        df = data_generators_p.create_df_2()

        x = InputChecker()

        x_fitted = x.fit(df)

        assert x_fitted is x, "Returned value from InputChecker.fit not as expected."

    def test_no_optional_calls_fit(self):
        """Test numerical_values and expected_values is not present after fit if parameters set to None"""

        x = InputChecker(
            numerical_columns=None, categorical_columns=None, datetime_columns=None
        )

        df = data_generators_p.create_df_2()

        x.fit(df)

        assert (
            hasattr(x, "numerical_values") is False
        ), "numerical_values attribute present with numerical_columns set to None"

        assert (
            hasattr(x, "expected_values") is False
        ), "expected_values attribute present with categorical_columns set to None"

        assert (
            hasattr(x, "datetime_values") is False
        ), "datetime_values attribute present with datetime_columns set to None"

    def test_compulsory_checks_generated_with_no_optional_calls_fit(self):
        """Test null_map and column_classes are present after fit when optional parameters set to None"""

        x = InputChecker(
            numerical_columns=None, categorical_columns=None, datetime_columns=None
        )

        df = data_generators_p.create_df_2()

        x.fit(df)

        assert (
            hasattr(x, "null_map") is True
        ), "null_map attribute not present when optional checks set to None"

        assert (
            hasattr(x, "column_classes") is True
        ), "column_classes attribute not present when optional checks set to None"

    def test_all_checks_generated(self):
        """Test all checks are generated when all optional parameters set"""

        x = InputChecker(
            columns=["a", "b", "c", "d"],
            numerical_columns=["a"],
            categorical_columns=["b", "c"],
            datetime_columns=["d"],
        )

        df = data_generators_p.create_df_2()

        df["d"] = pd.to_datetime(
            [
                "01/02/2020",
                "01/02/2021",
                "08/04/2019",
                "01/03/2020",
                "29/03/2019",
                "15/10/2018",
                "24/07/2020",
            ]
        )

        x.fit(df)

        assert (
            hasattr(x, "numerical_values") is True
        ), "numerical_values attribute not present after fit with numerical_columns set"

        assert (
            hasattr(x, "expected_values") is True
        ), "expected_values attribute not present after fit with categorical_columns set"

        assert (
            hasattr(x, "datetime_values") is True
        ), "expected_values attribute not present after fit with datetime_columns set"

        assert (
            hasattr(x, "null_map") is True
        ), "null_map attribute not present after fit"

        assert (
            hasattr(x, "column_classes") is True
        ), "column_classes attribute not present after fit"

    def test_check_df_is_empty_called(self, mocker):
        """Test check is df empty is called by the fit method."""

        x = InputChecker(
            columns=["a", "b", "c"],
            numerical_columns=["a"],
            categorical_columns=["b", "c"],
        )

        df = data_generators_p.create_df_2()

        spy = mocker.spy(input_checker.checker.InputChecker, "_df_is_empty")

        x.fit(df)

        assert (
            spy.call_count == 1
        ), "unexpected number of calls to InputChecker._df_is_empty with fit"

        call_0_args = spy.call_args_list[0]
        call_0_pos_args = call_0_args[0]

        expected_pos_args_0 = (x, "input dataframe", df)

        assert (
            expected_pos_args_0 == call_0_pos_args
        ), "positional args unexpected in _df_is_empty call for dataframe argument"


class TestTransformTypeChecker(object):
    """Tests for InputChecker._transform_type_checker()."""

    def test_arguments(self):
        """Test that InputChecker _transform_type_checker has expected arguments."""

        h.test_function_arguments(
            func=InputChecker._transform_type_checker,
            expected_arguments=["self", "X", "batch_mode"],
            expected_default_values=(False,),
        )

    def test_check_fitted_called(self, mocker):
        """Test that transform calls BaseTransformer.check_is_fitted."""

        expected_call_args = {0: {"args": (["column_classes"],), "kwargs": {}}}

        x = InputChecker()

        df = data_generators_p.create_df_2()

        x.fit(df)

        with h.assert_function_call(
            mocker, tubular.base.BaseTransformer, "check_is_fitted", expected_call_args
        ):
            x._transform_type_checker(df)

    def test_transform_returns_failed_checks_dict(self):
        """Test _transform_type_checker returns results dictionary"""

        df = data_generators_p.create_df_2()

        x = InputChecker()

        x.fit(df)

        type_checker_failed_checks = x._transform_type_checker(df)

        assert isinstance(
            type_checker_failed_checks, dict
        ), f"incorrect type results type identified - expected: dict but got: {type(type_checker_failed_checks)}"

    def test_transform_passes(self):
        """Test _transform_type_checker passes all the checks on the training dataframe"""

        df = data_generators_p.create_df_2()

        x = InputChecker()

        x.fit(df)

        type_checker_failed_checks = x._transform_type_checker(df)

        assert (
            type_checker_failed_checks == {}
        ), f"Type checker found failed tests - {list(type_checker_failed_checks.keys())}"

    def test_transform_passes_column_all_nulls(self):
        """Test _transform_type_checker passes all the checks on the training dataframe when a column contains only nulls"""

        df = data_generators_p.create_df_2()

        x = InputChecker()

        x.fit(df)

        df["c"] = np.nan

        type_checker_failed_checks = x._transform_type_checker(df)

        assert (
            type_checker_failed_checks == {}
        ), f"Type checker found failed tests - {list(type_checker_failed_checks.keys())}"

    def test_transform_captures_failed_test(self):
        """Test _transform_type_checker captures a failed check"""

        df = data_generators_p.create_df_2()

        x = InputChecker()

        x.fit(df)

        exp_type = df["a"].dtypes

        df.loc[5, "a"] = "a"

        type_checker_failed_checks = x._transform_type_checker(df)

        assert (
            type_checker_failed_checks["a"]["actual"] == df["a"].dtypes
        ), f"incorrect values saved to type_checker_failed_checks bad types - expected: [{type('a')}] but got: {type_checker_failed_checks['a']['types']}"

        assert (
            type_checker_failed_checks["a"]["expected"] == exp_type
        ), f"incorrect values saved to type_checker_failed_checks expected types - expected: [{exp_type}] but got: {type_checker_failed_checks['a']['types']}"

    def test_transform_passes_batch_mode(self):
        """Test _transform_type_checker passes all the checks on the training dataframe"""

        df = data_generators_p.create_df_2()

        x = InputChecker()

        x.fit(df)

        type_checker_failed_checks = x._transform_type_checker(df, batch_mode=True)

        assert (
            type_checker_failed_checks == {}
        ), f"Type checker found failed tests - {list(type_checker_failed_checks.keys())}"

    def test_transform_captures_failed_test_batch_mode(self):
        """Test _transform_type_checker handles mixed types"""

        df = data_generators_p.create_df_2()

        df["d"] = pd.to_datetime(
            [
                "01/02/2020",
                "01/02/2021",
                "08/04/2019",
                "01/03/2020",
                "29/03/2019",
                "15/10/2018",
                "24/07/2020",
            ]
        )

        print(df)

        x = InputChecker()

        x.fit(df)

        exp_type = df["a"].dtypes
        print(exp_type)

        df.loc[5, "a"] = "a"
        df.loc[1, "d"] = "a"
        df.loc[3, "b"] = 1

        type_checker_failed_checks = x._transform_type_checker(df, batch_mode=True)

        expected_output = {
            "a": {"idxs": [5], "actual": {5: "str"}, "expected": "float"},
            "b": {"idxs": [3], "actual": {3: "int"}, "expected": "str"},
            "d": {"idxs": [1], "actual": {1: "str"}, "expected": "Timestamp"},
        }

        for k, v in expected_output.items():

            assert (
                k in type_checker_failed_checks.keys()
            ), f"expected column {k} in type_checker_failed_checks output"

            assert (
                type(type_checker_failed_checks[k]) == dict
            ), f"expected dict for column {k} in type_checker_failed_checks output"

            for sub_k, sub_v in expected_output[k].items():

                assert (
                    sub_k in type_checker_failed_checks[k].keys()
                ), f"expected {sub_k} as dict key in type_checker_failed_checks output"

                assert (
                    sub_v == type_checker_failed_checks[k][sub_k]
                ), f"expected {sub_v} as value for {sub_k} in column {k} output of type_checker_failed_checks output"


class TestTransformNullChecker(object):
    """Tests for InputChecker._transform_null_checker()."""

    def test_arguments(self):
        """Test that InputChecker _transform_null_checker has expected arguments."""

        h.test_function_arguments(
            func=InputChecker._transform_null_checker, expected_arguments=["self", "X"]
        )

    def test_check_fitted_called(self, mocker):
        """Test that transform calls BaseTransformer.check_is_fitted."""

        expected_call_args = {0: {"args": (["null_map"],), "kwargs": {}}}

        x = InputChecker()

        df = data_generators_p.create_df_2()

        x.fit(df)

        with h.assert_function_call(
            mocker, tubular.base.BaseTransformer, "check_is_fitted", expected_call_args
        ):
            x._transform_null_checker(df)

    def test_transform_returns_failed_checks_dict(self):
        """Test _transform_null_checker returns results dictionary"""

        df = data_generators_p.create_df_2()

        x = InputChecker()

        x.fit(df)

        null_checker_failed_checks = x._transform_null_checker(df)

        assert isinstance(
            null_checker_failed_checks, dict
        ), f"incorrect null results type identified - expected: dict but got: {type(null_checker_failed_checks)}"

    def test_transform_passes(self):
        """Test _transform_null_checker passes all the checks on the training dataframe"""

        df = data_generators_p.create_df_2()

        df["b"] = df["b"].fillna("a")

        x = InputChecker()

        x.fit(df)

        null_checker_failed_checks = x._transform_null_checker(df)

        assert (
            null_checker_failed_checks == {}
        ), f"Null checker found failed tests - {list(null_checker_failed_checks.keys())}"

    def test_transform_captures_failed_test(self):
        """Test _transform_null_checker captures a failed check"""

        df = data_generators_p.create_df_2()

        df["b"] = df["b"].fillna("a")

        x = InputChecker()

        x.fit(df)

        df.loc[5, "b"] = np.nan

        null_checker_failed_checks = x._transform_null_checker(df)

        assert null_checker_failed_checks["b"] == [
            5
        ], f"incorrect values saved to value_checker_failed_checks - expected: [5] but got: {null_checker_failed_checks['b']}"


class TestTransformNumericalChecker(object):
    """Tests for InputChecker._transform_numerical_checker()."""

    def test_arguments(self):
        """Test that InputChecker _transform_numerical_checker has expected arguments."""

        h.test_function_arguments(
            func=InputChecker._transform_numerical_checker,
            expected_arguments=["self", "X", "type_fails", "batch_mode"],
            expected_default_values=(
                {},
                False,
            ),
        )

    def test_check_fitted_called(self, mocker):
        """Test that transform calls BaseTransformer.check_is_fitted."""

        expected_call_args = {0: {"args": (["numerical_values"],), "kwargs": {}}}

        x = InputChecker(numerical_columns=["a"])

        df = data_generators_p.create_df_2()

        x.fit(df)

        with h.assert_function_call(
            mocker, tubular.base.BaseTransformer, "check_is_fitted", expected_call_args
        ):
            x._transform_numerical_checker(df, {})

    def test_transform_returns_failed_checks_dict(self):
        """Test _transform_numerical_checker returns results dictionary"""

        df = data_generators_p.create_df_2()

        x = InputChecker(numerical_columns=["a"])

        x.fit(df)

        numerical_checker_failed_checks = x._transform_numerical_checker(df, {})

        assert isinstance(
            numerical_checker_failed_checks, dict
        ), f"incorrect numerical results type identified - expected: dict but got: {type(numerical_checker_failed_checks)}"

    def test_transform_passes(self):
        """Test _transform_numerical_checker passes all the numerical checks on the training dataframe"""

        df = data_generators_p.create_df_2()

        x = InputChecker(numerical_columns=["a"])

        x.fit(df)

        numerical_checker_failed_checks = x._transform_numerical_checker(df, {})

        assert (
            numerical_checker_failed_checks == {}
        ), f"Numerical checker found failed tests - {list(numerical_checker_failed_checks.keys())}"

    def test_transform_captures_failed_test(self):
        """Test _transform_numerical_checker captures a failed check"""

        df = data_generators_p.create_df_2()

        x = InputChecker(numerical_columns=["a"])

        x.fit(df)

        df.loc[0, "a"] = -1
        df.loc[5, "a"] = 7

        numerical_checker_failed_checks = x._transform_numerical_checker(df, {})

        expected_max = {5: 7.0}
        expected_min = {0: -1.0}

        assert (
            numerical_checker_failed_checks["a"]["maximum"] == expected_max
        ), f"incorrect values saved to numerical_checker_failed_checks - expected: {expected_max} but got: {numerical_checker_failed_checks['a']['maximum']}"

        assert (
            numerical_checker_failed_checks["a"]["minimum"] == expected_min
        ), f"incorrect values saved to numerical_checker_failed_checks - expected: {expected_min} but got: {numerical_checker_failed_checks['a']['minimum']}"

    def test_transform_captures_failed_test_only_maximum(self):
        """Test _transform_numerical_checker captures a failed check when the check includes a maximum value but no minimum value"""

        df = data_generators_p.create_df_2()

        numerical_dict = {}
        numerical_dict["a"] = {}
        numerical_dict["a"]["maximum"] = True
        numerical_dict["a"]["minimum"] = False

        x = InputChecker(numerical_columns=numerical_dict)

        x.fit(df)

        df.loc[0, "a"] = -1
        df.loc[5, "a"] = 7

        expected_max = {5: 7.0}

        numerical_checker_failed_checks = x._transform_numerical_checker(df, {})

        assert (
            numerical_checker_failed_checks["a"]["maximum"] == expected_max
        ), f"incorrect values saved to numerical_checker_failed_checks - expected: {expected_max} but got: {numerical_checker_failed_checks['a']['maximum']}"

        assert (
            "minimum" not in numerical_checker_failed_checks["a"]
        ), "No minimum value results expected given input the numerical dict"

    def test_transform_captures_failed_test_only_minimum(self):
        """Test _transform_numerical_checker captures a failed check when the check includes a minimum value but no maximum value"""

        df = data_generators_p.create_df_2()

        numerical_dict = {}
        numerical_dict["a"] = {}
        numerical_dict["a"]["maximum"] = False
        numerical_dict["a"]["minimum"] = True

        x = InputChecker(numerical_columns=numerical_dict)

        x.fit(df)

        df.loc[0, "a"] = -1
        df.loc[5, "a"] = 7

        numerical_checker_failed_checks = x._transform_numerical_checker(df, {})

        expected_min = {0: -1.0}

        assert (
            numerical_checker_failed_checks["a"]["minimum"] == expected_min
        ), f"incorrect values saved to numerical_checker_failed_checks - expected: {expected_min} but got: {numerical_checker_failed_checks['a']['minimum']}"

        assert (
            "maximum" not in numerical_checker_failed_checks["a"]
        ), "No maximum value results expected given input the numerical dict"

    def test_transform_skips_failed_type_checks_batch_mode(self):
        """Test _transform_numerical_checker skips checks for rows which aren't numerical
        when operating in batch mode"""

        df = data_generators_p.create_df_2()

        x = InputChecker(numerical_columns=["a"])

        x.fit(df)

        df.loc[4, "a"] = "z"
        df.loc[1, "a"] = 1
        df.loc[2, "a"] = 100

        type_fails_dict = {
            "a": {"idxs": [1, 4], "actual": {1: "int", 4: "str"}, "expected": "float"}
        }

        expected_output = {"a": {"max idxs": [2], "maximum": {2: 100}}}

        numerical_checker_failed_checks = x._transform_numerical_checker(
            df, type_fails_dict, batch_mode=True
        )

        h.assert_equal_dispatch(
            actual=numerical_checker_failed_checks,
            expected=expected_output,
            msg="rows failing type check have not been removed by _transform_numerical_checker",
        )

    def test_transform_skips_failed_type_checks(self):
        """Test _transform_numerical_checker skips checks for columns which aren't numerical
        when not operating in batch mode"""

        df = data_generators_p.create_df_2()

        x = InputChecker(numerical_columns=["a"])

        x.fit(df)

        # Case 1: check will not be performed as column a is not numerical

        df_test = pd.DataFrame({"a": ["z", "zz", "zzz"]})

        type_fails_dict = {
            "a": {"actual": df_test["a"].dtypes, "expected": df["a"].dtypes}
        }

        numerical_checker_failed_checks = x._transform_numerical_checker(
            df_test, type_fails_dict, batch_mode=False
        )

        h.assert_equal_dispatch(
            actual=numerical_checker_failed_checks,
            expected={},
            msg="rows failing type check have not been removed by _transform_numerical_checker",
        )

        # Case 2: column a should still get checked because even though type does not match,
        # int != float the column is still numerical

        df_test2 = pd.DataFrame({"a": [5, 3, 222]})

        type_fails_dict2 = {
            "a": {"actual": df_test2["a"].dtypes, "expected": df["a"].dtypes}
        }

        numerical_checker_failed_checks2 = x._transform_numerical_checker(
            df_test2, type_fails_dict2, batch_mode=False
        )

        h.assert_equal_dispatch(
            actual=numerical_checker_failed_checks2,
            expected={"a": {"max idxs": [2], "maximum": {2: 222}}},
            msg="rows failing type check have not been removed by _transform_numerical_checker",
        )


class TestTransformValueChecker(object):
    """Tests for InputChecker._transform_value_checker()."""

    def test_arguments(self):
        """Test that InputChecker _transform_value_checker has expected arguments."""

        h.test_function_arguments(
            func=InputChecker._transform_value_checker, expected_arguments=["self", "X"]
        )

    def test_check_fitted_called(self, mocker):
        """Test that transform calls BaseTransformer.check_is_fitted."""

        expected_call_args = {0: {"args": (["expected_values"],), "kwargs": {}}}

        x = InputChecker(categorical_columns=["b", "c"])

        df = data_generators_p.create_df_2()

        x.fit(df)

        with h.assert_function_call(
            mocker, tubular.base.BaseTransformer, "check_is_fitted", expected_call_args
        ):
            x._transform_value_checker(df)

    def test_transform_returns_failed_checks_dict(self):
        """Test _transform_value_checker returns results dictionary"""

        df = data_generators_p.create_df_2()

        x = InputChecker(categorical_columns=["b", "c"])

        x.fit(df)

        value_checker_failed_checks = x._transform_value_checker(df)

        assert isinstance(
            value_checker_failed_checks, dict
        ), f"incorrect numerical results type identified - expected: dict but got: {type(value_checker_failed_checks)}"

    def test_transform_passes(self):
        """Test _transform_value_checker passes all the categorical checks on the training dataframe"""

        df = data_generators_p.create_df_2()

        x = InputChecker(categorical_columns=["b", "c"])

        x.fit(df)

        value_checker_failed_checks = x._transform_value_checker(df)

        assert (
            value_checker_failed_checks == {}
        ), f"Categorical checker found failed tests - {list(value_checker_failed_checks.keys())}"

    def test_transform_captures_failed_test(self):
        """Test _transform_value_checker captures a failed check"""

        df = data_generators_p.create_df_2()

        x = InputChecker(categorical_columns=["b", "c"])

        x.fit(df)

        df.loc[5, "b"] = "u"

        value_checker_failed_checks = x._transform_value_checker(df)

        assert value_checker_failed_checks["b"]["values"] == [
            "u"
        ], f"incorrect values saved to value_checker_failed_checks - expected: ['u'] but got: {value_checker_failed_checks['b']['values']}"

        assert value_checker_failed_checks["b"]["idxs"] == [
            5
        ], f"incorrect values saved to value_checker_failed_checks - expected: [5] but got: {value_checker_failed_checks['b']['idxs']}"


class TestTransformDatetimeChecker(object):
    """Tests for InputChecker._transform_datetime_checker()."""

    def test_arguments(self):
        """Test that InputChecker _transform_datetime_checker has expected arguments."""

        h.test_function_arguments(
            func=InputChecker._transform_datetime_checker,
            expected_arguments=["self", "X", "type_fails", "batch_mode"],
            expected_default_values=(
                {},
                False,
            ),
        )

    def test_check_fitted_called(self, mocker):
        """Test that transform calls BaseTransformer.check_is_fitted."""

        expected_call_args = {0: {"args": (["datetime_values"],), "kwargs": {}}}

        x = InputChecker(datetime_columns=["d"])

        df = data_generators_p.create_df_2()
        df["d"] = pd.to_datetime(
            [
                "01/02/2020",
                "01/02/2021",
                "08/04/2019",
                "01/03/2020",
                "29/03/2019",
                "15/10/2018",
                np.NAN,
            ]
        )

        x.fit(df)

        with h.assert_function_call(
            mocker, tubular.base.BaseTransformer, "check_is_fitted", expected_call_args
        ):
            x._transform_datetime_checker(df, {})

    def test_transform_returns_failed_checks_dict(self):
        """Test _transform_datetime_checker returns results dictionary"""

        df = data_generators_p.create_df_2()

        df["d"] = pd.to_datetime(
            [
                "01/02/2020",
                "01/02/2021",
                "08/04/2019",
                "01/03/2020",
                "29/03/2019",
                "15/10/2018",
                np.NAN,
            ]
        )

        x = InputChecker(datetime_columns=["d"])

        x.fit(df)

        datetime_checker_failed_checks = x._transform_datetime_checker(df, {})

        assert isinstance(
            datetime_checker_failed_checks, dict
        ), f"incorrect datetime results type identified - expected: dict but got: {type(datetime_checker_failed_checks)}"

    def test_transform_passes(self):
        """Test _transform_datetime_checker passes all the numerical checks on the training dataframe"""

        df = data_generators_p.create_df_2()

        df["d"] = pd.to_datetime(
            [
                "01/02/2020",
                "01/02/2021",
                "08/04/2019",
                "01/03/2020",
                "29/03/2019",
                "15/10/2018",
                np.NAN,
            ]
        )

        x = InputChecker(datetime_columns=["d"])

        x.fit(df)

        datetime_checker_failed_checks = x._transform_datetime_checker(df, {})

        assert (
            datetime_checker_failed_checks == {}
        ), f"Datetime checker found failed tests - {list(datetime_checker_failed_checks.keys())}"

    def test_transform_captures_failed_test(self):
        """Test _transform_datetime_checker captures a failed check"""

        df = data_generators_p.create_df_2()

        df["d"] = pd.to_datetime(
            [
                "01/02/2020",
                "01/02/2021",
                "08/04/2019",
                "01/03/2020",
                "29/03/2019",
                "15/10/2018",
                np.NAN,
            ]
        )

        x = InputChecker(datetime_columns=["d"])

        x.fit(df)

        outliers_1 = pd.to_datetime("15/09/2017", utc=False)
        outliers_2 = pd.to_datetime("13/09/2017", utc=False)

        df.loc[0, "d"] = outliers_1
        df.loc[1, "d"] = outliers_2

        datetime_checker_failed_checks = x._transform_datetime_checker(df, {})

        results = datetime_checker_failed_checks["d"]["minimum"]

        assert results[0] == outliers_1, (
            f"incorrect values saved to datetime_checker_failed_checks - "
            f"expected: {outliers_1} but got: {results[0]} "
        )

        assert results[1] == outliers_2, (
            f"incorrect values saved to datetime_checker_failed_checks - "
            f"expected: {outliers_2} but got: {results[1]} "
        )

    def test_transform_captures_failed_test_both_minimum_and_maximum(self):
        """Test _transform_datetime_checker captures a failed check when the check includes a maximum value and a
        minimum value"""

        df = data_generators_p.create_df_2()

        df["d"] = pd.to_datetime(
            [
                "01/02/2020",
                "01/02/2021",
                "08/04/2019",
                "01/03/2020",
                "29/03/2019",
                "15/10/2018",
                "24/07/2020",
            ]
        )

        datetime_dict = {"d": {"maximum": True, "minimum": True}}

        x = InputChecker(datetime_columns=datetime_dict)

        x.fit(df)

        lower_outliers = pd.to_datetime("15/09/2017", utc=False)
        upper_outliers = pd.to_datetime("20/01/2021", utc=False)

        df.loc[0, "d"] = lower_outliers
        df.loc[5, "d"] = upper_outliers

        datetime_checker_failed_checks = x._transform_datetime_checker(df, {})

        expected_min = {0: lower_outliers}
        expected_max = {5: upper_outliers}

        assert datetime_checker_failed_checks["d"]["maximum"] == expected_max, (
            f"incorrect values saved to "
            f"datetime_checker_failed_checks - "
            f"expected: {expected_max} but got: "
            f"{datetime_checker_failed_checks['d']['maximum']} "
        )

        assert datetime_checker_failed_checks["d"]["minimum"] == expected_min, (
            f"incorrect values saved to "
            f"datetime_checker_failed_checks - "
            f"expected: {expected_min} but got: "
            f"{datetime_checker_failed_checks['d']['minimum']} "
        )

    def test_transform_skips_failed_type_checks_batch_mode(self):
        """Test _transform_datetime_checker skips checks for rows which aren't datetime type
        when operating in batch mode"""

        df = data_generators_p.create_df_2()

        df["d"] = pd.to_datetime(
            [
                "01/02/2020",
                "01/02/2021",
                "08/04/2019",
                "01/03/2020",
                "29/03/2019",
                "15/10/2018",
                "24/07/2020",
            ]
        )

        x = InputChecker(datetime_columns=["d"])

        x.fit(df)

        df.loc[3, "d"] = 1
        df.loc[4, "d"] = "z"
        df.loc[5, "d"] = pd.to_datetime("20/09/2011", utc=False)

        type_fails_dict = {
            "d": {
                "idxs": [3, 4],
                "actual": {3: "int", 4: "str"},
                "expected": "Timestamp",
            }
        }

        datetime_checker_failed_checks = x._transform_datetime_checker(
            df, type_fails_dict, batch_mode=True
        )

        h.assert_equal_dispatch(
            actual=datetime_checker_failed_checks,
            expected={
                "d": {
                    "minimum": {5: pd.to_datetime("20/09/2011", utc=False)},
                    "min idxs": [5],
                }
            },
            msg="rows failing type check have not been removed by _transform_datetime_checker",
        )

    def test_transform_skips_failed_type_checks(self):
        """Test _transform_datetime_checker skips checks for columns which aren't datetime
        when not operating in batch mode"""

        df = data_generators_p.create_df_2()

        df["d"] = pd.to_datetime(
            [
                "01/02/2020",
                "01/02/2021",
                "08/04/2019",
                "01/03/2020",
                "29/03/2019",
                "15/10/2018",
                "24/07/2020",
            ]
        )

        x = InputChecker(datetime_columns=["d"])

        x.fit(df)

        df_test = pd.DataFrame({"d": ["z", "zz", "zzz"]})

        type_fails_dict = {
            "d": {"actual": df_test["d"].dtypes, "expected": df["d"].dtypes}
        }

        datetime_checker_failed_checks = x._transform_datetime_checker(
            df_test, type_fails_dict, batch_mode=False
        )

        h.assert_equal_dispatch(
            actual=datetime_checker_failed_checks,
            expected={},
            msg="rows failing type check have not been removed by _transform_datetime_checker",
        )


class TestTransform(object):
    """Tests for InputChecker.transform()."""

    def test_arguments(self):
        """Test that transform has expected arguments."""
        h.test_function_arguments(
            func=InputChecker.transform,
            expected_arguments=["self", "X", "batch_mode"],
            expected_default_values=(False,),
        )

    def test_super_transform_called(self, mocker):
        """Test super transform is called by the transform method."""

        x = InputChecker(
            columns=["a", "b", "c", "d"],
            numerical_columns=["a"],
            categorical_columns=["b", "c"],
            datetime_columns=["d"],
        )

        df = data_generators_p.create_df_2()

        df["d"] = pd.to_datetime(
            [
                "01/02/2020",
                "01/02/2021",
                "08/04/2019",
                "01/03/2020",
                "29/03/2019",
                "15/10/2018",
                "24/07/2020",
            ]
        )

        x.fit(df)

        spy = mocker.spy(tubular.base.BaseTransformer, "transform")

        df = x.transform(df)

        assert (
            spy.call_count == 1
        ), "unexpected number of calls to tubular.base.BaseTransformer.transform with transform"

    def test_transform_returns_df(self):
        """Test fit returns df"""

        df = data_generators_p.create_df_2()

        df["d"] = pd.to_datetime(
            [
                "01/02/2020",
                "01/02/2021",
                "08/04/2019",
                "01/03/2020",
                "29/03/2019",
                "15/10/2018",
                "24/07/2020",
            ]
        )

        x = InputChecker()

        x.fit(df)

        df_transformed = x.transform(df)

        assert df_transformed.equals(
            df
        ), "Returned value from InputChecker.transform not as expected."

    def test_batch_mode_transform_returns_df(self):
        """Test fit returns df"""

        df = data_generators_p.create_df_2()

        df["d"] = pd.to_datetime(
            [
                "01/02/2020",
                "01/02/2021",
                "08/04/2019",
                "01/03/2020",
                "29/03/2019",
                "15/10/2018",
                "24/07/2020",
            ]
        )

        x = InputChecker()

        x.fit(df)

        df_transformed, bad_df = x.transform(df, batch_mode=True)

        assert df_transformed.equals(
            df
        ), "Returned value from InputChecker.transform not as expected."

        h.assert_equal_dispatch(
            expected=df,
            actual=df_transformed,
            msg="Returned df of passed rows from InputChecker.transform not as expected.",
        )

        h.assert_equal_dispatch(
            expected=pd.DataFrame(
                columns=df.columns.values.tolist() + ["failed_checks"]
            ),
            actual=bad_df,
            msg="Returned df of failed rows from InputChecker.transform not as expected.",
        )

    def test_check_df_is_empty_called(self, mocker):
        """Test check is df empty is called by the transform method."""

        x = InputChecker(
            columns=["a", "b", "c", "d"],
            numerical_columns=["a"],
            categorical_columns=["b", "c"],
            datetime_columns=["d"],
        )

        df = data_generators_p.create_df_2()

        df["d"] = pd.to_datetime(
            [
                "01/02/2020",
                "01/02/2021",
                "08/04/2019",
                "01/03/2020",
                "29/03/2019",
                "15/10/2018",
                "24/07/2020",
            ]
        )

        x.fit(df)

        spy = mocker.spy(input_checker.checker.InputChecker, "_df_is_empty")

        df = x.transform(df)

        assert (
            spy.call_count == 1
        ), "unexpected number of calls to InputChecker._df_is_empty with transform"

        call_0_args = spy.call_args_list[0]
        call_0_pos_args = call_0_args[0]

        expected_pos_args_0 = (x, "scoring dataframe", df)

        h.assert_equal_dispatch(
            expected=expected_pos_args_0,
            actual=call_0_pos_args,
            msg="positional args unexpected in _df_is_empty call for scoring dataframe argument",
        )

    def test_non_optional_transforms_always_called(self, mocker):
        """Test non-optional checks are called by the transform method irrespective of categorical_columns,
        numerical_columns & datetime_columns values."""

        x = InputChecker(
            numerical_columns=None, categorical_columns=None, datetime_columns=None
        )

        df = data_generators_p.create_df_2()

        df["d"] = pd.to_datetime(
            [
                "01/02/2020",
                "01/02/2021",
                "08/04/2019",
                "01/03/2020",
                "29/03/2019",
                "15/10/2018",
                "24/07/2020",
            ]
        )

        x.fit(df)

        spy_null = mocker.spy(
            input_checker.checker.InputChecker, "_transform_null_checker"
        )

        spy_type = mocker.spy(
            input_checker.checker.InputChecker, "_transform_type_checker"
        )

        df = x.transform(df)

        assert spy_null.call_count == 1, (
            "unexpected number of calls to _transform_null_checker with transform when numerical_columns and "
            "categorical_columns set to None "
        )

        assert spy_type.call_count == 1, (
            "unexpected number of calls to _transform_type_checker with transform when numerical_columns and "
            "categorical_columns set to None "
        )

    def test_optional_transforms_not_called(self, mocker):
        """Test optional checks are not called by the transform method."""

        x = InputChecker(
            numerical_columns=None, categorical_columns=None, datetime_columns=None
        )

        df = data_generators_p.create_df_2()

        df["d"] = pd.to_datetime(
            [
                "01/02/2020",
                "01/02/2021",
                "08/04/2019",
                "01/03/2020",
                "29/03/2019",
                "15/10/2018",
                "24/07/2020",
            ]
        )

        x.fit(df)

        spy_numerical = mocker.spy(
            input_checker.checker.InputChecker, "_transform_numerical_checker"
        )

        spy_categorical = mocker.spy(
            input_checker.checker.InputChecker, "_transform_value_checker"
        )

        spy_datetime = mocker.spy(
            input_checker.checker.InputChecker, "_transform_datetime_checker"
        )

        df = x.transform(df)

        assert (
            spy_numerical.call_count == 0
        ), "unexpected number of calls to _transform_numerical_checker with transform when numerical_columns set to None"

        assert (
            spy_categorical.call_count == 0
        ), "unexpected number of calls to _transform_value_checker with transform when categorical_columns set to None"

        assert (
            spy_datetime.call_count == 0
        ), "unexpected number of calls to _transform_datetime_checker with transform when datetime_columns set to None"

    def test_raise_exception_if_checks_fail_called_no_optionals(self, mocker):
        """Test raise exception is called by the transform method when categorical, numerical_& datetime columns set
        to None."""

        x = InputChecker()

        df = data_generators_p.create_df_2()

        x.fit(df)

        spy = mocker.spy(
            input_checker.checker.InputChecker, "raise_exception_if_checks_fail"
        )

        df = x.transform(df)

        assert (
            spy.call_count == 1
        ), "unexpected number of calls to InputChecker.raise_exception_if_checks_fail with transform"

        call_0_args = spy.call_args_list[0]
        call_0_pos_args = call_0_args[0]

        value_failed_checks = {}
        numerical_failed_checks = {}
        datetime_failed_checks = {}
        type_failed_checks = x._transform_type_checker(df)
        null_failed_checks = x._transform_null_checker(df)

        expected_pos_args_0 = (
            x,
            type_failed_checks,
            null_failed_checks,
            value_failed_checks,
            numerical_failed_checks,
            datetime_failed_checks,
        )

        assert (
            expected_pos_args_0 == call_0_pos_args
        ), "positional args unexpected in raise_exception_if_checks_fail call in transform method"

    def test_raise_exception_if_checks_fail_called_all_checks(self, mocker):
        """Test raise exception is called by the transform method when categorical_columns and numerical_columns set
        to None."""

        x = InputChecker(
            numerical_columns=["a"],
            categorical_columns=["b", "c"],
            datetime_columns=["d"],
        )

        df = data_generators_p.create_df_2()

        df["d"] = pd.to_datetime(
            [
                "01/02/2020",
                "01/02/2021",
                "08/04/2019",
                "01/03/2020",
                "29/03/2019",
                "15/10/2018",
                "24/07/2020",
            ]
        )

        x.fit(df)

        spy = mocker.spy(
            input_checker.checker.InputChecker, "raise_exception_if_checks_fail"
        )

        df = x.transform(df)

        assert (
            spy.call_count == 1
        ), "unexpected number of calls to InputChecker.raise_exception_if_checks_fail with transform"

        call_0_args = spy.call_args_list[0]
        call_0_pos_args = call_0_args[0]

        value_failed_checks = x._transform_value_checker(df)
        numerical_failed_checks = x._transform_numerical_checker(df)
        datetime_failed_checks = x._transform_datetime_checker(df)
        type_failed_checks = x._transform_type_checker(df)
        null_failed_checks = x._transform_null_checker(df)

        expected_pos_args_0 = (
            x,
            type_failed_checks,
            null_failed_checks,
            value_failed_checks,
            numerical_failed_checks,
            datetime_failed_checks,
        )

        assert (
            expected_pos_args_0 == call_0_pos_args
        ), "positional args unexpected in raise_exception_if_checks_fail call in transform method"

    def test_separate_passes_and_fails_called_no_optionals(self, mocker):
        """Test raise exception is called by the transform method when categorical, numerical_& datetime columns set
        to None."""

        x = InputChecker()

        df = data_generators_p.create_df_2()

        orig_df = df.copy(deep=True)

        x.fit(df)

        spy = mocker.spy(
            input_checker.checker.InputChecker, "separate_passes_and_fails"
        )

        df, bad_df = x.transform(df, batch_mode=True)

        assert (
            spy.call_count == 1
        ), "unexpected number of calls to InputChecker.separate_passes_and_fails with transform"

        call_0_args = spy.call_args_list[0]
        call_0_pos_args = call_0_args[0]

        value_failed_checks = {}
        numerical_failed_checks = {}
        datetime_failed_checks = {}
        type_failed_checks = x._transform_type_checker(df)
        null_failed_checks = x._transform_null_checker(df)

        expected_pos_args_0 = (
            x,
            type_failed_checks,
            null_failed_checks,
            value_failed_checks,
            numerical_failed_checks,
            datetime_failed_checks,
            orig_df,
        )

        h.assert_equal_dispatch(
            expected=expected_pos_args_0,
            actual=call_0_pos_args,
            msg="positional args unexpected in separate_passes_and_fails call in transform method",
        )

    def test_separate_passes_and_fails_called_all_checks(self, mocker):
        """Test raise exception is called by the transform method when categorical_columns and numerical_columns set
        to None."""

        x = InputChecker(
            numerical_columns=["a"],
            categorical_columns=["b", "c"],
            datetime_columns=["d"],
        )

        df = data_generators_p.create_df_2()

        df["d"] = pd.to_datetime(
            [
                "01/02/2020",
                "01/02/2021",
                "08/04/2019",
                "01/03/2020",
                "29/03/2019",
                "15/10/2018",
                "24/07/2020",
            ]
        )

        orig_df = df.copy(deep=True)

        x.fit(df)

        spy = mocker.spy(
            input_checker.checker.InputChecker, "separate_passes_and_fails"
        )

        df, bad_df = x.transform(df, batch_mode=True)

        assert (
            spy.call_count == 1
        ), "unexpected number of calls to InputChecker.separate_passes_and_fails with transform"

        call_0_args = spy.call_args_list[0]
        call_0_pos_args = call_0_args[0]

        value_failed_checks = x._transform_value_checker(df)
        numerical_failed_checks = x._transform_numerical_checker(df)
        datetime_failed_checks = x._transform_datetime_checker(df)
        type_failed_checks = x._transform_type_checker(df)
        null_failed_checks = x._transform_null_checker(df)

        expected_pos_args_0 = (
            x,
            type_failed_checks,
            null_failed_checks,
            value_failed_checks,
            numerical_failed_checks,
            datetime_failed_checks,
            orig_df,
        )

        h.assert_equal_dispatch(
            expected=expected_pos_args_0,
            actual=call_0_pos_args,
            msg="positional args unexpected in separate_passes_and_fails call in transform method",
        )


class TestRaiseExceptionIfChecksFail(object):
    """Tests for InputChecker.raise_exception_if_checks_fail()."""

    def test_arguments(self):
        """Test that raise_exception_if_checks_fail has expected arguments."""
        h.test_function_arguments(
            func=InputChecker.raise_exception_if_checks_fail,
            expected_arguments=[
                "self",
                "type_failed_checks",
                "null_failed_checks",
                "value_failed_checks",
                "numerical_failed_checks",
                "datetime_failed_checks",
            ],
            expected_default_values=None,
        )

    def test_no_failed_checks_before_transform(self):
        """Test validation_failed_checks is not present before transform"""

        x = InputChecker()

        df = data_generators_p.create_df_2()

        x.fit(df)

        assert (
            hasattr(x, "validation_failed_checks") is False
        ), "validation_failed_checks attribute present before transform"

    def test_validation_failed_checks_saved(self):
        """Test raise_exception_if_checks_fail saves the validation results"""

        df = data_generators_p.create_df_2()

        x = InputChecker()

        x.fit(df)

        df = x.transform(df)

        assert (
            hasattr(x, "validation_failed_checks") is True
        ), "validation_failed_checks attribute not present after transform"

        assert isinstance(
            x.validation_failed_checks, dict
        ), f"incorrect validation results type identified - expected: dict but got: {type(x.validation_failed_checks)}"

    def test_correct_validation_failed_checks(self):
        """Test raise_exception_if_checks_fail saves and prints the correct error message"""

        df = data_generators_p.create_df_2()

        x = InputChecker()

        x.fit(df)

        df = x.transform(df)

        assert isinstance(
            x.validation_failed_checks["Failed type checks"], dict
        ), f"incorrect type validation results type identified - expected: dict but got: {type(x.validation_failed_checks['Failed type checks'])}"

        assert isinstance(
            x.validation_failed_checks["Failed null checks"], dict
        ), f"incorrect null validation results type identified - expected: dict but got: {type(x.validation_failed_checks['Failed null checks'])}"

        assert isinstance(
            x.validation_failed_checks["Failed categorical checks"], dict
        ), f"incorrect categorical validation results type identified - expected: dict but got: {type(x.validation_failed_checks['Failed categorical checks'])}"

        assert isinstance(
            x.validation_failed_checks["Failed numerical checks"], dict
        ), f"incorrect numerical validation results type identified - expected: dict but got: {type(x.validation_failed_checks['Failed numerical checks'])}"

        assert isinstance(
            x.validation_failed_checks["Failed datetime checks"], dict
        ), f"incorrect datetime validation results type identified - expected: dict but got: {type(x.validation_failed_checks['Failed datetime checks'])}"

        assert isinstance(
            x.validation_failed_checks["Exception message"], str
        ), f"incorrect exception message type identified - expected: str but got: {type(x.validation_failed_checks['Exception message'])}"

    def test_input_checker_error_raised_type(self):
        """Test InputCheckerError is raised if type test fails"""

        x = InputChecker()

        df = data_generators_p.create_df_2()

        x.fit(df)

        df.loc[5, "a"] = "a"

        with pytest.raises(InputCheckerError):
            df = x.transform(df)

    def test_input_checker_error_raised_nulls(self):
        """Test InputCheckerError is raised if null test fails"""

        x = InputChecker()

        df = data_generators_p.create_df_2()

        df["b"] = df["b"].fillna("a")

        x = InputChecker()

        x.fit(df)

        df.loc[5, "b"] = np.nan

        with pytest.raises(InputCheckerError):
            df = x.transform(df)

    def test_input_checker_error_raised_categorical(self):
        """Test InputCheckerError is raised if categorical test fails"""

        x = InputChecker(categorical_columns=["b"])

        df = data_generators_p.create_df_2()

        x.fit(df)

        df.loc[5, "b"] = "u"

        with pytest.raises(InputCheckerError):
            df = x.transform(df)

    def test_input_checker_error_raised_numerical(self):
        """Test InputCheckerError is raised if numerical test fails"""

        x = InputChecker(numerical_columns=["a"])

        df = data_generators_p.create_df_2()

        x.fit(df)

        df.loc[0, "a"] = -1

        with pytest.raises(InputCheckerError):
            df = x.transform(df)

    def test_input_checker_error_raised_datetime(self):
        """Test InputCheckerError is raised if datetime test fails"""

        df = data_generators_p.create_df_2()

        df["d"] = pd.to_datetime(
            [
                "01/02/2020",
                "01/02/2021",
                "08/04/2019",
                "01/03/2020",
                "29/03/2019",
                "15/10/2018",
                np.NAN,
            ]
        )

        x = InputChecker(datetime_columns=["d"])

        x.fit(df)

        outliers_1 = pd.to_datetime("15/09/2017")
        outliers_2 = pd.to_datetime("13/09/2017")

        df.loc[0, "d"] = outliers_1
        df.loc[1, "d"] = outliers_2

        with pytest.raises(InputCheckerError):
            df = x.transform(df)

    def test_validation_failed_checks_correctly_stores_fails(self):
        """Test correct data is saved in validation_failed_checks after a failed check exception"""

        x = InputChecker()

        df = data_generators_p.create_df_2()

        df["d"] = pd.to_datetime(
            [
                "01/02/2020",
                "01/02/2021",
                "08/04/2019",
                "01/03/2020",
                "29/03/2019",
                "15/10/2018",
                np.NAN,
            ]
        )

        df["b"] = df["b"].fillna("a")

        x.fit(df)

        df.loc[0, "a"] = -1

        df.loc[4, "b"] = "u"

        df.loc[5, "b"] = np.nan

        df["c"] = [True, True, False, True, True, False, np.nan]

        df["c"] = df["c"].astype("bool")

        df.loc[0, "d"] = pd.to_datetime("15/09/2017")

        with pytest.raises(InputCheckerError):
            df = x.transform(df)

        assert list(x.validation_failed_checks["Failed null checks"]) == [
            "b"
        ], f"incorrect failed null checks identified - expected: ['b'] but got: {list(x.validation_failed_checks['Failed null checks'])}"

        assert list(x.validation_failed_checks["Failed type checks"]) == [
            "c"
        ], f"incorrect failed type checks identified - expected: ['b'] but got: {list(x.validation_failed_checks['Failed null checks'])}"

        assert x.validation_failed_checks["Failed null checks"]["b"] == [
            5
        ], f"incorrect failed null checks error message - expected: [5] but got: {x.validation_failed_checks['Failed null checks']['b']}"

        expected_type_fail_chk = {
            "actual": np.dtype("bool"),
            "expected": pd.CategoricalDtype(
                categories=["a", "b", "c", "d", "e", "f"], ordered=False
            ),
        }

        assert (
            x.validation_failed_checks["Failed type checks"]["c"]
            == expected_type_fail_chk
        ), f"incorrect failed type checks error message - expected: (CategoricalDtype(categories=['a', 'b', 'c', 'd', 'e', 'f'], ordered=False), dtype('bool')) but got: {x.validation_failed_checks['Failed type checks']['c']}"

        assert (
            any(x.validation_failed_checks["Failed categorical checks"].values())
            is False
        ), f"incorrect failed categorical checks identified - expected: empty dict but got: {list(x.validation_failed_checks['Failed categorical checks'])}"

        assert (
            any(x.validation_failed_checks["Failed numerical checks"].values()) is False
        ), f"incorrect failed numerical checks identified - expected: empty dict but got: {list(x.validation_failed_checks['Failed numerical checks'])}"

        assert (
            any(x.validation_failed_checks["Failed datetime checks"].values()) is False
        ), f"incorrect failed datetime checks identified - expected: empty dict but got: {list(x.validation_failed_checks['Failed datetime checks'])}"


class TestSeparatePassAndFails(object):
    """Tests for InputChecker.separate_passes_and_fails()."""

    def test_arguments(self):
        """Test that separate_passes_and_fails has expected arguments."""
        h.test_function_arguments(
            func=InputChecker.separate_passes_and_fails,
            expected_arguments=[
                "self",
                "type_failed_checks",
                "null_failed_checks",
                "value_failed_checks",
                "numerical_failed_checks",
                "datetime_failed_checks",
                "X",
            ],
            expected_default_values=None,
        )

    def test_input_checker_type_errors_shape(self):
        """Test correct dataframes are returned if type test fails"""

        x = InputChecker()

        df = data_generators_p.create_df_2()

        x.fit(df)

        df.loc[5, "a"] = "a"

        good_df, bad_df = x.transform(df, batch_mode=True)

        assert not (
            5 in good_df.index.tolist()
        ), "Type failure does not remove the index"

        assert good_df.shape[0] == 6, "Wrong shape for the correct return dataframe"

        assert 5 in bad_df.index.tolist(), "Type failure does not track mixed index"

        assert (
            bad_df.shape[0] == 1
        ), f"Wrong number of rows for bad dataframe. Was expecting one row, instead return {bad_df.shape[0]}"

        assert bad_df.shape[1] == (
            df.shape[1] + 1
        ), f"Wrong number of columns for bad dataframe. Was expecting {df.shape[1]+1}, instead returned {bad_df.shape[1]}"

    def test_input_checker_type_errors_column(self):
        """Test correct error column message is returned if type test fails"""

        x = InputChecker()

        df = data_generators_p.create_df_2()

        x.fit(df)

        df.loc[5, "a"] = "a"
        df.loc[5, "b"] = 1

        good_df, bad_df = x.transform(df, batch_mode=True)

        assert (
            "failed_checks" in bad_df.columns.tolist()
        ), "Bad dataframe does not include the column 'failed_checks'"

        expected = "Failed type check for column: a; Expected: float, Found: str\nFailed type check for column: b; Expected: str, Found: int"

        actual = bad_df["failed_checks"].unique().tolist()

        assert (
            len(actual) == 1
        ), f"Values in failed_checks not as expected: actual: {actual} expected: {expected}"

        assert (
            actual[0] == expected
        ), f"Values in failed_checks not as expected: actual: {actual} expected: {expected}"

    def test_input_checker_null_errors_shape(self):
        """Test correct dataframes are returned if null test fails"""

        x = InputChecker()

        df = data_generators_p.create_df_2()

        df["b"] = df["b"].fillna("a")

        x.fit(df)

        df.loc[5, "b"] = np.nan

        good_df, bad_df = x.transform(df, batch_mode=True)

        assert not (
            5 in good_df.index.tolist()
        ), "Type failure does not remove the index"

        assert good_df.shape[0] == (
            df.shape[0] - 1
        ), "Wrong shape for the correct return dataframe"

        assert 5 in bad_df.index.tolist(), "Type failure does not track mixed index"

        assert (
            bad_df.shape[0] == 1
        ), f"Wrong number of rows for bad dataframe. Was expecting one row, instead return {bad_df.shape[0]}"

        assert bad_df.shape[1] == (
            df.shape[1] + 1
        ), f"Wrong number of columns for bad dataframe. Was expecting {df.shape[1]+1}, instead returned {bad_df.shape[1]}"

    def test_input_checker_null_errors_column(self):
        """Test correct error column message is returned if null test fails"""

        x = InputChecker()

        df = data_generators_p.create_df_2()

        df["b"] = df["b"].fillna("a")

        x.fit(df)

        df.loc[5, "b"] = np.nan

        good_df, bad_df = x.transform(df, batch_mode=True)

        assert (
            "failed_checks" in bad_df.columns.tolist()
        ), "Bad dataframe does not include the column 'failed_checks'"

        message = bad_df["failed_checks"].item()

        expected = "Failed null check for column: b"

        h.assert_equal_msg(message, expected, "Value in Reason Failed not as expected")

    def test_input_checker_categorical_errors_shape(self):
        """Test correct dataframes are returned if categorical test fails"""

        x = InputChecker(categorical_columns=["b"])

        df = data_generators_p.create_df_2()

        x.fit(df)

        df.loc[5, "b"] = "u"

        good_df, bad_df = x.transform(df, batch_mode=True)

        assert not (
            5 in good_df.index.tolist()
        ), "Type failure does not remove the index"

        assert good_df.shape[0] == (
            df.shape[0] - 1
        ), "Wrong shape for the correct return dataframe"

        assert 5 in bad_df.index.tolist(), "Type failure does not track mixed index"

        assert (
            bad_df.shape[0] == 1
        ), f"Wrong number of rows for bad dataframe. Was expecting one row, instead return {bad_df.shape[0]}"

        assert bad_df.shape[1] == (
            df.shape[1] + 1
        ), f"Wrong number of columns for bad dataframe. Was expecting {df.shape[1]+1}, instead returned {bad_df.shape[1]}"

    def test_input_checker_categorical_errors_column(self):
        """Test correct error column message is returned if categorical test fails"""

        x = InputChecker(categorical_columns=["b"])

        df = data_generators_p.create_df_2()

        x.fit(df)

        df.loc[5, "b"] = "u"

        good_df, bad_df = x.transform(df, batch_mode=True)

        assert (
            "failed_checks" in bad_df.columns.tolist()
        ), "Bad dataframe does not include the column 'failed_checks'"

        message = bad_df["failed_checks"].item()

        expected = "Failed categorical check for column: b. Unexpected values are ['u']"

        h.assert_equal_msg(message, expected, "Value in failed_checks not as expected")

    def test_input_checker_numerical_errors_shape(self):
        """Test correct dataframes are returned if numerical test fails"""

        x = InputChecker(numerical_columns=["a"])

        df = data_generators_p.create_df_2()

        x.fit(df)

        df.loc[0, "a"] = -1

        good_df, bad_df = x.transform(df, batch_mode=True)

        assert not (
            0 in good_df.index.tolist()
        ), "Type failure does not remove the index"

        assert good_df.shape[0] == (
            df.shape[0] - 1
        ), "Wrong shape for the correct return dataframe"

        assert 0 in bad_df.index.tolist(), "Type failure does not track mixed index"

        assert (
            bad_df.shape[0] == 1
        ), f"Wrong number of rows for bad dataframe. Was expecting one row, instead return {bad_df.shape[0]}"

        assert bad_df.shape[1] == (
            df.shape[1] + 1
        ), f"Wrong number of columns for bad dataframe. Was expecting {df.shape[1]+1}, instead returned {bad_df.shape[1]}"

    def test_input_checker_numerical_errors_column(self):
        """Test correct error column message is returned if numerical test fails"""

        x = InputChecker(numerical_columns=["a"])

        df = data_generators_p.create_df_2()

        x.fit(df)

        df.loc[0, "a"] = -1

        good_df, bad_df = x.transform(df, batch_mode=True)

        assert (
            "failed_checks" in bad_df.columns.tolist()
        ), "Bad dataframe does not include the column 'failed_checks'"

        message = bad_df["failed_checks"].item()

        expected = "Failed minimum value check for column: a; Value below minimum: -1.0"

        h.assert_equal_msg(message, expected, "Value in Reason Fails not as expected")

    def test_input_checker_datetime_errors_shape(self):
        """Test correct dataframes are returned if datetime test fails"""

        df = data_generators_p.create_df_2()

        df["d"] = pd.to_datetime(
            [
                "01/02/2020",
                "01/02/2021",
                "08/04/2019",
                "01/03/2020",
                "29/03/2019",
                "15/10/2018",
                np.NAN,
            ]
        )

        x = InputChecker(datetime_columns=["d"])

        x.fit(df)

        outliers_1 = pd.to_datetime("15/09/2017")
        outliers_2 = pd.to_datetime("13/09/2017")

        df.loc[0, "d"] = outliers_1
        df.loc[1, "d"] = outliers_2

        good_df, bad_df = x.transform(df, batch_mode=True)

        assert not (
            0 in good_df.index.tolist() and (1 in good_df.index.tolist())
        ), "Type failure does not remove the index"

        assert good_df.shape[0] == (
            df.shape[0] - 2
        ), "Wrong shape for the correct return dataframe"

        assert (0 in bad_df.index.tolist()) and (
            1 in bad_df.index.tolist()
        ), "Type failure does not track mixed index"

        assert (
            bad_df.shape[0] == 2
        ), f"Wrong number of rows for bad dataframe. Was expecting one row, instead return {bad_df.shape[0]}"

        assert bad_df.shape[1] == (
            df.shape[1] + 1
        ), f"Wrong number of columns for bad dataframe. Was expecting {df.shape[1]+1}, instead returned {bad_df.shape[1]}"

    def test_input_checker_datetime_errors_column(self):
        """Test correct error column message is returned if numerical test fails"""

        df = data_generators_p.create_df_2()

        df["d"] = pd.to_datetime(
            [
                "01/02/2020",
                "01/02/2021",
                "08/04/2019",
                "01/03/2020",
                "29/03/2019",
                "15/10/2018",
                np.NAN,
            ]
        )

        x = InputChecker(datetime_columns=["d"])

        x.fit(df)

        outliers_1 = pd.to_datetime("15/09/2017")
        outliers_2 = pd.to_datetime("13/09/2017")

        df.loc[0, "d"] = outliers_1
        df.loc[1, "d"] = outliers_2

        good_df, bad_df = x.transform(df, batch_mode=True)

        assert (
            "failed_checks" in bad_df.columns.tolist()
        ), "Bad dataframe does not include the column 'failed_checks'"

        message_0 = bad_df.loc[0, "failed_checks"]
        message_1 = bad_df.loc[1, "failed_checks"]

        expected_0 = (
            "Failed minimum value check for column: d; Value below minimum: 2017-09-15"
        )
        expected_1 = (
            "Failed minimum value check for column: d; Value below minimum: 2017-09-13"
        )

        h.assert_equal_msg(
            message_0, expected_0, "Value in Reason Failed not as expected"
        )
        h.assert_equal_msg(
            message_1, expected_1, "Value in Reason Failed not as expected"
        )

    def test_full_failed_checks(self):
        """Test correct data is outputted for multiple failed exceptions"""

        x = InputChecker(
            numerical_columns=["a"], datetime_columns=["d"], categorical_columns=["b"]
        )

        df = data_generators_p.create_df_2()

        df["d"] = pd.to_datetime(
            [
                "01/02/2020",
                "01/02/2021",
                "08/04/2019",
                "01/03/2020",
                "29/03/2019",
                "15/10/2018",
                np.NAN,
            ]
        )

        df["b"] = df["b"].fillna("a")

        x.fit(df)

        df.loc[0, "a"] = -1

        df.loc[4, "b"] = "u"

        df.loc[5, "b"] = None

        # for type check failues

        df["c"] = ["a", "b", "c", "d", True, "f", "e"]
        df.loc[2, "a"] = "z"
        df.loc[2, "d"] = 1

        df.loc[0, "d"] = pd.to_datetime("15/09/2017")

        good_df, bad_df = x.transform(df, batch_mode=True)

        assert good_df.shape[0] == (
            3
        ), f"Incorrect good df num rows. Expected {3} but got {good_df.shape[0]}"

        assert (
            bad_df.shape[0] == 4
        ), f"Incorred bad df num rows. Expected {4} but go {bad_df.shape[0]}"

        assert bad_df.shape[1] == (
            df.shape[1] + 1
        ), f"Expected bad df to have {df.shape[1]+1} columns, but got {bad_df.shape[1]} instead"

        expected_msg_0 = "Failed minimum value check for column: a; Value below minimum: -1.0\nFailed minimum value check for column: d; Value below minimum: 2017-09-15"

        expected_msg_2 = "Failed type check for column: a; Expected: float, Found: str\nFailed type check for column: d; Expected: Timestamp, Found: int"

        expected_msg_4 = "Failed categorical check for column: b. Unexpected values are ['u']\nFailed type check for column: c; Expected: str, Found: bool"

        expected_msg_5 = "Failed null check for column: b"

        h.assert_equal_msg(
            bad_df["failed_checks"].loc[0],
            expected_msg_0,
            "Wrong message in reason failed for index 0",
        )

        h.assert_equal_msg(
            bad_df["failed_checks"].loc[2],
            expected_msg_2,
            "Wrong message in reason failed for index 2",
        )

        h.assert_equal_msg(
            bad_df["failed_checks"].loc[4],
            expected_msg_4,
            "Wrong message in reason failed for index 4",
        )

        h.assert_equal_msg(
            bad_df["failed_checks"].loc[5],
            expected_msg_5,
            "Wrong message in reason failed for index 5",
        )

    def test_multiple_value_error_fails_on_same_row(self):
        """Test that failed checks are updated correctly for rows with multiple
        columns which fail _transform_value_checker"""

        df = pd.DataFrame({"col1": ["a", "b", "c"], "col2": ["a", "b", "c"]})

        checker = InputChecker(
            columns=["col1", "col2"],
            categorical_columns=["col1", "col2"],
        )

        checker.fit(df)

        df_new = pd.DataFrame({"col1": ["a", "d", "a"], "col2": ["a", "d", "a"]})

        good_df, bad_df = checker.transform(df_new, batch_mode=True)

        expected_msg = "Failed categorical check for column: col1. Unexpected values are ['d']\nFailed categorical check for column: col2. Unexpected values are ['d']"

        assert bad_df.index.tolist() == [
            1
        ], "Wrong rows in bad_df when a row fails multiple value checks"

        h.assert_equal_msg(
            bad_df["failed_checks"].loc[1],
            expected_msg,
            "Wrong message in reason failed when a row fails multiple value checks",
        )


class TestUpdateBadDF(object):
    """Tests for InputChecker._update_bad_df()."""

    def test_arguments(self):
        """Test that _update_bad_df has expected arguments."""
        h.test_function_arguments(
            func=InputChecker._update_bad_df,
            expected_arguments=[
                "self",
                "bad_df",
                "idxs",
                "reason_failed",
                "error_info_by_row",
            ],
            expected_default_values=(None,),
        )

    def test_expected_output(self):
        """Test that _update_bad_df works as expected."""

        x = InputChecker(numerical_columns=["u"])

        df = data_generators_p.create_df_2()

        df["failed_checks"] = "fail 1"

        bad_df = x._update_bad_df(df, [2, 4], "fail 2")

        # check message updated as expected
        h.assert_equal_dispatch(
            expected=[
                "fail 1",
                "fail 1",
                "fail 1\nfail 2",
                "fail 1",
                "fail 1\nfail 2",
                "fail 1",
                "fail 1",
            ],
            actual=bad_df["failed_checks"].values.tolist(),
            msg="failed_checks not updated as expected by _update_bad_df",
        )

        # check other columns unchanged
        h.assert_equal_dispatch(
            expected=df,
            actual=bad_df[df.columns],
            msg="other columns have been modified by _update_bad_df",
        )


class TestUpdateGoodBadDF(object):
    """Tests for InputChecker._update_good_bad_df()."""

    def test_arguments(self):
        """Test that _update_good_bad_df has expected arguments."""
        h.test_function_arguments(
            func=InputChecker._update_good_bad_df,
            expected_arguments=[
                "self",
                "good_df",
                "bad_df",
                "idxs",
                "reason_failed",
                "error_info_by_row",
            ],
            expected_default_values=(None,),
        )

    def test_expected_output(self):
        """Test that _update_good_bad_df works as expected."""

        x = InputChecker(numerical_columns=["u"])

        df = data_generators_p.create_df_2()

        bad_df = df.loc[[2, 4]]
        good_df = df.loc[[0, 1, 3, 5, 6]]
        bad_df["failed_checks"] = "fail 1"

        good_df_up, bad_df_up = x._update_good_bad_df(good_df, bad_df, [3, 6], "fail 2")

        # check message in bad_df updated as expected
        h.assert_equal_dispatch(
            expected=["fail 1", "fail 1", "fail 2", "fail 2"],
            actual=bad_df_up["failed_checks"].values.tolist(),
            msg="failed_checks not updated as expected by _update_good_bad_df",
        )

        # check other columns in bad_df unchanged
        h.assert_equal_dispatch(
            expected=df.loc[[2, 4, 3, 6], :],
            actual=bad_df_up[df.columns],
            msg="other columns have been modified in bad_df by _update_good_bad_df",
        )

        # check good_df
        h.assert_equal_dispatch(
            expected=df.loc[[0, 1, 5], :],
            actual=good_df_up,
            msg="wrong good_df returned by _update_good_bad_df",
        )


class TestCheckType(object):
    """Tests for InputChecker._check_type()."""

    def test_arguments(self):
        """Test that _check_type has expected arguments."""
        h.test_function_arguments(
            func=InputChecker._check_type,
            expected_arguments=["self", "obj", "obj_name", "options"],
            expected_default_values=None,
        )

    def test_exception(self):
        """Test that _check_type fails with the correct error."""

        with pytest.raises(TypeError):

            InputChecker(numerical_columns=pd.DataFrame())


class TestIsStringValue(object):
    """Tests for InputChecker._is_string_value()."""

    def test_arguments(self):
        """Test that _check_type has expected arguments."""
        h.test_function_arguments(
            func=InputChecker._is_string_value,
            expected_arguments=["self", "string", "string_name", "check_value"],
            expected_default_values=None,
        )

    def test_exception(self):
        """Test that _is_string_value fails with the correct error."""

        with pytest.raises(ValueError):
            InputChecker(numerical_columns="None")


class TestIsSubset(object):
    """Tests for InputChecker._is_subset()."""

    def test_arguments(self):
        """Test that _is_subset has expected arguments."""
        h.test_function_arguments(
            func=InputChecker._is_subset,
            expected_arguments=["self", "obj_name", "columns", "dataframe"],
            expected_default_values=None,
        )

    def test_exception(self):
        """Test that _is_subset fails with the correct error."""

        x = InputChecker(numerical_columns=["u"])

        with pytest.raises(ValueError):
            x.fit(data_generators_p.create_df_2())


class TestIsEmpty(object):
    """Tests for InputChecker._is_empty()."""

    def test_arguments(self):
        """Test that _is_empty has expected arguments."""
        h.test_function_arguments(
            func=InputChecker._is_empty,
            expected_arguments=["self", "obj_name", "obj"],
            expected_default_values=None,
        )

    def test_check_fails_empty_list(self):
        """Test that _is_empty fails with the correct error."""

        with pytest.raises(ValueError):
            InputChecker(columns=[])

    def test_check_fails_empty_dict(self):
        """Test that _is_empty fails with the correct error."""

        with pytest.raises(ValueError):
            InputChecker(numerical_columns={})


class TestIsListedInColumns(object):
    """Tests for InputChecker._is_listed_in_columns()."""

    def test_arguments(self):
        """Test that _is_empty has expected arguments."""
        h.test_function_arguments(
            func=InputChecker._is_listed_in_columns,
            expected_arguments=["self"],
            expected_default_values=None,
        )

    def test_check_fails_columns_not_listed(self):
        """Test that _is_listed_in_columns fails with the correct error."""

        diff_cols = ["b", "c"]
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Column(s); {diff_cols} are not listed when initialising column attribute"
            ),
        ):
            InputChecker(columns=["a"], numerical_columns=["a", "b", "c"])

    def test_check_fails_columns_not_listed_with_infer(self):
        """Test that _is_listed_in_columns fails with the correct error when one of the columns lists are set to infer."""

        diff_cols = ["b", "c"]
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Column(s); {diff_cols} are not listed when initialising column attribute"
            ),
        ):
            InputChecker(
                columns=["a"],
                numerical_columns=["a", "b", "c"],
                categorical_columns="infer",
            )

    def test_check_fails_columns_not_listed_with_none(self):
        """Test that _is_listed_in_columns fails with the correct error when one of the columns lists are set to None."""

        diff_cols = ["b", "c"]
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Column(s); {diff_cols} are not listed when initialising column attribute"
            ),
        ):
            InputChecker(
                columns=["a"],
                numerical_columns=["a", "b", "c"],
                categorical_columns=None,
            )


class TestDfIsEmpty(object):
    """Tests for InputChecker._df_is_empty()."""

    def test_arguments(self):
        """Test that _df_is_empty has expected arguments."""
        h.test_function_arguments(
            func=InputChecker._df_is_empty,
            expected_arguments=["self", "obj_name", "df"],
            expected_default_values=None,
        )

    def test_check_fails(self):
        """Test that _df_is_empty fails with the correct error."""

        x = InputChecker()

        with pytest.raises(ValueError):
            x.fit(pd.DataFrame())
