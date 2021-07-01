from tubular.base import BaseTransformer
from input_checker._version import __version__
from input_checker.exceptions import InputCheckerError

import numpy as np
import pandas as pd


class InputChecker(BaseTransformer):
    """Class to compare a dataframe against a benchmark

    The input checker class currently contains 5 different checks:
    1. Null checker: ensures that columns with missing values in the benchmark dataframe
    are the only columns with missing values in the comparison dataframe

    2. Dtype checker: ensures that columns in the comparison dataframe are of the same data type as
    in the benchmark dataframe

    3. Categorical value checker: ensures that categorical columns in the comparison dataframe only contain
    values that exist in the benchmark dataframe

    4. Numerical checker: ensures that the values of the numerical columns in
    the comparison dataframe lie within the minimum and maximum range of the numerical columns
    in the benchmark dataframe.

    5. Datetime checker: ensures that the values of datetime columns in the comparison
    dataframe lie beyond the minimum date (optionally maximum) of datetime columns
    in the benchmark dataframe.

    Checks 1 and 2 are completed for all the columns that are defined under the 'columns'
    variable. If this attribute is not set, all of the columns in the dataframe
    passed to the fit method will be taken into account. The numerical and
    categorical checks may be skipped by setting the categorical_columns and
    numerical_column variables to None. There is alternatively an 'infer' option which
    automatically finds the columns that are of a categorical or numerical type among the
    list of columns defined/set in the 'columns' attribute.

    The class is fitted to the benchmark dataframe by calling the fit method
    which calls all the individual fit methods for individual checks. The input checker
    class object can then be saved, later to be loaded, and called to compare a dataframe
    against the benchmark dataframe. For comparison, the transform method will get called,
    which runs every check in the fitted input checker class against the benchmark dataframe
    and returns an exception message stating which checks have failed if any.

    Parameters
    ----------

    columns : None, list or str
        The list of model input column names that the column name, null checker
        and data type checks are generated for. If None then all the columns
        in the (fitted) benchmark dataframe are included in the checks. If str of a column
        name then only that column is included in the check
    categorical_columns : list or 'infer'
        The list of model input column names containing categorical data that
        the categorical level checks are generated for. If the 'infer' option
        is defined instead, this list is inferred based on the column types of
        the benchmark dataframe (category, boolean or string)
    numerical_columns : list, 'infer' or dict
        The list of model input column names containing numerical data that
        the numerical range checks are generated for. If the 'infer' option
        is defined instead, this list is inferred based on the column types of
        the benchmark dataframe. If equal to a dict, then each key in the
        dictionary must be a column in the (fitted) benchmark dataframe, these must contain
        a 'maximum' and 'minimum' keys within them. These keys contain a boolean
        stating if a maximum and / or minimum value check is desired
     datetime_columns : list, 'infer'
        The list of model input column names containing datetime data that
        the datetime level checks are generated for. If the 'infer' option
        is defined instead, this list is inferred based on the column types of
        the (fitted) benchmark dataframe (datetime, object).
    skip_infer_columns : list
        The list of columns conttaining the names for dataframe columns that will
        have type and null checks applied to them but will not be included in
        the 'infer' calculation for the categorical and numerical columns check
        these should include id, datetime and text fields

    Attributes
    ----------

    Aside from the class parameters, these attributes are generated when the class
    is fitted to a benchmark dataframe

    null_map: dict
        Dictionary contain the null map for the specified columns, keys are the
        column names and the values are a 1 if the column can contain nulls and
        0 if the column is not allowed to contain any nulls

    expected_values: dict
        Dictionary contain the categorical map for the specified categorical columns,
        keys are the column names and the values are the various values that
        are allowed within each categorical column. Only generated if the
        categorical columns parameter is not set to None

    column_classes: dict
        Dictionary contain the data type map for the specified columns, keys are the
        column names and the values the column data types

    numerical_values: dict
        Dictionary contain the numerical map for the specified numerical columns,
        keys are the column names which themselves contain minimum and maximum
        allowables within each numerical column. Only generated if the
        numerical columns parameter is not set to None

    datetime_values: dict
        Dictionary contain the datetime map for the specified datetime columns,
        keys are the column names which themselves contain minimum and (optional)maximum
        allowables within each datetime column. Only generated if the
        datetime columns parameter is not set to None

    """

    def __init__(
        self,
        columns=None,
        categorical_columns=None,
        numerical_columns=None,
        datetime_columns=None,
        skip_infer_columns=None,
        **kwds,
    ):

        super().__init__(columns=columns, **kwds)

        self.columns = columns
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.datetime_columns = datetime_columns
        self.skip_infer_columns = skip_infer_columns

        # check that all the inputs are of the accepted formats
        self._check_type(self.columns, "input columns", [list, type(None), str])

        self._check_type(
            self.categorical_columns, "categorical columns", [list, str, type(None)]
        )
        if isinstance(self.categorical_columns, str):
            self._is_string_value(
                self.categorical_columns, "categorical columns", "infer"
            )

        self._check_type(
            self.numerical_columns, "numerical columns", [list, dict, str, type(None)]
        )
        if isinstance(self.numerical_columns, str):
            self._is_string_value(self.numerical_columns, "numerical columns", "infer")

        self._check_type(
            self.datetime_columns, "datetime columns", [list, dict, str, type(None)]
        )
        if isinstance(self.datetime_columns, str):
            self._is_string_value(self.datetime_columns, "datetime columns", "infer")

        self._check_type(
            self.skip_infer_columns, "skip infer columns", [list, type(None)]
        )

        # check if any of the inputs are empty
        self._is_empty("input columns", self.columns)
        self._is_empty("categorical columns", self.categorical_columns)
        self._is_empty("numerical columns", self.numerical_columns)
        self._is_empty("datetime columns", self.datetime_columns)

        # check if categorical/numerical/datetime/skip_infer columns are listed in columns (when all provided)
        if columns is not None:
            self._is_listed_in_columns()

        self.version_ = __version__

    def _consolidate_inputs(self, X):
        """Method to run checks on class inputs and convert them to the same format, if needed

        Parameters
        ----------
        X : pd.DataFrame
            The training input samples.

        """

        # set key column values to an empy list if equal to None
        if self.skip_infer_columns is None:
            self.skip_infer_columns = []
        else:
            self._is_subset("skip infer columns", self.skip_infer_columns, X)

        # if infer option is selected, generate list of categorical, numerical & datetime columns
        if self.categorical_columns == "infer":
            self.categorical_columns = []
            for column in self.columns:
                col_type = X[column].dtypes.name
                if (
                    col_type in ["category", "object", "bool"]
                    and column not in self.skip_infer_columns
                ):

                    self.categorical_columns.append(column)

        if self.numerical_columns == "infer":
            self.numerical_dict = {}
            for column in self.columns:
                if (
                    (str(X[column].dtype).startswith("int"))
                    or (str(X[column].dtype).startswith("float"))
                ) and (column not in self.skip_infer_columns):
                    self.numerical_dict[column] = {}
                    self.numerical_dict[column]["maximum"] = True
                    self.numerical_dict[column]["minimum"] = True

        if self.datetime_columns == "infer":
            self.datetime_dict = {}
            for column in self.columns:
                if (
                    str(X[column].dtype).startswith("datetime")
                    and column not in self.skip_infer_columns
                ):
                    self.datetime_dict[column] = {}
                    self.datetime_dict[column]["maximum"] = False
                    self.datetime_dict[column]["minimum"] = True

        # check that columns are a subset of the dataframe columns
        self._is_subset("input columns", self.columns, X)

        if isinstance(self.categorical_columns, list):
            self._is_subset("categorical columns", self.categorical_columns, X)

        # for numerical check, also store value ranges in a dictionary
        if isinstance(self.numerical_columns, list):
            self._is_subset("numerical columns", self.numerical_columns, X)
            self.numerical_dict = {}
            for column in self.numerical_columns:
                self.numerical_dict[column] = {}
                self.numerical_dict[column]["maximum"] = True
                self.numerical_dict[column]["minimum"] = True

        # for datetime check, also store value ranges in a dictionary
        if isinstance(self.datetime_columns, list):
            self._is_subset("datetime columns", self.datetime_columns, X)
            self.datetime_dict = {}
            for column in self.datetime_columns:
                self.datetime_dict[column] = {}
                self.datetime_dict[column]["maximum"] = False
                self.datetime_dict[column]["minimum"] = True

        # if numerical_columns attribute is a dictionary,
        # then save values and check keys are subset of dataframe columns
        if isinstance(self.numerical_columns, dict):
            self._is_subset(
                "numerical dictionary keys", list(self.numerical_columns.keys()), X
            )
            self.numerical_dict = self.numerical_columns

        if self.numerical_columns is not None:
            self.numerical_columns = list(self.numerical_dict.keys())

        # if datetime_columns attribute is a dictionary,
        # then save values and check keys are subset of dataframe columns
        if isinstance(self.datetime_columns, dict):
            self._is_subset(
                "datetime dictionary keys", list(self.datetime_columns.keys()), X
            )
            self.datetime_dict = self.datetime_columns

        if self.datetime_columns is not None:
            self.datetime_columns = list(self.datetime_dict.keys())

    def _fit_type_checker(self, X):
        """Sets the expected dtypes based on the benchmark dataframe, X.

        Parameters
        ----------
        X : pd.DataFrame
            Data to set expected dtypes from.

        """

        self.column_classes = X[self.columns].dtypes.to_dict()

    def _fit_null_checker(self, X):
        """Sets a lookup to check whether a column can have missing values or not.

        Based on the data of the benchmark dataframe X, this method initialises and
        sets the null_map attribute which is a dictionary with column names as keys and
        binary values set to indicate if a given column can contain missing values.

        Parameters
        ----------
        X : pd.DataFrame
            The training input samples.

        """

        self.null_map = {}

        for col in self.columns:

            if X[col].isnull().values.any():

                self.null_map[col] = 1

            else:

                self.null_map[col] = 0

    def _fit_value_checker(self, X):
        """Creates a dictionary to enable categorical value checks for the comparison dataframe.

        This method initialises and sets expected_values class attribute based on the categorical values
        in the benchmark dataframe, X.

        Parameters
        ----------
        X : pd.DataFrame
            The training input samples.
        """

        self.expected_values = {}

        for col in self.categorical_columns:
            self.expected_values[col] = X[col].unique().tolist()

    def _fit_numerical_checker(self, X):
        """Creates a dictionary to enable numerical value checks for the comparison dataframe.

        This method initialises and sets numerical_values class attribute based on the
        numerical values in the benchmark dataframe X. numerical_values is used to check that
        the values of the selected numerical variables of the comparison dataframe lie within a
        specified range based on the numerical values of the benchmark dataframe X.

        Parameters
        ----------
        X : pd.DataFrame
            The training input samples.

        """

        self.numerical_values = {}

        for col in self.numerical_dict:

            self.numerical_values[col] = {}
            if self.numerical_dict[col]["maximum"]:
                self.numerical_values[col]["maximum"] = X[col].max()
            else:
                self.numerical_values[col]["maximum"] = None

            if self.numerical_dict[col]["minimum"]:
                self.numerical_values[col]["minimum"] = X[col].min()
            else:
                self.numerical_values[col]["minimum"] = None

    def _fit_datetime_checker(self, X):
        """Creates a dictionary to enable datetime value checks for the comparison dataframe.

        This method initialises and sets datetime_values class attribute based on the
        datetime values in the benchmark dataframe X. datetime_values is used to check that
        the values of the datetime variables of the comparison dataframe lie within a
        specified range based on the datetime values of the benchmark dataframe X.

        Parameters
        ----------
        X : pd.DataFrame
            The training input samples.

        """

        self.datetime_values = {}

        for col in self.datetime_columns:

            self.datetime_values[col] = {}
            if self.datetime_dict[col]["maximum"]:
                self.datetime_values[col]["maximum"] = X[col].max()
            else:
                self.datetime_values[col]["maximum"] = None

            if self.datetime_dict[col]["minimum"]:
                self.datetime_values[col]["minimum"] = X[col].min()
            else:
                self.datetime_values[col]["minimum"] = None

    def fit(self, X, y=None):
        """Checks that the class inputs are of the correct format and then fits
        the different input checker methods to the benchmark dataframe

        Parameters
        ----------
        X : pd.DataFrame
            The training input samples.

        y : None
            y is not needed in this transformer, yet the sklearn pipeline API
            requires this parameter for checking.

        """
        if y is not None:
            raise ValueError(
                f"{y} is passed to the fit method which is not required for the input_checker"
            )

        super().fit(X, y)

        self._df_is_empty("input dataframe", X)

        self._consolidate_inputs(X)

        self._fit_type_checker(X)
        self._fit_null_checker(X)

        # only run the categorical, numerical & datetime checks if the columns have been specified
        if self.categorical_columns is not None:
            self._fit_value_checker(X)
        if self.numerical_columns is not None:
            self._fit_numerical_checker(X)
        if self.datetime_columns is not None:
            self._fit_datetime_checker(X)

        return self

    def _transform_type_checker(self, X, batch_mode=False):
        """Checks if columns in the comparison dataframe X are of the expected dtypes
        based on the (fitted) benchmark dataframe .

        Parameters
        ----------
        X : pd.DataFrame
            Input data to check column types.

        batch_mode: bool, default=False
            Flag indicating if transform is being run in batch mode

        Returns
        -------
        type_checker_failed_checks : dict
            Dictionary containing the failed tests, empty if none failed

        """

        self.check_is_fitted(["column_classes"])

        # mapping for pandas dtype to Python dtypes
        type_mappings = {
            "object": "str",
            "int": "int",
            "float": "float",
            "bool": "bool",
            "datetime[ns]": "Timestamp",
            "category": "str",
        }

        type_checker_failed_checks = {}

        for col in self.columns:

            # skip column if all values in column are missing as the expected
            # type will be float, nulls will be checked by null check either way
            if X[col].isnull().all():

                continue

            # compare types by row if operating in batch mode
            if batch_mode:

                # remove bytes part of type
                target_dtype_name = "".join(
                    i for i in self.column_classes[col].name if not i.isdigit()
                )

                # convert pandas dtype to python dtype
                target_dtype = type_mappings[target_dtype_name]

                current_dtype = X[col].apply(lambda x: type(x).__name__)

                # fix nulls dtype to target dtype before comparing actual to expected
                current_dtype[X[col].isnull()] = target_dtype

                if (current_dtype != target_dtype).any():

                    type_checker_failed_checks[col] = {}
                    type_checker_failed_checks[col]["idxs"] = X[
                        current_dtype != target_dtype
                    ].index.tolist()
                    type_checker_failed_checks[col]["actual"] = current_dtype[
                        type_checker_failed_checks[col]["idxs"]
                    ].to_dict()
                    type_checker_failed_checks[col]["expected"] = target_dtype

            # otherwise compare overall pandas dtype
            else:

                target_dtype = self.column_classes[col]

                current_dtype = X[col].dtypes

                if target_dtype.name == "category":

                    # checking object type == categorical type throws error
                    same_level_check = target_dtype == current_dtype

                else:

                    same_level_check = current_dtype == target_dtype

                if not same_level_check:

                    type_checker_failed_checks[col] = {}
                    type_checker_failed_checks[col]["actual"] = current_dtype
                    type_checker_failed_checks[col]["expected"] = target_dtype

        return type_checker_failed_checks

    def _transform_null_checker(self, X):
        """Checks if columns with missing values in the comparison dataframe X are the only columns that
        also contain missing values in the (fitted) benchmark dataframe.

        Parameters
        ----------
        X : pd.DataFrame
            Pandas dataframe containing columns to check null values.

        Returns
        -------
        null_checker_failed_checks : dict
           Dictionary containing the failed tests, empty if none failed

        """

        self.check_is_fitted(["null_map"])

        null_checker_failed_checks = {}

        for col in self.columns:

            if self.null_map[col] == 0 and X[col].isnull().any():
                null_checker_failed_checks[col] = X[X[col].isnull()].index.tolist()

        return null_checker_failed_checks

    def _transform_numerical_checker(self, X, type_fails={}, batch_mode=False):
        """Checks if values of numerical columns in the comparison dataframe X are
        inline with the benchmark dataframe.

        Please note that missing values are not checked as a part of this method,
        they are handled by the NullValueChecker.

        Parameters
        ----------
        X : pd.DataFrame
            The input samples to check take expected values.
        type_fails : dict, default={}
            Output dictionary from transform_type_checker.
        batch_mode : bool, default=False
            Flag indicating if transform is being run in batch mode

        Returns
        -------
        numerical_checker_failed_checks : dict
            Dictionary containing the failed tests, empty if none failed

        """

        self.check_is_fitted(["numerical_values"])

        numerical_checker_failed_checks = {}

        for col in self.numerical_columns:

            # remove rows which failed type checks

            X_filtered = X.copy()

            if col in type_fails.keys():

                if batch_mode:

                    # remove any rows where type is not float or int
                    ids_to_drop = [
                        k
                        for k, v in type_fails[col]["actual"].items()
                        if v not in ("float", "int")
                    ]
                    X_filtered = X_filtered.drop(ids_to_drop, axis=0)

                # if not batch mode and column is not a numerical dtype, drop all rows
                elif not type_fails[col]["actual"].name.startswith(
                    "float"
                ) and not type_fails[col]["actual"].name.startswith("int"):

                    X_filtered = X_filtered.drop(X.index, axis=0)

            min_value = self.numerical_values[col]["minimum"]
            max_value = self.numerical_values[col]["maximum"]

            if max_value:

                if (X_filtered[col] > max_value).any():

                    above_list = X_filtered[col][X_filtered[col] > max_value].to_dict()
                    above_idxs = X_filtered[col][
                        X_filtered[col] > max_value
                    ].index.tolist()

                    if col not in numerical_checker_failed_checks:
                        numerical_checker_failed_checks[col] = {}

                    numerical_checker_failed_checks[col]["max idxs"] = above_idxs
                    numerical_checker_failed_checks[col]["maximum"] = above_list

            if min_value:

                if (X_filtered[col] < min_value).any():

                    if col not in numerical_checker_failed_checks:
                        numerical_checker_failed_checks[col] = {}

                    below_list = X_filtered[col][X_filtered[col] < min_value].to_dict()

                    below_idxs = X_filtered[col][
                        X_filtered[col] < min_value
                    ].index.tolist()

                    numerical_checker_failed_checks[col]["minimum"] = below_list
                    numerical_checker_failed_checks[col]["min idxs"] = below_idxs

        return numerical_checker_failed_checks

    def _transform_value_checker(self, X):
        """Checks if values of categorical columns in the comparison dataframe X are
        inline with the benchmark dataframe using expected_values attribute.

        Please note that missing values are not checked as a part of this method,
        they are handled by using the NullValueChecker.

        Parameters
        ----------
        X : pd.DataFrame
            The input samples to check take expected values.

        Returns
        -------
        value_checker_failed_checks : dict
            Dictionary containing the failed tests, empty if none failed

        """

        self.check_is_fitted(["expected_values"])

        value_checker_failed_checks = {}

        for col in self.categorical_columns:

            v = self.expected_values[col]

            if (~X.loc[(~X[col].isnull()), col].isin(v)).any():

                unexpected_list_idx = X[
                    (~X[col].isnull()) & (~X[col].isin(v))
                ].index.tolist()

                value_checker_failed_checks[col] = {}
                value_checker_failed_checks[col]["idxs"] = unexpected_list_idx
                value_checker_failed_checks[col]["values"] = (
                    X[(~X[col].isnull()) & (~X[col].isin(v))][col].unique().tolist()
                )

        return value_checker_failed_checks

    def _transform_datetime_checker(self, X, type_fails={}, batch_mode=False):
        """Checks if values of datetime columns in the comparison dataframe X are
        inline with the benchmark dataframe using datetime_dict attribute.

        Please note that missing values are not checked as a part of this method,
        they are handled by using the NullValueChecker.

        Parameters
        ----------
        X : pd.DataFrame
            The input samples to check take expected values.
        type_fails : dict
            Output dictionary from transform_type_checker.
        batch_mode : bool
            Flag if transform is being run in batch mode


        Returns
        -------
        datetime_checker_failed_checks : dict
            Dictionary containing the failed tests, empty if none failed

        """

        self.check_is_fitted(["datetime_values"])

        datetime_checker_failed_checks = {}

        for col in self.datetime_columns:

            # remove rows which failed type checks

            X_filtered = X.copy()

            if col in type_fails.keys():

                if batch_mode:

                    # remove all rows where dtype was not Timestamp
                    X_filtered = X_filtered.drop(type_fails[col]["idxs"], axis=0)

                # if not batch mode and column is not a datetime dtype, drop all rows
                elif not type_fails[col]["actual"].name.startswith("datetime"):

                    X_filtered = X_filtered.drop(X.index, axis=0)

            min_value = self.datetime_values[col]["minimum"]
            max_value = self.datetime_values[col]["maximum"]

            if max_value:

                if (X_filtered[col] > max_value).any():

                    above_list = X_filtered[col][X_filtered[col] > max_value].to_dict()
                    above_idxs = X_filtered[col][
                        X_filtered[col] > max_value
                    ].index.tolist()

                    if col not in datetime_checker_failed_checks:
                        datetime_checker_failed_checks[col] = {}

                    datetime_checker_failed_checks[col]["maximum"] = above_list
                    datetime_checker_failed_checks[col]["max idxs"] = above_idxs

            if min_value:

                if (X_filtered[col] < min_value).any():

                    if col not in datetime_checker_failed_checks:
                        datetime_checker_failed_checks[col] = {}

                    below_list = X_filtered[col][X_filtered[col] < min_value].to_dict()
                    below_idxs = X_filtered[col][
                        X_filtered[col] < min_value
                    ].index.tolist()

                    datetime_checker_failed_checks[col]["minimum"] = below_list
                    datetime_checker_failed_checks[col]["min idxs"] = below_idxs

        return datetime_checker_failed_checks

    def raise_exception_if_checks_fail(
        self,
        type_failed_checks,
        null_failed_checks,
        value_failed_checks,
        numerical_failed_checks,
        datetime_failed_checks,
    ):
        """Method to combine all tests results from input checker tests and
        raise an InputChecker exception if any one of the checks fails.

        Parameters
        ----------
        type_failed_checks : dict
            Details of failed type checker tests, empty if no checks failed.

        null_failed_checks : dict
            Details of failed null checker tests, empty if no checks failed.

        value_failed_checks : dict
            Details of failed categorical checker tests, empty if no checks failed.

        numerical_failed_checks : dict
            Details of failed numerical checker tests, empty if no checks failed.

        datetime_failed_checks : dict
            Details of failed datetime checker tests, empty if no checks failed.

        """

        null_exception = ""
        for col in null_failed_checks:
            null_exception = null_exception + f"Failed null check for column: {col}\n"

        type_exception = ""
        for col, fails in type_failed_checks.items():
            type_exception = (
                type_exception
                + f"Failed type check for column: {col}; Expected: {fails['expected']}, Found: {fails['actual']}\n"
            )

        value_exception = ""
        for col, fails in value_failed_checks.items():
            value_exception = (
                value_exception
                + f"Failed categorical check for column: {col}; Unexpected values: {fails['values']}\n"
            )

        numerical_exception = ""
        for col, fails in numerical_failed_checks.items():
            if "maximum" in fails.keys():

                numerical_exception = (
                    numerical_exception
                    + f"Failed maximum value check for column: {col}; Values above maximum: {fails['maximum']}\n"
                )
            if "minimum" in fails.keys():

                numerical_exception = (
                    numerical_exception
                    + f"Failed minimum value check for column: {col}; Values below minimum: {fails['minimum']}\n"
                )

        datetime_exception = ""
        for col, fails in datetime_failed_checks.items():

            if "maximum" in fails.keys():

                datetime_exception = (
                    datetime_exception
                    + f"Failed maximum value check for column: {col}; Values above maximum: {fails['maximum']}\n"
                )
            if "minimum" in fails.keys():

                datetime_exception = (
                    datetime_exception
                    + f"Failed minimum value check for column: {col}; Values below minimum: {fails['minimum']}\n"
                )

        exception_message = (
            null_exception
            + type_exception
            + value_exception
            + numerical_exception
            + datetime_exception
        )

        self.validation_failed_checks = {}
        self.validation_failed_checks["Failed type checks"] = type_failed_checks
        self.validation_failed_checks["Failed null checks"] = null_failed_checks
        self.validation_failed_checks["Failed categorical checks"] = value_failed_checks
        self.validation_failed_checks[
            "Failed numerical checks"
        ] = numerical_failed_checks
        self.validation_failed_checks["Failed datetime checks"] = datetime_failed_checks
        self.validation_failed_checks["Exception message"] = exception_message

        if len(exception_message) > 0:
            raise InputCheckerError(exception_message)

    def separate_passes_and_fails(
        self,
        type_failed_checks,
        null_failed_checks,
        value_failed_checks,
        numerical_failed_checks,
        datetime_failed_checks,
        X,
    ):
        """Method to combine all tests results from input checker tests and
        separate rows which pass checks (good_df) from rows which fail checks
        (bad_df). Failing rows will have an extra column added called
        'failed_checks', which concatenates all the failing test information.

        Parameters
        ----------
        type_failed_checks : dict
            Details of failed type checker tests, empty if no checks failed.

        null_failed_checks : dict
            Details of failed null checker tests, empty if no checks failed.

        value_failed_checks : dict
            Details of failed categorical checker tests, empty if no checks failed.

        numerical_failed_checks : dict
            Details of failed numerical checker tests, empty if no checks failed.

        datetime_failed_checks : dict
            Details of failed datetime checker tests, empty if no checks failed.

        Returns:
        --------
        good_df, bad_df : tuple
            Dataframes containing rows which pass checks (good_df) and
            rows which fail checks (bad_df).

        """

        good_df = X.copy(deep=True)
        bad_df = pd.DataFrame(columns=X.columns.values.tolist() + ["failed_checks"])

        # add expected values check failures
        for col, fails in value_failed_checks.items():

            # if any of the failing rows have previously failed checks,
            # update these with the new failure
            bad_df = self._update_bad_df(
                bad_df,
                fails["idxs"],
                f"Failed categorical check for column: {col}. Unexpected values are {fails['values']}",
            )

            # separate failing rows from good_df and move to bad_df
            good_df, bad_df = self._update_good_bad_df(
                good_df,
                bad_df,
                fails["idxs"],
                f"Failed categorical check for column: {col}. Unexpected values are {fails['values']}",
            )

        # add numerical check failures
        for col, fails in numerical_failed_checks.items():

            if "maximum" in fails.keys():

                # check if some idxs have already been chosen
                bad_df = self._update_bad_df(
                    bad_df,
                    fails["max idxs"],
                    f"Failed maximum value check for column: {col}; Value above maximum: ",
                    error_info_by_row=fails["maximum"],
                )

                good_df, bad_df = self._update_good_bad_df(
                    good_df,
                    bad_df,
                    fails["max idxs"],
                    f"Failed maximum value check for column: {col}; Value above maximum: ",
                    error_info_by_row=fails["maximum"],
                )

            if "minimum" in fails.keys():

                # check if some idxs have already been chosen
                bad_df = self._update_bad_df(
                    bad_df,
                    fails["min idxs"],
                    f"Failed minimum value check for column: {col}; Value above minimum: ",
                    error_info_by_row=fails["minimum"],
                )

                good_df, bad_df = self._update_good_bad_df(
                    good_df,
                    bad_df,
                    fails["min idxs"],
                    f"Failed minimum value check for column: {col}; Value below minimum: ",
                    error_info_by_row=fails["minimum"],
                )

        # add datetime check failures
        for col, fails in datetime_failed_checks.items():

            if "maximum" in fails.keys():

                for k, v in fails["maximum"].items():

                    fails["maximum"][k] = np.datetime_as_string(
                        v.to_datetime64(), unit="D"
                    )

                # check if some idxs have already been chosen
                bad_df = self._update_bad_df(
                    bad_df,
                    fails["max idxs"],
                    f"Failed maximum value check for column: {col}; Value above maximum: ",
                    error_info_by_row=fails["maximum"],
                )

                good_df, bad_df = self._update_good_bad_df(
                    good_df,
                    bad_df,
                    fails["max idxs"],
                    f"Failed maximum value check for column: {col}; Value above maximum: ",
                    error_info_by_row=fails["maximum"],
                )

            if "minimum" in fails.keys():

                for k, v in fails["minimum"].items():

                    fails["minimum"][k] = np.datetime_as_string(
                        v.to_datetime64(), unit="D"
                    )

                # check if some idxs have already been chosen
                bad_df = self._update_bad_df(
                    bad_df,
                    fails["min idxs"],
                    f"Failed minimum value check for column: {col}; Value below minimum: ",
                    error_info_by_row=fails["minimum"],
                )

                good_df, bad_df = self._update_good_bad_df(
                    good_df,
                    bad_df,
                    fails["min idxs"],
                    f"Failed minimum value check for column: {col}; Value below minimum: ",
                    error_info_by_row=fails["minimum"],
                )

        # add null check failures
        for col, idxs in null_failed_checks.items():

            bad_df = self._update_bad_df(
                bad_df, idxs, f"Failed null check for column: {col}"
            )
            good_df, bad_df = self._update_good_bad_df(
                good_df, bad_df, idxs, f"Failed null check for column: {col}"
            )

        # add type check failures
        for col, fails in type_failed_checks.items():

            bad_df = self._update_bad_df(
                bad_df,
                fails["idxs"],
                f"Failed type check for column: {col}; Expected: {fails['expected']}, Found: ",
                fails["actual"],
            )

            good_df, bad_df = self._update_good_bad_df(
                good_df,
                bad_df,
                fails["idxs"],
                f"Failed type check for column: {col}; Expected: {fails['expected']}, Found: ",
                fails["actual"],
            )

        # indices in bad_df will be out of order, change to match order in original DF
        bad_df = bad_df.loc[[i for i in X.index if i in bad_df.index]]

        return good_df, bad_df

    def _update_bad_df(self, bad_df, idxs, reason_failed, error_info_by_row=None):
        """Method to update 'failed_checks' field of rows with indices in idxs.
        The field is updated by contentating reason_failed.

        Parameters
        ----------
        bad_df : pd.DataFrame
            The dataframe containing rows to update.

        idxs : list
            List of indices in bad_df to update.

        reason_failed : str
            String to concatenate to 'failed_checks' in bad_df.

        error_info_by_row: None or dict
            Additional error information for each record. Has actual value per row ID which is failing check

        Returns
        -------
        bad_df: pd.DataFrame
            Dataframe containing rows which failed checks

        """

        if error_info_by_row:
            if type(error_info_by_row) is not dict:
                raise TypeError("numerical should either be none or a dict")

        if sum(bad_df.index.isin(idxs)) == 0:

            return bad_df

        elif not error_info_by_row:

            bad_df.loc[bad_df.index.isin(idxs), "failed_checks"] = bad_df.loc[
                bad_df.index.isin(idxs), "failed_checks"
            ].apply(lambda x: x + "\n" + reason_failed)

        else:

            bad_df.loc[bad_df.index.isin(idxs), "failed_checks"] = bad_df.loc[
                bad_df.index.isin(idxs)
            ].apply(
                lambda x: x["failed_checks"]
                + "\n"
                + reason_failed
                + f"{error_info_by_row[x.name]}",
                axis=1,
            )

        return bad_df

    def _update_good_bad_df(
        self, good_df, bad_df, idxs, reason_failed, error_info_by_row=None
    ):
        """Function to separate rows from good_df with indices in idxs and add them
        to bad_df, along with an extra field 'failed_checks' set to reason failed

        Parameters
        ----------
        good_df : pd.DataFrame
            The dataframe containing rows to remove.

        bad_df : pd.DataFrame
            The dataframe to which rows will be added.

        idxs : list
            List of indices in good_df to remove and add to bad_df

        reason_failed : str
            String to assign as'failed_checks' in bad_df.

        error_info_by_row: None or dict, default=None
            Additional error information for each record. Has actual value per row ID which is failing check

        Returns
        -------
        good_df, bad_df : tuple
            Dataframes containing rows which pass checks (good_df) and
            rows which fail checks (bad_df).

        """

        if error_info_by_row:
            if type(error_info_by_row) is not dict:
                raise TypeError("numerical should either be none or a dict")

        bad_idxs = good_df.loc[good_df.index.isin(idxs)]

        if not error_info_by_row:
            bad_idxs = bad_idxs.assign(failed_checks=reason_failed)
        else:
            bad_idxs = bad_idxs.assign(failed_checks="")
            bad_idxs["failed_checks"] = bad_idxs.apply(
                lambda x: reason_failed + f"{error_info_by_row[x.name]}", axis=1
            )

        good_df = good_df.loc[~good_df.index.isin(idxs)]

        bad_df = bad_df.append(bad_idxs)

        return good_df, bad_df

    def transform(self, X, batch_mode=False):
        """Method to run the input checker tests that have set based on the fitted
        benchmark dataframe on the comparison dataframe.


        Parameters
        ----------
        X : pd.DataFrame
            The new dataframe to validate against the benchmark samples.
        batch_mode : bool, default=False
            When batch_mode = True, the dataframe is processed row-by-row. Two data frames
            are returned: a DF of the records that pass the checks and a DF of the records that
            fail the checks. The failed records have an extra column 'failed_checks' which
            contains reasons for the failed checks.
            When batch_mode = False, an exception will be raised if any of the rows fail
            the input checks, otherwise the comparison dataframe X is returned

        Returns
        -------
        good_df, bad_df or X: tuple or pd.DataFrame
            Returns a tuple of dataframes with rows passing and failing checks respectively
            if run in batch mode or the comparison dataframe X.
            If any of the checks fail when batch_mode=False, it will throw an InputChecker exception

        """

        if not isinstance(batch_mode, bool):

            raise ValueError("batch_mode must be either True or False")

        X = super().transform(X)

        # check that scoring dataframe is not empty
        self._df_is_empty("scoring dataframe", X)

        type_failed_checks = self._transform_type_checker(X, batch_mode)
        null_failed_checks = self._transform_null_checker(X)

        # only run the categorical and numerical checks if checks had been selected
        if self.categorical_columns is not None:
            value_failed_checks = self._transform_value_checker(X)
        else:
            value_failed_checks = {}
        if self.numerical_columns is not None:
            numerical_failed_checks = self._transform_numerical_checker(
                X, type_failed_checks, batch_mode
            )
        else:
            numerical_failed_checks = {}
        if self.datetime_columns is not None:
            datetime_failed_checks = self._transform_datetime_checker(
                X, type_failed_checks, batch_mode
            )
        else:
            datetime_failed_checks = {}

        if batch_mode:

            # read test results and raise exception if any have failed with check details
            good_df, bad_df = self.separate_passes_and_fails(
                type_failed_checks,
                null_failed_checks,
                value_failed_checks,
                numerical_failed_checks,
                datetime_failed_checks,
                X,
            )

            return good_df, bad_df

        else:

            # read test results and raise exception if any have failed with check details
            self.raise_exception_if_checks_fail(
                type_failed_checks,
                null_failed_checks,
                value_failed_checks,
                numerical_failed_checks,
                datetime_failed_checks,
            )

            return X

    def _check_type(self, obj, obj_name, options):
        """Method to check the type of a given object.

        Parameters
        ----------
        obj : any
            Object to check type of.

        obj_name : str
            Name of object, used in error message.

        options : list
            Expected options for obj. A single type may be passed in list or multiple
            options can be passed.

        """

        if type(obj) not in options:
            raise TypeError(
                f"unexpected type for {obj_name}\n  Expected: {options}\n  Actual: {type(obj)}"
            )

    def _is_string_value(self, string, string_name, check_value):
        """Method to check the value of a given string.

        Parameters
        ----------
        string : any
            string to check value.

        string_name : str
            Name of string, used in error message.

        check_value : str
            Expected value for string.

        """

        if string is not check_value:
            raise ValueError(
                f"unexpected str option for {string_name}\n  Expected: {check_value}\n  Actual: {string}"
            )

    def _is_subset(self, obj_name, columns, dataframe):
        """Method to check if columns are a subset of a dataframe columns.

        Parameters
        ----------
        obj_name : str
            Name of object, used in error message.

        columns : list
           Lists of subset columns.

        dataframe : pd.DataFrame
            Dataframe to check for subset of columns.

        """

        if not set(columns).issubset(dataframe.columns):
            unexpected_columns = list(set(columns) - set(dataframe.columns))

            raise ValueError(
                f"{obj_name} is not a subset of the training datframe columns\n  Unexpected columns: {unexpected_columns}"
            )

    def _is_empty(self, obj_name, obj):
        """Method to check if an object is empty.

        Parameters
        ----------
        obj_name : str
            Name of object, used in error message.

        obj : any
           object to run check on.

        """

        if obj is not None and not obj:
            raise ValueError(f"{obj_name} is empty")

    def _is_listed_in_columns(self):
        """Method to check if all columns passed are included in the columns attribute."""

        col_lst = []

        for cols in [
            self.categorical_columns,
            self.numerical_columns,
            self.datetime_columns,
            self.skip_infer_columns,
        ]:
            if cols is not None and cols != "infer":
                col_lst += cols

        cols_diff = sorted(set(col_lst) - set(self.columns))
        if len(cols_diff) > 0:
            raise ValueError(
                f"Column(s); {cols_diff} are not listed when initialising column attribute"
            )

    def _df_is_empty(self, obj_name, df):
        """Method to check if a dataframe is empty.

        Parameters
        ----------
        obj_name : str
            Name of object, used in error message.

        df : pd.DataFrame
           dataframe to run check on.

        """

        if df.empty:
            raise ValueError(f"{obj_name} is empty")
