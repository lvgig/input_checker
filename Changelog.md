# Change log

----
# 0.3.9
- Stripped of internal files and updated md documents

# 0.3.8
- Updated example notebook to use wine dataset, added new method to check column attributes and minor edits to prepare for open source release

# 0.3.7
- Updated separate_passes_and_fails to handle cases where the same row fails multiple expected values checks

# 0.3.6
- Added fix to transform_type_checker so it now skips a column if it consists of only missing values

# 0.3.5
- Changed type checker to compare type for each row individually when operating in batch mode
- Added fix to transform_numerical_checker and transform_datetime_checker to skip rows which fail type checks

# 0.3.3

- Changed transform method to have a batch mode, which returns two dataframes: rows which fail the input checks and rows which pass the input checks. Rows which fail are given an extra field called 'failed_checks' which contains information about the specific checks which were failed by that set of input values.

# 0.3.2

- Added checks for datetime columns

# 0.3.1

- Added skip_infer_columns parameter that solves issues with the infer column option for numerical_columns and categorical_columns

# 0.3.0

- Replaced checker classes with a single input checker class containing all checks
- Added numerical range checks
- Exceptions now raised simultaneously for failed checks

# 0.2.3

- Swap dependency from prepro to tubular

# 0.2.2

- Add build pipelines
- Flake8 style changes

# 0.2.1

- Update minimum required prepro version to be 0.2.4+ 

# 0.2.0

- Update minimum required prepro version to be 0.2.0+ 
- Update unit tests to be same standard as tubular

# 0.1.3

- Add custom exceptions
