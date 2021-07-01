# input_checker 

----

The `input_checker` package enables users to compare a given data frame against a benchmark data frame.
The package executes that by keeping track of the key information in the benchmark data frame and 
cross-checking the comparison data frame against those tracked characteristics.

The package currently contains five main checks;

- **Null checker:** ensures that columns with missing values in the benchmark data frame
are the only columns with missing values in the comparison data frame
- **Dtype checker:** ensures that columns in the comparison data frame are of the same data type as
in the benchmark data frame
- **Categorical value checker:** ensures that categorical columns in the comparison data frame only contain
values that exist in the benchmark data frame
- **Numerical checker:** ensures that the values of the numerical columns in
the comparison data frame lie within the minimum and maximum range of the numerical columns
in the benchmark data frame
- **Datetime checker:** ensures that the values of datetime columns in the
comparison data frame lie beyond the minimum date (optionally maximum) of datetime columns
in the benchmark data frame

The package has multiple usage areas including but not limited to ensuring that data points
sent to the model in live environment matches key characteristics of the data the model was 
initially trained on.

Here is a simple example of using input_checker to compare training data to test data;
```python
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

import input_checker
from input_checker.checker import InputChecker

# load and prepare sklearn wine dataset
wine = load_wine()

df_wine = pd.DataFrame(wine['data'], columns = wine['feature_names'])
df_wine['target'] = wine['target']

# split into train/test sets
df_train, df_test = train_test_split(df_wine, test_size=0.2)

# define numerical columns 
# please note; the original wine dataset only has numerical fields
# please refer to the example notebook under the examples folder for 
# using input_checker with different dtypes and missing values
numerical_columns = ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
       'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
       'proanthocyanins', 'color_intensity', 'hue',
       'od280/od315_of_diluted_wines', 'proline']

# define input_checker
checker = InputChecker(columns=numerical_columns,
                       numerical_columns=numerical_columns) 

# fitting input_checker
checker.fit(df_train)

# compare test data frame to the training data frame
df_test_checked = checker.transform(df_test)
```

# Installation

input_checker can be installed from PyPI simply with;

 `pip install input_checker`

# Documentation

Documentation for input_checker can be found on [readthedocs](https://input_checker.readthedocs.io/en/latest/).

# Examples

To help get started there is an example notebook in the [examples](https://github.com/lvgig/input_checker/tree/master/examples) folder that shows how to use input_checker.

# Build and test

The test framework we are using for this project is [pytest](https://docs.pytest.org/en/stable/), to run the tests follow the steps below.

First clone the repo and move to the root directory;

```shell
git clone https://github.com/lvgig/input_checker.git
cd input_checker
```

Then install input_checker in editable mode;

```shell
pip install -e . -r requirements-dev.txt
```

Then run the tests simply with pytest

```shell
pytest
```

# Contribute

`input_checker` is under active development, we're super excited if you're interested in contributing! See the `CONTRIBUTING.md` for the full details of our working practices. 

For bugs and feature requests please open an [issue](https://github.com/lvgig/input_checker/issues).

