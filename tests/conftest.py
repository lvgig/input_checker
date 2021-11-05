import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def df():
    """Create simple DataFrame to use in other tests"""

    data = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, np.NaN],
            "b": ["a", "b", "c", "d", "e", "f", np.NaN],
            "c": ["a", "b", "c", "d", "e", "f", np.NaN],
        }
    )

    data["c"] = data["c"].astype("category")

    return data
