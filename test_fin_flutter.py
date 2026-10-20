"""Contains tests for fin flutter calculation

Requires pytest to be installed (pip install pytest)
To run all tests, run the command "pytest" in terminal
To run a specific test file, run the command "pytest path/to/file" in terminal
To run a specific function in a test file, run the command "pytest path/to/file::function_name" in terminal
"""

import pytest
from fin_flutter import get_flutter_velocity

def test_something():
    assert get_flutter_velocity() == pytest.approx(3481.015629977519)