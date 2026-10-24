# Isi file test_app.py
import pytest
import pandas as pd

# Test sederhana untuk memastikan requirements terhubung
def test_dummy_pass():
    assert True == True

def test_dataframe_creation():
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    assert len(df) == 2