"""Tests for `nogan_synthesizer` package."""

import numpy as np
import pandas as pd
import pytest
import re
from nogan_synthesizer import NoGANSynth
from nogan_synthesizer.preprocessing import wrap_category_columns, unwrap_category_columns


# Define test cases
def test_nogan_synth():

    # Input DataSet is not a Pandas DataFrame (Expecting a TypeError)
    with pytest.raises(TypeError):
        df = [2, 3, 4]
        nogan = NoGANSynth(df)

    # Empty Pandas DataFrame (Expecting a ValueError)
    with pytest.raises(ValueError):
        df = pd.DataFrame()
        nogan = NoGANSynth(df)

    # Non Numerical Columns in DataFrame (Expecting a TypeError)
    with pytest.raises(TypeError):
        df = pd.DataFrame({"col1": [2, 5, 6],
                           "col2": [1.04, 4.22, 8.32],
                           "col3": ["abc", "ghd", "dds"]
                           })
        nogan = NoGANSynth(df)

    # Column Names with spaces and special characters (Expecting a ValueError)
    with pytest.raises(ValueError):
        df = pd.DataFrame(np.random.rand(10000, 5), columns=["x@1", "x'2", "xअ3", "x4", "x_5"])
        nogan = NoGANSynth(df)
        
    # No Errors expected
    real_data = pd.DataFrame(np.random.rand(10000, 5), columns=["x1", "x2", "x3", "x4", "x_5"])

    nogan = NoGANSynth(real_data)
    nogan.fit()

    n_synth_rows = len(real_data)
    synth_data = nogan.generate_synthetic_data(no_of_rows=n_synth_rows)

    assert len(synth_data.columns) == 5 and re.search(r'[^a-zA-Z0-9_]', 
                                                      "".join(synth_data.columns)) is None

def test_wrap_category_columns():
    
    # Input DataSet is not a Pandas DataFrame (Expecting a TypeError)
    with pytest.raises(TypeError):
        df = [2, 3, 4]
        cat_cols = {"col1", "col2", "col3"}
        wrapped_df, idx_to_key, key_to_idx = wrap_category_columns(df, cat_cols)

    # Empty Pandas DataFrame (Expecting a ValueError)
    with pytest.raises(ValueError):
        df = pd.DataFrame()
        cat_cols = {"col1", "col2", "col3"}
        wrapped_df, idx_to_key, key_to_idx = wrap_category_columns(df, cat_cols)

    # Column Names with spaces and special characters (Expecting a ValueError)
    with pytest.raises(ValueError):
        df = pd.DataFrame(np.random.rand(10000, 5), columns=["x@1", "x'2", "xअ3", "x4", "x_5"])
        cat_cols = {"col1", "col2", "col3"}
        wrapped_df, idx_to_key, key_to_idx = wrap_category_columns(df, cat_cols)

    # cat_cols not a list (Expecting a TypeError)
    with pytest.raises(TypeError):    
        df = pd.DataFrame({"col1": [2, 5, 6], 
                        "col2": [1.04, 4.22, 8.32], 
                        "col3": ["abc", "ghd", "dds"], 
                        "col4": ["high", "low", "medium"]})

        cat_cols = pd.DataFrame()
        wrapped_df, idx_to_key, key_to_idx = wrap_category_columns(df, cat_cols)

    # cat_cols is an empty list (Expecting a ValueError)
    with pytest.raises(ValueError):
        df = pd.DataFrame({"col1": [2, 5, 6], 
                        "col2": [1.04, 4.22, 8.32], 
                        "col3": ["abc", "ghd", "dds"], 
                        "col4": ["high", "low", "medium"]})

        cat_cols = []
        wrapped_df, idx_to_key, key_to_idx = wrap_category_columns(df, cat_cols)  

    # No Errors expected
    
    # 2 categorical columns (numerical and text) & 1 numerical column
    df = pd.DataFrame({"col1": [2, 5, 6], 
                       "col2": ["high", "low", "medium"],
                       "col5": [2, 3, 4]                       
                    })   
    cat_cols = ["col2", "col5"]
    wrapped_df, idx_to_key, key_to_idx = wrap_category_columns(df, cat_cols)
    assert len(wrapped_df.columns) == 2 
    
    # Only 2 categorical columns (numerical and text)
    df = pd.DataFrame({"col1": [2, 5, 6], 
                    "col2": ["high", "low", "medium"]
                    })   
    cat_cols = ["col1", "col2"]
    wrapped_df, idx_to_key, key_to_idx = wrap_category_columns(df, cat_cols)
    assert len(wrapped_df.columns) == 1

    # Only 2 categorical columns (Both text)
    df = pd.DataFrame({
                    "col3": ["abc", "ghd", "dds"], 
                    "col4": ["high", "low", "medium"]
                    })

    cat_cols = ["col3", "col4"]
    wrapped_df, idx_to_key, key_to_idx = wrap_category_columns(df, cat_cols)
    assert len(wrapped_df.columns) == 1    
    
    # 3 categorical columns (2 text & 1 numerical)
    df = pd.DataFrame({"col1": [2, 5, 6], 
                    "col2": [1.04, 4.22, 8.32], 
                    "col3": ["abc", "ghd", "dds"], 
                    "col4": ["high", "low", "medium"], 
                    "col5": [2, 3, 4]
                    })

    cat_cols = ["col3", "col4", "col5"]
    wrapped_df, idx_to_key, key_to_idx = wrap_category_columns(df, cat_cols)
    assert len(wrapped_df.columns) == 3

    # 3 categorical columns (All numerical)

    df = pd.DataFrame({"col1": np.round(np.random.uniform(-2.0, 2.0, 
                                                        size=400),
                                        2),
                        "col2": np.round(np.random.uniform(-2.0, 2.0,
                                                        size=400),
                                        2),
                        "col3": np.round(np.random.uniform(-2.0, 2.0, 
                                                        size=400)
                                        )
                        })

    cat_cols = ["col1", "col2", "col3"]
    wrapped_df, idx_to_key, key_to_idx = wrap_category_columns(df, cat_cols)
    assert len(wrapped_df.columns) == 1

    # 3 categorical columns (3 text)
    df = pd.DataFrame({"col1": [2, 5, 6], 
                    "col2": [1.04, 4.22, 8.32], 
                    "col3": ["abc", "ghd", "dds"], 
                    "col4": ["high (temp)", "low (temp)", "medium (temp)"], 
                    "col5": ["bad [rating]","good [rating]","satisfactory [rating]"]
                    })

    cat_cols = ["col3", "col4", "col5"]
    wrapped_df, idx_to_key, key_to_idx = wrap_category_columns(df, cat_cols)
    assert len(wrapped_df.columns) == 3


def test_unwrap_category_columns():
    
    # # Input DataSet is not a Pandas DataFrame (Expecting a TypeError)
    with pytest.raises(TypeError):
        df = ["col1","col2","cat_label"]
        idx_to_key = {1: ('abc', 'high', 2), 
                      2: ('ghd', 'low', 3), 
                      3: ('dds', 'medium', 4)}
        cat_cols = ["col3", "col4", "col5"]
        unwrapped_df = unwrap_category_columns(df, idx_to_key, cat_cols)

    # Empty Pandas DataFrame (Expecting a ValueError)
    with pytest.raises(ValueError):
        df = pd.DataFrame()
        idx_to_key = {1: ('abc', 'high', 2), 
                      2: ('ghd', 'low', 3), 
                      3: ('dds', 'medium', 4)}
        cat_cols = ["col3", "col4", "col5"]
        unwrapped_df = unwrap_category_columns(df, idx_to_key, cat_cols)
    
    # Column Names with spaces and special characters (Expecting a ValueError)
    with pytest.raises(ValueError):    
        df = pd.DataFrame({"col@1": [2, 5, 6],
                        "col'2": [1.04, 4.22, 8.32],
                        "cat_label": [1, 2, 3]})
        idx_to_key = {1: ('abc', 'high', 2), 
                    2: ('ghd', 'low', 3), 
                    3: ('dds', 'medium', 4)}
        cat_cols = ["col3", "col4", "col5"]
        unwrapped_df = unwrap_category_columns(df, idx_to_key, cat_cols)    

    # cat_label column not present (Expecting a TypeError)
    with pytest.raises(TypeError):    
        df = pd.DataFrame({"col1": [2, 5, 6],
                        "col2": [1.04, 4.22, 8.32],
                        "cat_name": [1, 2, 3]})
        idx_to_key = {1: ('abc', 'high', 2), 
                    2: ('ghd', 'low', 3), 
                    3: ('dds', 'medium', 4)}
        cat_cols = ["col3", "col4", "col5"]
        unwrapped_df = unwrap_category_columns(df, idx_to_key, cat_cols)

    # idx_to_key not a Dictionary (Expecting a TypeError)
    with pytest.raises(TypeError):    
        df = pd.DataFrame({"col1": [2, 5, 6],
                            "col2": [1.04, 4.22, 8.32],
                            "cat_label": [1, 2, 3]})
        idx_to_key = [1, 2, 3]
        cat_cols = ["col3", "col4", "col5"]
        unwrapped_df = unwrap_category_columns(df, idx_to_key, cat_cols)    
    
    # idx_to_key is empty (Expecting a ValueError)
    with pytest.raises(TypeError):    
        df = pd.DataFrame({"col1": [2, 5, 6],
                            "col2": [1.04, 4.22, 8.32],
                            "cat_label": [1, 2, 3]})
        idx_to_key = []
        cat_cols = ["col3", "col4", "col5"]
        unwrapped_df = unwrap_category_columns(df, idx_to_key, cat_cols)      
    
    # cat_cols not a list (Expecting a TypeError)
    with pytest.raises(TypeError):      
        df = pd.DataFrame({"col1": [2, 5, 6],
                           "col2": [1.04, 4.22, 8.32],
                           "cat_label": [1, 2, 3]})
        idx_to_key = {1: ('abc', 'high', 2), 
                      2: ('ghd', 'low', 3), 
                      3: ('dds', 'medium', 4)}
        cat_cols = {"col3", "col4", "col5"}
        unwrapped_df = unwrap_category_columns(df, idx_to_key, cat_cols)
         
    # cat_cols is empty (Expecting a ValueError)
    with pytest.raises(ValueError):      
        df = pd.DataFrame({"col1": [2, 5, 6],
                           "col2": [1.04, 4.22, 8.32],
                           "cat_label": [1, 2, 3]})
        idx_to_key = {1: ('abc', 'high', 2), 
                      2: ('ghd', 'low', 3), 
                      3: ('dds', 'medium', 4)}
        cat_cols = []
        unwrapped_df = unwrap_category_columns(df, idx_to_key, cat_cols)

    # No Errors expected
    df = pd.DataFrame({"col1": [2, 5, 6],
                        "col2": [1.04, 4.22, 8.32],
                        "cat_label": [1, 2, 3]})
    idx_to_key = {1: ('abc', 'high', 2), 
                    2: ('ghd', 'low', 3), 
                    3: ('dds', 'medium', 4)}
    cat_cols = ["col3", "col4", "col5"]
    unwrapped_df = unwrap_category_columns(df, idx_to_key, cat_cols)
    assert len(unwrapped_df.columns) == 5      