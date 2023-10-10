import pandas as pd
from typing import List, Tuple
import re


def wrap_category_columns(data: pd.DataFrame,
                          cat_cols: List[str]) -> Tuple:
    """
    Args:
        data (pd.DataFrame): Pandas DataFrame
        cat_cols (List[str]): List of all categorical columns

    Raises:
        TypeError: Throws error if Input Dataset is not a Pandas DataFrame
        ValueError: Throws error if Input Dataset is empty
        ValueError: Throws error if there are special characters or space in column 
                    names of Input Dataset
        TypeError: Throws error if 'cat_cols' is not a list
        ValueError: Throws error if 'cat_cols' is empty 

    Returns:
        Tuple: Returns a Pandas DataFrame with all category columns wrapped & Dictionaries 'idx_to_key' and 'key_to_idx' which contain key-index, index-key pairs of flag vector
    """
    
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input Dataset should be a Pandas DataFrame!!")

    if data.empty:
        raise ValueError("Input Dataset should not be empty!!")

    if re.search(r'[^a-zA-Z0-9_]', "".join(data.columns)):
        raise ValueError("There are special characters or space in the Column Names of Input Dataset. Please clean them before processing.")

    if not isinstance(cat_cols, list):
        raise TypeError("Input 'cat_cols' should a List!!")

    if not cat_cols:
        raise ValueError("'cat_cols' should not be Empty!!")

    df = data.copy()

    cat_data = df[cat_cols]
    num_cols = [f for f in data.columns if f not in cat_cols]

    flag_vector = [list(row) for row in 
                   cat_data.drop_duplicates().to_records(index=False)]

    key_to_idx = {str(v).replace("[","").replace("]",""):i 
                  for i, v in enumerate(flag_vector,1)}
    idx_to_key = {i:tuple(v) for i, v in enumerate(flag_vector,1)}

    df["cat_label"] = \
        [key_to_idx[str(tuple(row)).replace("(","").replace(")","").strip(",")]
         for row in cat_data.to_records(index=False)]

    df = df[num_cols + ["cat_label"]]

    return df, idx_to_key, key_to_idx


def unwrap_category_columns(data: pd.DataFrame, idx_to_key: dict,
                            cat_cols: List[str]) -> pd.DataFrame:
    """
    Args:
        data (pd.DataFrame): Pandas DataFrame
        idx_to_key (dict): Dictionary that holds the key-index pairs of the flag 
                            vector
        cat_cols (List[str]): List of all categorical columns

    Raises:
        TypeError: Throws error if Input Dataset is not a Pandas DataFrame
        ValueError: Throws error if Input Dataset is empty
        ValueError: Throws error if there are special characters or space in column 
                    names of Input Dataset
        TypeError: Throws error if 'cat_label' column is not present in the 
                    Input Dataset
        TypeError: Throws error if 'idx_to_key' is not a Dictionary
        ValueError: Throws error if 'idx_to_key' is empty
        TypeError: Throws error if 'cat_cols' is not a List
        ValueError: Throws error if 'cat_cols' is empty

    Returns:
        pd.DataFrame: Pandas DataFrame with expanded Categorical Columns
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input Dataset should be a Pandas DataFrame!!")

    if data.empty:
        raise ValueError("Input Dataset should not be empty!!")

    if re.search(r'[^a-zA-Z0-9_]', "".join(data.columns)):
        raise ValueError("There are special characters or space in the Column Names of Input Dataset. Please clean them before processing.")

    if "cat_label" not in data.columns:
        raise TypeError("Column named 'cat_label' is expected and not present!!")

    if not isinstance(idx_to_key, dict):
        raise TypeError("'idx_to_key' should a Dictionary!!")

    if not idx_to_key:
        raise ValueError("'idx_to_key' should not be empty!!")

    if not isinstance(cat_cols, list):
        raise TypeError("'cat_cols' should a List!!")

    if not cat_cols:
        raise ValueError("'cat_cols' should not be empty!!")

    df = data.copy()

    df_cat = pd.DataFrame([idx_to_key[idx] for idx in df.cat_label], 
                          columns=cat_cols)

    data_unwrapped = pd.concat([df, df_cat], axis=1)

    return data_unwrapped.drop(["cat_label"], axis=1)
