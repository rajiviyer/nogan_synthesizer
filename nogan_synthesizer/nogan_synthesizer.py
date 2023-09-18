"""Contains NoGANSynth Class to Generate Tabular Synthesis Data"""

import numpy as np
import pandas as pd
from typing import List
import re
from genai_evaluation import multivariate_ecdf, ks_statistic

def wrap_category_columns(data:pd.DataFrame,
                        cat_cols:List[str])->pd.DataFrame:

    df = data.copy()

    cat_data = data[cat_cols]

    flag_vector = [list(row) for row in cat_data.drop_duplicates().to_records(index=False)]

    key_to_idx = {str(v).replace("[","").replace("]",""):i for i, v in
                enumerate(flag_vector,1)}
    idx_to_key = {i:tuple(v) for i, v in enumerate(flag_vector,1)}
    
    df["cat_label"] = [key_to_idx[str(row).replace("(","").replace(")","")] 
                    for row in cat_data.to_records(index=False)]

    return df, idx_to_key, key_to_idx

def unwrap_category_columns(data:pd.DataFrame, idx_to_key: dict,
                            cat_cols:List[str])->pd.DataFrame:

    df = data.copy()
    #cat_data = df[cat_cols]

    df_cat = pd.DataFrame([idx_to_key[idx] for idx in df.cat_label], 
                          columns = cat_cols)

    data_unwrapped = pd.concat([df,df_cat],axis = 1)
    
    return data_unwrapped


class NoGANSynth:
    """
    The main NoGAN Synthesizer Class
    """
    def __init__(self, data: pd.DataFrame, cat_cols: List = None,
                 verbose: bool = False, random_seed: int = None) -> None:
        """
        Initialize Data, no of objects, features, no of features and epsilon

        Args:
            data (pd.DataFrame): Input Pandas DataFrame to be trained on
            cat_cols (List, optional): List of Categorical Columns. 
                                        Defaults to None.
            verbose (bool): Flag to display progress of the operations. 
                            Defaults to False
            random_seed (int, optional): Random seed to be set before 
                                        operations. If set random seed is set using `np.random.seed(random_seed)`. Defaults to None                             
        Raises:
            TypeError: Throws error if Input Dataset is not a Pandas DataFrame
            
        Returns:
            None          
        """
        
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input Dataset should be a Pandas DataFrame!!")
        
        self.data = data.copy()
        if len(data.select_dtypes(exclude=['number']).columns) != 0:
            raise TypeError("There are non numeric columns present in the Input Dataset. Please process or remove them before processing")
        
        if re.search(r'[^a-zA-Z0-9_]', "".join(data.columns)):
            raise ValueError("There are special characters or space in the Column Names of Input Dataset. Please remove them before processing.")

            
        self.features = data.columns        
        self.nobs = len(self.data)
        
        # Any special characters or space in column names will be cleaned up
        self.n_features = len(self.features)
        self.eps = 0.0000000001
        self.random_seed = random_seed
        


    def fit(self, bins: List = None) -> None:
        """
        Function to create bins for each Data column.
        
        Args:
            bins (List, optional): Bins List. Defaults to None. If it is None, then random bins between 50 to 100 will be assigned. Recommended to pass a tuned hyperparameter bins list
        """
        
        # Get bin indices for each row in the data
        if bins is None:
            self.bins_per_feature = [ np.random.randint(50,100) 
                                     for _ in range(self.n_features)]
        else:
            self.bins_per_feature = bins

        # self.bin_edges = np.array([(np.histogram(npdata[:,k], 
        #                                          bins=self.bins_per_feature[k]) \
        #                     )[1] for k in range(self.n_features)], dtype='object')
        self.bin_edges = [np.quantile(self.data.iloc[:,k], 
                                      np.arange(0, 1 + self.eps, 
                                                1/self.bins_per_feature[k]), 
                                      axis=0
                                      ) for k in range(self.n_features)]
                
        bin_indices = np.array([np.clip(np.searchsorted(self.bin_edges[col], 
                                                        self.data.iloc[:, col], side='right')-1,
                                        0,
                                        len(self.bin_edges[col])-2
                                        ) for col in range(self.data.shape[1])])
        
        bin_indices = bin_indices.T

        # Find the counts of all unique list entries
        unique_entries, counts = np.unique(bin_indices, axis=0, return_counts=True)

        # Create a dictionary having each entry as key and corresponging counts and actual lists as values
        bin_keys = {}
        for entry, count in zip(unique_entries, counts):
            key_str = ', '.join(map(str, entry))
            lower_val = [self.bin_edges[k][entry[k]] for k in range(len(entry))]
            upper_val = [self.bin_edges[k][1 + entry[k]] for k in range(len(entry))]
            bin_keys[key_str] = {'frequency': count, 'value': entry, 
                                    'lower_val': lower_val, 'upper_val': upper_val
                                    }

        self.bin_keys = bin_keys
    
    def random_bin_counts(self, no_of_rows: int) -> np.array:
        """
        Function to generate multinomial bin counts with same expectation as real counts

        Args:
            no_of_rows (int): Row Count

        Returns:
            np.array: Random Bin Count Array
        """
        pvals = []
        for key in self.bin_keys:
            #print(f"bin_count[{skey}] = {bin_count[skey]}, nobs: {nobs}, bin_count[{skey}]/nobs = {bin_count[skey]/nobs}")
            pvals.append(self.bin_keys[key]["frequency"]/self.nobs)
        return(np.random.multinomial(no_of_rows, pvals))

    def generate_synthetic_data(self, no_of_rows: int) -> pd.DataFrame:
        """
        The main function which Generates the Synthetic Data.
        It calls random bin to create the multinomial bin counts.
        Then for each key, gets the lower and upper bound and generates an observation (random uniform value) between those bounds
        Once the new observations list is generated, convert into a pandas synthetic dataframe and return.

        Args:
            no_of_rows (int): Row Count

        Returns:
            pd.DataFrame: Generate Synthetic Pandas DataFrame
        """
        #print("*"*40 + "Generating Synthetic Data" + "*"*40)
        bin_count_random = self.random_bin_counts(no_of_rows)
        data_synth = []
        for i, key in enumerate(self.bin_keys):
            lower_val = self.bin_keys[key]["lower_val"]
            upper_val = self.bin_keys[key]["upper_val"]
            count = bin_count_random[i]
            #print(key, count)
            for j in range(count):
                new_obs = np.empty(self.n_features) # synthesized obs
                for k in range(self.n_features):
                    new_obs[k] = np.random.uniform(lower_val[k],
                                                   upper_val[k])   
                    #print("adding new_obs")
                data_synth.append(new_obs)               
        data_synth = pd.DataFrame(data_synth, columns = self.features)
        return data_synth