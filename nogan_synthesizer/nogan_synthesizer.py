"""Contains NoGANSynth Class to Generate Tabular Synthesis Data"""

import numpy as np
import pandas as pd
from typing import List
import re


class NoGANSynth:
    """
    The main NoGAN Synthesizer Class
    """
    def __init__(self, data: pd.DataFrame, random_seed: int = None) -> None:
        """
        Initialize Data, no of objects, features, no of features and epsilon

        Args:
            data (pd.DataFrame): Input Pandas DataFrame to be trained on
            random_seed (int, optional): Random seed to be set before 
                                        operations. If set random seed is set using `np.random.seed(random_seed)`. Defaults to None
                                                
        Raises:
            TypeError: Throws error if Input Dataset is not a Pandas DataFrame
            ValueError: Throws error if Input Dataset is empty
            TypeError: Throws error if non numerical columns are present in the 
                        Input Dataset
            ValueError: Throws error if there are special characters or space in 
                        column names of Input Dataset

        Returns:
            None   
        """

        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input Dataset should be a Pandas DataFrame!!")

        if data.empty:
            raise ValueError("Input Dataset should not be empty!!") 

        if len(data.select_dtypes(exclude=['number']).columns) != 0:
            raise TypeError("There are non numeric columns present in the Input Dataset. Please process them using wrap_category_columns function")

        if re.search(r'[^a-zA-Z0-9_]', "".join(data.columns)):
            raise ValueError("There are special characters or space in the Column Names of Input Dataset. Please clean them before processing.")

        self.data = np.array(data.copy())
        self.features = data.columns
        self.int_columns = data.select_dtypes(include=['int']).columns  
        self.nobs = len(self.data)
        self.median = np.median(self.data, axis = 0)

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

        if self.random_seed:
            np.random.seed(self.random_seed)

        # Get bin indices for each row in the data
        if bins is None:
            self.bins_per_feature = [np.random.randint(50, 100) 
                                     for _ in range(self.n_features)]
        else:
            self.bins_per_feature = bins

        # create quantile table bin_edges, one row for each feature
        self.bin_edges = [np.quantile(self.data[:, k],
                                      np.arange(0, 1 + self.eps,
                                                1/self.bins_per_feature[k]),
                                      axis=0
                                      ) for k in range(self.n_features)]

        bin_keys = {}
        for obs in self.data:   
            # For each observation column get the respective bin index based 
            # on the quantile table bin_edges
            bin_indices = [np.clip(np.searchsorted(self.bin_edges[k], 
                                                   obs[k], side='right')-1,
                                   0,
                                   len(self.bin_edges[k])-2
                                   )
                           for k in range(self.n_features)]

            # Convert the bin_indices into a string of comma separated values
            # They are the multivariate keys used in bin_keys dictionary
            key_str = ', '.join(map(str, bin_indices))              

            # Calculate lower & upper bounds
            lower_val = [self.bin_edges[k][bin_indices[k]]
                         for k in range(self.n_features)]
            upper_val = [self.bin_edges[k][1 + bin_indices[k]]
                         for k in range(self.n_features)]

            # frequency & sum_obs are the running counts & sum of observations
            if key_str in bin_keys:
                bin_keys[key_str]["frequency"] += 1
                bin_keys[key_str]["sum_obs"] += obs
            else:
                bin_keys[key_str] = {"sum_obs": obs,
                                     "frequency": 1,
                                     "value": bin_indices,
                                     "lower_val": lower_val,
                                     "upper_val": upper_val}

        self.bin_keys = bin_keys


    def _random_bin_counts(self, no_of_rows: int) -> np.array:
        """
        Args:
            no_of_rows (int): Row Count

        Returns:
            np.array: Random Bin Count Array
        """
        pvals = []
        for key in self.bin_keys:
            pvals.append(self.bin_keys[key]["frequency"]/self.nobs)
        return(np.random.multinomial(no_of_rows, pvals))

    def generate_synthetic_data(self, no_of_rows: int = 100, 
                                stretch_type: List = None,
                                stretch: List = None,
                                gen_random_seed: int = None,
                                debug: bool = False
                                ) -> pd.DataFrame:
        """
        The main function which Generates the Synthetic Data.
        It calls random bin to create the multinomial bin counts.
        Then for each key, gets the lower and upper bound and generates an observation (random uniform value) between those bounds.
        Once the new observations list is generated, convert into a pandas synthetic dataframe and return.
        
        Args:
            no_of_rows (int): Row Count
            stretch_type (List): List of values {"Gaussian","Uniform"}. Specifies 
                                the Sampling Type for each column. Any value in List which is not `Uniform` will be treated as `Gaussian`. Default value is `Uniform` for all columns.
            stretch (List): Specifies the stretching factor (scale) for each 
                            column. Values between 0 and 1 with `Uniform` stretch type keeps generated observations inside each
                            hyperrectangle. Default value is 1.0 for all columns.
            gen_random_seed (int, optional): Random seed to be set before 
                                        generation. It is set using `np.random.seed(random_seed)`. Defaults to None. If set to None, the random seed set at instantiation will be used
            debug (bool): Flag to activate debugging. Default is False

        Returns:
            pd.DataFrame: Generate Synthetic Pandas DataFrame
        """
        if gen_random_seed:
            np.random.seed(gen_random_seed)
        elif self.random_seed:
            np.random.seed(self.random_seed)
            
        if not stretch_type:
            stretch_type = ["Uniform" for _ in range(self.n_features)]
        if not stretch:
            stretch = [1.0 for _ in range(self.n_features)]
        stretch = np.array(stretch, dtype = np.float32)
        
        if debug:
            print(f"List `stretch_type`: {stretch_type}")
            print(f"List `stretch`: {stretch}")
                
        bin_count_random = self._random_bin_counts(no_of_rows)
        data_synth = []
            
        for i, key in enumerate(self.bin_keys):
            lower_val = self.bin_keys[key]["lower_val"]
            upper_val = self.bin_keys[key]["upper_val"]
            mean_val = self.bin_keys[key]["sum_obs"] / self.bin_keys[key]["frequency"]
            count = bin_count_random[i]
            for j in range(count):
                new_obs = np.empty(self.n_features)  # synthesized obs
                for k in range(self.n_features):
                    if stretch[k] < 0:
                        new_obs[k] = np.random.uniform(lower_val[k], 
                                                       upper_val[k])
                    else:
                        if stretch_type[k] == 'Uniform':
                            deviate = np.random.uniform(-1, 1) 
                        else:
                            deviate = np.random.normal(0, 1)
                        dist_to_edge = \
                        min(mean_val[k] - lower_val[k], upper_val[k] - mean_val[k])
                        new_obs[k] = \
                        mean_val[k] + dist_to_edge * stretch[k] * deviate
                            
                data_synth.append(new_obs)
        data_synth = pd.DataFrame(data_synth, columns=self.features)

        for col in self.int_columns:
            data_synth[col] = data_synth[col].astype(int)

        return data_synth
