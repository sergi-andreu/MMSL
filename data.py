import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Data:
    
    def __init__(self, path):
        
        self.path = path
        self.df = pd.read_csv(path)
        self.column_names = list(self.df.columns)

        self.labels = self.df["Label"]
        

        self.num_bound_col_names = ["danceability", "energy", "speechiness","acousticness",
                              "instrumentalness", "liveness", "valence"]
        
        self.num_unbound_col_names = ["loudness", "tempo"]
        
        self.class_col_names = ["key"]
        self.binary_col_names = ["mode"]

        self.preprocessed = False

        self.bound_preprocessed = False
        self.unbound_preprocessed = False
        self.class_preprocessed  = False
        self.binary_preprocessed = False

        pass
    

    def __str__(self):
        return str(self.path)


    def _split_df(self):

        assert self.preprocessed == False
        
        assert(len( self.num_bound_col_names) + len(self.num_unbound_col_names) + 
               len(self.class_col_names) + len(self.binary_col_names) + 1 == len(self.column_names))
               
        self.num_bound_col = self.df[self.num_bound_col_names]
        self.num_unbound_col = self.df[self.num_unbound_col_names]
        self.class_col = self.df[self.class_col_names]
        self.binary_col = self.df[self.binary_col_names]
        pass

        
    def _normalize_num_bound_col(self):
        
        self.mean_num_bound_col = self.num_bound_col.mean()
        self.std_num_bound_col = self.num_bound_col.std()
        
        self.num_bound_col = (self.num_bound_col - self.mean_num_bound_col) / self.std_num_bound_col

        self.bound_preprocessed = True
        pass
        
    def _normalize_num_unbound_col(self):

        self.mean_num_unbound_col = self.num_unbound_col.mean()
        self.std_num_unbound_col = self.num_unbound_col.std()
        
        self.num_unbound_col = (self.num_unbound_col - self.mean_num_unbound_col) / self.std_num_unbound_col

        self.unbound_preprocessed = True
        pass
        
    def _preprocess_class_col(self):
        
        self.class_col = pd.get_dummies(self.class_col["key"], prefix='key')

        self.class_preprocessed = True
        pass
    
    def _normalize_binary(self):
        
        self.mean_binary_col = self.binary_col.mean()
        self.std_binary_col = self.binary_col.std()
        
        self.binary_col = (self.binary_col - self.mean_binary_col) / self.std_binary_col

        self.binary_preprocessed = True
        pass
        
    pass


    def _append_cols(self):
        
        self.df = pd.concat([self.num_bound_col, self.num_unbound_col, self.class_col, self.binary_col], axis=1)


    def _preprocess(self, bound_bool = True, unbound_bool = True, class_bool = True, binary_bool = True):

        """
        Booleans stand for normalising the data types or not normalising the data types
        """

        self._split_df()

        if bound_bool : self._normalize_num_bound_col()
        if unbound_bool : self._normalize_num_unbound_col()
        if class_bool : self._preprocess_class_col()
        if binary_bool : self._normalize_binary()

        self._append_cols()

        self.preprocessed = True

        pass


    def remove_duplicates(self):
        duplicated = self.df.duplicated()
        n_duplicated = np.sum(duplicated)

        idx = np.where(duplicated==True)[0]

        self.df = self.df.drop(idx).reset_index(drop=True)
        self.labels = self.labels.drop(idx).reset_index(drop=True)

        print(f"There were {n_duplicated} duplicated elements in the dataset, and have been removed from the dataframe")
        

    
    def visualize(self, cmap=None, labels=None):

        from matplotlib import colors

        try:
            if labels==None:
                labels = self.labels
        except: pass

        try:
            if cmap==None:
                cmap = colors.ListedColormap(['r', 'b'], 2)
        except: pass

        plotting_features = self.num_bound_col_names + self.num_unbound_col_names
        plotting_df = self.df[self.num_bound_col_names]
        pd.plotting.scatter_matrix(plotting_df, alpha=0.4, figsize=(17,14), c=labels, cmap=cmap)
        plt.show()

        pass


