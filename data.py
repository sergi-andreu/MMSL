import pandas as pd

class Data:
    
    def __init__(self, path):
        
        self.path = path
        self.df = pd.read_csv(path)
        self.column_names = list(self.df.columns)

        self.labels = self.df["Label"]
        

        self.num_bound_col = ["danceability", "energy", "speechiness","acousticness",
                              "instrumentalness", "liveness", "valence"]
        
        self.num_unbound_col = ["loudness", "tempo"]
        
        self.class_col = ["key"]
        self.binary_col = ["mode"]

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
        
        assert(len( self.num_bound_col) + len(self.num_unbound_col) + 
               len(self.class_col) + len(self.binary_col) + 1 == len(self.column_names))
               
        self.num_bound_col = self.df[self.num_bound_col]
        self.num_unbound_col = self.df[self.num_unbound_col]
        self.class_col = self.df[self.class_col]
        self.binary_col = self.df[self.binary_col]
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
    
    
    def train_val_split(self, train=0.8):
        
        assert self.preprocessed = True
        
        n_rows = self.df.count
        
        n_training_rows = int(train*n_rows)
        
        self.training_df = self.df[:n_training_rows, :]
        self.validation_df = self.df[n_training_rows:, :]
        
        pass

