import pandas as pd

class Data:
    
    def __init__(self, path):
        
        self.path = path
        self.df = pd.read_csv(path)
        self.column_names = list(self.df.columns)
        
        self.num_bound_col = ["danceability", "energy", "speechiness","acousticness",
                              "instrumentalness", "liveness", "valence"]
        
        self.num_unbound_col = ["loudness", "tempo"]
        
        self.class_col = ["key"]
        self.binary_col = ["mode"]
        pass
    
    def split_df(self):
        
        self.num_bound_col = ["danceability", "energy", "speechiness","acousticness",
                              "instrumentalness", "liveness", "valence"]
        
        
        assert(len( self.num_bound_col) + len(self.num_unbound_col) + 
               len(self.class_col) + len(self.binary_col) + 1 == len(self.column_names))
               
        
        self.num_bound_col = self.df[self.num_bound_col]
        self.num_unbound_col = self.df[self.num_unbound_col]
        self.class_col = self.df[self.class_col]
        self.binary_col = self.df[self.binary_col]
        pass
        
    def normalize_num_bound_col(self):
        
        self.mean_num_bound_col = self.num_bound_col.mean()
        self.std_num_bound_col = self.num_bound_col.std()
        
        self.num_bound_col = (self.num_bound_col - self.mean_num_bound_col) / self.std_num_bound_col
        pass
        
    def normalize_num_unbound_col(self):

        self.mean_num_unbound_col = self.num_unbound_col.mean()
        self.std_num_unbound_col = self.num_unbound_col.std()
        
        self.num_unbound_col = (self.num_unbound_col - self.mean_unnum_bound_col) / self.std_num_bound_col
        pass
        
    def preprocess_class_col(self):
        
        self.class_col = pd.get_dummies(self.class_col["key"], prefix='key')
        pass
    
    def normalize_binary(self):
        
        self.mean_binary_col = self.binary_col.mean()
        self.std_binary_col = self.binary_col.std()
        
        self.binary_col = (self.binary_col - self.mean_binary_col) / self.std_binary_col
        
        
        return col
    pass
