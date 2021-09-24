import pandas as pd

class Reader:
    
    def __init__(self, path):
        
        self.path = path
        self.df = pd.read_csv(path)
        self.column_names = list(self.df.columns)
        
        self.num_bound_col = ["danceability", "energy", "speechiness","acousticness",
                              "instrumentalness", "liveness", "valence"]
        
        self.num_unbound_col = ["loudness", "tempo"]
        
        self.class_col = ["key"]
        self.binary_col = ["mode"]
                                
    
    def split_df(self):
        
        print(len(self.num_bound_col) + len(self.num_unbound_col) + len(self.class_col) + len(self.binary_col))
                                
reader = Reader("project_train.csv")