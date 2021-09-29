class LearningMachine:
    
    def __init__(self, data_class):
        
        self.name = None
        
        self.input = data_class
        
        self.labels = data_class.labels
        
        self.methods = ["LDA", "RDA"]
        
        self.output = np.full(np.shape(self.labels), None)
        self.metrics = {}
        
        self.fitting_parameters = {}
        
        self.model = None
        
        
    def __str__(self):
        return str(self.name)
    
        
    def _fit(self):
        
        self.fitting_parameters = {}
        
        pass
    
        
    def _predict(self):
        
        self.output = self.output
        
        pass
        
        
    def _evaluate_training(self):
        
        self.metrics = {}
        
        pass
    
    def _evaluate_testing(self):
        
        self.metrics = {}
        
        pass
        
class LDA(LearningMachine):
    
    def __init__(self, data):
        super().__init__(data)
        self.name = "LDA"
        
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        
        self.model = LinearDiscriminantAnalysis()

        pass

    def _fit(self):
        
        self.fitting_parameters = self.model.fit(self.input.df, self.input.labels).get_params()
        
        print(f"The {self.name} model has been trained on the given data")
        pass
    
    def _predict(self, X):
        
        """
        X is a dataframe containing the features of the samples for which we want to predict their labels
        """
        
        self.model.predict(X)
        
        pass
    
    def _evaluate_training(self, measure="accuracy"):
        
        if measure == "accuracy":
            sc = self.model.score(self.input.df,self.labels)
            self.metrics["Train_Acc"]  = sc
        
    def _evaluate_testing(self, X, Y, measure="accuracy"):
        
        """
        X is a dataframe containing the features of the samples for which we want to predict their labels
        Y is a dataframe containing the true labels for the data X  (1 = like, 0 = dislike)
        """
        
        
        self.model.score(X, Y)