import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

class LearningMachine:
    def __init__(self, data):
        self.name = None  # name of ML algorithm
        self.model = None

        self.data = data  # training data object
        self.output = np.full(np.shape(self.data.labels), None)  # output of ML algorithm

        # self.metrics = {}  # dict of metrics, e.g. accuracy
        self.fitting_parameters = dict()
        self.run = 0
        pass

    def __str__(self):
        return str(self.name)

    def _fit(self):
        # self.fitting_parameters = {}
        pass

    def _predict(self):
        # self.output = self.output
        pass

    def _evaluate_training(self):
        # self.metrics = {}
        pass

    def _evaluate_testing(self):
        # self.metrics = {}
        pass

    def metrics_print(self):
        mat = confusion_matrix(self.data.labels, self.output)
        report = classification_report(self.data.labels, self.output)
        print("Confusion Matrix:\n",mat)
        print("Clasification Report:\n",report)
        pass


class LDA(LearningMachine):
    def __init__(self, data):
        super().__init__(data)

        self.name = "LDA"

        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        self.model = LinearDiscriminantAnalysis()

        # self._fit()
        pass

    def _fit(self):
        self.fitting_parameters[self.run] = self.model.fit(self.data.df, self.data.labels)
        # print(f"The {self.name} model has been trained on the given data")
        # print("with parameters:" + str(self.model.get_params()))
        self.run += 1
        pass

    def _refit(self):
        self.fitting_parameters = self.model.fit(self.data_class.df, self.data_class.labels).get_params()
        print(f"The {self.name} model has been trained on the given data")
        pass

    def _predict(self, X):
        """
        X is a dataframe containing the features of the samples for which we want to predict their labels
        """

        self.output = self.model.predict(X)
        print(f"The {self.name} model has predicted X")
        return self.output

    def _evaluate_training(self, measure="accuracy") -> float:
        if measure == "accuracy":
            score = self.model.score(self.data_class.df, self.data_class.labels)
            self.metrics[str(self.run)+"_train_Acc"] = score

        if measure == "confusionmatrix":
            cm = confusion_matrix(self.data.labels, self._predict(self.data.df))
            self.metrics[str(self.run)+"_cm"] = cm

        print(f"The {self.name} model has been trained on the given data")
        return score

    # def _evaluate_testing(self, X, Y, measure="accuracy"):
    #     """
    #     X is a dataframe containing the features of the samples for which we want to predict their labels
    #     Y is a dataframe containing the true labels for the data X  (1 = like, 0 = dislike)
    #     """
    #     print(f"The {self.name} model has been trained on the given data")
    #
    #     self.model.score(X, Y)
    # pass
