import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

import warnings

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

        self.run = 0
        self.run_cross_validation = 0

        
        
    def __str__(self):
        return str(self.name)
    
        
    def _fit(self):
        raise ValueError("The _fit method for the given subclass is not implemented.")
        pass

    def _get_fitting_parameters(self):
        return 0
    
    def _predict(self):
        raise ValueError("The _predict method for the given subclass is not implemented.")
        pass
        

    def _evaluate(self, X=None, Y=None, measure=None):

        pass

    def _get_confusion_matrix(self, X=None, Y=None):

        try:
            if X==None:
                X = self.input.df
        except: pass

        try:
            if Y==None:
                Y = self.input.labels
        except: pass

        prediction = self._predict(X)

        cm = confusion_matrix(Y, prediction)

        return cm

    def _compute_metrics(self, X=None, Y=None, compute_accuracy=True, compute_recall=True, compute_precision=True, compute_F1=True):

        metrics = {}
        cm = self._get_confusion_matrix(X=X, Y=Y)

        if compute_accuracy:
            acc = (cm[0][0] + cm[1][1])/np.sum(cm)
            metrics["accuracy"] = acc

        if compute_recall:
            rec = (cm[0][0]) / (cm[0][0] + cm[0][1])
            metrics["recall"] = rec

        if compute_precision:
            pre = (cm[0][0]) / (cm[0][0] + cm[1][0])
            metrics["precision"] = pre

        if compute_F1:
            fscore = (2*pre*rec)/(pre+rec)
            metrics["F1"] = fscore

        return metrics


    def train_once(self, fraction_test_data=0.1, shuffle=False):

        total_input = self.input.df
        total_labels = self.input.labels

        n_data = len(self.input.df)
        n_test_data = int(fraction_test_data * n_data)

        if shuffle:
            idx = np.random.permutation(n_data)
            total_input, total_labels = total_input.reindex(idx), total_labels.reindex(idx)

        train_df = total_input[:n_data-n_test_data]
        train_labels = total_labels[:n_data-n_test_data]
        
        test_df = total_input[n_data - n_test_data:]
        test_labels = total_labels[n_data - n_test_data:]

        self._fit(input=train_df, labels=train_labels)
        
        training_sc = self._evaluate(X=train_df, Y=train_labels, measure="accuracy")
        test_sc = self._evaluate(X=test_df, Y=test_labels, measure="accuracy")

        self.metrics["Training_Accuracy_"+str(self.run)] = training_sc
        self.metrics["Test_Accuracy_"+str(self.run)] = test_sc

        print(self._compute(X=train_df, Y=train_labels, accuracy=True))

        self.run += 1


    def cross_validation(self, k=6, shuffle=False, compute_accuracy=True, compute_recall=True, compute_precision=True, compute_F1=True):

        kfold_train_metrics = []
        kfold_test_metrics = []

        total_input = self.input.df

        total_labels = self.input.labels

        n_data = len(self.input.df)

        if shuffle:
            idx = np.random.permutation(n_data)
            total_input, total_labels = total_input.reindex(idx), total_labels.reindex(idx)

        cv = KFold(n_splits=k)

        try: self._fit()
        except: pass

        for train_index, test_index in cv.split(total_input):

            train_df, train_labels = total_input.iloc[train_index], total_labels.iloc[train_index]
            test_df, test_labels = total_input.iloc[test_index], total_labels.iloc[test_index]

            mean = train_df.mean()
            std = train_df.std()

            train_df = (train_df - mean) / std
            test_df = (test_df - mean) / std

            self._fit(input=train_df, labels=train_labels, print_message=False)

            kfold_train_metrics.append(self._compute_metrics(X=train_df, Y=train_labels, 
                                                            compute_accuracy=compute_accuracy, compute_recall=compute_recall,
                                                            compute_precision=compute_precision, compute_F1=compute_F1))
            kfold_test_metrics.append(self._compute_metrics(X=test_df, Y=test_labels, 
                                                            compute_accuracy=compute_accuracy, compute_recall=compute_recall,
                                                            compute_precision=compute_precision, compute_F1=compute_F1))
            
        
        assert len(kfold_train_metrics) == len(kfold_test_metrics)

        L = len(kfold_train_metrics)

        kfold_metrics = {}

        if compute_accuracy: 
            train_acc_list = [kfold_train_metrics[m]["accuracy"] for m in range(L)]
            test_acc_list = [kfold_test_metrics[m]["accuracy"] for m in range(L)]

            train_acc = [np.mean(train_acc_list), np.std(train_acc_list)]
            test_acc = [np.mean(test_acc_list), np.std(test_acc_list)]

            kfold_metrics["Train Accuracy"] = train_acc
            kfold_metrics["Test Accuracy"] = test_acc

            #self.metrics[self.run_cross_validation] = {"Train Accuracy": train_acc, "Test Accuracy": test_acc}


        if compute_recall:
            train_rec_list = [kfold_train_metrics[m]["recall"] for m in range(L)]
            test_rec_list = [kfold_test_metrics[m]["recall"] for m in range(L)]

            train_rec = [np.mean(train_rec_list), np.std(train_rec_list)]
            test_rec = [np.mean(test_rec_list), np.std(test_rec_list)]

            kfold_metrics["Train Recall"] = train_rec
            kfold_metrics["Test Recall"] = test_rec

            #self.metrics[self.run_cross_validation] = {"Train Recall": train_rec, "Test Recall": test_rec}


        if compute_precision:
            train_prec_list = [kfold_train_metrics[m]["precision"] for m in range(L)]
            test_prec_list = [kfold_test_metrics[m]["precision"] for m in range(L)]

            train_prec = [np.mean(train_prec_list), np.std(train_prec_list)]
            test_prec = [np.mean(test_prec_list), np.std(test_prec_list)]

            kfold_metrics["Train Precision"] = train_prec
            kfold_metrics["Test Precision"] = test_prec

            #self.metrics[self.run_cross_validation] = {"Train Precision": train_prec, "Test Precision": test_prec}


        if compute_F1:
            train_F1_list = [kfold_train_metrics[m]["F1"] for m in range(L)]
            test_F1_list = [kfold_test_metrics[m]["F1"] for m in range(L)]

            train_F1 = [np.mean(train_F1_list), np.std(train_F1_list)]
            test_F1 = [np.mean(test_F1_list), np.std(test_F1_list)]

            kfold_metrics["Train F1 score"] = train_F1
            kfold_metrics["Test F1 score"] = test_F1

            #self.metrics[self.run_cross_validation] = {"Train F1 score": train_F1, "Test F1 score": test_F1}

        self.metrics[self.run_cross_validation] = kfold_metrics
        self.fitting_parameters[self.run_cross_validation] = self._get_fitting_parameters()

        self.run_cross_validation += 1


    def fit_with_all_data(self, visualize = True, print_metrics = True):

        self._fit()

        if print_metrics:
            metrics = self._compute_metrics()

            for key in metrics:
                print(f"{key} : {metrics[key]}")


        if visualize:
            from matplotlib import colors

            pred = self._predict()
            labels = self.input.labels

            pred_0 = (pred == 0)
            pred_1 = (pred == 1)

            labels_0 = (labels == 0)
            labels_1 = (labels == 1)

            TP = (pred_1 & labels_1)
            FP = (pred_1 & labels_0)
            TN = (pred_0 & labels_0)
            FN = (pred_0 & labels_1)

            C = 1*TP + 2*FP + 3*TN + 4*FN

            cmap = colors.ListedColormap(['g', 'k', 'b', 'r'], 4)
            
            self.input.visualize(cmap=cmap, labels=C)


        pass





        
class LDA(LearningMachine):
    
    def __init__(self, data):
        super().__init__(data)
        self.name = "LDA"
        
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        
        self.model = LinearDiscriminantAnalysis()

        pass

    def _fit(self, input = None, labels = None, print_message=True):

        try: 
            if input == None: input = self.input.df
        except: pass

        try: 
            if labels == None: labels = self.input.labels
        except: pass

        self.model.fit(input, labels)
        
        if print_message:
            print(f"The {self.name} model has been trained on the given data")
        pass
    
    def _predict(self, X=None):
        
        """
        X is a dataframe containing the features of the samples for which we want to predict their labels
        """
        
        try:
            if X == None:
                X = self.input.df
        except:
            pass


        return self.model.predict(X)
        
        pass

    def _get_fitting_parameters(self):

        return self.model.get_params()


class QDA(LearningMachine):
    
    def __init__(self, data):
        super().__init__(data)
        self.name = "LDA"
        
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
        
        self.model = QuadraticDiscriminantAnalysis()

        pass

    def _fit(self, input = None, labels = None, print_message=True):

        warnings.filterwarnings("ignore", message="Variables are collinear")

        try: 
            if input == None: input = self.input.df
        except: pass

        try: 
            if labels == None: labels = self.input.labels
        except: pass

        self.model.fit(input, labels)
        
        if print_message:
            print(f"The {self.name} model has been trained on the given data")
        pass
    
    def _predict(self, X=None):
        
        """
        X is a dataframe containing the features of the samples for which we want to predict their labels
        """
        
        try:
            if X == None:
                X = self.input.df
        except:
            pass


        return self.model.predict(X)
        
        pass

    def _get_fitting_parameters(self):

        return self.model.get_params()