from data import Data
from learning_machine import LearningMachine, LDA
import numpy as np


if __name__ == "__main__":
    #initializing data
    data = Data('project_train.csv')
    data.df.loc[84, "energy"] = 0.734
    data.df.loc[94, "loudness"] = -6.542
    data._preprocess()

    #creating Learning Machine
    lda = LDA(data)
    lda._fit()
    # for i in ("svd","lsqr","eigen"):
    #     lda.model.solver = i
    #     lda._fit()
    #     score = lda._evaluate_training()
    #     print(i+"-score:"+str(score))
    #     print(lda.model.get_params())
    #     print()
    # print(lda.metrics)

    prediction = lda.model.predict(lda.data_class.df)

    color_tp = [prediction == 1 and lda.data_class.labels == 1]
    print(color_tp)
    exit()
    print(prediction==lda.data_class.labels)


