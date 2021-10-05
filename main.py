from data import Data
from learning_machine import LearningMachine, LDA
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


if __name__ == "__main__":
    #initializing data
    data = Data('project_train.csv')
    data.df.loc[84, "energy"] = 0.734
    data.df.loc[94, "loudness"] = -6.542
    data._preprocess()

    model = LogisticRegression(solver='liblinear', random_state=0)
    model_fit = model.fit(data.df,data.labels)
    score = model.score(data.df,data.labels)
    mat = confusion_matrix(data.labels, model.predict(data.df))
    report = classification_report(data.labels, model.predict(data.df))
    print(mat)
    print(report)

    print("hola Sergi, que tal?")
    ### COLORING UNPROCESSED DATA
    # color_wheel = {0: "darkred", 1: "darkblue"}
    # colors = data.labels.map(lambda x: color_wheel.get(x))
    #
    # list_of_numerical_column_names = list(data.num_bound_col)
    #
    # print(list_of_numerical_column_names)
    #
    # scatter_matrix(data.df[list_of_numerical_column_names],
    #                alpha=0.5,
    #                c=colors,
    #                s=7.5,
    #                diagonal='kde')
    # plt.savefig('scatter_matrix_numerical_features.png',dpi = 1600)
    # plt.show()
    #
    # exit()
    #
    # list_of_column_names =  list(data.df)
    # scatter_matrix(data.df[list_of_column_names])
    # plt.savefig('scatter_matrix_features.png',dpi = 1600, groups = data.labels)
    # plt.show()


    # list_of_cat_names = list([data.binary_col,data.class_col])
    # scatter_matrix(data.df[list_of_cat_names])
    # plt.savefig('scatter_matrix_binary_features.png')
    # plt.show()

    # #creating Learning Machine
    # lda = LDA(data)
    # lda._fit()
    # # for i in ("svd","lsqr","eigen"):
    # #     lda.model.solver = i
    # #     lda._fit()
    # #     score = lda._evaluate_training()
    # #     print(i+"-score:"+str(score))
    # #     print(lda.model.get_params())
    # #     print()
    # # print(lda.metrics)
    #
    # prediction = lda.model.predict(lda.data_class.df)
    #
    # color_tp = [prediction == 1 and lda.data_class.labels == 1]
    # print(color_tp)
    # exit()
    # print(prediction==lda.data_class.labels)


