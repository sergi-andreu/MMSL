from data import Data
from learning_machine import LearningMachine, LDA, Logistic_regression
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

if __name__ == "__main__":
    #initializing data
    data = Data('project_train.csv')
    data._preprocess()
    # data.show_scattermatrix()

    from sklearn.tree import DecisionTreeClassifier

    model = DecisionTreeClassifier(random_state=0)
    print(data.df.columns)
    model.fit(data.df,data.labels)

    output = model.predict(data.df)
    true_false = np.array(output == data.labels)
    print(true_false)
    print(output)
    print(data.labels)

    # ### COLOUR WHEEL FOR OUPUT
    # color_labels = [None for i in LM.output]
    # print(LM.output,LM.data.labels)
    # print("\n")
    # for i in range(0,sizeOfdata):
    #     if LM.output[i] == LM.data.labels[i]:
    #         if LM.output[i] == 1:
    #             color_labels[i] = 1
    #         else:
    #             color_labels[i] = 4
    #     else:
    #         if LM.output[i] == 1:
    #             color_labels[i] = 2
    #         else:
    #             color_labels[i] = 3
    #
    # color_wheel = {1: "green", 2: "yellow", 3: "purple", 4: "red"}
    # colors = [color_wheel.get(x) for x in color_labels]
    #
    # list_of_numerical_column_names = list(LM.data.num_bound_col)
    #
    # print(list_of_numerical_column_names)
    #
    # pd.plotting.scatter_matrix(LM.data.df[list_of_numerical_column_names],
    #                            alpha=0.5,
    #                            c=colors,
    #                            s=7.5,
    #                            diagonal='kde')
    # plt.savefig('scatter_matrix_numerical_features_LR.png', dpi=1600)
    # plt.show()
    #
    # exit()
