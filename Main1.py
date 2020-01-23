import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# get DataSet
data = pd.read_csv("datasets/irisDataset.csv")
x = data.iloc[:, 0:4]
y = data.iloc[:, 4]

# split data to train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25, random_state=0)

