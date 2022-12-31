#!C:\Users\Lenovo\AppData\Local\Programs\Python\Python37-32\python.exe

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import warnings
import pickle
warnings.filterwarnings("ignore")

data = pd.read_csv("Forest_fire.csv")
data = np.array(data)

X = data.iloc[:, 4:-1]
y = data.iloc[:, -1]
print(X.head())
print(y.head())
y = y.astype('int')
X = X.astype('int')
# print(X,y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
decision_tree = DecisionTreeClassifier(max_depth=7,criterion='gini')


decision_tree.fit(X_train, y_train)
pickle.dump(decision_tree,open('model.txt','wb'))
model=pickle.load(open('model.txt','rb'))


