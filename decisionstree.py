import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix


df = pd.read_csv('kyphosis.csv')

print(df.info())
print(df.head(5))


X = df.drop('Kyphosis', axis =1 )
y = df['Kyphosis']

X_train , X_test , y_train, y_test = train_test_split(X, y, test_size = 0.3)


treeDecision = DecisionTreeClassifier()

treeDecision.fit(X_train, y_train)

predictions = treeDecision.predict(X_test)

print(classification_report(y_test, predictions))
print(pd.crosstab(y_test, predictions, rownames=['Real'], colnames=['Predito']))

