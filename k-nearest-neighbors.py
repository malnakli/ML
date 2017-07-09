import numpy as np
from sklearn import  model_selection 
from sklearn import preprocessing , neighbors
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.csv')
df.replace('?',-99999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,1,1,2,3,2,1]])
# example_measures = example_measures.reshape(len(example_measures), -1)
prediction = clf.predict(example_measures)

print(prediction)