import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# df = pd.read_csv('WildBlueberryPollinationSimulationData.csv')
# df = df.drop(columns='Row#')
# # print(df.head())
# #
# X = df.drop(columns = ['yield'])
# y = df['yield']
#

# X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
#
# from sklearn.ensemble import RandomForestRegressor
#
# rf = RandomForestRegressor()
# rf.fit(X_train,y_train)
#
# print(rf.predict(X_test))
# #
import joblib
# joblib.dump(rf,'randomforest.joblib')
#
# print(X_test.head())

df = pd.read_csv(csv_file)
df.head()

X = df.drop(columns=['class'])
y = df['class']

from sklearn.ensemble import RandomForestClassifier
rf1 = RandomForestClassifier()
rf1.fit(X,y)

# joblib.dump(rf1,'rf1.joblib')
