import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
df=pd.read_csv('train.csv')
df.drop(columns=['Cabin'],axis=1,inplace=True)
df['Age'].fillna(df['Age'].mean(),inplace=True)
df['Embarked'].fillna(df['Embarked'].value_counts().idxmax(),inplace=True)
df.replace({'Sex':{'male':0,'female':1},'Embarked':{'S':0,'C':1,'Q':2}},inplace=True)
x=df.drop(columns=['PassengerId','Survived','Name','Ticket'],axis=1)
y=df['Survived']
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.80,random_state=2)
model=LogisticRegression()
model.fit(x_train,y_train)
pickle.dump(model,open('titanic_prediction_model.pkl','wb'))