import numpy as np
import pandas as pd
import matplotlib.pyplot as mt

dataset=pd.read_csv("50_startup.csv")
x=dataset.iloc[:,:-1]
y=dataset.iloc[:,4]

states=pd.get_dummies(x['State'],drop_first=True)
x=x.drop('State',axis=1)

x=pd.concat([x,states],axis=1)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regression=LinearRegression()
regression.fit(x_train,y_train)
y_pred=regression.predict(x_test)

mt.scatter(x_train,y_train,color="red")
mt.plot(x_train,regression.predict(x_train))
mt.show()


from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)

