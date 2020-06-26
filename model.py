import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
data=pd.read_csv('data.csv')
print(data.head())
x=data.iloc[:,[0,1,2,3]]
y=data.iloc[:,[4]]
model=LogisticRegression().fit(x,y)
pred_y=model.predict(x)
from sklearn.metrics import confusion_matrix as cm
print(cm(y,pred_y))
import pickle
with open ('log_model','wb') as file:
    pickle.dump(model,file)
