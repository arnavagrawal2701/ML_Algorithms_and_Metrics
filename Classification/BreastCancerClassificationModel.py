#Import required modules
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

#Import Data
data=load_breast_cancer()

#Display Data
df=pd.DataFrame(data.data, columns = data.feature_names)
df['Target']=data.target
print(df)

#Splitting data into Train and Test Data
X=data.data
y=data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Creating model
model=LogisticRegression()

#Training model
model.fit(X_train, y_train)

#Prediction
y_pred=model.predict(X_test)

#Accuracy Score
acc = accuracy_score(y_pred,y_test)
print(f"Accurace Score: {acc: 0.3f}")


# In[ ]:




