#Import required modules
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd

#Import Data
housing=fetch_california_housing()

#Display Data
df=pd.DataFrame(housing.data, columns = housing.feature_names)
df['Target']=housing.target
print(df)

#Splitting Train and Test data
X=housing.data
y=housing.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Creating and fiting model to train data
model=LinearRegression()
model.fit(X_train,y_train)

#Prediction
y_pred = model.predict(X_test)

#Mean Squared Error
mse=mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse: .3f}.")

