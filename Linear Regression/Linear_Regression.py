import pandas as pd
from sklearn.linear_model import LinearRegression   #Linear models ( Regression )
from sklearn.cross_validation import train_test_split #module to split dataset
import matplotlib.pyplot as plt

dataset = pd.read_csv('salary.csv')
print("Rows:",dataset.shape[0],"Columns:",dataset.shape[0])

#separating out independent variable
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,1].values
#print(X,"\n\n",Y)

#splitting dataset into training and test set
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=0)
#print("test X:",X_test,"\ntrain X:\n",X_train)
#print("test Y:",Y_test,"\ntrain Y:",Y_train)
print(X_train,X_test,Y_train,Y_test)

#Defining regressor to train and predict
regressor = LinearRegression()
regressor.fit(X_train,Y_train)
PredY = regressor.predict(X_train)

#Visualisong data for training set
plt.plot(X_train,Y_train,'bo')
plt.plot(X_train,PredY,'r')
plt.title("Experience vs Salary (Training set)")
plt.xlabel("Experience (No. of Years)")
plt.ylabel("Salary (lakhs per annum)")
plt.show()

#Visualisong data for test set
plt.plot(X_test,Y_test,'bo')
plt.plot(X_train,PredY,'r')
plt.title("Experience vs Salary (Test set)")
plt.xlabel("Experience (No. of Years)")
plt.ylabel("Salary (lakhs per annum)")
plt.show()
