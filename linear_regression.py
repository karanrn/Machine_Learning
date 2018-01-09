import pandas as pd
from sklearn import linear_model

dataset = pd.read_csv('salary.csv')
print("Rows:",dataset.shape[0],"Columns:",dataset.shape[1])

#separating out independent variable
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,1].values
#print(X,"\n\n",Y)

#splitting dataset into training and test set
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0,random_state=0)
#print("test X:",X_test,"\ntrain X:\n",X_train)
#print("test Y:",Y_test,"\ntrain Y:",Y_train)
print(X_train,X_test,Y_train,Y_test)


n=len(X_train)
sX,sY,sX2,sY2,sXY=0,0,0,0,0
for i in range(n):
	sX=sX+X_train[i]
	sY=sY+Y_train[i]
	sX2=sX2+X_train[i]**2
	sY2=sY2+Y_train[i]**2
	sXY=sXY+X_train[i]*Y_train[i]

#Calculate slope for equation
m=(n*sXY-sX*sY)/(n*sX2-sX**2)
#Calculate intercept 
a=(sY/n)-m*(sX/n)
#equation
print("Equation: ",a,"+",m,"X")

#predicting values from test values( test_set)
Y_pred=[] #predicted list
for i in X_test:
	pred=a+m*i
	Y_pred.append(pred)
print("Actual:",Y_test,"\nPredicted:",Y_pred)

#Finding accuracy
x,y=0,0
mean=sY/n
for i in range(len(Y_test)):
	x=x+(Y_pred[i]-mean)**2
	y=y+(Y_test[i]-mean)**2

Accuracy=x/y
print("Accuracy:",Accuracy*100)
