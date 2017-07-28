#import data preprocessing template
import data_preprocessing_template as dpt
#preprocess("/path/to/file",no_of_dependant_variable,no_of_independent_variable,test_split_ratio)
X_train,X_test,Y_train,Y_test=dpt.preprocess("Datasets/Simple_Linear_Regression/Salary_Data.csv",1,1,0.2)
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
