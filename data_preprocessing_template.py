import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd #library to manage and import data sets

def preprocess(path,xn,dep_index,tsize):
	#Importing data sets
	dataset=pd.read_csv(path)
	#print("Rows:",dataset.shape[0],"Columns:",dataset.shape[1])

	#separating out independent variable
	X=dataset.iloc[:,:-xn].values
	Y=dataset.iloc[:,dep_index].values
	#print(X,"\n\n",Y)

	#splitting dataset into training and test set
	from sklearn.cross_validation import train_test_split
	X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=tsize,random_state=0)
	#print("test X:",X_test,"\ntrain X:\n",X_train)
	#print("test Y:",Y_test,"\ntrain Y:",Y_train)
	return X_train,X_test,Y_train,Y_test

	'''
	#Feature scaling
	from sklearn.preprocessing import StandardScaler
	s_X=StandardScaler()
	X_train=s_X.fit_transform(X_train)
	X_test=s_X.transform(X_test) #No need to fit test vector to standardscaler object
	print(X_train,"\n\n",X_test)
	'''
