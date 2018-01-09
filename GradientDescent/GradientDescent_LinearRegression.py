'''
Author : Karan R Nadagoudar
Description : Gradient Descent for Linear Regression

'''
import pandas as pd
import numpy.random as random
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression   #Linear models ( Regression )
from sklearn.cross_validation import train_test_split #module to split dataset

def GradientDescentLinearRegression(alpha,X,Y,ep,max_iter):
	converged = False
	iteration = 0
	m = X.shape[0]
	theta0 = random.random(X.shape[1])
	theta1 = random.random(X.shape[1])
	print(theta0,theta1)

	#Initial Cost function calaculation
	J = sum([(theta0 + theta1*X[i] - Y[i])**2 for i in range(m)])
	
	#Gradient Descent until it converges
	while not converged:
		temp0 = theta0 - alpha * (1.0/m)*sum([(theta0 + theta1 * X[i] - Y[i]) for i in range(m)])
		temp1 = theta1 - alpha *(1.0/m)*sum([(theta0 + theta1 * X[i] - Y[i])*X[i] for i in range(m)])
		theta0 = temp0
		theta1 = temp1

		#mean squared  error
		e = sum([((theta0 + theta1*X[i] - Y[i])**2) for i in range(m)]) 
		if abs(J-e) <= ep:
			print("Converged at %s" %iteration)
			converged = True

		J = e
		iteration+=1

		if iteration == max_iter :
			print("maximum iterations done")
			converged = True

	return theta0,theta1


if __name__ == '__main__':
	dataset = pd.read_csv('salary.csv')
	#separating out independent variable
	X=dataset.iloc[:,:-1].values
	Y=dataset.iloc[:,1].values
	
	alpha = 0.01
	ep = 0.001

	theta0,theta1 = GradientDescentLinearRegression(alpha,X,Y,ep,1000)
	print("theta0:%s \ntheta1:%s" %(theta0,theta1))

	#statistical values from function
	slope, intercept, r_value, p_value, slope_std_error = stats.linregress(X[:,0], Y)
	print("intercept:%s \nSlope:%s" %(intercept,slope))

	#Visualising Gradient Descsent
	for i in range(X.shape[0]):
		y_predict = theta0 + theta1 * X
		Y1_predict = intercept + slope * X
	plt.plot(X,Y,'ro')
	plt.plot(X,y_predict,'k-')
	plt.plot(X,Y1_predict,'b-')
	plt.xlabel("Experience")
	plt.ylabel("Salary")
	plt.show()