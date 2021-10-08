import pandas as pd
import numpy as np
import sys
import random
import matplotlib
matplotlib.use('TkAgg') # change the backend
import matplotlib.pyplot as plt

from data_helper import generate_q4_data, load_simulate_data

from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.metrics import mean_squared_error


# load data
dpa = pd.read_csv('./data/house-votes-84.complete.csv')
dpa['Class'] = dpa['Class'].map({'republican': 0, 'democrat': 1})
for i in range(16):
	index = 'A'+ str(i+1)
	dpa[index] = dpa[index].map({'y': 1, 'n': 0})
#dpa.info()

pay = dpa.Class
paX = dpa.drop('Class', axis = 1)



'''
  10-cv with house-votes-84.complete.csv using LASSO
  - train_subset: train the classifier on a smaller subset of the training
    data
  -subset_size: the size of subset when train_subset is true 
  NOTE you do *not* need to modify this function
  '''
def lasso_evaluate(train_subset=False, subset_size = 0):
	sample_size = pay.shape[0]
	tot_incorrect=0
	tot_test=0
	tot_train_incorrect=0
	tot_train=0
	step = int( sample_size/ 10 + 1)
	for holdout_round, i in enumerate(range(0, sample_size, step)):
		#print("CV round: %s." % (holdout_round + 1))
		if(i==0):
			X_train = paX.iloc[i+step:sample_size]
			y_train = pay.iloc[i+step:sample_size]
		else:
			X_train =paX.iloc[0:i]  
			X_train = X_train.append(paX.iloc[i+step:sample_size], ignore_index=True)
			y_train = pay.iloc[0:i]
			y_train = y_train.append(pay.iloc[i+step:sample_size], ignore_index=True)
		X_test = paX.iloc[i: i+step]
		y_test = pay.iloc[i: i+step]
		if(train_subset):
			X_train = X_train.iloc[0:subset_size]
			y_train = y_train.iloc[0:subset_size]
		#print(" Samples={} test = {}".format(y_train.shape[0],y_test.shape[0]))
		# train the classifiers
		lasso = Lasso(alpha = 0.001)
		lasso.fit(X_train, y_train)            
		lasso_predit = lasso.predict(X_test)           # Use this model to predict the test data
		lasso_result = [1 if x>0.5 else 0 for x in lasso_predit]
		error = 0
		for (index, num) in enumerate(lasso_result):
			if(y_test.values.tolist()[index] != num):
				error+=1
		tot_incorrect += error
		tot_test += len(lasso_result)
		#print('Error rate {}'.format(1.0*error/len(lasso_result)))
		lasso_predit = lasso.predict(X_train)           # Use this model to get the training error
		lasso_result = [1 if x>0.5 else 0 for x in lasso_predit]
		error = 0
		for (index, num) in enumerate(lasso_result):
			if(y_train.values.tolist()[index] != num):
				error+=1
		tot_train_incorrect+= error
		tot_train += len(lasso_result)
		#print('Train Error rate {}'.format(1.0*error/len(lasso_result)))		

	#print('10CV Error rate {}'.format(1.0*tot_incorrect/tot_test))
	#print('10CV train Error rate {}'.format(1.0*tot_train_incorrect/tot_train))

	return 1.0*tot_incorrect/tot_test, 1.0*tot_train_incorrect/tot_train

def lasso_evaluate_incomplete_entry():
	# get incomplete data
	dpc = pd.read_csv('./data/house-votes-84.incomplete.csv')
	for i in range(16):
		index = 'A'+ str(i+1)
		dpc[index] = dpc[index].map({'y': 1, 'n': 0})
		
	lasso = Lasso(alpha = 0.001)
	lasso.fit(paX, pay)
	lasso_predit = lasso.predict(dpc)
	print(lasso_predit)


# A helper method for pretty-printing the coefficients
def pretty_print_coefs(coefs, names = None, sort = False):
    if names == None:
        names = ["X%s" % x for x in range(len(coefs))]
    lst = zip(coefs, names)
    if sort:
        lst = sorted(lst,  key = lambda x:-np.abs(x[0]))
    return " + ".join("%s * %s" % (round(coef, 3), name)
                                   for coef, name in lst)


def main():
	'''
	TODO modify or use the following code to evaluate your implemented
	classifiers
	Suggestions on how to use the starter code for Q2, Q3, and Q5:
	'''
	
	#For Q2
	print('---------------------------------- Question 2_LASSO ----------------------------------')
	error_rate, unused = lasso_evaluate()
	print('10CV Error rate {}'.format(error_rate))

	
	#For Q3
	print('---------------------------------- Question 3_LASSO ----------------------------------')
	print("check lasso_q3.png")
	train_error = np.zeros(10)
	test_error = np.zeros(10)
	sample_size = np.zeros(10)

	for i in range(10):
		sample_size[i] = (i+1)*10
		x, y =lasso_evaluate(train_subset=True, subset_size=i*10+10)
		train_error[i] = y
		test_error[i] = x

	plt.figure()
	plt.subplot(211)
	plt.plot(sample_size, train_error)
	plt.ylabel('training error')
	# plt.xlabel('sample size')
	plt.title('LASSO')

	plt.subplot(212)
	plt.plot(sample_size, test_error)
	plt.ylabel('test error')
	plt.xlabel('sample size')
	plt.savefig('lasso_q3.png')
	# plt.show()
	
	
	#Q4
	#TODO 
	print('---------------------------------- Question 4_LASSO ----------------------------------')
	print("check lasso_q4.png")
	random.seed(1234)
	np.random.seed(1234)
	EPS = sys.float_info.epsilon
	n_nonpartisan = 12
	training_set_size = []
	fraction_ignored_arr = []
	for i in range(400, 4001, 400):
		training_set_size.append(i)
		file_name = "q4_data_{}.csv".format(i)
		generate_q4_data(i, file_name)

		# load data
		dpa = pd.read_csv(file_name)
		dpa['Class'] = dpa['Class'].map({'republican': 0, 'democrat': 1})
		for i in range(16):
			index = 'A'+ str(i+1)
			dpa[index] = dpa[index].map({'y': 1, 'n': 0})
		#dpa.info()

		pay = dpa.Class
		paX = dpa.drop('Class', axis = 1)


		lasso = Lasso(alpha = 0.001)
		lasso.fit(paX, pay)

		coefs = lasso.coef_
		# print Lasso model's coefficient
		# print ("LASSO model:", pretty_print_coefs(coefs))
		
		n_nonpartisan_ignored = 0
		for j in range(4, 16):
			if abs(coefs[j])<EPS:
				n_nonpartisan_ignored += 1
				
		fraction_ignored = float(n_nonpartisan_ignored)/n_nonpartisan
		fraction_ignored_arr.append(fraction_ignored)

	# Plot
	plt.figure()
	plt.plot(training_set_size, fraction_ignored_arr)
	plt.ylabel('Fraction of ignored')
	plt.xlabel('training set size')
	plt.title('Fraction of ignored nonpartisan bills (LASSO)')
	plt.savefig('lasso_q4.png')
	# plt.show()
	

	#Q5
	print('---------------------------------- Question 5_LASSO ----------------------------------')
	print('LASSO  P(C=1|A_observed) ')
	lasso_evaluate_incomplete_entry()
	

if __name__ == "__main__":
    main()