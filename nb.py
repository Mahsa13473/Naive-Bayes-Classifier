import os
import sys
import numpy as np
from scipy.special import logsumexp
from collections import Counter
import random
import pdb
import math
import itertools
from itertools import product
# helpers to load data
from data_helper import load_vote_data, load_incomplete_entry,load_simulate_data
from data_helper import generate_q4_data, load_simulate_data
# helpers to learn and traverse the tree over attributes
import matplotlib
matplotlib.use('TkAgg') # change the backend
import matplotlib.pyplot as plt

# pseudocounts for uniform dirichlet prior
alpha = 0.1


#--------------------------------------------------------------------------
# Naive bayes CPT and classifier
#--------------------------------------------------------------------------
# inspired by https://github.com/kushagra06/CS228_PGM/tree/master/hw

class NBCPT(object):
  '''
  NB Conditional Probability Table (CPT) for a child attribute.  Each child
  has only the class variable as a parent
  '''

  def __init__(self, A_i):
    '''
    TODO create any persistent instance variables you need that hold the
    state of the learned parameters for this CPT
        - A_i: the index of the child variable
    '''
    super(NBCPT, self).__init__()

    self.index = A_i
    # use Beta prior
    self.c_count = [2*alpha, 2*alpha]
    self.pseudocounts = [alpha, alpha]

  def learn(self, A, C):
    '''
    TODO
    populate any instance variables specified in __init__ to learn
    the parameters for this CPT
        - A: a 2-d numpy array where each row is a sample of assignments
        - C: a 1-d n-element numpy where the elements correspond to the
          class labels of the rows in A
    '''
    for i in range(2):
      self.c_count[i] += len(C[C == i])
      self.pseudocounts[i] += len(C[(A[:, self.index] == 1) & (C == i)])

  def get_cond_prob(self, entry, c):
    ''' TODO
    return the conditional probability P(A_i|C=c) for the values
    specified in the example entry and class label c
        - entry: full assignment of variables
            e.g. entry = np.array([0,1,1]) means A_0 = 0, A_1 = 1, A_2 = 1
        - c: the class
    '''
    if entry[self.index] == 1:
      return float(self.pseudocounts[c]) / self.c_count[c]
    else:
      return 1 - float(self.pseudocounts[c]) / self.c_count[c]

class NBClassifier(object):
  '''
  NB classifier class specification
  '''

  def __init__(self, A_train, C_train):
    '''
    TODO create any persistent instance variables you need that hold the
    state of the trained classifier and populate them with a call to
    Suggestions for the attributes in the classifier:
        - P_c: the probabilities for the class variable C
        - cpts: a list of NBCPT objects
    '''
    super(NBClassifier, self).__init__()

    self.P_c = 0.0
    self.n_features = A_train.shape[1]
    self.cpts = []
    for i in range(self.n_features):
      self.cpts.append(NBCPT(i))
   
    self._train(A_train, C_train)

  def _train(self, A_train, C_train):
    ''' TODO
    train your NB classifier with the specified data and class labels
    hint: learn the parameters for the required CPTs
        - A_train: a 2-d numpy array where each row is a sample of assignments
        - C_train: a 1-d n-element numpy where the elements correspond to
          the class labels of the rows in A
    '''
    self.P_c = len(C_train[C_train == 1]) / float(len(C_train))

    for i in range(self.n_features):
        self.cpts[i].learn(A_train, C_train)


  def classify(self, entry):
    ''' TODO
    return the log probabilites for class == 0 and class == 1 as a
    tuple for the given entry
    - entry: full assignment of variables
    e.g. entry = np.array([0,1,1]) means variable A_0 = 0, A_1 = 1, A_2 = 1
    NOTE this must return both the predicated label {0,1} for the class
    variable and also the log of the conditional probability of this
    assignment in a tuple, e.g. return (c_pred, logP_c_pred)
    '''

    P_c_pred = [0, 0]

    unobserved_idx = []
    for i, e in enumerate(entry):
      if e == -1:
        unobserved_idx.append(i)

    # Add empty set to all settings
    unobserved_assigments = list(itertools.product((0, 1), repeat=len(unobserved_idx))) + [[]]

    for unobserved in unobserved_assigments:
      for i, value in enumerate(unobserved):
        entry[unobserved_idx[i]] = value

      full_P_c_pred = [1 - self.P_c, self.P_c]
      for cpt in self.cpts:
        for i in range(2):
          full_P_c_pred[i] *= cpt.get_cond_prob(entry, i)

      for i in range(2):
        P_c_pred[i] += full_P_c_pred[i]


    # normalize
    P_c_pred /= np.sum(P_c_pred)

    c_pred = np.argmax(P_c_pred)
    return (c_pred, np.log(P_c_pred[c_pred]))


  def predict_unobserved(self, entry, index):
    ''' TODO
    Predicts P(A_index  | mid entry)
    Return a tuple of probabilities for A_index=0  and  A_index = 1
    We only use the 2nd value (P(A_index =1 |entry)) in this assignment
    '''
    #
    if entry[index] == 0 or entry[index] == 1:
      return [1 - entry[index], entry[index]]

    # if not observed use model to predict
    P_pred_index = [0.0, 0.0]

    unobserved_idx = []
    for i, e in enumerate(entry):
      if e == -1 and i != index:
        unobserved_idx.append(i)
    # Add empty set to all settings
    unobserved_assigments = list(itertools.product((0, 1), repeat=len(unobserved_idx))) + [[]]

    for p in range(2):
      entry[index] = p
      for unobserved in unobserved_assigments:
        for i, value in enumerate(unobserved):
          entry[unobserved_idx[i]] = value

        full_P_c_pred = [1 - self.P_c, self.P_c]
        for cpt in self.cpts:
          for i in range(2):
            full_P_c_pred[i] *= cpt.get_cond_prob(entry, i)
        P_pred_index[p] += np.sum(full_P_c_pred)

    # normalize
    P_pred_index /= np.sum(P_pred_index)
    return P_pred_index


# load data
A_data, C_data = load_vote_data()


def evaluate(classifier_cls, train_subset=False, subset_size = 0):
  '''
  evaluate the classifier specified by classifier_cls using 10-fold cross
  validation
  - classifier_cls: either NBClassifier or other classifiers
  - train_subset: train the classifier on a smaller subset of the training
    data
  -subset_size: the size of subset when train_subset is true
  NOTE you do *not* need to modify this function
  '''
  global A_data, C_data

  A, C = A_data, C_data


  # partition train and test set for 10 rounds
  M, N = A.shape
  tot_correct = 0
  tot_test = 0
  train_correct = 0
  train_test = 0
  step = int(M / 10 + 1)
  for holdout_round, i in enumerate(range(0, M, step)):
    # print("Holdout round: %s." % (holdout_round + 1))
    A_train = np.vstack([A[0:i, :], A[i+step:, :]])
    C_train = np.hstack([C[0:i], C[i+step:]])
    A_test = A[i: i+step, :]
    C_test = C[i: i+step]
    if train_subset:
      A_train = A_train[: subset_size, :]
      C_train = C_train[: subset_size]
    # train the classifiers
    classifier = classifier_cls(A_train, C_train)

    train_results = get_classification_results(classifier, A_train, C_train)
    # print(
    #    '  train correct {}/{}'.format(np.sum(train_results), A_train.shape[0]))
    test_results = get_classification_results(classifier, A_test, C_test)
    tot_correct += sum(test_results)
    tot_test += len(test_results)
    train_correct += sum(train_results)
    train_test += len(train_results)

  return 1.*tot_correct/tot_test, 1.*train_correct/train_test


  # score classifier on specified attributes, A, against provided labels,
  # C
def get_classification_results(classifier, A, C):
  results = []
  pp = []
  for entry, c in zip(A, C):
    c_pred, unused = classifier.classify(entry)
    results.append((c_pred == c))
    pp.append(unused)
    # print('logprobs', np.array(pp))
  return results



def evaluate_incomplete_entry(classifier_cls):

  global A_data, C_data

  # train a classifier on the full dataset
  classifier = classifier_cls(A_data, C_data)

  # load incomplete entry 1
  entry = load_incomplete_entry()

  c_pred, logP_c_pred = classifier.classify(entry)
  print("  P(C={}|A_observed) = {:2.4f}".format(c_pred, np.exp(logP_c_pred)))

  return


def predict_unobserved(classifier_cls, index=11):
  global A_data, C_data

  # train a classifier on the full dataset
  classifier = classifier_cls(A_data, C_data)
  # load incomplete entry 1
  entry = load_incomplete_entry()

  a_pred = classifier.predict_unobserved(entry, index)
  print("  P(A{}=1|A_observed) = {:2.4f}".format(index+1, a_pred[1]))

  return


def main():

  '''
  TODO modify or use the following code to evaluate your implemented
  classifiers
  Suggestions on how to use the starter code for Q2, Q3, and Q5:
  '''
  
  ##For Q1
  print('---------------------------------- Question 1_NB ----------------------------------')
  print('Naive Bayes')
  accuracy, num_examples = evaluate(NBClassifier, train_subset=False)
  print('  10-fold cross validation total test error {:2.4f}'.format(1 - accuracy))

  
  ##For Q3
  print('---------------------------------- Question 3_NB ----------------------------------')
  print('Naive Bayes (Small Data)')
  print("check nb_q3.png")
  train_error = np.zeros(10)
  test_error = np.zeros(10)
  sample_size = np.zeros(10)

  for x in range(10):
    sample_size[x] = (x+1)*10
    accuracy, train_accuracy = evaluate(NBClassifier, train_subset=True, subset_size = (x+1)*10)
    train_error[x] = 1-train_accuracy
    test_error[x] = 1- accuracy
    print('  10-fold cross validation total test error {:2.4f} total train error {:2.4f}on {} ''examples'.format(1 - accuracy, 1- train_accuracy  ,(x+1)*10))

  plt.figure()
  plt.subplot(211)
  plt.plot(sample_size, train_error)
  plt.ylabel('training error')
  # plt.xlabel('sample size')
  plt.title('NB')

  plt.subplot(212)
  plt.plot(sample_size, test_error)
  plt.ylabel('test error')
  plt.xlabel('sample size')
  plt.savefig('nb_q3.png')
  # plt.show()
  

  ##For Q4 TODO
  print('---------------------------------- Question 4_NB ----------------------------------')
  print("check nb_q4.png")
  random.seed(1234)
  np.random.seed(1234)
  n_nonpartisan = 12
  feature_size = 16
  training_set_size = []
  fraction_ignored_arr = []
  diff = np.zeros((10, feature_size))
  probabilites = np.zeros((10, 2, feature_size))
  threshold = 0.1
  i = 0
  entry = [1]*feature_size
  for size in range(400, 4001, 400):
    i += 1
    training_set_size.append(size)
    file_name = "q4_data_{}.csv".format(size)
    generate_q4_data(size, file_name)
    A, C = load_simulate_data(file_name)

    classifier = NBClassifier(A, C)

    for index, cpt in enumerate(classifier.cpts):
      for c in [0, 1]:
        probabilites[i-1, c, index] = cpt.get_cond_prob(entry, c)


    n_nonpartisan_ignored = 0
    for j in range(4, feature_size):
      diff[i-1, j] = np.abs(probabilites[i-1, 1, j] - probabilites[i-1, 0, j])

      if diff[i-1, j] < threshold:
        n_nonpartisan_ignored += 1

    fraction_ignored = float(n_nonpartisan_ignored)/n_nonpartisan
    fraction_ignored_arr.append(fraction_ignored)
    
    
  # Plot
  plt.figure()
  plt.plot(training_set_size, fraction_ignored_arr)
  plt.ylabel('Fraction of ignored')
  plt.xlabel('training set size')
  plt.title('Fraction of ignored nonpartisan bills (NB)')
  plt.savefig('nb_q4.png')
  # plt.show()
  

  ##For Q5
  print('---------------------------------- Question 5_NB ----------------------------------')
  print('Naive Bayes Classifier on missing data')
  evaluate_incomplete_entry(NBClassifier)

  index = 11
  print('Prediting vote of A%s using NBClassifier on missing data' % (
      index + 1))
  predict_unobserved(NBClassifier, index)
 

if __name__ == '__main__':
  main()
