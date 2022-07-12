"""Emulates the filter class
If we pass in a generic classifier obejct, should be able to execute standard sklearn commands

Creating a kernel to handle verification and execution should be straight-foward

Create a classifier object and enter
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 10:51:07 2019

filter.py - Defines the filter Class for BCIP

@author: ivanovn
"""

from bcip import BCIP
from bcip_enums import BcipEnums
from sklearn import *

class Classifier(BCIP):
    """
    A classifier that can be used by different BCIP kernels
    """
    
    # these are the possible internal methods for storing the filter 
    # parameters which determine how it will be executed
    ctypes = ['lda', 'svm', 'custom']

    def __init__(self,sess, ctype,classifier):
        """
        Create a new filter object
        """
        self._ctype = ctype
        self._classifier = classifier

        super().__init__(BcipEnums.FILTER,sess)
        
        
    def __str__(self):
        return "BCIP {} Classifier with following" + \
               "attributes:\nClassifier Type: {}\n".format(self.ctype)

    # API Getters
    @property
    def ctype(self):
        return self.ctype
    
    @classmethod
    def create_SVM(cls, sess, C=1, kernel="rbf", degree=3, gamma="scale", coef0=0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape="ovr", break_ties=False, random_state=None):
        svm_object = svm.SVC(C, kernel, degree, gamma, coef0, shrinking, probability, tol, cache_size, class_weight, verbose, max_iter, decision_function_shape, break_ties, random_state)
        f = cls(sess, 'svm', svm_object)


    @classmethod
    def create_LDA(cls,sess, solver="svd", shrinkage=None, priors=None, n_components=None, store_covariance=False, tol=0.0001, covariance_estimator=None):
        """Factory method to create an LDA BCIP Classifier object"""
        lda_object = discriminant_analysis.LinearDiscriminantAnalysis(solver, shrinkage, priors, n_components, store_covariance, tol, covariance_estimator)
        f = cls(sess, 'lda', lda_object)

        sess.add_misc_bcip_obj(f)

        return f

    @classmethod
    def create_classifier(cls, sess, classifier_object, classifier_type):
        f = cls(sess, classifier_type, classifier_object)
        sess.add_misc_bcip_obj(f)

        return f