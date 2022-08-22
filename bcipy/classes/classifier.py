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

from .bcip import BCIP
from .bcip_enums import BcipEnums
from sklearn import *
from pyriemann import classification
from sklearn.linear_model import LogisticRegression

class Classifier(BCIP):
    """
    A classifier that can be used by different BCIP kernels

    Parameters
    ----------
    sess : Session object
        - Session where the Array object will exist

    ctype : str
        - The name of the classifier to be created

    classifier : Classifier object
        - The classifier object to be used within the node (should be the return from a BCIP kernel)

    Attributes
    ----------
    _ctype : ['lda', 'svm', 'logistic regression', 'custom']
        - Indicates the type of classifier

    _classifier : Classifier object
        - The Classifier object (ie. Scipy classifier object) that will dictate this node's function
    
    Examples
    --------


    Return
    ------
    BCIP Classifier object

    """
    
    # these are the possible internal methods for storing the filter 
    # parameters which determine how it will be executed
    ctypes = ['lda', 'svm', 'logistic regression' 'custom']

    def __init__(self,sess, ctype,classifier):
        """
        Create a new filter object
        """
        self._ctype = ctype
        self._classifier = classifier

        super().__init__(BcipEnums.CLASSIFIER,sess)
        
        
    def __str__(self):
        return "BCIP {} Classifier with following" + \
               "attributes:\nClassifier Type: {}\n".format(self.ctype)

    # API Getters
    @property
    def ctype(self):
        return self.ctype
    
    @classmethod
    def create_SVM(cls, sess, C=1, kernel="rbf", degree=3, gamma="scale", coef0=0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape="ovr", break_ties=False, random_state=None):
        """
        Factory Method to create an SVM BCIP Classifier object.
        
        C-Support Vector Classification. The implementation is based on libsvm. The fit time scales at least 
        quadratically with the number of samples and may be impractical beyond tens of thousands of samples. 
        The multiclass support is handled according to a one-vs-one scheme.

        Parameters
        ----------
        sess : session object
            Session where the SVM BCIP Classifier object will exist

        C : float, default=1.0
            Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive. The penalty is a squared l2 penalty.

        kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'} or callable, default='rbf'
            Specifies the kernel type to be used in the algorithm. If none is given, 'rbf' will be used. If a callable is given it is used to pre-compute the kernel matrix from data matrices; that matrix should be an array of shape (n_samples, n_samples).

        degree : int, default=3
            Degree of the polynomial kernel function ('poly'). Ignored by all other kernels.

        gamma : {'scale', 'auto'} or float, default='scale'
            Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.

            if gamma='scale' (default) is passed then it uses 1 / (n_features * X.var()) as value of gamma,
            if 'auto', uses 1 / n_features.

        coef0 : float, default=0.0
            Independent term in kernel function. It is only significant in 'poly' and 'sigmoid'.

        shrinking : bool, default=True
            Whether to use the shrinking heuristic. See the User Guide <shrinking_svm>.

        probability : bool, default=False
            Whether to enable probability estimates. This must be enabled prior to calling fit, will slow down that method as it internally uses 5-fold cross-validation, and predict_proba may be inconsistent with predict. Read more in the User Guide <scores_probabilities>.

        tol : float, default=1e-3
            Tolerance for stopping criterion.

        cache_size : float, default=200
            Specify the size of the kernel cache (in MB).

        class_weight : dict or 'balanced', default=None
            Set the parameter C of class i to class_weight[i]*C for SVC. If not given, all classes are supposed to have weight one. The "balanced" mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y)).

        verbose : bool, default=False
            Enable verbose output. Note that this setting takes advantage of a per-process runtime setting in libsvm that, if enabled, may not work properly in a multithreaded context.

        max_iter : int, default=-1
            Hard limit on iterations within solver, or -1 for no limit.

        decision_function_shape : {'ovo', 'ovr'}, default='ovr'
            Whether to return a one-vs-rest ('ovr') decision function of shape (n_samples, n_classes) as all other classifiers, or the original one-vs-one ('ovo') decision function of libsvm which has shape (n_samples, n_classes * (n_classes - 1) / 2). However, note that internally, one-vs-one ('ovo') is always used as a multi-class strategy to train models; an ovr matrix is only constructed from the ovo matrix. The parameter is ignored for binary classification.

        break_ties : bool, default=False
            If true, decision_function_shape='ovr', and number of classes > 2, predict will break ties according to the confidence values of decision_function; otherwise the first class among the tied classes is returned. Please note that breaking ties comes at a relatively high computational cost compared to a simple predict.

        random_state : int, RandomState instance or None, default=None
            Controls the pseudo random number generation for shuffling the data for probability estimates. Ignored when probability is False. Pass an int for reproducible output across multiple function calls. See Glossary <random_state>.

        Examples
        --------

        Return
        ------
        Bcip Classifier Object
        """
        svm_object = svm.SVC(C, kernel, degree, gamma, coef0, shrinking, probability, tol, cache_size, class_weight, verbose, max_iter, decision_function_shape, break_ties, random_state)
        f = cls(sess, 'svm', svm_object)


    @classmethod
    def create_LDA(cls,sess, solver="svd", shrinkage=None, priors=None, n_components=None, store_covariance=False, tol=0.0001, covariance_estimator=None):
        """
        Factory method to create an LDA BCIP Classifier object. 

        A classifier with a linear decision boundary, generated by fitting class conditional densities to the 
        data and using Bayes' rule. The model fits a Gaussian density to each class, assuming that all classes 
        share the same covariance matrix.
        
        Parameters
        ----------
        sess : session object
            Session where the SVM BCIP Classifier object will exist

        solver : {'svd', 'lsqr', 'eigen'}, default='svd'
            Solver to use, possible values:

            'svd': Singular value decomposition (default). Does not compute the covariance matrix, therefore this solver is recommended for data with a large number of features.
            'lsqr': Least squares solution. Can be combined with shrinkage or custom covariance estimator.
            'eigen': Eigenvalue decomposition. Can be combined with shrinkage or custom covariance estimator.

        shrinkage : 'auto' or float, default=None
            Shrinkage parameter, possible values:

            None: no shrinkage (default).
            'auto': automatic shrinkage using the Ledoit-Wolf lemma.
            float between 0 and 1: fixed shrinkage parameter.
                This should be left to None if covariance_estimator is used. Note that shrinkage works only with 'lsqr' and 'eigen' solvers.

        priors : array-like of shape (n_classes,), default=None
            The class prior probabilities. By default, the class proportions are inferred from the training data.

        n_components : int, default=None
            Number of components (<= min(n_classes - 1, n_features)) for dimensionality reduction. If None, will be set to min(n_classes - 1, n_features). This parameter only affects the transform method.

        store_covariance : bool, default=False
            If True, explicitly compute the weighted within-class covariance matrix when solver is 'svd'. The matrix is always computed and stored for the other solvers.

        tol : float, default=1.0e-4
            Absolute threshold for a singular value of X to be considered significant, used to estimate the rank of X. Dimensions whose singular values are non-significant are discarded. Only used if solver is 'svd'.

        covariance_estimator : covariance estimator, default=None
            If not None, covariance_estimator is used to estimate the covariance matrices instead of relying on the empirical covariance estimator (with potential shrinkage). The object should have a fit method and a covariance_ attribute like the estimators in sklearn.covariance. if None the shrinkage parameter drives the estimate.

            This should be left to None if shrinkage is used. Note that covariance_estimator works only with 'lsqr' and 'eigen' solvers.

        """
        lda_object = discriminant_analysis.LinearDiscriminantAnalysis(solver, shrinkage, priors, n_components, store_covariance, tol, covariance_estimator)
        #clf = classification.TSclassifier(clf=lda_object)
        f = cls(sess, 'lda', lda_object)

        sess.add_misc_bcip_obj(f)

        return f

    @classmethod
    def create_logistic_regression(cls,sess, penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None):
        log_reg_object = LogisticRegression(penalty, dual=dual, tol=tol, C=C, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight=class_weight, random_state=random_state, solver=solver, max_iter=max_iter, multi_class=multi_class, verbose=0, warm_start=warm_start, n_jobs=n_jobs, l1_ratio=l1_ratio)
        f = cls(sess, 'logistic regression', log_reg_object)
        sess.add_misc_bcip_obj(f)

        return f

    @classmethod
    def create_custom_classifier(cls, sess, classifier_object, classifier_type):
        """ 
        Factory method to create a generic BCIP Classifier object. 
        
        Parameters
        """
        f = cls(sess, classifier_type, classifier_object)
        sess.add_misc_bcip_obj(f)

        return f