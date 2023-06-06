"""Emulates the filter class
If we pass in a generic classifier obejct, should be able to execute standard sklearn commands

Creating a kernel to handle verification and execution should be straight-foward

Create a classifier object and enter
"""

from .core import BCIP, BcipEnums
from pyriemann import classification
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

class Classifier(BCIP):
    """
    A classifier that can be used by different BCIP kernels

    Parameters
    ----------
    sess : Session object
        - Session where the Array object will exist

    ctype : str
        - The name of the classifier to be created

    classifier : BCIPy Classifier object
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

        Args:
            sess (session object): Session where the SVM BCIP Classifier object will exist
        
        

        Examples
        --------

        Return
        ------
        Bcip Classifier Object
        """
        svm_object = SVC(C, kernel, degree, gamma, coef0, shrinking, probability, tol, cache_size, class_weight, verbose, max_iter, decision_function_shape, break_ties, random_state)
        f = cls(sess, 'svm', svm_object)


    @classmethod
    def create_LDA(cls,sess, solver="svd", shrinkage=None, priors=None, n_components=None, store_covariance=False, tol=0.0001, covariance_estimator=None):
        """
        Factory method to create an LDA BCIP Classifier object. 
        
        Parameters
        ----------

        sess : session object
            * Session where the SVM BCIP Classifier object will exist

        solver : {'svd', 'lsqr', 'eigen'}, default='svd'
            * Solver to use, possible values:
                * 'svd' : Singular value decomposition (default). Does not compute the covariance matrix, therefore this solver is recommended for data with a large number of features.
                * 'lsqr' : Least squares solution. Can be combined with shrinkage or custom covariance estimator.
                * 'eigen' : Eigenvalue decomposition. Can be combined with shrinkage or custom covariance estimator.
            
        shrinkage : 'auto' or float, default=None
            * Shrinkage parameter, possible values:
                * None: no shrinkage (default).
                * 'auto': automatic shrinkage using the Ledoit-Wolf lemma.
                * float between 0 and 1: fixed shrinkage parameter.
                    * This should be left to None if covariance_estimator is used. Note that shrinkage works only with 'lsqr' and 'eigen' solvers.

        priors : array-like of shape (n_classes,), default=None
            * The class prior probabilities. By default, the class proportions are inferred from the training data.

        n_components : int, default=None
            * Number of components (should be of dimension at least min(n_classes - 1, n_features)) for dimensionality reduction. If None, will be set to min(n_classes - 1, n_features). This parameter only affects the transform method.

        store_covariance: bool, default=False
            * If True, explicitly compute the weighted within-class covariance matrix when solver is 'svd'. The matrix is always computed and stored for the other solvers.

        tol : float, default=.0001
            * Absolute threshold for a singular value of X to be considered significant, used to estimate the rank of X. Dimensions whose singular values are non-significant are discarded. Only used if solver is 'svd'.

        covariance_estimator : covariance estimator, default=None
            * If not None, covariance_estimator is used to estimate the covariance matrices instead of relying on the empirical covariance estimator (with potential shrinkage). The object should have a fit method and a covariance\_ attribute like the estimators in sklearn.covariance. if None the shrinkage parameter drives the estimate.
                * This should be left to None if shrinkage is used. Note that covariance_estimator works only with 'lsqr' and 'eigen' solvers.

        """
        lda_object = LinearDiscriminantAnalysis(solver, shrinkage, priors, n_components, store_covariance, tol, covariance_estimator)
        #clf = classification.TSclassifier(clf=lda_object)
        f = cls(sess, 'lda', lda_object)

        sess.add_misc_bcip_obj(f)

        return f

    @classmethod
    def create_logistic_regression(cls,sess, penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None):
        """
        Parameters
        ----------
        penalty : {'l1', 'l2', 'elasticnet', 'none'}, default='l2'
            Specify the norm of the penalty:

            'none': no penalty is added;
            'l2': add a L2 penalty term and it is the default choice;
            'l1': add a L1 penalty term;
            'elasticnet': both L1 and L2 penalty terms are added.

        dual : bool, default=False
            Dual or primal formulation. Dual formulation is only implemented for l2 penalty with liblinear solver. Prefer dual=False when n_samples > n_features.

        tol : float, default=1e-4
            Tolerance for stopping criteria.

        C : float, default=1.0
            Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.

        fit_intercept : bool, default=True
            Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function.

        intercept_scaling : float, default=1
            Useful only when the solver 'liblinear' is used and self.fit_intercept is set to True. In this case, x becomes [x, self.intercept_scaling], i.e. a "synthetic" feature with constant value equal to intercept_scaling is appended to the instance vector. The intercept becomes intercept_scaling * synthetic_feature_weight.

            Note! the synthetic feature weight is subject to l1/l2 regularization as all other features. To lessen the effect of regularization on synthetic feature weight (and therefore on the intercept) intercept_scaling has to be increased.

        class_weight : dict or 'balanced', default=None
            Weights associated with classes in the form {class_label: weight}. If not given, all classes are supposed to have weight one.

            The "balanced" mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y)).

            Note that these weights will be multiplied with sample_weight (passed through the fit method) if sample_weight is specified.

        random_state : int, RandomState instance, default=None
            Used when solver == 'sag', 'saga' or 'liblinear' to shuffle the data. See Glossary <random_state> for details.

        solver : {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'}, default='lbfgs'

            Algorithm to use in the optimization problem. Default is 'lbfgs'. To choose a solver, you might want to consider the following aspects:

            - For small datasets, 'liblinear' is a good choice, whereas 'sag'and 'saga' are faster for large ones; - For multiclass problems, only 'newton-cg', 'sag', 'saga' and 'lbfgs' handle multinomial loss; - 'liblinear' is limited to one-versus-rest schemes.

        max_iter : int, default=100
            Maximum number of iterations taken for the solvers to converge.

        multi_class : {'auto', 'ovr', 'multinomial'}, default='auto'
            If the option chosen is 'ovr', then a binary problem is fit for each label. For 'multinomial' the loss minimised is the multinomial loss fit across the entire probability distribution, *even when the data is binary*. 'multinomial' is unavailable when solver='liblinear'. 'auto' selects 'ovr' if the data is binary, or if solver='liblinear', and otherwise selects 'multinomial'.

        verbose : int, default=0
            For the liblinear and lbfgs solvers set verbose to any positive number for verbosity.

        warm_start : bool, default=False
            When set to True, reuse the solution of the previous call to fit as initialization, otherwise, just erase the previous solution. Useless for liblinear solver. See the Glossary <warm_start>.

        n_jobs : int, default=None
            Number of CPU cores used when parallelizing over classes if multi_class='ovr'". This parameter is ignored when the solver is set to 'liblinear' regardless of whether 'multi_class' is specified or not. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors. See Glossary <n_jobs> for more details.

        l1_ratio : float, default=None
            The Elastic-Net mixing parameter, with 0 <= l1_ratio <= 1. Only used if penalty='elasticnet'. Setting l1_ratio=0 is equivalent to using penalty='l2', while setting l1_ratio=1 is equivalent to using penalty='l1'. For 0 < l1_ratio <1, the penalty is a combination of L1 and L2.
                
        """
        
        
        log_reg_object = LogisticRegression(penalty, dual=dual, tol=tol, C=C, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight=class_weight, random_state=random_state, solver=solver, max_iter=max_iter, multi_class=multi_class, verbose=0, warm_start=warm_start, n_jobs=n_jobs, l1_ratio=l1_ratio)
        f = cls(sess, 'logistic regression', log_reg_object)
        sess.add_misc_bcip_obj(f)

        return f

    @classmethod
    def create_custom_classifier(cls, sess, classifier_object, classifier_type):
        """ 
        Factory method to create a generic BCIP Classifier object. 
        
        Parameters
        ----------
        sess : BCIP Session Object
            The BCIP Session object to which the classifier will be added.
        classifier_object : BCIP Classifier object
            The classifier object to be added to the BCIP Session.
        classifier_type : str
            The type of classifier to be added to the BCIP Session.            
        """
        f = cls(sess, classifier_type, classifier_object)
        sess.add_misc_bcip_obj(f)

        return f
