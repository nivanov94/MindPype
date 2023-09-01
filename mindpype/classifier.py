"""

Use this class to create a classifier object that can be used by the MindPype Classifier kernel.

.. note:: 
   MindPype Classifier objects must be created in order to be used by the MindPype Classifier kernel.

"""

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from .core import MPBase, MPEnums


class Classifier(MPBase):
    """
    A classifier that can be used by different MindPype kernels

    Args:
        MPBase (MPBase): The base class for all MindPype objects

    Parameters
    ----------
    sess : Session 
       Session where the Array object will exist
    ctype : str
       The name of the classifier to be created
    classifier : Classifier 
       The classifier object to be used within the node (should be the return from a MindPype kernel)

    Attributes
    ----------
    _ctype : str
       One of ['lda', 'svm', 'logistic regression', 'custom'], corresponding to the type of classifier
    _classifier : Classifier
       The Classifier object (ie. Scipy classifier object) that will dictate this node's function

    Examples
    --------

    .. code:: python
        
        from bcipy import Classifier
        
        # Create a MindPype Classifier object using the factory method
        classifier_object = Classifier.create_LDA(sess, solver='svd', shrinkage=None, priors=None,
        n_components=None, store_covariance=False, tol=0.0001)

    Return
    ------
    MindPype Classifier object : Classifier
        The MindPype Classifier object that can be used by the MindPype Classifier kernel



    """

    # these are the possible internal methods for storing the filter
    # parameters which determine how it will be executed
    ctypes = ["lda", "svm", "logistic regression", "custom"]

    def __init__(self, sess, ctype, classifier):
        """
        Constructor to create a new filter object
        """
        self._ctype = ctype
        self._classifier = classifier

        if hasattr(classifier, 'n_classes_'):
            self.n_classes = classifier.n_classes_
        else:
            self.n_classes = 2

        super().__init__(MPEnums.CLASSIFIER, sess)

    def __str__(self):
        return (
            "MindPype {} Classifier with following"
            + f"attributes:\nClassifier Type: {self.ctype}\n"
        )

    # API Getters
    @property
    def ctype(self):
        """
        Getter for the classifier type
        
        Returns
        -------
        The classifier type : str

        """
        return self.ctype

    @classmethod
    def create_SVM(
        cls,
        sess,
        C=1,
        kernel="rbf",
        degree=3,
        gamma="scale",
        coef0=0,
        shrinking=True,
        probability=False,
        tol=0.001,
        cache_size=200,
        class_weight=None,
        verbose=False,
        max_iter=-1,
        decision_function_shape="ovr",
        break_ties=False,
        random_state=None,
    ):
        """
        Factory Method to create an SVM MindPype Classifier object.

        .. note::
           C-Support Vector Classification. The implementation is based on libsvm. 
           The fit time scales at least quadratically with the number of samples 
           and may be impractical beyond tens of thousands of samples.
           The multiclass support is handled according to a one-vs-one scheme.

           
        .. note:: 
            All unlisted parameters are the same as the sklearn SVC object. 
            Check out the `sklearn documentation <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`_
            for more information.

        Parameters
        ----------
        sess : session object
            Session where the SVM MindPype Classifier object will exist

    
        Examples
        --------
        >>> from bcipy import Classifier
        >>> classifier_object = Classifier.create_SVM(sess)

        Return
        ------
        MindPype Classifier Object : Classifier
            MindPype Classifier Object containing the SVM classifier

        """
        
        svm_object = SVC(
            C,
            kernel,
            degree,
            gamma,
            coef0,
            shrinking,
            probability,
            tol,
            cache_size,
            class_weight,
            verbose,
            max_iter,
            decision_function_shape,
            break_ties,
            random_state,
        )
        f = cls(sess, "svm", svm_object)
        sess.add_misc_mp_obj(f)
        return f

    @classmethod
    def create_LDA(
        cls,
        sess,
        solver="svd",
        shrinkage=None,
        priors=None,
        n_components=None,
        store_covariance=False,
        tol=0.0001,
        covariance_estimator=None,
    ):
        """
        Factory method to create an LDA MindPype Classifier object.

        .. note:: 
           All unlisted parameters are the same as the LDA Scikit-Learn object
           `linked here <https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html>`_
           for more information.

        Parameters
        ----------

        sess : Session
            Session where the SVM MindPype Classifier object will exist

    
        Return
        ------
        MindPype Classifier : Classifier
            MindPype Classifier Object containing the LDA classifier


        Examples
        --------
        >>> from bcipy import Classifier
        >>> classifier_object = Classifier.create_LDA(sess)

        """
        lda_object = LinearDiscriminantAnalysis(
            solver,
            shrinkage,
            priors,
            n_components,
            store_covariance,
            tol,
            covariance_estimator,
        )
        # clf = classification.TSclassifier(clf=lda_object)
        f = cls(sess, "lda", lda_object)

        sess.add_misc_mp_obj(f)

        return f

    @classmethod
    def create_logistic_regression(
        cls,
        sess,
        penalty="l2",
        dual=False,
        tol=0.0001,
        C=1.0,
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        random_state=None,
        solver="lbfgs",
        max_iter=100,
        multi_class="auto",
        verbose=0,
        warm_start=False,
        n_jobs=None,
        l1_ratio=None,
    ):
        """

        .. note::
           The unlisted parameters accepted by this function are specified in the documentation 
           for the 
           `SKLearn Logistic Regression classifier <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`_
                

        Parameters
        ----------
        sess : session object
            Session where the Logistic Regression MindPype Classifier object will exist

        
    
        Return
        ------
        MindPype Classifier Object : Classifier
            MindPype Classifier Object containing the Logistic Regression classifier

        Examples
        --------
        >>> from bcipy import Classifier
        >>> classifier_object = Classifier.create_logistic_regression(sess)

        """

        log_reg_object = LogisticRegression(
            penalty,
            dual=dual,
            tol=tol,
            C=C,
            fit_intercept=fit_intercept,
            intercept_scaling=intercept_scaling,
            class_weight=class_weight,
            random_state=random_state,
            solver=solver,
            max_iter=max_iter,
            multi_class=multi_class,
            verbose=verbose,
            warm_start=warm_start,
            n_jobs=n_jobs,
            l1_ratio=l1_ratio,
        )
        f = cls(sess, "logistic regression", log_reg_object)
        sess.add_misc_mp_obj(f)

        return f

    @classmethod
    def create_custom_classifier(cls, sess, classifier_object, classifier_type):
        """
        Factory method to create a generic MindPype Classifier object.

        Parameters
        ----------
        sess : MindPype Session Object
            The MindPype Session object to which the classifier will be added.
        classifier_object : Sklearn Classifier object
            The classifier object to be added to the MindPype Session.
        classifier_type : str
            The type of classifier to be added to the MindPype Session.

        Return
        ------
        Classifier Object : Classifier
            MindPype Classifier object that contains the classifier object and type.

        Examples
        --------
        >>> from bcipy import Classifier
        >>> from sklearn.svm import SVC
        >>> svm_object = SVC()
        >>> classifier_object = Classifier.create_custom_classifier(sess, svm_object, 'svm')
        """
        f = cls(sess, classifier_type, classifier_object)
        sess.add_misc_mp_obj(f)

        return f
