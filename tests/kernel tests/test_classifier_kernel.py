import sklearn.discriminant_analysis
import mindpype as mp
import numpy as np
import pytest

class ClassifierKernelUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestClassifierKernelExecution(self, raw_data, init_data, init_labels_data, num_classes=4, test_invalid_input=False, test_invalid_classifiers=False):
        if test_invalid_input:
            inTensor = mp.Scalar.create_from_value(self.__session, "test")
        else:
            inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        if test_invalid_classifiers:
            mp_clsf = mp.Scalar.create_from_value(self.__session, "test")
        else:
            mp_clsf = mp.Classifier.create_LDA(self.__session, shrinkage='auto', solver='lsqr')
 
        predictions = mp.Tensor.create(self.__session, init_labels_data.shape)
        init_tensor = mp.Tensor.create_from_data(self.__session, init_data)
        init_labels = mp.Tensor.create_from_data(self.__session, init_labels_data)
        node = mp.kernels.ClassifierKernel.add_to_graph(self.__graph,inTensor,mp_clsf,predictions,num_classes=num_classes,initialization_data=init_tensor,labels=init_labels)
        self.__graph.verify()
        self.__graph.initialize(init_tensor, init_labels)
        self.__graph.execute()
        return predictions.data

    
def test_execute():
    np.random.seed(44)
    raw_data = np.random.randn(50,26)
    init_data = np.random.randn(50,26)
    init_labels_data = np.random.randint(0,4, (50,))
    KernelExecutionUnitTest_Object = ClassifierKernelUnitTest()
    res = KernelExecutionUnitTest_Object.TestClassifierKernelExecution(raw_data, init_data, init_labels_data)
    
    classifier = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(shrinkage='auto', solver='lsqr')
    classifier.fit(init_data, init_labels_data)
    expected_output = classifier.predict(raw_data)
    assert (res == expected_output).all()
    
    # test errors
    with pytest.raises(TypeError) as e_info:
        res = KernelExecutionUnitTest_Object.TestClassifierKernelExecution(raw_data, init_data, init_labels_data, test_invalid_input=True)
    del KernelExecutionUnitTest_Object
    
    KernelExecutionUnitTest_Object = ClassifierKernelUnitTest()
    with pytest.raises(TypeError) as e_info:
        res = KernelExecutionUnitTest_Object.TestClassifierKernelExecution(raw_data, init_data, init_labels_data, test_invalid_classifiers=True)
    del KernelExecutionUnitTest_Object
    
    init_data = np.random.randn(2,2)
    init_labels_2d = np.random.randint(0,4, (2,2))
    KernelExecutionUnitTest_Object = ClassifierKernelUnitTest()
    with pytest.raises(ValueError) as e_info:
        res = KernelExecutionUnitTest_Object.TestClassifierKernelExecution(raw_data, init_data, init_labels_2d)
    del KernelExecutionUnitTest_Object
    
test_execute()