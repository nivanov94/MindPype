import sklearn.discriminant_analysis
import mindpype as mp
import numpy as np

class ClassifierKernelUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestClassifierKernelExecution(self, raw_data, init_data, init_labels_data):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        mp_clsf = mp.Classifier.create_LDA(self.__session, shrinkage='auto', solver='lsqr')
        predictions = mp.Tensor.create(self.__session, init_labels_data.shape)
        init_tensor = mp.Tensor.create_from_data(self.__session, init_data)
        init_labels = mp.Tensor.create_from_data(self.__session, init_labels_data)
        node = mp.kernels.ClassifierKernel.add_to_graph(self.__graph,inTensor,mp_clsf,predictions,num_classes=4,initialization_data=init_tensor,labels=init_labels)
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
    del KernelExecutionUnitTest_Object
    