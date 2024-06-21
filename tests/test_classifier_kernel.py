import sklearn.discriminant_analysis
import mindpype as mp
import sys, os
import numpy as np

class ClassifierKernelCreationUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestClassifierKernelCreation(self):
        inTensor = mp.Tensor.create(self.__session, (50,26))
        mp_clsf = mp.Classifier.create_LDA(self.__session, shrinkage='auto', solver='lsqr')
        prediction = mp.Tensor.create(self.__session, (50,))
        init_tensor = mp.Tensor.create(self.__session, (50,26))
        labels = mp.Tensor.create(self.__session,(50,))
        node = mp.kernels.ClassifierKernel.add_to_graph(self.__graph,inTensor,mp_clsf,prediction,init_tensor,labels)
        return node.mp_type
    
class ClassifierKernelExecutionUnitTest:

    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestClassifierKernelExecution(self):
        np.random.seed(44)
        raw_data = np.random.randint(-10,10, size=(50,26))
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        mp_clsf = mp.Classifier.create_LDA(self.__session, shrinkage='auto', solver='lsqr')
        
        predictions = mp.Tensor.create(self.__session, (50,))
        init_data = np.random.randint(-10,10, size=(50,26))
        init_tensor = mp.Tensor.create_from_data(self.__session, init_data)
        init_labels_data = np.random.randint(0,4, (50,))
        init_labels = mp.Tensor.create_from_data(self.__session, init_labels_data)
        
        node = mp.kernels.ClassifierKernel.add_to_graph(self.__graph,inTensor,mp_clsf,predictions,num_classes=4,initialization_data=init_tensor,labels=init_labels)

        sys.stdout = open(os.devnull, 'w')
        self.__graph.verify()
        self.__graph.initialize(init_tensor, init_labels)
        self.__graph.execute()
        sys.stdout = sys.__stdout__

        return (init_data, init_labels_data, raw_data, predictions.data)

def test_create():
    KernelUnitTest_Object = ClassifierKernelCreationUnitTest()
    assert KernelUnitTest_Object.TestClassifierKernelCreation() == mp.MPEnums.NODE
    del KernelUnitTest_Object


def test_execute():
    KernelExecutionUnitTest_Object = ClassifierKernelExecutionUnitTest()
    res = KernelExecutionUnitTest_Object.TestClassifierKernelExecution()
    
    classifier = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(shrinkage='auto', solver='lsqr')
    classifier.fit(res[0], res[1])
    expected_output = classifier.predict(res[2])
    assert (res[3] == expected_output).all()
    del KernelExecutionUnitTest_Object
    