import mindpype as mp
import sys, os
import numpy as np

class FeatureNormalizationKernelCreationUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestFeatureNormalizationKernelCreation(self):
        inTensor = mp.Tensor.create(self.__session, (10,10,10))
        outTensor = mp.Tensor.create(self.__session,(10,10,10))
        initialization_tensor = mp.Tensor.create(self.__session, (10,10,10))
        labels = mp.Tensor.create(self.__session, (10,))
        node = mp.kernels.FeatureNormalizationKernel.add_to_graph(self.__graph,inTensor,outTensor,method='zscore-norm',axis=0,init_data=initialization_tensor,labels=labels)
        # node = mp.kernels.FeatureNormalizationKernel.add_to_graph(self.__graph,inTensor,outTensor,method='min-max',axis=0)
        return node.mp_type
    
class FeatureNormalizationKernelExecutionUnitTest:

    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestFeatureNormalizationKernelExecution(self):
        np.random.seed(44)
        raw_data = np.random.randint(-10,10, size=(10,10,10))
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session,(10,10,10))
        init_data = np.random.randint(-10,10, size=(10,10,10))
        initialization_tensor = mp.Tensor.create_from_data(self.__session, init_data)
        labels_data = np.random.randint(0, 4, size=(10,))
        labels = mp.Tensor.create_from_data(self.__session, labels_data)
        node = mp.kernels.FeatureNormalizationKernel.add_to_graph(self.__graph,inTensor,outTensor,method='zscore-norm',axis=0,init_data=initialization_tensor,labels=labels)
        # node = mp.kernels.FeatureNormalizationKernel.add_to_graph(self.__graph,inTensor,outTensor,axis=0)
        sys.stdout = open(os.devnull, 'w')
        self.__graph.verify()
        self.__graph.initialize(initialization_tensor, labels)
        self.__graph.execute()
        sys.stdout = sys.__stdout__

        return (raw_data, init_data, outTensor.data)

def test_create():
    KernelUnitTest_Object = FeatureNormalizationKernelCreationUnitTest()
    assert KernelUnitTest_Object.TestFeatureNormalizationKernelCreation() == mp.MPEnums.NODE
    del KernelUnitTest_Object


def test_execute():
    KernelExecutionUnitTest_Object = FeatureNormalizationKernelExecutionUnitTest()
    res = KernelExecutionUnitTest_Object.TestFeatureNormalizationKernelExecution()
    expected_output = (res[0] - np.mean(res[1], axis=0)) / np.std(res[1], axis=0)
    assert (res[2] == expected_output).all()
    del KernelExecutionUnitTest_Object
    