import mindpype as mp
import numpy as np

class FeatureNormalizationKernelUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestFeatureNormalizationKernelExecution(self, raw_data, init_data, labels_data):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session,raw_data.shape)
        initialization_tensor = mp.Tensor.create_from_data(self.__session, init_data)
        labels = mp.Tensor.create_from_data(self.__session, labels_data)
        node = mp.kernels.FeatureNormalizationKernel.add_to_graph(self.__graph,inTensor,outTensor,method='zscore-norm',axis=0,init_data=initialization_tensor,labels=labels)
        self.__graph.verify()
        self.__graph.initialize(initialization_tensor, labels)
        self.__graph.execute()
        return (raw_data, init_data, outTensor.data)

def test_execute():
    np.random.seed(44)
    raw_data = np.random.randn(10,10,10)
    init_data = np.random.randn(10,10,10)
    labels_data = np.random.randint(0, 4, size=(10,))
    KernelExecutionUnitTest_Object = FeatureNormalizationKernelUnitTest()
    res = KernelExecutionUnitTest_Object.TestFeatureNormalizationKernelExecution(raw_data, init_data, labels_data)
    expected_output = (res[0] - np.mean(res[1], axis=0)) / np.std(res[1], axis=0)
    assert (res[2] == expected_output).all()
    del KernelExecutionUnitTest_Object
    