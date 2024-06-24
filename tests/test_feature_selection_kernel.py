import mindpype as mp
import numpy as np
from sklearn.feature_selection import SelectKBest

class FeatureSelectionKernelUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestFeatureSelectionKernelExecution(self, raw_data, init_data, labels_data):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session,raw_data.shape)
        initialization_tensor = mp.Tensor.create_from_data(self.__session, init_data)
        labels = mp.Tensor.create_from_data(self.__session, labels_data)
        node = mp.kernels.FeatureSelectionKernel.add_to_graph(self.__graph,inTensor,outTensor,k=10,init_inputs=initialization_tensor,labels=labels)
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        return (raw_data, init_data, labels_data, outTensor.data)

def test_execute():
    np.random.seed(44)
    raw_data = np.random.randn(10,10)
    init_data = np.random.randn(10,10)
    labels_data = np.random.randint(0, 4, size=(10,))
    KernelExecutionUnitTest_Object = FeatureSelectionKernelUnitTest()
    res = KernelExecutionUnitTest_Object.TestFeatureSelectionKernelExecution(raw_data, init_data, labels_data)

    model = SelectKBest(k=10)
    model.fit(res[1], res[2])
    expected_output = model.transform(res[0])
    assert(res[3] == expected_output).all()
    del KernelExecutionUnitTest_Object
    