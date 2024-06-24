import pyriemann.utils
import pyriemann.utils.covariance
import mindpype as mp
import numpy as np
import pyriemann

class TangentSpaceKernelUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TangentSpaceKernelExecution(self, raw_data, initialization_data):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, (2,3))
        init_in = mp.Tensor.create_from_data(self.__session, initialization_data)
        tensor_test_node = mp.kernels.TangentSpaceKernel.add_to_graph(self.__graph,inTensor, outTensor, initialization_data=init_in)

        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()

        return outTensor.data

def test_execute():
    np.random.seed(44)
    raw_data = np.random.randn(2,2,2)
    raw_data = pyriemann.utils.covariance.covariances(raw_data)
    r = 0.001
    raw_data = (1-r)*raw_data + r*np.diag(np.ones(raw_data.shape[-1]))
    initialization_data = np.random.randn(2,2,2)
    initialization_data = pyriemann.utils.covariance.covariances(initialization_data)
    initialization_data = (1-r)*initialization_data + r*np.diag(np.ones(initialization_data.shape[-1]))
    KernelExecutionUnitTest_Object = TangentSpaceKernelUnitTest()
    res = KernelExecutionUnitTest_Object.TangentSpaceKernelExecution(raw_data, initialization_data)

    tangent_space = pyriemann.tangentspace.TangentSpace()
    tangent_space.fit(initialization_data)
    output = tangent_space.transform(raw_data)
    assert (res == output).all()

    del KernelExecutionUnitTest_Object
