import mindpype as mp
import numpy as np

class ReducedSumKernelUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestReducedSumKernelExecution(self, raw_data):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, (1,))
        tensor_test_node = mp.kernels.ReducedSumKernel.add_to_graph(self.__graph,inTensor,outTensor,axis=None)

        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        
        return (raw_data, outTensor.data)

def test_execute():
    np.random.seed(44)
    raw_data = np.random.randn(2,2,2)
    KernelExecutionUnitTest_Object = ReducedSumKernelUnitTest()
    res = KernelExecutionUnitTest_Object.TestReducedSumKernelExecution(raw_data)
    expected_output = np.sum(res[0], axis = None)
    assert (res[1] == expected_output).all()
    del KernelExecutionUnitTest_Object