import mindpype as mp
import sys, os
import numpy as np

class ReducedSumKernelCreationUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestReducedSumKernelCreation(self):
        inTensor = mp.Tensor.create(self.__session, (2,2,2))
        outTensor = mp.Tensor.create(self.__session, (1,))
        node = mp.kernels.ReducedSumKernel.add_to_graph(self.__graph,inTensor,outTensor,axis=None)
        return node.mp_type
    
class ReducedSumKernelExecutionUnitTest:

    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestReducedSumKernelExecution(self):
        np.random.seed(44)
        raw_data = np.random.randint(-10,10, size=(2,2,2))
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, (1,))
        tensor_test_node = mp.kernels.ReducedSumKernel.add_to_graph(self.__graph,inTensor,outTensor,axis=None)

        sys.stdout = open(os.devnull, 'w')
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        sys.stdout = sys.__stdout__

        return (raw_data, outTensor.data)

def test_create():
    KernelUnitTest_Object = ReducedSumKernelCreationUnitTest()
    assert KernelUnitTest_Object.TestReducedSumKernelCreation() == mp.MPEnums.NODE
    del KernelUnitTest_Object
    
def test_execute():
    KernelExecutionUnitTest_Object = ReducedSumKernelExecutionUnitTest()
    res = KernelExecutionUnitTest_Object.TestReducedSumKernelExecution()
    expected_output = np.sum(res[0], axis = None)
    assert (res[1] == expected_output).all()
    del KernelExecutionUnitTest_Object