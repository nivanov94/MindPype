import mindpype as mp
import sys, os
import numpy as np

class MaxKernelCreationUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestMaxKernelCreation(self):
        inTensor = mp.Tensor.create(self.__session, (2,2))
        outTensor = mp.Tensor.create(self.__session, (1,))
        node = mp.kernels.MaxKernel.add_to_graph(self.__graph,inTensor,outTensor)
        return node.mp_type
    
class MaxKernelExecutionUnitTest:

    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestMaxKernelExecution(self):
        np.random.seed(44)
        raw_data = np.random.randint(-10,10, size=(2,2))
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, (1,))
        tensor_test_node = mp.kernels.MaxKernel.add_to_graph(self.__graph,inTensor,outTensor)

        sys.stdout = open(os.devnull, 'w')
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        sys.stdout = sys.__stdout__

        return (inTensor.data, outTensor.data)
    
class MinKernelCreationUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestMinKernelCreation(self):
        inTensor = mp.Tensor.create(self.__session, (2,2))
        outTensor = mp.Tensor.create(self.__session, (1,))
        node = mp.kernels.MinKernel.add_to_graph(self.__graph,inTensor,outTensor)
        return node.mp_type
    
class MinKernelExecutionUnitTest:

    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestMinKernelExecution(self):
        np.random.seed(44)
        raw_data = np.random.randint(-10,10, size=(2,2))
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, (1,))
        tensor_test_node = mp.kernels.MinKernel.add_to_graph(self.__graph,inTensor,outTensor)

        sys.stdout = open(os.devnull, 'w')
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        sys.stdout = sys.__stdout__

        return (inTensor.data, outTensor.data)

def test_create():
    KernelUnitTest_Object = MaxKernelCreationUnitTest()
    assert KernelUnitTest_Object.TestMaxKernelCreation() == mp.MPEnums.NODE
    del KernelUnitTest_Object
    
    KernelUnitTest_Object = MinKernelCreationUnitTest()
    assert KernelUnitTest_Object.TestMinKernelCreation() == mp.MPEnums.NODE
    del KernelUnitTest_Object


def test_execute():
    KernelExecutionUnitTest_Object = MinKernelExecutionUnitTest()
    res = KernelExecutionUnitTest_Object.TestMinKernelExecution()
    
    expected_output = np.min(res[0])
    
    assert (res[1] == expected_output).all()
    del KernelExecutionUnitTest_Object