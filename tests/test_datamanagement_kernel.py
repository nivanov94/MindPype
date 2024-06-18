import mindpype as mp
import sys, os
import numpy as np

class ConcatenationKernelCreationUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestConcatenationKernelCreation(self):
        inTensor1 = mp.Tensor.create(self.__session, (2,2,2))
        inTensor2 = mp.Tensor.create(self.__session, (2,2,2))
        outTensor = mp.Tensor.create(self.__session, (4,2,2))
        node = mp.kernels.ConcatenationKernel.add_to_graph(self.__graph,inTensor1,inTensor2,outTensor,axis=0)
        return node.mp_type
    
class ConcatenationKernelExecutionUnitTest:

    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestConcatenationKernelExecution(self):
        np.random.seed(44)
        raw_data = np.random.randint(-10,10, size=(2,2,2))
        inTensor1 = mp.Tensor.create_from_data(self.__session, raw_data)
        inTensor2 = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, (4,2,2))
        tensor_test_node = mp.kernels.ConcatenationKernel.add_to_graph(self.__graph,inTensor1,inTensor2,outTensor,axis=0)

        sys.stdout = open(os.devnull, 'w')
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        sys.stdout = sys.__stdout__

        return (raw_data, outTensor.data)

def test_create():
    KernelUnitTest_Object = ConcatenationKernelCreationUnitTest()
    assert KernelUnitTest_Object.TestConcatenationKernelCreation() == mp.MPEnums.NODE
    del KernelUnitTest_Object


def test_execute():
    KernelExecutionUnitTest_Object = ConcatenationKernelExecutionUnitTest()
    res = KernelExecutionUnitTest_Object.TestConcatenationKernelExecution()
    
    # manual concatenation
    output = np.concatenate((res[0], res[0]), axis = 0)
    assert (res[1] == output).all()
    del KernelExecutionUnitTest_Object