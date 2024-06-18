import mindpype as mp
import sys, os
import numpy as np

class PadKernelCreationUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestPadKernelCreation(self):
        raw_data = np.ones(1)
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, (3,))
        node = mp.kernels.PadKernel.add_to_graph(self.__graph,inTensor,outTensor)
        return node.mp_type
    
class PadKernelExecutionUnitTest:

    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestPadKernelExecution(self):
        np.random.seed(44)
        raw_data = np.ones(1)
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, (3,))
        tensor_test_node = mp.kernels.PadKernel.add_to_graph(self.__graph,inTensor,outTensor, pad_width=1, mode = 'constant', constant_values = 0)

        sys.stdout = open(os.devnull, 'w')
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        sys.stdout = sys.__stdout__

        return (raw_data, outTensor.data)

def test_create():
    KernelUnitTest_Object = PadKernelCreationUnitTest()
    assert KernelUnitTest_Object.TestPadKernelCreation() == mp.MPEnums.NODE
    del KernelUnitTest_Object


def test_execute():
    KernelExecutionUnitTest_Object = PadKernelExecutionUnitTest()
    res = KernelExecutionUnitTest_Object.TestPadKernelExecution()
    assert (res[1] == np.pad(res[0], pad_width=1, mode="constant", constant_values=0)).all()
    del KernelExecutionUnitTest_Object