
import mindpype as mp
import sys, os
import numpy as np

class EpochKernelCreationUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestEpochKernelCreation(self):
        inTensor = mp.Tensor.create(self.__session, (2,2,2))
        outTensor = mp.Tensor.create(self.__session,(2,2,1,2))
        node = mp.kernels.EpochKernel.add_to_graph(self.__graph,inTensor,outTensor, epoch_len=2,epoch_stride=1)
        return node.mp_type
    
class EpochKernelExecutionUnitTest:

    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestEpochKernelExecution(self):
        np.random.seed(44)
        raw_data = np.random.randint(-10,10, size=(2,2,2))
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session,(2,2,1,2))
        
        node = mp.kernels.EpochKernel.add_to_graph(self.__graph,inTensor,outTensor,epoch_len=2, epoch_stride=1)

        sys.stdout = open(os.devnull, 'w')
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        sys.stdout = sys.__stdout__

        return (raw_data, outTensor, outTensor.data)

def test_create():
    KernelUnitTest_Object = EpochKernelCreationUnitTest()
    assert KernelUnitTest_Object.TestEpochKernelCreation() == mp.MPEnums.NODE
    del KernelUnitTest_Object


def test_execute():
    KernelExecutionUnitTest_Object = EpochKernelExecutionUnitTest()
    res = KernelExecutionUnitTest_Object.TestEpochKernelExecution()
    
    # manually epoch data
    expected_output = np.zeros((2,2,1,2))
    src_slc = [slice(None)] * len(res[0].shape)
    dst_slc = [slice(None)] * len(res[1].shape)
    Nepochs = int(res[0].shape[2] - 2) // 1 + 1
    for i_e in range(Nepochs):
        src_slc[2] = slice(i_e*1,
                                i_e*1 + 2)
        dst_slc[2] = i_e
        expected_output[tuple(dst_slc)] = res[0][tuple(src_slc)]

    assert (res[2] == expected_output).all()
    del KernelExecutionUnitTest_Object
    
test_execute()