
import mindpype as mp
import numpy as np

class EpochKernelUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestEpochKernelExecution(self, raw_data):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session,(2,2,1,2))
        node = mp.kernels.EpochKernel.add_to_graph(self.__graph,inTensor,outTensor,epoch_len=2, epoch_stride=1)
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        return outTensor.data

def test_execute():
    np.random.seed(44)
    raw_data = np.random.randn(2,2,2)
    KernelExecutionUnitTest_Object = EpochKernelUnitTest()
    res = KernelExecutionUnitTest_Object.TestEpochKernelExecution(raw_data)
    
    # manually epoch data
    expected_output = np.zeros((2,2,1,2))
    src_slc = [slice(None)] * len(raw_data.shape)
    dst_slc = [slice(None)] * len(expected_output.shape)
    Nepochs = int(res[0].shape[2] - 2) // 1 + 1
    for i_e in range(Nepochs):
        src_slc[2] = slice(i_e*1,
                                i_e*1 + 2)
        dst_slc[2] = i_e
        expected_output[tuple(dst_slc)] = raw_data[tuple(src_slc)]

    assert (res == expected_output).all()
    del KernelExecutionUnitTest_Object
