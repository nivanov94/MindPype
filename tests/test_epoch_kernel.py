
import mindpype as mp
import numpy as np

class EpochKernelUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestEpochKernelExecution(self, raw_data, epoch_length, epoch_stride, ax):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        # compute outTensor shape
        output_shape = list(inTensor.shape)
        output_shape[ax] = epoch_length
        output_shape.insert(ax, int(inTensor.shape[ax] - epoch_length) // epoch_stride + 1)
        
        outTensor = mp.Tensor.create(self.__session,tuple(output_shape))
        node = mp.kernels.EpochKernel.add_to_graph(self.__graph,inTensor,outTensor,epoch_len=epoch_length, epoch_stride=epoch_stride, axis=ax)
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        return outTensor.data

def test_execute():
    np.random.seed(44)
    raw_data = np.random.randn(2,2,2)
    epoch_length = 2
    epoch_stride = 1
    axis = -1
    KernelExecutionUnitTest_Object = EpochKernelUnitTest()
    res = KernelExecutionUnitTest_Object.TestEpochKernelExecution(raw_data, epoch_length, epoch_stride, axis)
    
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
