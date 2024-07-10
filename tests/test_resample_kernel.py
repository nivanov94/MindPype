import mindpype as mp
import numpy as np
from scipy import signal

class ResampleKernelUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestResampleKernelExecution(self, raw_data, factor):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        # compute output shape
        output_shape = (inTensor.shape[0], inTensor.shape[1] * 2, inTensor.shape[2])
        outTensor = mp.Tensor.create(self.__session, output_shape)
        tensor_test_node = mp.kernels.ResampleKernel.add_to_graph(self.__graph,inA=inTensor,factor=factor,outA=outTensor)

        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        
        return outTensor.data

def test_execute():
    np.random.seed(44)
    raw_data = np.random.randn(3,3,3)
    factor = 2

    KernelExecutionUnitTest_Object = ResampleKernelUnitTest()
    res = KernelExecutionUnitTest_Object.TestResampleKernelExecution(raw_data, factor)
    
    expected_output = signal.resample(raw_data, np.ceil(raw_data.shape[1] * 2).astype(int),axis=1)
    assert(res == expected_output).all()

    del KernelExecutionUnitTest_Object
    