import mindpype as mp
import numpy as np

class PadKernelUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestPadKernelExecution(self, raw_data, output_sz):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, output_sz)
        tensor_test_node = mp.kernels.PadKernel.add_to_graph(self.__graph,inTensor,outTensor, pad_width=1, mode = 'constant', constant_values = 0)

        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()

        return outTensor.data

def test_execute():
    np.random.seed(44)
    raw_data = np.ones(1)
    expected_output = np.pad(raw_data, pad_width=1, mode="constant", constant_values=0)
    output_sz = expected_output.shape
    KernelExecutionUnitTest_Object = PadKernelUnitTest()
    res = KernelExecutionUnitTest_Object.TestPadKernelExecution(raw_data, output_sz)
    assert (res == expected_output).all()
    del KernelExecutionUnitTest_Object
    