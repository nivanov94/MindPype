import mindpype as mp
import numpy as np

class PadKernelUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestPadKernelExecution(self, raw_data, output_sz):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, output_sz)
        
        virtual_tensors = [
            mp.Tensor.create_virtual(self.__session),
            mp.Tensor.create_virtual(self.__session),
            mp.Tensor.create_virtual(self.__session),
        ]
        
        node1 = mp.kernels.PadKernel.add_to_graph(self.__graph,inTensor,virtual_tensors[0], pad_width=1, mode = 'constant', constant_values = 0)
        node2 = mp.kernels.PadKernel.add_to_graph(self.__graph,virtual_tensors[0],virtual_tensors[1], pad_width=1, mode = 'linear_ramp', end_values=(5, -4))
        node3 = mp.kernels.PadKernel.add_to_graph(self.__graph,virtual_tensors[1],virtual_tensors[2], pad_width=1, mode = 'reflect')
        node4 = mp.kernels.PadKernel.add_to_graph(self.__graph,virtual_tensors[2],outTensor, pad_width=1, mode = 'wrap')

        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()

        return outTensor.data

def test_execute():
    np.random.seed(44)
    raw_data = np.ones(1)
    expected_output = np.pad(raw_data, pad_width=1, mode="constant", constant_values=0)
    expected_output = np.pad(expected_output, pad_width=1, mode="linear_ramp", end_values=(5, -4))
    expected_output = np.pad(expected_output, pad_width=1, mode="reflect")
    expected_output = np.pad(expected_output, pad_width=1, mode="wrap")
    
    output_sz = expected_output.shape
    KernelExecutionUnitTest_Object = PadKernelUnitTest()
    res = KernelExecutionUnitTest_Object.TestPadKernelExecution(raw_data, output_sz)
    assert (res == expected_output).all()
    del KernelExecutionUnitTest_Object
    