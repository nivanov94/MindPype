import mindpype as mp
import numpy as np

class ReducedSumKernelUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestReducedSumKernelExecution(self, raw_data, ax, keep_dims):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        
        # compute outTensor shape
        input_sz = inTensor.shape
        if ax is not None:
            reduced_axes = [a==ax for a in range(len(input_sz))]
        else:
            reduced_axes = [True] * len(input_sz)

        output_sz = []
        for i in range(len(input_sz)):
            if reduced_axes[i] and keep_dims:
                output_sz.append(1)
            elif not reduced_axes[i]:
                output_sz.append(input_sz[i])

            
        outTensor = mp.Tensor.create(self.__session, output_sz)
        tensor_test_node = mp.kernels.ReducedSumKernel.add_to_graph(self.__graph,inTensor,outTensor,axis=ax,keep_dims=keep_dims)

        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        
        return outTensor.data

def test_execute():
    np.random.seed(44)
    raw_data = np.random.randn(2,2,2)
    axis = None
    keep_dims = True
    KernelExecutionUnitTest_Object = ReducedSumKernelUnitTest()
    res = KernelExecutionUnitTest_Object.TestReducedSumKernelExecution(raw_data, axis, keep_dims)
    expected_output = np.sum(raw_data, axis = None)
    assert (res == expected_output).all()
    del KernelExecutionUnitTest_Object
    