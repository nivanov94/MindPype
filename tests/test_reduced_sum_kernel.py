import mindpype as mp
import numpy as np

class ReducedSumKernelUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestReducedSumKernelExecution(self, raw_data, ax, keep_dims):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        # compute outTensor shape
        if ax != None:
            axis = (ax,)
        else:
            axis = ()
        out_shape = None
        if keep_dims:
            # all reduced dimensions will be '1'
            out_shape = tuple([1 if i in axis else inTensor.shape[i]
                                          for i in range(len(inTensor.shape))])
        elif axis == ():
            out_shape = (1,)
        else:
            out_shape = tuple([inTensor.shape[i] for i in range(len(inTensor.shape))
                                                   if i not in axis])
            
        outTensor = mp.Tensor.create(self.__session, out_shape)
        tensor_test_node = mp.kernels.ReducedSumKernel.add_to_graph(self.__graph,inTensor,outTensor,axis=ax,keep_dims=keep_dims)

        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        
        return outTensor.data

def test_execute():
    np.random.seed(44)
    raw_data = np.random.randn(2,2,2)
    axis = None
    keep_dims = False
    KernelExecutionUnitTest_Object = ReducedSumKernelUnitTest()
    res = KernelExecutionUnitTest_Object.TestReducedSumKernelExecution(raw_data, axis, keep_dims)
    expected_output = np.sum(raw_data, axis = None)
    assert (res == expected_output).all()
    del KernelExecutionUnitTest_Object
    
