import mindpype as mp
import numpy as np
import pytest

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
    
    def TestNonTensorInputError(self):
        input = mp.Scalar.create_from_value(self.__session, "test")
        outTensor = mp.Tensor.create(self.__session, (1,1,1))
        tensor_test_node = mp.kernels.ReducedSumKernel.add_to_graph(self.__graph,input,outTensor)
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        
        return outTensor.data
    
    def TestInvalidOutputType(self, raw_data):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        output = mp.Array.create(self.__session, 3, inTensor)
        tensor_test_node = mp.kernels.ReducedSumKernel.add_to_graph(self.__graph,  inTensor, output)
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        
        return output
        
    def TestNonNumericScalarOutputError(self, raw_data):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        output = mp.Scalar.create(self.__session, str)
        tensor_test_node = mp.kernels.ReducedSumKernel.add_to_graph(self.__graph,  inTensor, output)
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        
        return output
    
    def TestInvalidOutputShapeError(self, raw_data):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, raw_data.shape*100)
        tensor_test_node = mp.kernels.ReducedSumKernel.add_to_graph(self.__graph,  inTensor, outTensor)
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        
        return outTensor.data
    
    def TestMultidimensionalScalarOutputError(self, raw_data):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        output = mp.Scalar.create(self.__session, int)
        tensor_test_node = mp.kernels.ReducedSumKernel.add_to_graph(self.__graph,  inTensor, output, keep_dims=True)
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        
        return output
    
        
def test_execute():
    np.random.seed(44)
    raw_data = np.random.randn(2,2,2)
    axis = None
    keep_dims = True
    KernelExecutionUnitTest_Object = ReducedSumKernelUnitTest()
    res = KernelExecutionUnitTest_Object.TestReducedSumKernelExecution(raw_data, axis, keep_dims)
    expected_output = np.sum(raw_data, axis = None)
    assert (res == expected_output).all()
    
    with pytest.raises(TypeError) as e_info:
        res= KernelExecutionUnitTest_Object.TestNonTensorInputError()
    del KernelExecutionUnitTest_Object
    
    KernelExecutionUnitTest_Object = ReducedSumKernelUnitTest()
    with pytest.raises(TypeError) as e_info:
        res= KernelExecutionUnitTest_Object.TestInvalidOutputType(raw_data)
    del KernelExecutionUnitTest_Object
    
    KernelExecutionUnitTest_Object = ReducedSumKernelUnitTest()
    with pytest.raises(TypeError) as e_info:
        res= KernelExecutionUnitTest_Object.TestNonNumericScalarOutputError(raw_data)
    del KernelExecutionUnitTest_Object
    
    KernelExecutionUnitTest_Object = ReducedSumKernelUnitTest()
    with pytest.raises(ValueError) as e_info:
        res= KernelExecutionUnitTest_Object.TestInvalidOutputShapeError(raw_data)
    del KernelExecutionUnitTest_Object
    
    KernelExecutionUnitTest_Object = ReducedSumKernelUnitTest()
    with pytest.raises(ValueError) as e_info:
        res= KernelExecutionUnitTest_Object.TestMultidimensionalScalarOutputError(raw_data)
    del KernelExecutionUnitTest_Object