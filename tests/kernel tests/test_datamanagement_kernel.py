import mindpype as mp
import numpy as np
import pytest

class ConcatenationKernelUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestConcatenationKernelExecution(self, raw_data, ax):
        inTensor1 = mp.Tensor.create_from_data(self.__session, raw_data)
        inTensor2 = mp.Tensor.create_from_data(self.__session, raw_data)
        # compute outTensor size
        if len(inTensor1.shape) == len(inTensor2.shape):
            noncat_sz_A = [d for i,d in enumerate(inTensor1.shape) if i!=ax]
            noncat_sz_B = [d for i,d in enumerate(inTensor2.shape) if i!=ax]
            output_sz = noncat_sz_A[:]
            output_sz.insert(ax,inTensor1.shape[ax]+inTensor2.shape[ax])
        elif len(inTensor1.shape) == len(inTensor2.shape)+1:
            # appending B to A
            noncat_sz_A = [d for i,d in enumerate(inTensor1.shape) if i!=ax]
            noncat_sz_B = inTensor2.shape
            output_sz = noncat_sz_A[:]
            output_sz.insert(ax,inTensor1.shape[ax]+1)
        elif len(inTensor1.shape) == len(inTensor2.shape)-1:
            noncat_sz_B = [d for i,d in enumerate(inTensor2.shape) if i!=ax]
            noncat_sz_A = inTensor1.shape
            output_sz = noncat_sz_B[:]
            output_sz.insert(ax,inTensor2.shape[ax]+1)
        else:
            output_sz = []
            
        outTensor = mp.Tensor.create(self.__session, tuple(output_sz))
        tensor_test_node = mp.kernels.ConcatenationKernel.add_to_graph(self.__graph,inTensor1,inTensor2,outTensor,axis=ax)
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        return outTensor.data

    def TestWrongOutputSizeError(self, raw_data):
        inTensor1 = mp.Tensor.create_from_data(self.__session, raw_data)
        inTensor2 = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, (1,2,3))
        tensor_test_node = mp.kernels.ConcatenationKernel.add_to_graph(self.__graph,inTensor1,inTensor2,outTensor,axis=1)
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        
    def TestConcatNonTensorError(self):
        input1 = mp.Scalar.create_from_value(self.__session, 2)
        input2 = mp.Scalar.create_from_value(self.__session, 2)
        output = mp.Scalar.create(self.__session, int)
        tensor_test_node = mp.kernels.ConcatenationKernel.add_to_graph(self.__graph,input1,input2,output,axis=1)
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
    
    def TestUnequalDimensionsError(self):
        inTensor1 = mp.Tensor.create_from_data(self.__session, np.zeros((2,3)))
        inTensor2 = mp.Tensor.create_from_data(self.__session, np.zeros((4,5)))
        outTensor = mp.Tensor.create(self.__session, (2,2))
        tensor_test_node = mp.kernels.ConcatenationKernel.add_to_graph(self.__graph,inTensor1,inTensor2,outTensor,axis=1)
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
    
class EnqueueKernelUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestEnqueueKernelExecution(self, raw_data):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        template = mp.Tensor.create(self.__session, inTensor.shape)
        queue = mp.CircleBuffer.create(self.__session, 3, template)
        tensor_test_node = mp.kernels.EnqueueKernel.add_to_graph(self.__graph,inTensor,queue)
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        return queue
    
    def TestInvalidCapacity(self, raw_data):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        template = mp.Tensor.create(self.__session, inTensor.shape)
        queue = mp.CircleBuffer.create(self.__session, 0, template)
        tensor_test_node = mp.kernels.EnqueueKernel.add_to_graph(self.__graph,inTensor,queue)
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        return queue
    
    def TestInvalidOutputType(self, raw_data):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        queue = mp.Tensor.create(self.__session, inTensor.shape)
        tensor_test_node = mp.kernels.EnqueueKernel.add_to_graph(self.__graph,inTensor,queue)
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        return queue.data
class ExtractKernelUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestExtractKernelExecution(self, raw_data):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        indices = [0, slice(None)] 
            
        # calculate outTensor shape
        output_sz = []
        ix_grid = []
        for axis in range(len(indices)):
            axis_indices = indices[axis]
            if isinstance(axis_indices, int):
                axis_indices = (axis_indices,)
            elif isinstance(axis_indices, slice):
                axis_indices = range(*axis_indices.indices(inTensor.shape[axis]))
            for index in axis_indices:
                if index < -inTensor.shape[axis] or index >= inTensor.shape[axis]:
                    raise ValueError("ExtractKernel extraction index in dimension {} exceeds the input Tensor's shape".format(axis))
            ix_grid.append(axis_indices)
            # if len(indices[axis]) > 1:
            output_sz.append(len(axis_indices)) 
                
        outTensor = mp.Tensor.create(self.__session, tuple(output_sz))
        tensor_test_node = mp.kernels.ExtractKernel.add_to_graph(self.__graph,inTensor,indices,outTensor,reduce_dims=False)
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        return outTensor.data
    
class StackKernelUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)
        
    def TestStackKernelExecution(self, raw_data1, raw_data2, ax):
        tensor1 = mp.Tensor.create_from_data(self.__session, raw_data1)
        tensor2 = mp.Tensor.create_from_data(self.__session, raw_data2)
        inArray = mp.Array.create(self.__session, 2, tensor1)
        inArray.set_element(0,tensor1)
        inArray.set_element(1,tensor2)
        # compute outTensor shape
        output_shape = (tensor1.shape[:ax] + (inArray.capacity,)
                         + tensor1.shape[ax:])
        outTensor = mp.Tensor.create(self.__session, output_shape)
        tensor_test_node = mp.kernels.StackKernel.add_to_graph(self.__graph,inArray,outTensor)
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        return outTensor.data
    
class TensorStackKernelUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestTensorStackKernelExecution(self, raw_data1, raw_data2, ax):
        inTensor1 = mp.Tensor.create_from_data(self.__session, raw_data1)
        inTensor2 = mp.Tensor.create_from_data(self.__session, raw_data2)
        # compute outTensor shape
        output_shape = inTensor1.shape[:ax] + (2,) + inTensor1.shape[ax:]
        outTensor = mp.Tensor.create(self.__session, output_shape)
        tensor_test_node = mp.kernels.TensorStackKernel.add_to_graph(self.__graph,inTensor1,inTensor2, outTensor,axis=ax)
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        return outTensor.data
    
class ReshapeKernelUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestReshapeKernelExecution(self, raw_data, new_shape):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, new_shape)
        tensor_test_node = mp.kernels.ReshapeKernel.add_to_graph(self.__graph,inTensor, outTensor, shape=(2,8))
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        return outTensor.data

def test_execute():
    np.random.seed(44)
    raw_data = np.random.randn(2,2,2)
    raw_data2 = np.random.randn(4,4)
    KernelExecutionUnitTest_Object = ConcatenationKernelUnitTest()
    axis = 0
    res = KernelExecutionUnitTest_Object.TestConcatenationKernelExecution(raw_data, axis)
    output = np.concatenate((raw_data, raw_data), axis = axis)
    assert (res == output).all()
    
    with pytest.raises(TypeError) as e_info:
        res = KernelExecutionUnitTest_Object.TestConcatNonTensorError(raw_data)
        
    with pytest.raises(ValueError) as e_info:
        res = KernelExecutionUnitTest_Object.TestWrongOutputSizeError(raw_data)
        
    with pytest.raises(ValueError) as e_info:
        res = KernelExecutionUnitTest_Object.TestUnequalDimensionsError()
    del KernelExecutionUnitTest_Object

    KernelExecutionUnitTest_Object = EnqueueKernelUnitTest()
    res = KernelExecutionUnitTest_Object.TestEnqueueKernelExecution(raw_data)
    assert np.all(res.peek().data == raw_data)
    with pytest.raises(TypeError):
        res = KernelExecutionUnitTest_Object.TestInvalidOutputType(raw_data)
    del KernelExecutionUnitTest_Object
    KernelExecutionUnitTest_Object = EnqueueKernelUnitTest()
    with pytest.raises(ValueError) as e_info:
        res = KernelExecutionUnitTest_Object.TestInvalidCapacity(raw_data)
    del KernelExecutionUnitTest_Object
    
    KernelExecutionUnitTest_Object = ExtractKernelUnitTest()
    res = KernelExecutionUnitTest_Object.TestExtractKernelExecution(raw_data2)
    extracted_data = raw_data2[0,:]
    assert np.all(res == extracted_data)
    del KernelExecutionUnitTest_Object
    
    KernelExecutionUnitTest_Object = StackKernelUnitTest()
    res = KernelExecutionUnitTest_Object.TestStackKernelExecution(raw_data2, raw_data2, axis)
    expected_output = np.stack([raw_data2, raw_data2], axis = axis)
    assert (res == expected_output).all()
    del KernelExecutionUnitTest_Object
    
    KernelExecutionUnitTest_Object = TensorStackKernelUnitTest()
    res = KernelExecutionUnitTest_Object.TestTensorStackKernelExecution(raw_data2, raw_data2, axis)
    expected_output = np.stack([raw_data2, raw_data2], axis = 0)
    assert (res == expected_output).all()
    del KernelExecutionUnitTest_Object
    
    KernelExecutionUnitTest_Object = ReshapeKernelUnitTest()
    new_shape = (2,8)
    res = KernelExecutionUnitTest_Object.TestReshapeKernelExecution(raw_data2, new_shape)
    expected_output = np.reshape(raw_data2, newshape=new_shape)
    assert (res == expected_output).all()
    del KernelExecutionUnitTest_Object
test_execute()