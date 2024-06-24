import mindpype as mp
import numpy as np

class ConcatenationKernelUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestConcatenationKernelExecution(self, raw_data):
        inTensor1 = mp.Tensor.create_from_data(self.__session, raw_data)
        inTensor2 = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, (4,2,2))
        tensor_test_node = mp.kernels.ConcatenationKernel.add_to_graph(self.__graph,inTensor1,inTensor2,outTensor,axis=0)
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        return outTensor.data
    
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
    
class ExtractKernelUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestExtractKernelExecution(self, raw_data):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        indices = [0, slice(None)]
        outTensor = mp.Tensor.create(self.__session, (1,4))
        tensor_test_node = mp.kernels.ExtractKernel.add_to_graph(self.__graph,inTensor,indices,outTensor,reduce_dims=False)
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        return outTensor.data
    
class StackKernelUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)
        
    def TestStackKernelExecution(self, raw_data1, raw_data2):
        tensor1 = mp.Tensor.create_from_data(self.__session, raw_data1)
        tensor2 = mp.Tensor.create_from_data(self.__session, raw_data2)
        inArray = mp.Array.create(self.__session, 2, tensor1)
        inArray.set_element(0,tensor1)
        inArray.set_element(1,tensor2)
        outTensor = mp.Tensor.create(self.__session, (2,4,4))
        tensor_test_node = mp.kernels.StackKernel.add_to_graph(self.__graph,inArray,outTensor)
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        return outTensor.data
    
class TensorStackKernelUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestTensorStackKernelExecution(self, raw_data1, raw_data2):
        inTensor1 = mp.Tensor.create_from_data(self.__session, raw_data1)
        inTensor2 = mp.Tensor.create_from_data(self.__session, raw_data2)
        outTensor = mp.Tensor.create(self.__session, (2,4,4))
        tensor_test_node = mp.kernels.TensorStackKernel.add_to_graph(self.__graph,inTensor1,inTensor2, outTensor,axis=0)
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        return outTensor.data
    
class ReshapeKernelUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestReshapeKernelExecution(self, raw_data):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, (2,8))
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
    res = KernelExecutionUnitTest_Object.TestConcatenationKernelExecution(raw_data)
    output = np.concatenate((raw_data, raw_data), axis = 0)
    assert (res == output).all()
    del KernelExecutionUnitTest_Object

    KernelExecutionUnitTest_Object = EnqueueKernelUnitTest()
    res = KernelExecutionUnitTest_Object.TestEnqueueKernelExecution(raw_data)
    assert np.all(res.peek().data == raw_data)
    del KernelExecutionUnitTest_Object
    
    KernelExecutionUnitTest_Object = ExtractKernelUnitTest()
    res = KernelExecutionUnitTest_Object.TestExtractKernelExecution(raw_data2)
    extracted_data = raw_data2[0,:]
    assert np.all(res == extracted_data)
    del KernelExecutionUnitTest_Object
    
    KernelExecutionUnitTest_Object = StackKernelUnitTest()
    res = KernelExecutionUnitTest_Object.TestStackKernelExecution(raw_data2, raw_data2)
    expected_output = np.stack([raw_data2, raw_data2], axis = 0)
    assert (res == expected_output).all()
    del KernelExecutionUnitTest_Object
    
    KernelExecutionUnitTest_Object = TensorStackKernelUnitTest()
    res = KernelExecutionUnitTest_Object.TestTensorStackKernelExecution(raw_data2, raw_data2)
    expected_output = np.stack([raw_data2, raw_data2], axis = 0)
    assert (res == expected_output).all()
    del KernelExecutionUnitTest_Object
    
    KernelExecutionUnitTest_Object = ReshapeKernelUnitTest()
    res = KernelExecutionUnitTest_Object.TestReshapeKernelExecution(raw_data2)
    expected_output = np.reshape(raw_data2, newshape=(2,8))
    assert (res == expected_output).all()
    del KernelExecutionUnitTest_Object