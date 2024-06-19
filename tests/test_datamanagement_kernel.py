import mindpype as mp
import sys, os
import numpy as np

class ConcatenationKernelCreationUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestConcatenationKernelCreation(self):
        inTensor1 = mp.Tensor.create(self.__session, (2,2,2))
        inTensor2 = mp.Tensor.create(self.__session, (2,2,2))
        outTensor = mp.Tensor.create(self.__session, (4,2,2))
        node = mp.kernels.ConcatenationKernel.add_to_graph(self.__graph,inTensor1,inTensor2,outTensor,axis=0)
        return node.mp_type
    
class ConcatenationKernelExecutionUnitTest:

    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestConcatenationKernelExecution(self):
        np.random.seed(44)
        raw_data = np.random.randint(-10,10, size=(2,2,2))
        inTensor1 = mp.Tensor.create_from_data(self.__session, raw_data)
        inTensor2 = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, (4,2,2))
        tensor_test_node = mp.kernels.ConcatenationKernel.add_to_graph(self.__graph,inTensor1,inTensor2,outTensor,axis=0)

        sys.stdout = open(os.devnull, 'w')
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        sys.stdout = sys.__stdout__

        return (raw_data, outTensor.data)
    
class EnqueueKernelCreationUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestEnqueueKernelCreation(self):
        inTensor = mp.Tensor.create(self.__session, (2,2,2))
        template = mp.Tensor.create(self.__session, inTensor.shape)
        queue = mp.CircleBuffer.create(self.__session, 3, template)
        node = mp.kernels.EnqueueKernel.add_to_graph(self.__graph,inTensor,queue)
        return node.mp_type
    
class EnqueueKernelExecutionUnitTest:

    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestEnqueueKernelExecution(self):
        np.random.seed(44)
        raw_data = np.random.randint(-10,10, size=(2,2,2))
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        template = mp.Tensor.create(self.__session, inTensor.shape)
        queue = mp.CircleBuffer.create(self.__session, 3, template)
        tensor_test_node = mp.kernels.EnqueueKernel.add_to_graph(self.__graph,inTensor,queue)

        sys.stdout = open(os.devnull, 'w')
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        sys.stdout = sys.__stdout__
        
        expected_buffer = mp.CircleBuffer.create(self.__session, 3, template)
        expected_buffer.enqueue(inTensor)
        
        return (expected_buffer, queue)
    
class ExtractKernelCreationUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestExtractKernelCreation(self):
        inTensor = mp.Tensor.create(self.__session, (4,4))
        indices = [0]
        outTensor = mp.Tensor.create(self.__session, (1,))
        node = mp.kernels.ExtractKernel.add_to_graph(self.__graph,inTensor,indices,outTensor,reduce_dims=False)
        return node.mp_type
    
class ExtractKernelExecutionUnitTest:

    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestExtractKernelExecution(self):
        np.random.seed(44)
        raw_data = np.random.randint(-10,10, size=(4,4))
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        indices = [0]
        outTensor = mp.Tensor.create(self.__session, (1,))
        tensor_test_node = mp.kernels.ExtractKernel.add_to_graph(self.__graph,inTensor,indices,outTensor,reduce_dims=False)

        sys.stdout = open(os.devnull, 'w')
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        sys.stdout = sys.__stdout__

        return (raw_data, outTensor.data)
    
class TensorStackKernelCreationUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestTensorStackKernelCreation(self):
        inTensor1 = mp.Tensor.create(self.__session, (4,4))
        inTensor2 = mp.Tensor.create(self.__session, (4,4))
        outTensor = mp.Tensor.create(self.__session, (2,4,4))
        node = mp.kernels.TensorStackKernel.add_to_graph(self.__graph,inTensor1,inTensor2,outTensor,axis=0)
        return node.mp_type
    
class TensorStackKernelExecutionUnitTest:

    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestTensorStackKernelExecution(self):
        np.random.seed(44)
        raw_data = np.random.randint(-10,10, size=(4,4))
        inTensor1 = mp.Tensor.create_from_data(self.__session, raw_data)
        inTensor2 = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, (2,4,4))
        tensor_test_node = mp.kernels.TensorStackKernel.add_to_graph(self.__graph,inTensor1,inTensor2, outTensor,axis=0)

        sys.stdout = open(os.devnull, 'w')
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        sys.stdout = sys.__stdout__

        return (inTensor1.data, inTensor2.data, outTensor.data)
    
class ReshapeKernelCreationUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestReshapeKernelCreation(self):
        inTensor = mp.Tensor.create(self.__session, (4,4))
        outTensor = mp.Tensor.create(self.__session, (2,8))
        node = mp.kernels.ReshapeKernel.add_to_graph(self.__graph,inTensor,outTensor, shape=(2,8))
        return node.mp_type
    
class ReshapeKernelExecutionUnitTest:

    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestReshapeKernelExecution(self):
        np.random.seed(44)
        raw_data = np.random.randint(-10,10, size=(4,4))
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, (2,8))
        tensor_test_node = mp.kernels.ReshapeKernel.add_to_graph(self.__graph,inTensor, outTensor, shape=(2,8))

        sys.stdout = open(os.devnull, 'w')
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        sys.stdout = sys.__stdout__

        return (raw_data, outTensor.data)

def test_create():
    KernelUnitTest_Object = ConcatenationKernelCreationUnitTest()
    assert KernelUnitTest_Object.TestConcatenationKernelCreation() == mp.MPEnums.NODE
    del KernelUnitTest_Object

    KernelUnitTest_Object = EnqueueKernelCreationUnitTest()
    assert KernelUnitTest_Object.TestEnqueueKernelCreation() == mp.MPEnums.NODE
    del KernelUnitTest_Object
    
    KernelUnitTest_Object = ExtractKernelCreationUnitTest()
    assert KernelUnitTest_Object.TestExtractKernelCreation() == mp.MPEnums.NODE
    del KernelUnitTest_Object
    
    KernelUnitTest_Object = TensorStackKernelCreationUnitTest()
    assert KernelUnitTest_Object.TestTensorStackKernelCreation() == mp.MPEnums.NODE
    del KernelUnitTest_Object
    
    KernelUnitTest_Object = ReshapeKernelCreationUnitTest()
    assert KernelUnitTest_Object.TestReshapeKernelCreation() == mp.MPEnums.NODE
    del KernelUnitTest_Object



def test_execute():
    KernelExecutionUnitTest_Object = ConcatenationKernelExecutionUnitTest()
    res = KernelExecutionUnitTest_Object.TestConcatenationKernelExecution()
    output = np.concatenate((res[0], res[0]), axis = 0)
    assert (res[1] == output).all()
    del KernelExecutionUnitTest_Object

    # KernelExecutionUnitTest_Object = EnqueueKernelExecutionUnitTest()
    # res = KernelExecutionUnitTest_Object.TestEnqueueKernelExecution()
    # assert res[0].peek() == res[1].peek()
    # del KernelExecutionUnitTest_Object
    
    # KernelExecutionUnitTest_Object = ExtractKernelExecutionUnitTest()
    # res = KernelExecutionUnitTest_Object.TestExtractKernelExecution()
    # npixgrid = np.ix_(res[1])
    # extracted_data = res[0][npixgrid]
    # assert res[2] == extracted_data
    # del KernelExecutionUnitTest_Object
    
    KernelExecutionUnitTest_Object = TensorStackKernelExecutionUnitTest()
    res = KernelExecutionUnitTest_Object.TestTensorStackKernelExecution()
    expected_output = np.stack([res[0], res[1]], axis = 0)
    assert (res[2] == expected_output).all()
    del KernelExecutionUnitTest_Object
    
    KernelExecutionUnitTest_Object = ReshapeKernelExecutionUnitTest()
    res = KernelExecutionUnitTest_Object.TestReshapeKernelExecution()
    expected_output = np.reshape(res[0], newshape=(2,8))
    assert (res[1] == expected_output).all()
    del KernelExecutionUnitTest_Object