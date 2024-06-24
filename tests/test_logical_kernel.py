import mindpype as mp
import numpy as np

class LogicalKernelUnitTest:

    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestNotKernelExecution(self, raw_data):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, raw_data.shape)
        tensor_test_node = mp.kernels.logical.NotKernel.add_to_graph(self.__graph,inTensor,outTensor)
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        return (raw_data, outTensor.data)

    def TestAndKernelExecution(self, raw_data):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        inTensor2 = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, raw_data.shape)
        tensor_node = mp.kernels.logical.AndKernel.add_to_graph(self.__graph, inTensor, inTensor2, outTensor)
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        return (raw_data, outTensor.data)
    
    def TestOrKernelExecution(self, raw_data):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        inTensor2 = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, raw_data.shape)
        tensor_node = mp.kernels.logical.OrKernel.add_to_graph(self.__graph, inTensor, inTensor2, outTensor)
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        return (raw_data, outTensor.data)

    def TestXorKernelExecution(self, raw_data):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        inTensor2 = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, raw_data.shape)
        tensor_node = mp.kernels.logical.XorKernel.add_to_graph(self.__graph, inTensor, inTensor2, outTensor)
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        return (raw_data, outTensor.data)

    def TestGreaterKernelExecution(self, raw_data_1, raw_data_2):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data_1)
        inTensor2 = mp.Tensor.create_from_data(self.__session, raw_data_2)
        outTensor = mp.Tensor.create(self.__session, raw_data_1.shape)
        tensor_node = mp.kernels.logical.GreaterKernel.add_to_graph(self.__graph, inTensor, inTensor2, outTensor)
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        return (raw_data_1, raw_data_2, outTensor.data)

    def TestLessKernelExecution(self, raw_data_1, raw_data_2):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data_1)
        inTensor2 = mp.Tensor.create_from_data(self.__session, raw_data_2)
        outTensor = mp.Tensor.create(self.__session, raw_data_1.shape)
        tensor_node = mp.kernels.logical.LessKernel.add_to_graph(self.__graph, inTensor, inTensor2, outTensor)
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        return (raw_data_1, raw_data_2, outTensor.data)

    def TestEqualKernelExecution(self, raw_data):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        inTensor2 = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, raw_data.shape)
        tensor_node = mp.kernels.logical.EqualKernel.add_to_graph(self.__graph, inTensor, inTensor2, outTensor)
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        return (inTensor.data, inTensor2.data, outTensor.data)

def test_execute():
    np.random.seed(7)
    raw_data = np.random.randn(2,2,2)
    
    # KernelExecutionUnitTest_Object = LogicalKernelUnitTest()
    # res = KernelExecutionUnitTest_Object.TestNotKernelExecution(raw_data)
    # assert (res[1] == np.logical_not(res[0])).all()
    # del KernelExecutionUnitTest_Object

    KernelExecutionUnitTest_Object = LogicalKernelUnitTest()
    res = KernelExecutionUnitTest_Object.TestAndKernelExecution(raw_data)
    assert (res[1] == np.logical_and(res[0], res[0])).all()
    del KernelExecutionUnitTest_Object

    KernelExecutionUnitTest_Object = LogicalKernelUnitTest()
    res = KernelExecutionUnitTest_Object.TestOrKernelExecution(raw_data)
    assert (res[1] == np.logical_or(res[0], res[0])).all()
    del KernelExecutionUnitTest_Object

    KernelExecutionUnitTest_Object = LogicalKernelUnitTest()
    res = KernelExecutionUnitTest_Object.TestXorKernelExecution(raw_data)
    assert (res[1] == np.logical_xor(res[0], res[0])).all()
    del KernelExecutionUnitTest_Object

    KernelExecutionUnitTest_Object = LogicalKernelUnitTest()
    np.random.seed(7)
    raw_data_1 = np.random.randn(2,2,2)
    raw_data_2 = np.random.randn(2,2,2)
    res = KernelExecutionUnitTest_Object.TestGreaterKernelExecution(raw_data_1, raw_data_2)
    assert (res[2] == (res[0] > res[1])).all()
    del KernelExecutionUnitTest_Object

    KernelExecutionUnitTest_Object = LogicalKernelUnitTest()
    np.random.seed(7)
    raw_data_1 = np.random.randn(2,2,2)
    raw_data_2 = np.random.randn(2,2,2)
    res = KernelExecutionUnitTest_Object.TestLessKernelExecution(raw_data_1, raw_data_2)
    assert (res[2] == (res[0] < res[1])).all()
    del KernelExecutionUnitTest_Object

    KernelExecutionUnitTest_Object = LogicalKernelUnitTest()
    res = KernelExecutionUnitTest_Object.TestEqualKernelExecution(raw_data)
    assert ((res[0] == res[1]) == res[2]).all()
    del KernelExecutionUnitTest_Object


