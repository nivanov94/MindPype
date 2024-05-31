import mindpype as mp
import sys, os
import numpy as np

class LogicalKernelCreationUnitTest:

    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestNotKernelCreation(self):
        inTensor = mp.Tensor.create(self.__session, (1,1))
        outTensor = mp.Tensor.create(self.__session, (1,1))
        node = mp.kernels.logical.NotKernel.add_to_graph(self.__graph,inTensor,outTensor)
        return node.mp_type

    def TestAndKernelCreation(self):
        inTensor = mp.Tensor.create(self.__session, (1,1))
        inTensor2 = mp.Tensor.create(self.__session, (1,1))
        outTensor = mp.Tensor.create(self.__session, (1,1))
        node = mp.kernels.logical.AndKernel.add_to_graph(self.__graph,inTensor, inTensor2, outTensor)
        return node.mp_type

    def TestOrKernelCreation(self):
        inTensor = mp.Tensor.create(self.__session, (1,1))
        inTensor2 = mp.Tensor.create(self.__session, (1,1))
        outTensor = mp.Tensor.create(self.__session, (1,1))
        node = mp.kernels.logical.OrKernel.add_to_graph(self.__graph,inTensor, inTensor2, outTensor)
        return node.mp_type

    def TestXorCreation(self):
        inTensor = mp.Tensor.create(self.__session, (1,1))
        inTensor2 = mp.Tensor.create(self.__session, (1,1))
        outTensor = mp.Tensor.create(self.__session, (1,1))
        node = mp.kernels.logical.XorKernel.add_to_graph(self.__graph,inTensor,inTensor2,outTensor)
        return node.mp_type

    def TestGreaterKernelCreation(self):
        inTensor = mp.Tensor.create(self.__session, (1,1))
        inTensor2 = mp.Tensor.create(self.__session, (1,1))
        outTensor = mp.Tensor.create(self.__session, (1,1))
        node = mp.kernels.logical.GreaterKernel.add_to_graph(self.__graph,inTensor,inTensor2,outTensor)
        return node.mp_type
    
    def TestLessKernelCreation(self):
        inTensor = mp.Tensor.create(self.__session, (1,1))
        inTensor2 = mp.Tensor.create(self.__session, (1,1))
        outTensor = mp.Tensor.create(self.__session, (1,1))
        node = mp.kernels.logical.LessKernel.add_to_graph(self.__graph,inTensor,inTensor2,outTensor)
        return node.mp_type

    def TestEqualKernelCreation(self):
        inTensor = mp.Tensor.create(self.__session, (1,1))
        inTensor2 = mp.Tensor.create(self.__session, (1,1))
        outTensor = mp.Tensor.create(self.__session, (1,1))
        node = mp.kernels.logical.EqualKernel.add_to_graph(self.__graph,inTensor,inTensor2,outTensor)
        return node.mp_type


class LogicalKernelExecutionUnitTest:

    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestNotKernelExecution(self):
        np.random.seed(44)
        raw_data = np.random.randint(-10,10, size=(2,2,2))
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, (2,2,2))
        tensor_test_node = mp.kernels.logical.NotKernel.add_to_graph(self.__graph,inTensor,outTensor)

        sys.stdout = open(os.devnull, 'w')
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        sys.stdout = sys.__stdout__

        return (raw_data, outTensor.data)

    def TestAndKernelExecution(self):
        np.random.seed(7)
        raw_data = np.random.randint(-10,10, size=(2,2,2))
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        inTensor2 = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, raw_data.shape)
        tensor_node = mp.kernels.logical.AndKernel.add_to_graph(self.__graph, inTensor, inTensor2, outTensor)
        sys.stdout = open(os.devnull, 'w')
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        sys.stdout = sys.__stdout__
        return (raw_data, outTensor.data)
    
    def TestOrKernelExecution(self):
        np.random.seed(7)
        raw_data = np.random.randint(-10,10, size=(2,2,2))
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        inTensor2 = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, raw_data.shape)
        tensor_node = mp.kernels.logical.OrKernel.add_to_graph(self.__graph, inTensor, inTensor2, outTensor)
        sys.stdout = open(os.devnull, 'w')
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        sys.stdout = sys.__stdout__
        return (raw_data, outTensor.data)

    def TestXorKernelExecution(self):
        np.random.seed(7)
        raw_data = np.random.randint(-10,10, size=(2,2,2))
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        inTensor2 = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, raw_data.shape)
        tensor_node = mp.kernels.logical.XorKernel.add_to_graph(self.__graph, inTensor, inTensor2, outTensor)
        sys.stdout = open(os.devnull, 'w')
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        sys.stdout = sys.__stdout__
        return (raw_data, outTensor.data)

    def TestGreaterKernelExecution(self):
        np.random.seed(7)
        raw_data_1 = np.random.randint(0, 10, size=(2,2,2))
        raw_data_2 = np.random.randint(-10, 0, size=(2,2,2))
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data_1)
        inTensor2 = mp.Tensor.create_from_data(self.__session, raw_data_2)
        outTensor = mp.Tensor.create(self.__session, raw_data_1.shape)
        tensor_node = mp.kernels.logical.GreaterKernel.add_to_graph(self.__graph, inTensor, inTensor2, outTensor)
        sys.stdout = open(os.devnull, 'w')
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        sys.stdout = sys.__stdout__
        return (raw_data_1, raw_data_2, outTensor.data)

    def TestLessKernelExecution(self):
        np.random.seed(7)
        raw_data_1 = np.random.randint(-10, 0, size=(2,2,2))
        raw_data_2 = np.random.randint(0, 10, size=(2,2,2))
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data_1)
        inTensor2 = mp.Tensor.create_from_data(self.__session, raw_data_2)
        outTensor = mp.Tensor.create(self.__session, raw_data_1.shape)
        tensor_node = mp.kernels.logical.LessKernel.add_to_graph(self.__graph, inTensor, inTensor2, outTensor)
        sys.stdout = open(os.devnull, 'w')
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        sys.stdout = sys.__stdout__
        return (raw_data_1, raw_data_2, outTensor.data)

    def TestEqualKernelExecution(self):
        np.random.seed(7)
        raw_data = np.random.randint(-10,10, size=(2,2,2))
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        inTensor2 = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, raw_data.shape)
        tensor_node = mp.kernels.logical.EqualKernel.add_to_graph(self.__graph, inTensor, inTensor2, outTensor)
        sys.stdout = open(os.devnull, 'w')
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        sys.stdout = sys.__stdout__
        return (raw_data, outTensor.data)


def test_create():
    KernelUnitTest_Object = LogicalKernelCreationUnitTest()
    assert KernelUnitTest_Object.TestNotKernelCreation == mp.MPEnums.NODE
    del KernelUnitTest_Object

    KernelUnitTest_Object = LogicalKernelCreationUnitTest()
    assert KernelUnitTest_Object.TestAndKernelCreation == mp.MPEnums.NODE
    del KernelUnitTest_Object

    KernelUnitTest_Object = LogicalKernelCreationUnitTest()
    assert KernelUnitTest_Object.TestOrKernelCreation == mp.MPEnums.NODE
    del KernelUnitTest_Object

    KernelUnitTest_Object = LogicalKernelCreationUnitTest()
    assert KernelUnitTest_Object.TestXorCreation == mp.MPEnums.NODE
    del KernelUnitTest_Object

    KernelUnitTest_Object = LogicalKernelCreationUnitTest()
    assert KernelUnitTest_Object.TestGreaterKernelCreation == mp.MPEnums.NODE
    del KernelUnitTest_Object

    KernelUnitTest_Object = LogicalKernelCreationUnitTest()
    assert KernelUnitTest_Object.TestLessKernelCreation == mp.MPEnums.NODE
    del KernelUnitTest_Object

    KernelUnitTest_Object = LogicalKernelCreationUnitTest()
    assert KernelUnitTest_Object.TestEqualKernelCreation == mp.MPEnums.NODE
    del KernelUnitTest_Object

def test_execute():
    KernelExecutionUnitTest_Object = LogicalKernelExecutionUnitTest()
    res = KernelExecutionUnitTest_Object.TestNotKernelExecution()
    assert res[1].all() == np.logical_not(res[0].all())
    del KernelExecutionUnitTest_Object

    KernelExecutionUnitTest_Object = LogicalKernelExecutionUnitTest()
    res = KernelExecutionUnitTest_Object.TestAndKernelExecution()
    assert res[1].all() == np.logical_and(res[0], res[0])
    del KernelExecutionUnitTest_Object

    KernelExecutionUnitTest_Object = LogicalKernelExecutionUnitTest()
    res = KernelExecutionUnitTest_Object.TestOrKernelExecution()
    assert res[1].all() == np.logical_or(res[0], res[0])
    del KernelExecutionUnitTest_Object

    KernelExecutionUnitTest_Object = LogicalKernelExecutionUnitTest()
    res = KernelExecutionUnitTest_Object.TestXorKernelExecution()
    assert res[1].all() == np.logical_xor(res[0], res[0])
    del KernelExecutionUnitTest_Object

    KernelExecutionUnitTest_Object = LogicalKernelExecutionUnitTest()
    res = KernelExecutionUnitTest_Object.TestGreaterKernelExecution()
    assert res[2].all() == (res[0] > res[1])
    del KernelExecutionUnitTest_Object

    KernelExecutionUnitTest_Object = LogicalKernelExecutionUnitTest()
    res = KernelExecutionUnitTest_Object.TestLessKernelExecution()
    assert res[2].all() == (res[0] < res[1])
    del KernelExecutionUnitTest_Object

    KernelExecutionUnitTest_Object = LogicalKernelExecutionUnitTest()
    res = KernelExecutionUnitTest_Object.TestEqualKernelExecution()
    assert res[1].all() == res[0].all()
    del KernelExecutionUnitTest_Object


