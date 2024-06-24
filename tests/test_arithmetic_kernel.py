import mindpype as mp
import sys, os
import numpy as np

class ArithmeticKernelUnitTest:

    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestTensorAbsoluteKernelExecution(self, raw_data):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, (2,2,2))
        tensor_test_node = mp.kernels.arithmetic.AbsoluteKernel.add_to_graph(self.__graph,inTensor,outTensor)
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        return outTensor.data

    def TestScalarAbsoluteKernelExecution(self, raw_data):
        inScalar = mp.Scalar.create_from_value(self.__session, raw_data)
        outScalar = mp.Scalar.create(self.__session, int)
        scalar_test_node = mp.kernels.arithmetic.AbsoluteKernel.add_to_graph(self.__graph,inScalar,outScalar)
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        return outScalar.data

    def TestAdditionKernelExecution(self, raw_data1, raw_data2):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data1)
        inTensor2 = mp.Tensor.create_from_data(self.__session, raw_data2)
        outTensor = mp.Tensor.create(self.__session, raw_data1.shape)
        tensor_node = mp.kernels.arithmetic.AdditionKernel.add_to_graph(self.__graph, inTensor, inTensor2, outTensor)
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        return outTensor.data

    def TestSubtractionKernelExecution(self, raw_data1, raw_data2):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data1)
        inTensor2 = mp.Tensor.create_from_data(self.__session, raw_data2)
        outTensor = mp.Tensor.create(self.__session, raw_data1.shape)
        tensor_node = mp.kernels.arithmetic.SubtractionKernel.add_to_graph(self.__graph, inTensor, inTensor2, outTensor)
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        return outTensor.data

    def TestMultiplicationKernelExecution(self, raw_data1, raw_data2):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data1)
        inTensor2 = mp.Tensor.create_from_data(self.__session, raw_data2)
        outTensor = mp.Tensor.create(self.__session, raw_data1.shape)
        tensor_node = mp.kernels.arithmetic.MultiplicationKernel.add_to_graph(self.__graph, inTensor, inTensor2, outTensor)
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        return outTensor.data

    def TestDivisionKernelExecution(self, raw_data1, raw_data2):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data1)
        inTensor2 = mp.Tensor.create_from_data(self.__session, raw_data2)
        outTensor = mp.Tensor.create(self.__session, raw_data1.shape)
        tensor_node = mp.kernels.arithmetic.DivisionKernel.add_to_graph(self.__graph, inTensor, inTensor2, outTensor)
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        return outTensor.data

    def TestLogKernelExecution(self, raw_data):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, raw_data.shape)
        tensor_node = mp.kernels.arithmetic.LogKernel.add_to_graph(self.__graph, inTensor, outTensor)
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        return outTensor.data

def test_execute():
    np.random.seed(7)
    raw_data1 = np.random.randint(-10,10, size=(2,2,2))
    raw_data2 = raw_data1
    raw_scalar_data = -10
    
    KernelExecutionUnitTest_Object = ArithmeticKernelUnitTest()
    res = KernelExecutionUnitTest_Object.TestTensorAbsoluteKernelExecution(raw_data1)
    assert res.all() == np.absolute(raw_data1).all()
    del KernelExecutionUnitTest_Object

    KernelExecutionUnitTest_Object = ArithmeticKernelUnitTest()
    res = KernelExecutionUnitTest_Object.TestScalarAbsoluteKernelExecution(raw_scalar_data)
    assert res == np.absolute(raw_scalar_data)
    del KernelExecutionUnitTest_Object

    KernelExecutionUnitTest_Object = ArithmeticKernelUnitTest()
    res = KernelExecutionUnitTest_Object.TestAdditionKernelExecution(raw_data1, raw_data2)
    assert res.all() == (raw_data1 + raw_data2).all()
    del KernelExecutionUnitTest_Object

    KernelExecutionUnitTest_Object = ArithmeticKernelUnitTest()
    res = KernelExecutionUnitTest_Object.TestSubtractionKernelExecution(raw_data1, raw_data2)
    assert res.all() == (raw_data1 - raw_data2).all()
    del KernelExecutionUnitTest_Object

    KernelExecutionUnitTest_Object = ArithmeticKernelUnitTest()
    res = KernelExecutionUnitTest_Object.TestMultiplicationKernelExecution(raw_data1, raw_data2)
    assert res.all() == (raw_data1 * raw_data2).all()
    del KernelExecutionUnitTest_Object

    KernelExecutionUnitTest_Object = ArithmeticKernelUnitTest()
    res = KernelExecutionUnitTest_Object.TestDivisionKernelExecution(raw_data1, raw_data2)
    assert res.all() == (raw_data1 / raw_data2).all()
    del KernelExecutionUnitTest_Object

    KernelExecutionUnitTest_Object = ArithmeticKernelUnitTest()
    res = KernelExecutionUnitTest_Object.TestLogKernelExecution(raw_data1)
    assert res.all() == np.log(raw_data1).all()
    del KernelExecutionUnitTest_Object


