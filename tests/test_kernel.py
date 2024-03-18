import mindpype as mp
import sys, os
import numpy as np

class KernelCreationUnitTest:

    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestAbsoluteKernelCreation(self):
        inTensor = mp.Tensor.create(self.__session, (1,1))
        outTensor = mp.Tensor.create(self.__session, (1,1))
        node = mp.kernels.arithmetic.AbsoluteKernel.add_to_graph(self.__graph,inTensor,outTensor)
        return node.mp_type

    def TestAdditionKernelCreation(self):
        inTensor = mp.Tensor.create(self.__session, (1,1))
        inTensor2 = mp.Tensor.create(self.__session, (1,1))
        outTensor = mp.Tensor.create(self.__session, (1,1))
        node = mp.kernels.arithmetic.AdditionKernel.add_to_graph(self.__graph,inTensor, inTensor2, outTensor)
        return node.mp_type

    def TestSubtractionKernelCreation(self):
        inTensor = mp.Tensor.create(self.__session, (1,1))
        inTensor2 = mp.Tensor.create(self.__session, (1,1))
        outTensor = mp.Tensor.create(self.__session, (1,1))
        node = mp.kernels.arithmetic.SubtractionKernel.add_to_graph(self.__graph,inTensor, inTensor2, outTensor)
        return node.mp_type

    def TestMultiplicationKernelCreation(self):
        inTensor = mp.Tensor.create(self.__session, (1,1))
        inTensor2 = mp.Tensor.create(self.__session, (1,1))
        outTensor = mp.Tensor.create(self.__session, (1,1))
        node = mp.kernels.arithmetic.MultiplicationKernel.add_to_graph(self.__graph,inTensor,inTensor2,outTensor)
        return node.mp_type

    def TestDivisionKernelCreation(self):
        inTensor = mp.Tensor.create(self.__session, (1,1))
        inTensor2 = mp.Tensor.create(self.__session, (1,1))
        outTensor = mp.Tensor.create(self.__session, (1,1))
        node = mp.kernels.arithmetic.DivisionKernel.add_to_graph(self.__graph,inTensor,inTensor2,outTensor)
        return node.mp_type

    def TestLogKernelCreation(self):
        inTensor = mp.Tensor.create(self.__session, (1,1))
        outTensor = mp.Tensor.create(self.__session, (1,1))
        node = mp.kernels.arithmetic.LogKernel.add_to_graph(self.__graph,inTensor,outTensor)
        return node.mp_type


class KernelExecutionUnitTest:

    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestTensorAbsoluteKernelExecution(self):
        np.random.seed(44)
        raw_data = np.random.randint(-10,10, size=(2,2,2))
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, (2,2,2))
        tensor_test_node = mp.kernels.arithmetic.AbsoluteKernel.add_to_graph(self.__graph,inTensor,outTensor)

        sys.stdout = open(os.devnull, 'w')
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        sys.stdout = sys.__stdout__

        return (raw_data, outTensor.data)

    def TestScalarAbsoluteKernelExecution(self):
        raw_data = -10
        inScalar = mp.Scalar.create_from_value(self.__session, raw_data)
        outScalar = mp.Scalar.create(self.__session, int)
        scalar_test_node = mp.kernels.arithmetic.AbsoluteKernel.add_to_graph(self.__graph,inScalar,outScalar)

        sys.stdout = open(os.devnull, 'w')
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        sys.stdout = sys.__stdout__
        return (raw_data, outScalar.data)

    def TestAdditionKernelExecution(self):
        np.random.seed(7)
        raw_data = np.random.randint(-10,10, size=(2,2,2))
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        inTensor2 = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, raw_data.shape)
        tensor_node = mp.kernels.arithmetic.AdditionKernel.add_to_graph(self.__graph, inTensor, inTensor2, outTensor)
        sys.stdout = open(os.devnull, 'w')
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        sys.stdout = sys.__stdout__
        return (raw_data, outTensor.data)

    def TestSubtractionKernelExecution(self):
        inTensor = mp.Tensor.create(self.__session, (1,1))
        inTensor2 = mp.Tensor.create(self.__session, (1,1))
        outTensor = mp.Tensor.create(self.__session, (1,1))
        node = mp.kernels.arithmetic.SubtractionKernel.add_to_graph(self.__graph,inTensor, inTensor2, outTensor)
        return node.mp_type

    def TestMultiplicationKernelExecution(self):
        inTensor = mp.Tensor.create(self.__session, (1,1))
        inTensor2 = mp.Tensor.create(self.__session, (1,1))
        outTensor = mp.Tensor.create(self.__session, (1,1))
        node = mp.kernels.arithmetic.MultiplicationKernel.add_to_graph(self.__graph,inTensor,inTensor2,outTensor)
        return node.mp_type

    def TestDivisionKernelExecution(self):
        inTensor = mp.Tensor.create(self.__session, (1,1))
        inTensor2 = mp.Tensor.create(self.__session, (1,1))
        outTensor = mp.Tensor.create(self.__session, (1,1))
        node = mp.kernels.arithmetic.DivisionKernel.add_to_graph(self.__graph,inTensor,inTensor2,outTensor)
        return node.mp_type

    def TestLogKernelExecution(self):
        inTensor = mp.Tensor.create(self.__session, (1,1))
        outTensor = mp.Tensor.create(self.__session, (1,1))
        node = mp.kernels.arithmetic.LogKernel.add_to_graph(self.__graph,inTensor,outTensor)
        return node.mp_type

"""
def runCreationTests():
    KernelUnitTest_Object = KernelCreationUnitTest()

    assert KernelUnitTest_Object.TestAbsoluteKernelCreation() == mp.MPEnums.NODE
    #print("Absolute Kernel Creation Test: PASSED")
    #except AssertionError:
    #    print("Absolute Kernel Creation Test: FAILED")

    try:
        assert KernelUnitTest_Object.TestAdditionKernelCreation() == mp.MPEnums.NODE
        print("Addition Kernel Creation Test: PASSED")
    except AssertionError:
        print("Addition Kernel Creation Test: FAILED")

    try:
        assert KernelUnitTest_Object.TestSubtractionKernelCreation() == mp.MPEnums.NODE
        print("Subtraction Kernel Creation Test: PASSED")
    except AssertionError:
        print("Subtraction Kernel Creation Test: FAILED")

    try:
        assert KernelUnitTest_Object.TestMultiplicationKernelCreation() == mp.MPEnums.NODE
        print("Multiplication Kernel Creation Test: PASSED")
    except AssertionError:
        print("Multiplication Kernel Creation Test: FAILED")

    try:
        assert KernelUnitTest_Object.TestDivisionKernelCreation() == mp.MPEnums.NODE
        print("Division Kernel Creation Test: PASSED")
    except AssertionError:
        print("Division Kernel Creation Test: FAILED")

    try:
        assert KernelUnitTest_Object.TestLogKernelCreation() == mp.MPEnums.NODE
        print("Log Kernel Creation Test: PASSED")
    except AssertionError:
        print("Log Kernel Creation Test: FAILED")
"""
"""
def runExecutionTests():

    try:
        KernelExecutionUnitTest_Object = KernelExecutionUnitTest()
        res = KernelExecutionUnitTest_Object.TestTensorAbsoluteKernelExecution()
        assert res[1].all() == np.absolute(res[0]).all()

        print("Tensor Absolute Kernel Execution Test: PASSED")


    except AssertionError:
        print("Tensor Absolute Kernel Execution Test: FAILED")

    try:
        KernelExecutionUnitTest_Object = KernelExecutionUnitTest()
        res = KernelExecutionUnitTest_Object.TestScalarAbsoluteKernelExecution()
        assert res[1] == np.absolute(res[0])
        print("Scalar Absolute Kernel Execution Test: PASSED")

    except AssertionError:
        print("Scalar Absolute Kernel Execution Test: FAILED")

    try:
        KernelExecutionUnitTest_Object = KernelExecutionUnitTest()
        res = KernelExecutionUnitTest_Object.TestAdditionKernelExecution()
        assert res[1].all() == (res[0] + res[0]).all()
        print("Addition Kernel Execution Test: PASSED")

    except AssertionError:
        print("Addition Kernel Execution Test: FAILED")
"""

def test_create():
    KernelUnitTest_Object = KernelCreationUnitTest()
    assert KernelUnitTest_Object.TestAbsoluteKernelCreation() == mp.MPEnums.NODE
    del KernelUnitTest_Object

    KernelUnitTest_Object = KernelCreationUnitTest()
    assert KernelUnitTest_Object.TestAdditionKernelCreation() == mp.MPEnums.NODE
    del KernelUnitTest_Object

    KernelUnitTest_Object = KernelCreationUnitTest()
    assert KernelUnitTest_Object.TestSubtractionKernelCreation() == mp.MPEnums.NODE
    del KernelUnitTest_Object

    KernelUnitTest_Object = KernelCreationUnitTest()
    assert KernelUnitTest_Object.TestMultiplicationKernelCreation() == mp.MPEnums.NODE
    del KernelUnitTest_Object

    KernelUnitTest_Object = KernelCreationUnitTest()
    assert KernelUnitTest_Object.TestDivisionKernelCreation() == mp.MPEnums.NODE
    del KernelUnitTest_Object

    KernelUnitTest_Object = KernelCreationUnitTest()
    assert KernelUnitTest_Object.TestLogKernelCreation() == mp.MPEnums.NODE
    del KernelUnitTest_Object

def test_execute():
    KernelExecutionUnitTest_Object = KernelExecutionUnitTest()
    res = KernelExecutionUnitTest_Object.TestTensorAbsoluteKernelExecution()
    assert res[1].all() == np.absolute(res[0]).all()
    del KernelExecutionUnitTest_Object

    KernelExecutionUnitTest_Object = KernelExecutionUnitTest()
    res = KernelExecutionUnitTest_Object.TestScalarAbsoluteKernelExecution()
    assert res[1] == np.absolute(res[0])
    del KernelExecutionUnitTest_Object

    KernelExecutionUnitTest_Object = KernelExecutionUnitTest()
    res = KernelExecutionUnitTest_Object.TestAdditionKernelExecution()
    assert res[1].all() == (res[0] + res[0]).all()
    del KernelExecutionUnitTest_Object

