from bcipy import bcipy
import sys, os
import numpy as np
import pytest



class KernelCreationUnitTest:

    def __init__(self):
        self.__session = bcipy.Session.create()
        self.__graph = bcipy.Graph.create(self.__session)
        
    def TestAbsoluteKernelCreation(self):
        inTensor = bcipy.Tensor.create(self.__session, (1,1))
        outTensor = bcipy.Tensor.create(self.__session, (1,1))
        node = bcipy.kernels.arithmetic.AbsoluteKernel.add_absolute_node(self.__graph,inTensor,outTensor)
        return node._bcip_type
    
    def TestAdditionKernelCreation(self):
        inTensor = bcipy.Tensor.create(self.__session, (1,1))
        inTensor2 = bcipy.Tensor.create(self.__session, (1,1))
        outTensor = bcipy.Tensor.create(self.__session, (1,1))
        node = bcipy.kernels.arithmetic.AdditionKernel.add_addition_node(self.__graph,inTensor, inTensor2, outTensor)
        return node._bcip_type
    
    def TestSubtractionKernelCreation(self):
        inTensor = bcipy.Tensor.create(self.__session, (1,1))
        inTensor2 = bcipy.Tensor.create(self.__session, (1,1))
        outTensor = bcipy.Tensor.create(self.__session, (1,1))
        node = bcipy.kernels.arithmetic.SubtractionKernel.add_subtraction_node(self.__graph,inTensor, inTensor2, outTensor)
        return node._bcip_type
    
    def TestMultiplicationKernelCreation(self):
        inTensor = bcipy.Tensor.create(self.__session, (1,1))
        inTensor2 = bcipy.Tensor.create(self.__session, (1,1))
        outTensor = bcipy.Tensor.create(self.__session, (1,1))
        node = bcipy.kernels.arithmetic.MultiplicationKernel.add_multiplication_node(self.__graph,inTensor,inTensor2,outTensor)
        return node._bcip_type
    
    def TestDivisionKernelCreation(self):
        inTensor = bcipy.Tensor.create(self.__session, (1,1))
        inTensor2 = bcipy.Tensor.create(self.__session, (1,1))
        outTensor = bcipy.Tensor.create(self.__session, (1,1))
        node = bcipy.kernels.arithmetic.DivisionKernel.add_division_node(self.__graph,inTensor,inTensor2,outTensor)
        return node._bcip_type
    
    def TestLogKernelCreation(self):
        inTensor = bcipy.Tensor.create(self.__session, (1,1))
        outTensor = bcipy.Tensor.create(self.__session, (1,1))
        node = bcipy.kernels.arithmetic.LogKernel.add_log_node(self.__graph,inTensor,outTensor)
        return node._bcip_type
    

class KernelExecutionUnitTest:

    def __init__(self):
        self.__session = bcipy.Session.create()
        self.__graph = bcipy.Graph.create(self.__session)
        
    def TestTensorAbsoluteKernelExecution(self):
        raw_data = np.random.randint(-10,10, size=(2,2,2))
        inTensor = bcipy.Tensor.create_from_data(self.__session, raw_data.shape, raw_data)
        outTensor = bcipy.Tensor.create(self.__session, (2,2,2))
        tensor_test_node = bcipy.kernels.arithmetic.AbsoluteKernel.add_absolute_node(self.__graph,inTensor,outTensor)
        
        sys.stdout = open(os.devnull, 'w')
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        sys.stdout = sys.__stdout__

        return (raw_data, outTensor.data)
    
    def TestScalarAbsoluteKernelExecution(self):
        raw_data = np.random.randint(-10, high=0)
        inScalar = bcipy.Scalar.create_from_value(self.__session, raw_data)
        outScalar = bcipy.Scalar.create(self.__session, int)
        scalar_test_node = bcipy.kernels.arithmetic.AbsoluteKernel.add_absolute_node(self.__graph,inScalar,outScalar)

        sys.stdout = open(os.devnull, 'w')
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        sys.stdout = sys.__stdout__
        return (raw_data, outScalar.data)
    
    def TestAdditionKernelExecution(self):
        raw_data = np.random.randint(-10,10, size=(2,2,2))
        inTensor = bcipy.Tensor.create_from_data(self.__session, raw_data.shape, raw_data)
        inTensor2 = bcipy.Tensor.create_from_data(self.__session, raw_data.shape, raw_data)
        outTensor = bcipy.Tensor.create(self.__session, raw_data.shape)
        tensor_node = bcipy.kernels.arithmetic.AdditionKernel.add_addition_node(self.__graph, inTensor, inTensor2, outTensor)
        sys.stdout = open(os.devnull, 'w')
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        sys.stdout = sys.__stdout__
        return (raw_data, outTensor.data)
    
    def TestSubtractionKernelExecution(self):
        inTensor = bcipy.Tensor.create(self.__session, (1,1))
        inTensor2 = bcipy.Tensor.create(self.__session, (1,1))
        outTensor = bcipy.Tensor.create(self.__session, (1,1))
        node = bcipy.kernels.arithmetic.SubtractionKernel.add_subtraction_node(self.__graph,inTensor, inTensor2, outTensor)
        return node._bcip_type
    
    def TestMultiplicationKernelExecution(self):
        inTensor = bcipy.Tensor.create(self.__session, (1,1))
        inTensor2 = bcipy.Tensor.create(self.__session, (1,1))
        outTensor = bcipy.Tensor.create(self.__session, (1,1))
        node = bcipy.kernels.arithmetic.MultiplicationKernel.add_multiplication_node(self.__graph,inTensor,inTensor2,outTensor)
        return node._bcip_type
    
    def TestDivisionKernelExecution(self):
        inTensor = bcipy.Tensor.create(self.__session, (1,1))
        inTensor2 = bcipy.Tensor.create(self.__session, (1,1))
        outTensor = bcipy.Tensor.create(self.__session, (1,1))
        node = bcipy.kernels.arithmetic.DivisionKernel.add_division_node(self.__graph,inTensor,inTensor2,outTensor)
        return node._bcip_type
    
    def TestLogKernelExecution(self):
        inTensor = bcipy.Tensor.create(self.__session, (1,1))
        outTensor = bcipy.Tensor.create(self.__session, (1,1))
        node = bcipy.kernels.arithmetic.LogKernel.add_log_node(self.__graph,inTensor,outTensor)
        return node._bcip_type
    
"""
def runCreationTests():
    KernelUnitTest_Object = KernelCreationUnitTest()

    assert KernelUnitTest_Object.TestAbsoluteKernelCreation() == bcipy.BcipEnums.NODE
    #print("Absolute Kernel Creation Test: PASSED")
    #except AssertionError:
    #    print("Absolute Kernel Creation Test: FAILED")

    try:
        assert KernelUnitTest_Object.TestAdditionKernelCreation() == bcipy.BcipEnums.NODE
        print("Addition Kernel Creation Test: PASSED")
    except AssertionError:
        print("Addition Kernel Creation Test: FAILED")

    try:
        assert KernelUnitTest_Object.TestSubtractionKernelCreation() == bcipy.BcipEnums.NODE
        print("Subtraction Kernel Creation Test: PASSED")
    except AssertionError:
        print("Subtraction Kernel Creation Test: FAILED")

    try:
        assert KernelUnitTest_Object.TestMultiplicationKernelCreation() == bcipy.BcipEnums.NODE
        print("Multiplication Kernel Creation Test: PASSED")
    except AssertionError:
        print("Multiplication Kernel Creation Test: FAILED")

    try:
        assert KernelUnitTest_Object.TestDivisionKernelCreation() == bcipy.BcipEnums.NODE
        print("Division Kernel Creation Test: PASSED")
    except AssertionError:
        print("Division Kernel Creation Test: FAILED")

    try:
        assert KernelUnitTest_Object.TestLogKernelCreation() == bcipy.BcipEnums.NODE
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
    assert KernelUnitTest_Object.TestAbsoluteKernelCreation() == bcipy.BcipEnums.NODE
    del KernelUnitTest_Object

    KernelUnitTest_Object = KernelCreationUnitTest()
    assert KernelUnitTest_Object.TestAdditionKernelCreation() == bcipy.BcipEnums.NODE
    del KernelUnitTest_Object

    KernelUnitTest_Object = KernelCreationUnitTest()
    assert KernelUnitTest_Object.TestSubtractionKernelCreation() == bcipy.BcipEnums.NODE
    del KernelUnitTest_Object

    KernelUnitTest_Object = KernelCreationUnitTest()
    assert KernelUnitTest_Object.TestMultiplicationKernelCreation() == bcipy.BcipEnums.NODE
    del KernelUnitTest_Object

    KernelUnitTest_Object = KernelCreationUnitTest()
    assert KernelUnitTest_Object.TestDivisionKernelCreation() == bcipy.BcipEnums.NODE
    del KernelUnitTest_Object

    KernelUnitTest_Object = KernelCreationUnitTest()
    assert KernelUnitTest_Object.TestLogKernelCreation() == bcipy.BcipEnums.NODE
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

    