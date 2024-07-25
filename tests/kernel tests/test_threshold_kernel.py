import mindpype as mp
import pytest
import numpy as np

class ThresholdKernelUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def ThresholdKernelExecution(self, raw_data, thresh_val):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, raw_data.shape)
        thresh = mp.Scalar.create_from_value(self.__session, thresh_val)
        tensor_test_node = mp.kernels.ThresholdKernel.add_to_graph(self.__graph,inTensor, outTensor, thresh)

        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()

        return outTensor.data
    
    def TestThresholdValueError(self, raw_data, thresh_val):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, shape=(3,5,6))
        thresh = mp.Scalar.create_from_value(self.__session, thresh_val)
        tensor_test_node = mp.kernels.ThresholdKernel.add_to_graph(self.__graph,inTensor, outTensor, thresh)

        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        
    def TestThresholdTypeError(self, raw_data):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, raw_data.shape)
        thresh = mp.Scalar.create_from_value(self.__session, "1")
        tensor_test_node = mp.kernels.ThresholdKernel.add_to_graph(self.__graph,inTensor, outTensor, thresh)

        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        
    def TestNonScalarThresh(self, raw_data, thresh_val):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, raw_data.shape)
        thresh = mp.Tensor.create_from_data(self.__session, np.zeros(2,2))        
        thresh = mp.Scalar.create_from_value(self.__session, thresh_val)
        tensor_test_node = mp.kernels.ThresholdKernel.add_to_graph(self.__graph,inTensor, outTensor, thresh)

        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        
    def TestMismatchOutputTypes(self, raw_data):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Scalar.create(self.__session, int)
        thresh = mp.Tensor.create_from_data(self.__session, np.zeros(2,2))
        tensor_test_node = mp.kernels.ThresholdKernel.add_to_graph(self.__graph,inTensor, outTensor, thresh)

        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()

def test_execute():
    np.random.seed(44)
    raw_data = np.random.randn(2,2,2)
    thresh_val = 1
    
    KernelExecutionUnitTest_Object = ThresholdKernelUnitTest()
    res = KernelExecutionUnitTest_Object.ThresholdKernelExecution(raw_data, thresh_val)
    output = raw_data > thresh_val
    assert (res == output).all()
    
    with pytest.raises(TypeError) as e_info:
        res = KernelExecutionUnitTest_Object.TestThresholdTypeError(raw_data)
    del KernelExecutionUnitTest_Object
    
    KernelExecutionUnitTest_Object = ThresholdKernelUnitTest()
    with pytest.raises(ValueError) as e_info:
        res = KernelExecutionUnitTest_Object.TestThresholdValueError(raw_data, thresh_val)  
    del KernelExecutionUnitTest_Object

    KernelExecutionUnitTest_Object = ThresholdKernelUnitTest()
    with pytest.raises(TypeError) as e_info:
        res = KernelExecutionUnitTest_Object.TestMismatchOutputTypes(raw_data)  
    del KernelExecutionUnitTest_Object