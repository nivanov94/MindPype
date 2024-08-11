import mindpype as mp
import numpy as np
import pytest

class BaselineCorrectionKernelUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestBaselineCorrectionKernelExecution(self, raw_data, baseline):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, raw_data.shape)
        node = mp.kernels.BaselineCorrectionKernel.add_to_graph(self.__graph,inTensor,outTensor,baseline_period=baseline)
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        return outTensor.data
    
    def TestNonTensorBaseline(self):
        input = mp.Scalar.create_from_value(self.__session, "test")
        output = mp.Scalar.create(self.__session, str)
        node = mp.kernels.BaselineCorrectionKernel.add_to_graph(self.__graph,input,output,baseline_period=[0,10])
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        return output
        
def test_execute():
    np.random.seed(44)
    raw_data = np.random.randn(10,10)
    KernelExecutionUnitTest_Object = BaselineCorrectionKernelUnitTest()
    res = KernelExecutionUnitTest_Object.TestBaselineCorrectionKernelExecution(raw_data, baseline=[0,10])
    expected_output = raw_data - np.mean(raw_data, axis=-1, keepdims=True)
    assert (res == expected_output).all()
    
    # test invalid baseline period
    with pytest.raises(ValueError) as e_info:
        res = KernelExecutionUnitTest_Object.TestBaselineCorrectionKernelExecution(raw_data, baseline=[5,3])
    del KernelExecutionUnitTest_Object
    
    KernelExecutionUnitTest_Object = BaselineCorrectionKernelUnitTest()      
    with pytest.raises(TypeError) as e_info:
        res = KernelExecutionUnitTest_Object.TestNonTensorBaseline()
    del KernelExecutionUnitTest_Object
    
    # test a differnt type of invalid baseline period
    KernelExecutionUnitTest_Object = BaselineCorrectionKernelUnitTest()   
    with pytest.raises(ValueError) as e_info:
        res = KernelExecutionUnitTest_Object.TestBaselineCorrectionKernelExecution(raw_data, baseline=[5])
    del KernelExecutionUnitTest_Object
    