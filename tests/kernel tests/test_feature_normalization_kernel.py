import mindpype as mp
import numpy as np
import pytest

class FeatureNormalizationKernelUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestFeatureNormalizationKernelExecution(self, raw_data, init_data, labels_data, method, test_invalid_input=False, test_invalid_output_shape=False):
        if test_invalid_input:
            inTensor = mp.Scalar.create_from_value(self.__session, "test")
        else:
            inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        if test_invalid_output_shape:
            outTensor = mp.Tensor.create(self.__session, raw_data.shape*100)
        else:
            outTensor = mp.Tensor.create(self.__session,raw_data.shape)
        initialization_tensor = mp.Tensor.create_from_data(self.__session, init_data)
        labels = mp.Tensor.create_from_data(self.__session, labels_data)
        node = mp.kernels.FeatureNormalizationKernel.add_to_graph(self.__graph,inTensor,outTensor,method=method,axis=0,init_data=initialization_tensor,labels=labels)
        self.__graph.verify()
        self.__graph.initialize(initialization_tensor, labels)
        self.__graph.execute()
        return outTensor.data

def test_execute():
    np.random.seed(44)
    raw_data = np.random.randn(10,10,10)
    init_data = np.random.randn(10,10,10)
    labels_data = np.random.randint(0, 4, size=(10,))
    KernelExecutionUnitTest_Object = FeatureNormalizationKernelUnitTest()
    res = KernelExecutionUnitTest_Object.TestFeatureNormalizationKernelExecution(raw_data, init_data, labels_data, 'zscore-norm')
    expected_output = (raw_data - np.mean(init_data, axis=0)) / np.std(init_data, axis=0)
    assert (res == expected_output).all()
    
    res = KernelExecutionUnitTest_Object.TestFeatureNormalizationKernelExecution(raw_data, init_data, labels_data, 'min-max')
    expected_output = (raw_data - np.min(init_data, axis=0)) / (np.max(init_data, axis=0) - np.min(init_data, axis=0))
    assert (res == expected_output).all()
    
    res = KernelExecutionUnitTest_Object.TestFeatureNormalizationKernelExecution(raw_data, init_data, labels_data, 'mean-norm')
    expected_output = (raw_data - np.mean(init_data, axis=0)) / (np.max(init_data, axis=0) - np.min(init_data, axis=0))
    assert (res == expected_output).all()
    
    with pytest.raises(TypeError) as e_info:
        res = KernelExecutionUnitTest_Object.TestFeatureNormalizationKernelExecution(raw_data, init_data, labels_data, 'mean-norm', test_invalid_input=True)
    del KernelExecutionUnitTest_Object
    
    KernelExecutionUnitTest_Object = FeatureNormalizationKernelUnitTest()
    with pytest.raises(ValueError) as e_info:
        res = KernelExecutionUnitTest_Object.TestFeatureNormalizationKernelExecution(raw_data, init_data, labels_data, 'mean-norm', test_invalid_output_shape=True)
    del KernelExecutionUnitTest_Object