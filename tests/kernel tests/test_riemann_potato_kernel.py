import pyriemann.utils.covariance
import pyriemann.utils.mean
import mindpype as mp
import numpy as np
import pyriemann
import pytest

class RiemannPotatoKernelUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestRiemannPotatoKernelExecution(self, raw_data, init_data, max_iter, thresh):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        initTensor = mp.Tensor.create_from_data(self.__session, init_data)
        # calculate outTensor shape
        output_shape = (inTensor.shape[0],)
        outTensor = mp.Tensor.create(self.__session, output_shape)
        tensor_test_node = mp.kernels.RiemannPotatoKernel.add_to_graph(self.__graph,inTensor,outTensor,initTensor, thresh=thresh, max_iter=max_iter, regularization=0)

        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        
        return outTensor.data
    
    def TestNonTensorOutputError(self, raw_data, init_data, max_iter, thresh):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        initTensor = mp.Tensor.create_from_data(self.__session, init_data)
        outTensor = mp.Array.create(self.__session, capacity=3, element_template=inTensor)
        tensor_test_node = mp.kernels.RiemannPotatoKernel.add_to_graph(self.__graph,inTensor,outTensor,initTensor, thresh=thresh, max_iter=max_iter, regularization=0)

        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        
        return outTensor.data
    
    def TestNonTensorInputError(self, init_data, max_iter, thresh):
        inTensor = mp.Scalar.create_from_value(self.__session, "test")
        initTensor = mp.Tensor.create_from_data(self.__session, init_data)
        output_shape = (initTensor.shape[0],)
        outTensor = mp.Tensor.create(self.__session, output_shape)
        tensor_test_node = mp.kernels.RiemannPotatoKernel.add_to_graph(self.__graph,inTensor,outTensor,initTensor, thresh=thresh, max_iter=max_iter, regularization=0)

        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        
        return outTensor.data

def test_execute():
    np.random.seed(44)
    raw_data = np.random.randn(3,3,3)
    raw_data = pyriemann.utils.covariance.covariances(raw_data)
    r = 0.001
    raw_data = (1-r)*raw_data + r*np.diag(np.ones(raw_data.shape[-1]))
       
    init_data = np.random.randn(3,3,3)
    init_data = pyriemann.utils.covariance.covariances(init_data)
    init_data = (1-r)*init_data + r*np.diag(np.ones(init_data.shape[-1]))   
    
    thresh = 3
    KernelExecutionUnitTest_Object = RiemannPotatoKernelUnitTest()
    res = KernelExecutionUnitTest_Object.TestRiemannPotatoKernelExecution(raw_data, init_data, max_iter=100, thresh=thresh)
    
    potato_filter = pyriemann.clustering.Potato()
    potato_filter.fit(init_data)
    expected_output = potato_filter.predict(raw_data)
    
    assert(res == expected_output).all()
    del KernelExecutionUnitTest_Object

    # test max iter error
    KernelExecutionUnitTest_Object = RiemannPotatoKernelUnitTest()
    with pytest.raises(ValueError) as e_info:
        res = KernelExecutionUnitTest_Object.TestRiemannPotatoKernelExecution(raw_data, init_data, max_iter= -1, thresh=thresh)
    del KernelExecutionUnitTest_Object
    
    # test thresh error
    KernelExecutionUnitTest_Object = RiemannPotatoKernelUnitTest()
    with pytest.raises(ValueError) as e_info:
        res = KernelExecutionUnitTest_Object.TestRiemannPotatoKernelExecution(raw_data, init_data, max_iter= 100, thresh=-1)
    del KernelExecutionUnitTest_Object
    
    #test non 3d init data
    KernelExecutionUnitTest_Object = RiemannPotatoKernelUnitTest()
    invalid_init = np.random.randn(3,3)
    with pytest.raises(ValueError) as e_info:
        res = KernelExecutionUnitTest_Object.TestRiemannPotatoKernelExecution(raw_data, invalid_init, max_iter= 100, thresh=thresh)
    del KernelExecutionUnitTest_Object
    
    KernelExecutionUnitTest_Object = RiemannPotatoKernelUnitTest()
    with pytest.raises(TypeError) as e_info:
        res = KernelExecutionUnitTest_Object.TestNonTensorInputError(init_data, max_iter= 100, thresh=thresh)
    del KernelExecutionUnitTest_Object
    
    KernelExecutionUnitTest_Object = RiemannPotatoKernelUnitTest()
    with pytest.raises(TypeError) as e_info:
        res = KernelExecutionUnitTest_Object.TestNonTensorOutputError(raw_data, init_data, max_iter= 100, thresh=thresh)
    del KernelExecutionUnitTest_Object
    
test_execute()