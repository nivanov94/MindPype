import pyriemann.utils
import pyriemann.utils.covariance
import pyriemann.utils.distance
import mindpype as mp
import numpy as np
import pyriemann
import pytest

class RiemannDistanceKernelUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestRiemannDistanceKernelExecution(self, raw_data1, raw_data2):
        inTensor1 = mp.Tensor.create_from_data(self.__session, raw_data1)
        inTensor2 = mp.Tensor.create_from_data(self.__session, raw_data2)
        
        # compute outTensor shape
        out_sz = []
        mat_sz = None
        for param in (inTensor1,inTensor2):
            param_rank = len(param.shape)
            if mat_sz == None:
                mat_sz = param.shape[-2:]
            elif param.shape[-2:] != mat_sz:
                return ()

            if param_rank == 3:
                out_sz.append(param.shape[0])
            else:
                out_sz.append(1)

        outTensor = mp.Tensor.create(self.__session, tuple(out_sz))
        tensor_test_node = mp.kernels.RiemannDistanceKernel.add_to_graph(self.__graph,inTensor1,inTensor2,outTensor)

        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()

        return outTensor.data
    
    def TestArrayRiemannDistance(self, raw_data):
        template = mp.Tensor.create_from_data(self.__session, raw_data)
        input1 = mp.Array.create(self.__session, 3, template)
        input2 = mp.Array.create(self.__session, 3, template)
        output = mp.Array.create(self.__session, 3, template)
        tensor_test_node = mp.kernels.RiemannDistanceKernel.add_to_graph(self.__graph,input1,input2,output)

        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        return input1
        
    def TestNonTensorInput(self):
        input1 = mp.Scalar.create_from_value(self.__session, 1)
        input2 = mp.Scalar.create_from_value(self.__session, 1)
        output = mp.Scalar.create(self.__session, int)
        tensor_test_node = mp.kernels.RiemannDistanceKernel.add_to_graph(self.__graph,input1,input2,output)
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        return input1
    
    def TestWrongOutputShape(self, raw_data1, raw_data2):
        inTensor1 = mp.Tensor.create_from_data(self.__session, raw_data1)
        inTensor2 = mp.Tensor.create_from_data(self.__session, raw_data2)
        outTensor = mp.Tensor.create(self.__session, (1,2))
        tensor_test_node = mp.kernels.RiemannDistanceKernel.add_to_graph(self.__graph,inTensor1,inTensor2,outTensor)

        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
            
def test_execute():
    np.random.seed(44)
    raw_data1 = np.random.randn(10,10,10)
    raw_data1 = pyriemann.utils.covariance.covariances(raw_data1)
    raw_data2 = np.random.randn(10,10,10)
    raw_data2 = pyriemann.utils.covariance.covariances(raw_data2)
    r = 0.001
    raw_data1 = (1-r)*raw_data1 + r*np.diag(np.ones(raw_data1.shape[-1]))
    raw_data2 = (1-r)*raw_data2 + r*np.diag(np.ones(raw_data2.shape[-1]))
    KernelExecutionUnitTest_Object = RiemannDistanceKernelUnitTest()
    res = KernelExecutionUnitTest_Object.TestRiemannDistanceKernelExecution(raw_data1, raw_data2)
    
    expected_output = np.empty((10,10))
    for i in range(10):
            # extract the ith element from inA
            x = raw_data1[i,:,:]
            for j in range(10):
                # extract the jth element from inB
                y = raw_data2[j,:,:]
                expected_output[i,j] = pyriemann.utils.distance.distance_riemann(x,y)
    assert(res[0,0] == expected_output[0,0])
    assert (res == expected_output).all()
    
    with pytest.raises(ValueError) as e_info:
        res = KernelExecutionUnitTest_Object.TestWrongOutputShape(raw_data1,raw_data2)
    del KernelExecutionUnitTest_Object
    
    KernelExecutionUnitTest_Object = RiemannDistanceKernelUnitTest()
    with pytest.raises(TypeError) as e_info:
        res = KernelExecutionUnitTest_Object.TestNonTensorInput()
    del KernelExecutionUnitTest_Object
    
    KernelExecutionUnitTest_Object = RiemannDistanceKernelUnitTest()
    with pytest.raises(TypeError) as e_info:
        res = KernelExecutionUnitTest_Object.TestArrayRiemannDistance(raw_data1)
    del KernelExecutionUnitTest_Object
    
test_execute()