import pyriemann.utils.covariance
import pyriemann.utils.mean
import mindpype as mp
import numpy as np
import pyriemann

class RiemannMeanKernelUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestRiemannMeanKernelExecution(self, raw_data):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        # compute outTensor shape
        output_shape = inTensor.shape[-2:]
        outTensor = mp.Tensor.create(self.__session, output_shape)
        tensor_test_node = mp.kernels.RiemannMeanKernel.add_to_graph(self.__graph,inTensor,outTensor)

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
    KernelExecutionUnitTest_Object = RiemannMeanKernelUnitTest()
    res = KernelExecutionUnitTest_Object.TestRiemannMeanKernelExecution(raw_data)
    
    expected_output = pyriemann.utils.mean.mean_riemann(raw_data)
    assert(res == expected_output).all()

    del KernelExecutionUnitTest_Object
    