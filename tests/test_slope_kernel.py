import mindpype as mp
import numpy as np

class SlopeKernelUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestSlopeKernelExecution(self, raw_data):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, (2,1))
        # tensor_test_node = mp.kernels.SlopeKernel.add_to_graph(self.__graph,inTensor,outTensor,Fs=1,axis=-1)
        tensor_test_node = mp.kernels.SlopeKernel.add_to_graph(self.__graph,inTensor,outTensor,Fs=1,axis=0)

        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()

        return (inTensor, outTensor.data)
    
def test_execute():
    KernelExecutionUnitTest_Object = SlopeKernelUnitTest()
    np.random.seed(44)
    raw_data = np.random.randn(2,2)
    res = KernelExecutionUnitTest_Object.TestSlopeKernelExecution(raw_data)
    
    # manual calculation
    axis = 0
    # axis = -1
    Fs = 1
    Ns = res[0].shape[axis]
    X = np.linspace(0, Ns/Fs, Ns)
    Y = np.moveaxis(res[0].data, axis, -1)
    X -= np.mean(X)
    Y -= np.mean(Y, axis=-1, keepdims=True)
    output = np.expand_dims(Y.dot(X) / X.dot(X), axis=-1)
    
    assert (res[1] == output).all()
    del KernelExecutionUnitTest_Object
