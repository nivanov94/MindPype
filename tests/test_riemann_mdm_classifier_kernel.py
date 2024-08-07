import pyriemann
import pyriemann.utils
import pyriemann.utils.covariance
import mindpype as mp
import numpy as np

class RiemannMDMKernelUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestRiemannMDMKernelExecution(self, raw_data, init_data, init_label_data):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        # compute outTensor shape
        output_shape = (inTensor.shape[0],)
        outTensor = mp.Tensor.create(self.__session, output_shape)
        init_inputs = mp.Tensor.create_from_data(self.__session, init_data)
        labels = mp.Tensor.create_from_data(self.__session, init_label_data)
        node = mp.kernels.RiemannMDMClassifierKernel.add_to_graph(self.__graph,inTensor,outTensor,num_classes=2,initialization_data=init_inputs,labels=labels)

        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()

        return outTensor.data

def test_execute():
    np.random.seed(42)
    raw_data = np.random.randn(10,10,10)
    raw_data = pyriemann.utils.covariance.covariances(raw_data)
    r = 0.001
    raw_data = (1-r)*raw_data + r*np.diag(np.ones(raw_data.shape[-1]))
    init_data = np.random.randn(10,10,10)
    init_data = pyriemann.utils.covariance.covariances(init_data)
    init_data = (1-r)*init_data + r*np.diag(np.ones(init_data.shape[-1]))
    init_label_data = np.random.randint(0,2, size=(10,))
    
    KernelExecutionUnitTest_Object = RiemannMDMKernelUnitTest()
    res = KernelExecutionUnitTest_Object.TestRiemannMDMKernelExecution(raw_data, init_data, init_label_data)
    
    model = pyriemann.classification.MDM()
    model.fit(init_data, init_label_data)
    expected_output = model.predict(raw_data)
    assert(res == expected_output).all()
    del KernelExecutionUnitTest_Object