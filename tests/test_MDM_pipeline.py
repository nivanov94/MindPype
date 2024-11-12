import pyriemann
import pyriemann.utils
import pyriemann.utils.covariance
import mindpype as mp
import numpy as np
from scipy import signal

class MDMPipelineUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestMDMPipelineExecution(self, input_data, training_data, labels):    
        training_data = mp.Tensor.create_from_data(self.__session, training_data)
        labels = mp.Tensor.create_from_data(self.__session, labels)
        
        inTensor = mp.Tensor.create_from_data(self.__session, input_data)
        outTensor = mp.Tensor.create(self.__session, shape=(inTensor.shape[0],))
        
        virtual_tensors = [
            mp.Tensor.create_virtual(self.__session),
            mp.Tensor.create_virtual(self.__session)
        ]
        
        # create a filter
        order = 4
        bandpass = (8,35)
        fs = 250
        filter = mp.Filter.create_butter(self.__session,order,bandpass,btype='bandpass',fs=fs,implementation='sos')

        node1 = mp.kernels.FilterKernel.add_to_graph(self.__graph,inTensor, filter, virtual_tensors[0])
        node2 = mp.kernels.CovarianceKernel.add_to_graph(self.__graph,virtual_tensors[0],virtual_tensors[1], regularization=0.001, init_input=training_data, init_labels=labels)
        node3 = mp.kernels.RiemannMDMClassifierKernel.add_to_graph(self.__graph,virtual_tensors[1],outTensor,num_classes=3)
        
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()

        return outTensor.data

def test_execute():
    # create fake initialization and input data
    training_data = np.random.rand(180,12,12)
    
    labels = np.asarray([0]*60 + [1]*60 + [2]*60)
        
    input_data = np.random.randn(180,12,12)
    
    KernelExecutionUnitTest_Object = MDMPipelineUnitTest()
    res = KernelExecutionUnitTest_Object.TestMDMPipelineExecution(input_data, training_data, labels)
    
    # manually perform operations
    filter = signal.butter(4,(8,35),btype='bandpass',output='sos',fs=250)
    filtered_data = signal.sosfilt(filter,input_data,axis=1)
    r = 0.001
    cov_data = pyriemann.utils.covariance.covariances(filtered_data)
    cov_data = (1-r)*cov_data + r*np.diag(np.ones(cov_data.shape[-1]))
    cov_init_data = pyriemann.utils.covariance.covariances(training_data)
    cov_init_data = (1-r)*cov_init_data + r*np.diag(np.ones(cov_init_data.shape[-1]))
    model = pyriemann.classification.MDM()
    model.fit(cov_init_data, labels)
    expected_output = model.predict(cov_data)
    
    assert(res == expected_output).all()
    del KernelExecutionUnitTest_Object