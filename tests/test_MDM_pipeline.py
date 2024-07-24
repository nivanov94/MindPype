import pyriemann
import pyriemann.utils
import pyriemann.utils.covariance
import mindpype as mp
import numpy as np
from scipy import signal

class MDMPipelinenitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestMDMPipelineExecution(self, input_data, training_data, labels):    
        training_data = mp.Tensor.create_from_data(self.__session, training_data)
        labels = mp.Tensor.create_from_data(self.__session, labels)
        
        inTensor = mp.Tensor.create_from_data(self.__session, input_data)
        outTensor = mp.Scalar.create_from_value(self.__session,-1)
        
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
        node2 = mp.kernels.CovarianceKernel.add_to_graph(self.__graph,virtual_tensors[0],virtual_tensors[1])
        node3 = mp.kernels.RiemannMDMClassifierKernel.add_to_graph(self.__graph,virtual_tensors[1],outTensor,num_classes=3,initialization_data=training_data,labels=labels)
        
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()

        return outTensor.data

def test_execute():
    # create fake initialization and input data
    raw_training_data = np.random.normal(loc=0.0,scale=1.0,size=(180,250,12))
    training_data = np.zeros((180,12,12))
    for i in range(180):
        training_data[i,:,:] = np.cov(raw_training_data[i,:,:],rowvar=False)
        
    labels = np.asarray([0]*60 + [1]*60 + [2]*60)
        
    input_data = np.random.randn(12,500)
    
    KernelExecutionUnitTest_Object = MDMPipelinenitTest()
    res = KernelExecutionUnitTest_Object.TestMDMPipelineExecution(input_data, training_data, labels)
    
    # manually perform operations
    filter = signal.butter(4,(8,35),btype='bandpass',output='sos',fs=250)
    filtered_data = signal.sosfilt(filter,input_data)
    cov_data = np.cov(filtered_data)
    model = pyriemann.classification.MDM()
    model.fit(training_data, labels)
    expected_output = model.predict(cov_data)
    
    assert(res == expected_output).all()
    del KernelExecutionUnitTest_Object
