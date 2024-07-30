import mindpype as mp
import numpy as np
from scipy import signal
import mne
import sklearn

class CSPPipelinenitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestCSPPipelineExecution(self, input_data, training_data, labels, order, bandpass, fs):    
        training_data = mp.Tensor.create_from_data(self.__session, training_data)
        labels = mp.Tensor.create_from_data(self.__session, labels)
        
        inTensor = mp.Tensor.create_from_data(self.__session, input_data)
        # outTensor = mp.Scalar.create_from_value(self.__session,-1)
        outTensor = mp.Tensor.create(self.__session, shape=(inTensor.shape[0],))
        
        virtual_tensors = [
            mp.Tensor.create_virtual(self.__session),
            mp.Tensor.create_virtual(self.__session),
            mp.Tensor.create_virtual(self.__session),
            mp.Tensor.create_virtual(self.__session)
        ]
        
        # create a filter
        filter = mp.Filter.create_butter(self.__session,order,bandpass,btype='bandpass',fs=fs,implementation='sos')

        # create classifier
        classifier = mp.Classifier.create_LDA(self.__session)

        node1 = mp.kernels.FilterKernel.add_to_graph(self.__graph,inTensor, filter, virtual_tensors[0], init_input=training_data, init_labels=labels)
        node2 = mp.kernels.CommonSpatialPatternKernel.add_to_graph(self.__graph, virtual_tensors[0], virtual_tensors[1])
        node3 = mp.kernels.VarKernel.add_to_graph(self.__graph, virtual_tensors[1], virtual_tensors[2], axis=1)
        node4 = mp.kernels.LogKernel.add_to_graph(self.__graph, virtual_tensors[2], virtual_tensors[3])
        node5 = mp.kernels.ClassifierKernel.add_to_graph(self.__graph, virtual_tensors[3], classifier, outTensor)
       
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()

        return outTensor.data

def test_execute():
    # create fake initialization and input data
    training_data = np.random.random((120,12,12))
    labels = np.asarray([0]*60 + [1]*60)
        
    input_data = np.random.randn(120, 12, 12)
    
    order = 4
    bandpass = (8,35)
    fs = 250
    
    KernelExecutionUnitTest_Object = CSPPipelinenitTest()
    res = KernelExecutionUnitTest_Object.TestCSPPipelineExecution(input_data, training_data, labels, order, bandpass, fs)
    
    # manually perform operations
    filter = signal.butter(order,bandpass,btype='bandpass',output='sos',fs=fs)
    filtered_data = signal.sosfilt(filter,input_data, axis=1)
    filtered_init_data = signal.sosfilt(filter, training_data, axis=1)
    csp = mne.decoding.CSP(cov_est='concat', transform_into='csp_space')
    csp.fit(filtered_init_data, labels)
    
    csp_training_data = csp_data = csp.transform(filtered_init_data)
    var_training_data = np.var(csp_training_data, axis=1)
    log_training_data = np.log(var_training_data)
    
    csp_data = csp.transform(filtered_data)
    var_data = np.var(csp_data, axis=1)
    log_data = np.log(var_data)
    
    classifier = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(solver='svd')
    classifier.fit(log_training_data, labels)
    expected_output = classifier.predict(log_data)
    
    assert(res == expected_output).all()
    del KernelExecutionUnitTest_Object