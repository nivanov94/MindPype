import pyriemann.utils
import pyriemann.utils.covariance
import mindpype as mp
import numpy as np
from pyriemann.estimation import XdawnCovariances
from scipy import signal
from sklearn.feature_selection import SelectKBest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pyriemann
import mne

class MiscPipelineUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestMiscPipelineExecution(self, input_data, init_data, init_label_data):    
        init_data = mp.Tensor.create_from_data(self.__session, init_data)
        init_labels = mp.Tensor.create_from_data(self.__session, init_label_data)
        inTensor = mp.Tensor.create_from_data(self.__session, input_data)
        outTensor = mp.Tensor.create(self.__session, shape=(10,4,10))
        
        virtual_tensors = [
            mp.Tensor.create_virtual(self.__session),
        ]
        node1 = mp.kernels.TransposeKernel.add_to_graph(self.__graph, inTensor, virtual_tensors[0], axes=[0,1,2], init_input=init_data, init_labels=init_labels)
        node2 = mp.kernels.CommonSpatialPatternKernel.add_to_graph(self.__graph, virtual_tensors[0], outTensor)

        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()

        return outTensor.data

class Misc2PipelineUnitTest():
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestMisc2PipelineExecution(self, input_data, init_data, init_label_data):    
        init_data = mp.Tensor.create_from_data(self.__session, init_data)
        init_labels = mp.Tensor.create_from_data(self.__session, init_label_data)
        inTensor = mp.Tensor.create_from_data(self.__session, input_data)
        outTensor = mp.Tensor.create(self.__session, (10,16,16))
        
        virtual_tensors = [
            mp.Tensor.create_virtual(self.__session),
        ]
        node1 = mp.kernels.BaselineCorrectionKernel.add_to_graph(self.__graph,inTensor,virtual_tensors[0],baseline_period=[0,10], init_input=init_data, init_labels=init_labels)
        node2 = mp.kernels.XDawnCovarianceKernel.add_to_graph(self.__graph, virtual_tensors[0], outTensor)

        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()

        return outTensor.data
    
class Misc3PipelineUnitTest():
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestMisc3PipelineExecution(self, input_data, thresh, init_data, init_label_data, Nfeats):    
        init_in = mp.Tensor.create_from_data(self.__session, init_data[0])
        init_thresholds = mp.Tensor.create_from_data(self.__session, init_data[1])
        init_labels = mp.Tensor.create_from_data(self.__session, init_label_data)
        inTensor = mp.Tensor.create_from_data(self.__session, input_data)
        thresh = mp.Scalar.create_from_value(self.__session, thresh)
        v_tensor = mp.Tensor.create_virtual(self.__session)
        outTensor = mp.Tensor.create(self.__session, (10,2))
        
        mp.kernels.ThresholdKernel.add_to_graph(self.__graph,inTensor,v_tensor,thresh=thresh, init_inputs=(init_in, init_thresholds), init_labels=init_labels)
        mp.kernels.FeatureSelectionKernel.add_to_graph(self.__graph,v_tensor,outTensor,k=Nfeats)
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()

        return outTensor.data
    
class Misc4PipelineUnitTest():
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestMisc4PipelineExecution(self, input_data, init_data, init_label_data, factor):    
        init_data = mp.Tensor.create_from_data(self.__session, init_data)
        init_labels = mp.Tensor.create_from_data(self.__session, init_label_data)
        inTensor = mp.Tensor.create_from_data(self.__session, input_data)
        outTensor = mp.Tensor.create(self.__session, (10,16,16))
        
        virtual_tensors = [
            mp.Tensor.create_virtual(self.__session),
        ]
        node1 = mp.kernels.ResampleKernel.add_to_graph(self.__graph,inTensor, factor, virtual_tensors[0], init_input=init_data, init_labels=init_labels)
        node2 = mp.kernels.XDawnCovarianceKernel.add_to_graph(self.__graph, virtual_tensors[0], outTensor)

        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()

        return outTensor.data
    
class Misc5PipelineUnitTest():
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestMisc5PipelineExecution(self, input_data, init_data, init_label_data):    
        init_data = mp.Tensor.create_from_data(self.__session, init_data)
        init_labels = mp.Tensor.create_from_data(self.__session, init_label_data)
        inTensor = mp.Tensor.create_from_data(self.__session, input_data)
        outTensor = mp.Tensor.create(self.__session, (inTensor.shape[0],))
        
        virtual_tensors = [
            mp.Tensor.create_virtual(self.__session),
        ]
        mp_clsf = mp.Classifier.create_LDA(self.__session, shrinkage='auto', solver='lsqr')
        node1 = mp.kernels.FeatureSelectionKernel.add_to_graph(self.__graph,inTensor, virtual_tensors[0], init_inputs=init_data, labels=init_labels)
        node2 = mp.kernels.ClassifierKernel.add_to_graph(self.__graph, virtual_tensors[0], mp_clsf, outTensor)

        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()

        return outTensor.data   
    
class Misc6PipelineUnitTest():
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestMisc6PipelineExecution(self, input_data, init_data):    
        init_data = mp.Tensor.create_from_data(self.__session, init_data)
        inTensor = mp.Tensor.create_from_data(self.__session, input_data)
        outTensor = mp.Tensor.create(self.__session, (4,10))
        
        virtual_tensors = [
            mp.Tensor.create_virtual(self.__session),
            mp.Tensor.create_virtual(self.__session)
        ]
        
        node1 = mp.kernels.PadKernel.add_to_graph(self.__graph,inTensor, virtual_tensors[0], pad_width=1, init_input=init_data)
        node2 = mp.kernels.CovarianceKernel.add_to_graph(self.__graph, virtual_tensors[0], virtual_tensors[1], regularization=0.001)
        node3 = mp.kernels.TangentSpaceKernel.add_to_graph(self.__graph, virtual_tensors[1], outTensor)

        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()

        return outTensor.data  
    
class Misc7PipelineUnitTest():
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestMisc7PipelineExecution(self, input_data, init_data, labels):    
        init_data = mp.Tensor.create_from_data(self.__session, init_data)
        init_labels = mp.Tensor.create_from_data(self.__session, labels)
        inTensor = mp.Tensor.create_from_data(self.__session, input_data)
        outTensor = mp.Tensor.create(self.__session, (inTensor.shape[0],))
        
        virtual_tensors = [
            mp.Tensor.create_virtual(self.__session)
        ]
        
        classifier = mp.Classifier.create_LDA(self.__session, shrinkage='auto', solver='lsqr')
        
        node1 = mp.kernels.SlopeKernel.add_to_graph(self.__graph, inTensor, virtual_tensors[0], init_input=init_data, init_labels=init_labels)
        node2 = mp.kernels.ClassifierKernel.add_to_graph(self.__graph, virtual_tensors[0], classifier, outTensor)
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()

        return outTensor.data     
    
class Misc8PipelineUnitTest():
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestMisc8PipelineExecution(self, input_data, init_data, labels):    
        init_data = mp.Tensor.create_from_data(self.__session, init_data)
        init_labels = mp.Tensor.create_from_data(self.__session, labels)
        inTensor = mp.Tensor.create_from_data(self.__session, input_data)
        outTensor = mp.Tensor.create(self.__session, (52,))
        
        virtual_tensors = [
            mp.Tensor.create_virtual(self.__session)
        ]
        
        classifier = mp.Classifier.create_LDA(self.__session, shrinkage='auto', solver='lsqr')
        
        node1 = mp.kernels.PadKernel.add_to_graph(self.__graph,inTensor, virtual_tensors[0], pad_width=1, init_input=init_data, init_labels=init_labels)
        node2 = mp.kernels.ClassifierKernel.add_to_graph(self.__graph, virtual_tensors[0], classifier, outTensor)
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()

        return outTensor.data   
    
class Misc9PipelineUnitTest():
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestMisc9PipelineExecution(self, input_data, init_data, labels):    
        init_data = mp.Tensor.create_from_data(self.__session, init_data)
        init_labels = mp.Tensor.create_from_data(self.__session, labels)
        inTensor = mp.Tensor.create_from_data(self.__session, input_data)
        outTensor = mp.Tensor.create(self.__session, (4,))
        
        virtual_tensors = [
            mp.Tensor.create_virtual(self.__session),
            mp.Tensor.create_virtual(self.__session)
        ]
        
        classifier = mp.Classifier.create_LDA(self.__session, shrinkage='auto', solver='lsqr')
        
        node1 = mp.kernels.RiemannPotatoKernel.add_to_graph(self.__graph, inTensor, virtual_tensors[0]) #, initialization_data=init_data, labels=init_labels)
        node1.add_initialization_data([init_data], init_labels)
        node2 = mp.kernels.ReshapeKernel.add_to_graph(self.__graph, virtual_tensors[0], virtual_tensors[1], shape=(4,4))
        node3 = mp.kernels.ClassifierKernel.add_to_graph(self.__graph, virtual_tensors[1], classifier, outTensor)
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()

        return outTensor.data    
    
class Misc10PipelineUnitTest():
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestMisc10PipelineExecution(self, input_data_A, input_data_B, init_data_A, init_data_B, labels, axis):    
        initA = mp.Tensor.create_from_data(self.__session, init_data_A)
        initB = mp.Tensor.create_from_data(self.__session, init_data_B)
        init_labels = mp.Tensor.create_from_data(self.__session, labels)
        inTensor1 = mp.Tensor.create_from_data(self.__session, input_data_A)
        inTensor2 = mp.Tensor.create_from_data(self.__session, input_data_B)
        outTensor = mp.Tensor.create(self.__session, (4,))
        
        virtual_tensors = [
            mp.Tensor.create_virtual(self.__session)
        ]
        
        lda_object = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        classifier = mp.Classifier.create_custom_classifier(self.__session, lda_object, 'lda')
        
        node1 = mp.kernels.ConcatenationKernel.add_to_graph(self.__graph, inTensor1, inTensor2, virtual_tensors[0], axis=axis, init_inputs=[initA, initB], init_labels=init_labels)
        node2 = mp.kernels.ClassifierKernel.add_to_graph(self.__graph, virtual_tensors[0], classifier, outTensor)
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()

        return outTensor.data  
    
class Misc11PipelineUnitTest():
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestMisc11PipelineExecution(self, input_data1, input_data2, init_data1, init_data2, labels):    
        init_data1 = mp.Tensor.create_from_data(self.__session, init_data1)
        init_data2 = mp.Tensor.create_from_data(self.__session, init_data2)
        init_labels = mp.Tensor.create_from_data(self.__session, labels)
        inTensor1 = mp.Tensor.create_from_data(self.__session, input_data1)
        inTensor2 = mp.Tensor.create_from_data(self.__session, input_data2)
        outTensor = mp.Tensor.create(self.__session, (4,))
        
        virtual_tensors = [
            mp.Tensor.create_virtual(self.__session),
        ]
        
        classifier = mp.Classifier.create_logistic_regression(self.__session)
        
        node1 = mp.kernels.RiemannDistanceKernel.add_to_graph(self.__graph, inTensor1, inTensor2, virtual_tensors[0], init_inputs=[init_data1, init_data2], init_labels=init_labels)
        node2 = mp.kernels.ClassifierKernel.add_to_graph(self.__graph, virtual_tensors[0], classifier, outTensor)
        self.__graph.initialize()
        self.__graph.execute()

        return outTensor.data    

class Misc12PipelineUnitTest():
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestMisc12PipelineExecution(self, input_data, init_data, labels):    
        init_data = mp.Tensor.create_from_data(self.__session, init_data)
        init_labels = mp.Tensor.create_from_data(self.__session, labels)
        inTensor = mp.Tensor.create_from_data(self.__session, input_data)
        outTensor = mp.Tensor.create(self.__session, (inTensor.shape[0],))
        
        virtual_tensors = [
            mp.Tensor.create_virtual(self.__session)
        ]
        
        classifier = mp.Classifier.create_SVM(self.__session)
        
        node1 = mp.kernels.RiemannMeanKernel.add_to_graph(self.__graph, inTensor, virtual_tensors[0], init_input=init_data, init_labels=init_labels)
        node2 = mp.kernels.ClassifierKernel.add_to_graph(self.__graph, virtual_tensors[0], classifier, outTensor)
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()

        return outTensor.data      

class Misc13PipelineUnitTest():
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestMisc13PipelineExecution(self, input_data, init_data, labels, epoch_stride, epoch_length, ax):    
        init_data = mp.Tensor.create_from_data(self.__session, init_data)
        init_labels = mp.Tensor.create_from_data(self.__session, labels)
        inTensor = mp.Tensor.create_from_data(self.__session, input_data)
        outTensor = mp.Tensor.create(self.__session, (4,))
        
        virtual_tensors = [
            mp.Tensor.create_virtual(self.__session),
            mp.Tensor.create_virtual(self.__session)
        ]
        
        classifier = mp.Classifier.create_SVM(self.__session)
        
        node1 = mp.kernels.EpochKernel.add_to_graph(self.__graph,inTensor,virtual_tensors[0],epoch_len=epoch_length, epoch_stride=epoch_stride, axis=ax, init_input=init_data, labels=init_labels)
        node2 = mp.kernels.ReshapeKernel.add_to_graph(self.__graph, virtual_tensors[0], virtual_tensors[1], (4,2))
        node3 = mp.kernels.ClassifierKernel.add_to_graph(self.__graph, virtual_tensors[1], classifier, outTensor)
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()

        return outTensor.data  
    
class Misc14PipelineUnitTest():
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestMisc14PipelineExecution(self, input_data, init_data, labels):    
        init_data = mp.Tensor.create_from_data(self.__session, init_data)
        init_labels = mp.Tensor.create_from_data(self.__session, labels)
        inTensor1 = mp.Tensor.create_from_data(self.__session, input_data)
        inTensor2 = mp.Tensor.create_from_data(self.__session, input_data)
        outTensor = mp.Tensor.create(self.__session, (input_data.shape[0],))
        
        virtual_tensors = [
            mp.Tensor.create_virtual(self.__session)
        ]
        
        classifier = mp.Classifier.create_SVM(self.__session)
        
        node1 = mp.kernels.AdditionKernel.add_to_graph(self.__graph,inTensor1, inTensor2, virtual_tensors[0], init_inputs=[init_data, init_data], init_labels=init_labels)
        node2 = mp.kernels.ClassifierKernel.add_to_graph(self.__graph, virtual_tensors[0], classifier, outTensor)
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()

        return outTensor.data      

class Misc15PipelineUnitTest():
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestMisc15PipelineExecution(self, input_data, init_data, labels):    
        inTensor = mp.Tensor.create_from_data(self.__session, input_data)
        init_tensor = mp.Tensor.create_from_data(self.__session, init_data)
        labels = mp.Tensor.create_from_data(self.__session, labels)
        outTensor = mp.Tensor.create(self.__session, (input_data.shape[0],))
        
        virtual_tensors = [
            mp.Tensor.create_virtual(self.__session)
        ]
        
        classifier = mp.Classifier.create_SVM(self.__session)
        
        node1 = mp.kernels.ReducedSumKernel.add_to_graph(self.__graph, inTensor, virtual_tensors[0], init_input=init_tensor, init_labels=labels)
        node2 = mp.kernels.ClassifierKernel.add_to_graph(self.__graph, virtual_tensors[0], classifier, outTensor)
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()

        return outTensor.data 
    
class Misc16PipelineUnitTest():
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestMisc16PipelineExecution(self, input_data, init_data, labels, indices):    
        inTensor = mp.Tensor.create_from_data(self.__session, input_data)
        init_tensor = mp.Tensor.create_from_data(self.__session, init_data)
        labels = mp.Tensor.create_from_data(self.__session, labels)
        outTensor = mp.Tensor.create(self.__session, (4,))
        
        virtual_tensors = [
            mp.Tensor.create_virtual(self.__session)
        ]
        
        classifier = mp.Classifier.create_SVM(self.__session)
        
        node1 = mp.kernels.ExtractKernel.add_to_graph(self.__graph, inTensor, indices, virtual_tensors[0], init_input=init_tensor, init_labels=labels)
        node2 = mp.kernels.ClassifierKernel.add_to_graph(self.__graph, virtual_tensors[0], classifier, outTensor)
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()

        return outTensor.data 

def test_execute():
    np.random.seed(44)
    
    test_transpose_csp_graph()
    test_baseline_xdawn_graph()
    test_threshold_feature_selection_kernel_graph() 
    test_resample_xdawn_graph()
    test_feature_selection_classifier_graph()
    test_slope_classifier_graph()
    test_pad_classifier_graph()
    
    # padded_init = np.pad(init_data, pad_width=1, mode="constant", constant_values=0)
    # KernelExecutionUnitTest_Object = Misc6PipelineUnitTest()
    # raw_data = np.random.randn(2,2,2)
    # init_data = np.random.randn(2,2,2)
    # r = 0.001
    # res = KernelExecutionUnitTest_Object.TestMisc6PipelineExecution(raw_data, init_data)
    # padded_data = np.pad(raw_data, pad_width=1, mode="constant", constant_values=0)
    # padded_init = np.pad(init_data, pad_width=1, mode="constant", constant_values=0)
    # cov_data = pyriemann.utils.covariance.covariances(padded_data)
    # cov_data = (1-r)*cov_data + r*np.diag(np.ones(cov_data.shape[-1]))
    # cov_init = pyriemann.utils.covariance.covariances(padded_init)
    # cov_init = (1-r)*cov_init + r*np.diag(np.ones(cov_init.shape[-1]))
    # tangent_space = pyriemann.tangentspace.TangentSpace()
    # tangent_space.fit(cov_init)
    # output = tangent_space.transform(cov_data)
    # assert (res == output).all()
    
    test_riemann_potato_classifier_graph()
    test_concatenation_classifier_graph()
    test_riemann_distance_classifier_graph()
    test_riemann_mean_classifier_graph()
    test_epoch_classifier_graph()
    test_addition_classifier_graph()
    # test_reduced_sum_classifier_graph()
    test_extract_classifier_graph()

def test_transpose_csp_graph():
    """ Test passing init data to transpose kernel that will be passed downstream to csp kernel """
    raw_data = np.random.randn(10,10,10)
    init_data = np.random.randn(10,10,10)
    init_label_data = np.concatenate((np.zeros((5,)), np.ones((5,))))
    KernelExecutionUnitTest_Object = MiscPipelineUnitTest()
    res = KernelExecutionUnitTest_Object.TestMiscPipelineExecution(raw_data, init_data, init_label_data)
    transposed_data = np.transpose(raw_data, [0,1,2])
    transposed_init = np.transpose(init_data, [0,1,2])
    csp = mne.decoding.CSP(transform_into='csp_space')
    csp.fit(transposed_init, init_label_data)
    expected_output = csp.transform(transposed_data)
    assert (res == expected_output).all()
    del KernelExecutionUnitTest_Object
    
def test_baseline_xdawn_graph():
    """ Test passing init data to baseline kernel that will be passed downstream to xdawn kernel """
    KernelExecutionUnitTest_Object = Misc2PipelineUnitTest()
    raw_data = np.random.randn(10,10,10)
    init_data = np.random.randn(10,10,10)
    init_label_data = np.concatenate((np.zeros((5,)), np.ones((5,)))) 
    res = KernelExecutionUnitTest_Object.TestMisc2PipelineExecution(raw_data, init_data, init_label_data)
    baseline = np.mean(raw_data[..., 0:10],
                        axis=-1,
                        keepdims=True)
    expected_output = raw_data - baseline
    baseline = np.mean(init_data[..., 0:10],
                        axis=-1,
                        keepdims=True)
    init_after_baseline = init_data - baseline
    xdawn_estimator = XdawnCovariances()
    xdawn_estimator.fit(init_after_baseline, init_label_data)
    expected_output = xdawn_estimator.transform(expected_output)
    assert (res == expected_output).all()
    del KernelExecutionUnitTest_Object

def test_threshold_feature_selection_kernel_graph():
    """ Test passing init data to threshold kernel"""
    KernelExecutionUnitTest_Object = Misc3PipelineUnitTest()
    thresh_val=0.5
    raw_data = np.random.randn(10,10)
    init_data = np.random.randn(10,10)
    init_thresholds = np.random.randn(10,1)
    init_label_data = np.concatenate((np.zeros((5,)), np.ones((5,))))
    Nfeats = 2
    res = KernelExecutionUnitTest_Object.TestMisc3PipelineExecution(raw_data, thresh_val, (init_data, init_thresholds), init_label_data, Nfeats)
    data_after_thresh = raw_data > thresh_val
    init_data_after_thresh = init_data > init_thresholds
    feat_selector = SelectKBest(k=Nfeats).fit(init_data_after_thresh, init_label_data)
    expected_output = feat_selector.transform(data_after_thresh)
    assert (res == expected_output).all()
    del KernelExecutionUnitTest_Object
    
def test_resample_xdawn_graph():
    """ Test passing init data to resample kernel that will be passed downstream to xdawn kernel """
    KernelExecutionUnitTest_Object = Misc4PipelineUnitTest()
    factor = 1
    raw_data = np.random.randn(10,10,10)
    init_data = np.random.randn(10,10,10)
    init_label_data = np.concatenate((np.zeros((5,)), np.ones((5,)))) 
    res = KernelExecutionUnitTest_Object.TestMisc4PipelineExecution(raw_data, init_data, init_label_data, factor)
    resampled_data = signal.resample(raw_data, np.ceil(raw_data.shape[1] * factor).astype(int),axis=1)
    resampled_init = signal.resample(init_data, np.ceil(init_data.shape[1] * factor).astype(int),axis=1)
    xdawn_estimator = XdawnCovariances()
    xdawn_estimator.fit(resampled_init, init_label_data)
    expected_output = xdawn_estimator.transform(resampled_data)
    assert (res == expected_output).all()
    del KernelExecutionUnitTest_Object
    
def test_feature_selection_classifier_graph():
    """ Test passing init data to feature selection kernel that will be passed downstream to classifier kernel """
    KernelExecutionUnitTest_Object = Misc5PipelineUnitTest()
    raw_data = np.random.randn(50,26)
    init_data = np.random.randn(50,26)
    init_labels_data = np.random.randint(0,2, (50,))
    res = KernelExecutionUnitTest_Object.TestMisc5PipelineExecution(raw_data, init_data, init_labels_data)
    model = SelectKBest(k=10)
    model.fit(init_data, init_labels_data)
    init_data_after_feature_sel = model.transform(init_data)
    rwa_data_after_feature_sel = model.transform(raw_data)
    classifier = LinearDiscriminantAnalysis(shrinkage='auto', solver='lsqr')
    classifier.fit(init_data_after_feature_sel, init_labels_data)
    expected_output = classifier.predict(rwa_data_after_feature_sel)
    assert (res == expected_output).all()
    del KernelExecutionUnitTest_Object
    
def test_slope_classifier_graph():
    """ Test passing init data to slope kernel that will be passed downstream to classifier kernel """
    KernelExecutionUnitTest_Object = Misc7PipelineUnitTest()
    raw_data = np.random.randn(50,26)
    init_data = np.random.randn(50,26)
    init_labels_data = np.random.randint(0,2, (50,))
    res = KernelExecutionUnitTest_Object.TestMisc7PipelineExecution(raw_data, init_data, init_labels_data)
    axis = -1
    Fs = 1
    Ns = raw_data.shape[axis]
    X = np.linspace(0, Ns/Fs, Ns)
    Y = np.moveaxis(raw_data, axis, -1)
    X -= np.mean(X)
    Y -= np.mean(Y, axis=-1, keepdims=True)
    raw_data_after_slope = np.expand_dims(Y.dot(X) / X.dot(X), axis=-1)
    Ns = init_data.shape[axis]
    X = np.linspace(0, Ns/Fs, Ns)
    Y = np.moveaxis(init_data, axis, -1)
    X -= np.mean(X)
    Y -= np.mean(Y, axis=-1, keepdims=True)
    init_data_after_slope = np.expand_dims(Y.dot(X) / X.dot(X), axis=-1)
    classifier = LinearDiscriminantAnalysis(shrinkage='auto', solver='lsqr')
    classifier.fit(init_data_after_slope, init_labels_data)
    expected_output = classifier.predict(raw_data_after_slope)
    assert (res == expected_output).all()
    del KernelExecutionUnitTest_Object
    
def test_pad_classifier_graph():
    """ Test passing init data to pad kernel that will be passed downstream to classifier kernel """
    KernelExecutionUnitTest_Object = Misc8PipelineUnitTest()
    raw_data = np.random.randn(50,26)
    init_data = np.random.randn(50,26)
    init_labels_data = np.random.randint(0,2, (52,))
    res = KernelExecutionUnitTest_Object.TestMisc8PipelineExecution(raw_data, init_data, init_labels_data)
    padded_data = np.pad(raw_data, pad_width=1, mode="constant", constant_values=0)
    padded_init = np.pad(init_data, pad_width=1, mode="constant", constant_values=0)
    classifier = LinearDiscriminantAnalysis(shrinkage='auto', solver='lsqr')
    classifier.fit(padded_init, init_labels_data)
    expected_output = classifier.predict(padded_data)
    assert (res == expected_output).all()
    del KernelExecutionUnitTest_Object
    
def test_riemann_potato_classifier_graph():
    """ Test passing init data to riemann potato kernel that will be passed downstream to classifier kernel """
    KernelExecutionUnitTest_Object = Misc9PipelineUnitTest()
    raw_data = np.random.randn(16,10,10)
    init_data = np.random.randn(16,10,10)
    r = 0.001
    raw_data = pyriemann.utils.covariance.covariances(raw_data)
    raw_data = (1-r)*raw_data + r*np.diag(np.ones(raw_data.shape[-1]))
    init_data = pyriemann.utils.covariance.covariances(init_data)
    init_data = (1-r)*init_data + r*np.diag(np.ones(init_data.shape[-1]))  
    init_labels_data = np.concatenate((np.zeros((2,)), np.ones((2,))))
    res = KernelExecutionUnitTest_Object.TestMisc9PipelineExecution(raw_data, init_data, init_labels_data)
    potato_filter = pyriemann.clustering.Potato()
    potato_filter.fit(init_data)
    init_after_potato = potato_filter.predict(init_data)
    data_after_potato = potato_filter.predict(raw_data)
    reshaped_init = np.reshape(init_after_potato, (4,4))
    reshaped_data = np.reshape(data_after_potato, (4,4))
    classifier = LinearDiscriminantAnalysis(shrinkage='auto', solver='lsqr')
    classifier.fit(reshaped_init, init_labels_data)
    expected_output = classifier.predict(reshaped_data)
    assert (res == expected_output).all()
    del KernelExecutionUnitTest_Object
    
def test_concatenation_classifier_graph():
    """ Test passing init data to concatenation kernel that will be passed downstream to classifier kernel """
    KernelExecutionUnitTest_Object = Misc10PipelineUnitTest()
    raw_data1 = np.random.randn(4,2)
    raw_data2 = np.random.randn(4,2)
    init_data1 = np.random.randn(4,2)
    init_data2 = np.random.randn(4,2)
    axis = 1
    init_labels_data = np.concatenate((np.zeros((2,)), np.ones((2,))))
    res = KernelExecutionUnitTest_Object.TestMisc10PipelineExecution(raw_data1, raw_data2, init_data1, init_data2, init_labels_data, axis)
    concatenated_data = np.concatenate([raw_data1, raw_data2], axis=1)
    concatenated_init = np.concatenate([init_data1, init_data2], axis=1)
    classifier = LinearDiscriminantAnalysis(shrinkage='auto', solver='lsqr')
    classifier.fit(concatenated_init, init_labels_data)
    expected_output = classifier.predict(concatenated_data)
    assert (res == expected_output).all()
    del KernelExecutionUnitTest_Object
    
def test_riemann_distance_classifier_graph():
    """ Test passing init data to riemann distance kernel that will be passed downstream to classifier kernel """
    KernelExecutionUnitTest_Object = Misc11PipelineUnitTest()
    raw_data1 = np.random.randn(4,4,4)
    raw_data2 = np.random.randn(4,4,4)
    init_data1 = np.random.randn(4,4,4)
    init_data2 = np.random.randn(4,4,4)
    raw_data1 = pyriemann.utils.covariance.covariances(raw_data1)
    raw_data2 = pyriemann.utils.covariance.covariances(raw_data2)
    init_data1 = pyriemann.utils.covariance.covariances(init_data1)
    init_data2 = pyriemann.utils.covariance.covariances(init_data2)
    r = 0.001
    raw_data1 = (1-r)*raw_data1 + r*np.diag(np.ones(raw_data1.shape[-1]))
    raw_data2 = (1-r)*raw_data2 + r*np.diag(np.ones(raw_data2.shape[-1]))
    init_data1 = (1-r)*init_data1 + r*np.diag(np.ones(init_data1.shape[-1]))
    init_data2 = (1-r)*init_data2 + r*np.diag(np.ones(init_data2.shape[-1]))
    init_labels_data = np.concatenate((np.zeros((2,)), np.ones((2,))))
    res = KernelExecutionUnitTest_Object.TestMisc11PipelineExecution(raw_data1, raw_data2, init_data1, init_data2, init_labels_data)
    data_after_riem_dist = np.empty((4,4))
    init_after_riem_dist = np.empty((4,4))
    for i in range(4):
        # extract the ith element from inA
        x = raw_data1[i,:,:]
        x_i = init_data1[i,:,:]
        for j in range(4):
            # extract the jth element from inB
            y = raw_data2[j,:,:]
            y_i = init_data2[j,:,:]
            data_after_riem_dist[i,j] = pyriemann.utils.distance.distance_riemann(x,y)
            init_after_riem_dist[i,j] = pyriemann.utils.distance.distance_riemann(x_i,y_i)
    classifier = LogisticRegression()
    classifier.fit(init_after_riem_dist, init_labels_data)
    expected_output = classifier.predict(data_after_riem_dist)
    assert (res == expected_output).all()
    del KernelExecutionUnitTest_Object
    
def test_riemann_mean_classifier_graph():
    """ Test passing init data to riemann mean kernel that will be passed downstream to classifier kernel """
    KernelExecutionUnitTest_Object = Misc12PipelineUnitTest()
    init_data = np.random.randn(10,10,10)
    raw_data = np.random.randn(10,10,10)
    raw_data = pyriemann.utils.covariance.covariances(raw_data)
    init_data = pyriemann.utils.covariance.covariances(init_data)
    r = 0.001
    raw_data = (1-r)*raw_data + r*np.diag(np.ones(raw_data.shape[-1]))
    init_data = (1-r)*init_data + r*np.diag(np.ones(init_data.shape[-1]))
    init_labels_data = np.concatenate((np.zeros((5,)), np.ones((5,))))
    res = KernelExecutionUnitTest_Object.TestMisc12PipelineExecution(raw_data, init_data, init_labels_data)
    data_after_riem_mean = pyriemann.utils.mean.mean_riemann(raw_data)
    init_after_riem_mean = pyriemann.utils.mean.mean_riemann(init_data)
    classifier = SVC()
    classifier.fit(init_after_riem_mean, init_labels_data)
    expected_output = classifier.predict(data_after_riem_mean)
    assert (res == expected_output).all()
    del KernelExecutionUnitTest_Object
    
def test_epoch_classifier_graph():
    """ Test passing init data to epcoh kernel that will be passed downstream to classifier kernel """
    raw_data = np.random.randn(2,2,2)
    init_data = np.random.randn(2,2,2)
    labels = np.concatenate((np.zeros((2,)), np.ones((2,))))
    epoch_length = 2
    epoch_stride = 1
    axis = -1
    KernelExecutionUnitTestObject = Misc13PipelineUnitTest()
    res = KernelExecutionUnitTestObject.TestMisc13PipelineExecution(raw_data, init_data, labels, epoch_stride, epoch_length, axis)
    
    # manually epoch data
    epoched_data = np.zeros((2,2,1,2))
    epoched_init = np.zeros((2,2,1,2))

    # manually epoch data
    epoched_data = np.zeros((2,2,1,2))
    src_slc = [slice(None)] * len(raw_data.shape)
    dst_slc = [slice(None)] * len(epoched_data.shape)
    Nepochs = 1 #int(res[0].shape[2] - 2) // 1 + 1
    for i_e in range(Nepochs):
        src_slc[2] = slice(i_e*1,
                                i_e*1 + 2)
        dst_slc[2] = i_e
        epoched_data[tuple(dst_slc)] = raw_data[tuple(src_slc)]
        
    epoched_init = np.zeros((2,2,1,2))
    src_slc = [slice(None)] * len(init_data.shape)
    dst_slc = [slice(None)] * len(epoched_init.shape)
    Nepochs = 1 #int(res[0].shape[2] - 2) // 1 + 1
    for i_e in range(Nepochs):
        src_slc[2] = slice(i_e*1,
                                i_e*1 + 2)
        dst_slc[2] = i_e
        epoched_init[tuple(dst_slc)] = init_data[tuple(src_slc)]

    reshaped_data = np.reshape(epoched_data, (4,2))
    reshaped_init = np.reshape(epoched_init, (4,2))
    classifier = SVC()
    classifier.fit(reshaped_init, labels)
    expected_output = classifier.predict(reshaped_data)
    assert (res == expected_output).all()
    del KernelExecutionUnitTestObject
    
def test_addition_classifier_graph():
    """ Test passing init data to addition kernel that will be passed downstream to classifier kernel """
    KernelExecutionUnitTest_Object = Misc14PipelineUnitTest()
    init_data = np.random.randn(10,10)
    raw_data = np.random.randn(10,10)
    init_labels_data = np.concatenate((np.zeros((5,)), np.ones((5,))))
    
    res = KernelExecutionUnitTest_Object.TestMisc14PipelineExecution(raw_data, init_data, init_labels_data)
    data_after_addition = raw_data + raw_data
    init_after_addition = init_data + init_data
    classifier = SVC()
    classifier.fit(init_after_addition, init_labels_data)
    expected_output = classifier.predict(data_after_addition)
    assert (res == expected_output).all()
    del KernelExecutionUnitTest_Object
    
# def test_reduced_sum_classifier_graph():
#     """ Test passing init data to resample kernel that will be passed downstream to classifier kernel """
#     KernelExecutionUnitTest_Object = Misc15PipelineUnitTest()
#     init_data = np.random.randn(10,10)
#     raw_data = np.random.randn(10,10)
#     init_labels_data = np.concatenate((np.zeros((5,)), np.ones((5,))))
    
#     res = KernelExecutionUnitTest_Object.TestMisc15PipelineExecution(raw_data, init_data, init_labels_data)
#     init_after_reduced_sum = np.sum(init_data, axis = None)
#     data_after_reduced_sum = np.sum(raw_data, axis = None)
#     classifier = SVC()
#     classifier.fit(init_after_reduced_sum, init_labels_data)
#     expected_output = classifier.predict(data_after_reduced_sum)
#     assert (res == expected_output).all()
#     del KernelExecutionUnitTest_Object

def test_extract_classifier_graph():
    """ Test passing init data to addition kernel that will be passed downstream to classifier kernel """
    KernelExecutionUnitTest_Object = Misc16PipelineUnitTest()
    init_data = np.random.randn(10,10)
    raw_data = np.random.randn(10,10)
    init_labels_data = np.concatenate((np.zeros((2,)), np.ones((2,))))
    indices = [slice(4), slice(None)] 
    
    res = KernelExecutionUnitTest_Object.TestMisc16PipelineExecution(raw_data, init_data, init_labels_data, indices)
    data_after_extract =  raw_data[0:4:]
    init_after_extract = init_data[0:4,:]
    classifier = SVC()
    classifier.fit(init_after_extract, init_labels_data)
    expected_output = classifier.predict(data_after_extract)
    assert (res == expected_output).all()
    del KernelExecutionUnitTest_Object
    
test_execute()