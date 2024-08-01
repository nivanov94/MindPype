import pyriemann.utils
import pyriemann.utils.covariance
import mindpype as mp
import numpy as np
from pyriemann.estimation import XdawnCovariances
from scipy import signal
from sklearn.feature_selection import SelectKBest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pyriemann

class MiscPipelineUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestMiscPipelineExecution(self, input_data, init_data, init_label_data):    
        init_data = mp.Tensor.create_from_data(self.__session, init_data)
        init_labels = mp.Tensor.create_from_data(self.__session, init_label_data)
        inTensor = mp.Tensor.create_from_data(self.__session, input_data)
        outTensor = mp.Tensor.create(self.__session, shape=(inTensor.shape))
        
        virtual_tensors = [
            mp.Tensor.create_virtual(self.__session),
            mp.Tensor.create_virtual(self.__session),
            mp.Tensor.create_virtual(self.__session),
            mp.Tensor.create_virtual(self.__session)
        ]
        # node1 = mp.kernels.TransposeKernel.add_to_graph(self.__graph, inTensor, virtual_tensors[0], axes=[0,1], init_input=init_data)
        node2 = mp.kernels.CommonSpatialPatternKernel.add_to_graph(self.__graph, virtual_tensors[0], virtual_tensors[1])

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

    def TestMisc3PipelineExecution(self, input_data, thresh, init_data, init_label_data):    
        init_data = mp.Tensor.create_from_data(self.__session, init_data)
        init_labels = mp.Tensor.create_from_data(self.__session, init_label_data)
        inTensor = mp.Tensor.create_from_data(self.__session, input_data)
        thresh = mp.Scalar.create_from_value(self.__session, thresh)
        outTensor = mp.Tensor.create(self.__session, (10,10,10))
        
        virtual_tensors = [
            mp.Tensor.create_virtual(self.__session),
        ]
        node1 = mp.kernels.ThresholdKernel.add_to_graph(self.__graph,inTensor,virtual_tensors[0],thresh=thresh, init_input=init_data, init_labels=init_labels)
        node2 = mp.kernels.XDawnCovarianceKernel.add_to_graph(self.__graph, virtual_tensors[0], outTensor)

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
        
        classifier = mp.Classifier.create_LDA(self.__session, shrinkage='auto', solver='lsqr')
        
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
        
        classifier = mp.Classifier.create_LDA(self.__session, shrinkage='auto', solver='lsqr')
        
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
        
        classifier = mp.Classifier.create_LDA(self.__session, shrinkage='auto', solver='lsqr')
        
        node1 = mp.kernels.RiemannMeanKernel.add_to_graph(self.__graph, inTensor, virtual_tensors[0], init_input=init_data, init_labels=init_labels)
        node2 = mp.kernels.ClassifierKernel.add_to_graph(self.__graph, virtual_tensors[0], classifier, outTensor)
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()

        return outTensor.data      

def test_execute():
    np.random.seed(44)
    raw_data = np.random.randn(10,10,10)
    init_data = np.random.randn(10,10,10)
    # init_label_data = np.concatenate((np.zeros((6,)), np.ones((6,))))
    # factor = 1
    # KernelExecutionUnitTest_Object = MiscPipelineUnitTest()
    # res = KernelExecutionUnitTest_Object.TestMiscPipelineExecution(raw_data, init_data, init_label_data)
    # expected_output = 1
    # assert(res == expected_output).all()
    # del KernelExecutionUnitTest_Object
    
    # test pipeline with baseline and xDawn nodes
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
    
    # KernelExecutionUnitTest_Object = Misc3PipelineUnitTest()
    # thresh_val=1
    # res = KernelExecutionUnitTest_Object.TestMisc3PipelineExecution(raw_data, thresh_val, init_data, init_label_data)
    # data_after_thresh = raw_data > thresh_val
    # init_after_thresh = init_data > thresh_val
    # xdawn_estimator = XdawnCovariances()
    # xdawn_estimator.fit(init_after_thresh, init_label_data)
    # expected_output = xdawn_estimator.transform(data_after_thresh)
    # assert (res == expected_output).all()
    # del KernelExecutionUnitTest_Object
    
    # test pipeline with resample and xDawn nodes
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
    
    # test pipeline with feature selection and classifier nodes
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
    
    # test pipeline with slope and classifier nodes
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
    
    # test pipeline with pad and classifier nodes
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
    
    # test pipeline with riemann potato, reshape, and classifier nodes
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
    
    # test pipeline with concatenation and classifier nodes
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
    
    # test pipeline with riemann distance and classifier pipeline
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
    classifier = LinearDiscriminantAnalysis(shrinkage='auto', solver='lsqr')
    classifier.fit(init_after_riem_dist, init_labels_data)
    expected_output = classifier.predict(data_after_riem_dist)
    assert (res == expected_output).all()
    del KernelExecutionUnitTest_Object
    
    KernelExecutionUnitTest_Object = Misc12PipelineUnitTest()
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
    classifier = LinearDiscriminantAnalysis(shrinkage='auto', solver='lsqr')
    classifier.fit(init_after_riem_mean, init_labels_data)
    expected_output = classifier.predict(data_after_riem_mean)
    assert (res == expected_output).all()
    del KernelExecutionUnitTest_Object
    
test_execute()