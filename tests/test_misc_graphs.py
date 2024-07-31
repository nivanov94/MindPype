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
        node1 = mp.kernels.PadKernel.add_to_graph(self.__graph, inTensor, virtual_tensors[0], pad_width=1, init_input=init_data, init_labels=init_labels)
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
        outTensor = mp.Tensor.create(self.__session, (inTensor.shape[0],))
        
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
    
def test_execute():
    np.random.seed(44)
    raw_data = np.random.randn(10,10,10)
    init_data = np.random.randn(10,10,10)
    init_label_data = np.concatenate((np.zeros((6,)), np.ones((6,))))
    factor = 1
    # KernelExecutionUnitTest_Object = MiscPipelineUnitTest()
    # res = KernelExecutionUnitTest_Object.TestMiscPipelineExecution(raw_data, init_data, init_label_data)
    
    # expected_output = 1
    # assert(res == expected_output).all()

    # del KernelExecutionUnitTest_Object
    
    init_label_data = np.concatenate((np.zeros((5,)), np.ones((5,)))) 
    KernelExecutionUnitTest_Object = Misc2PipelineUnitTest()
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
    
    KernelExecutionUnitTest_Object = Misc4PipelineUnitTest()
    res = KernelExecutionUnitTest_Object.TestMisc4PipelineExecution(raw_data, init_data, init_label_data, factor)
    output1 = signal.resample(raw_data, np.ceil(raw_data.shape[1] * factor).astype(int),axis=1)
    resampled_init = signal.resample(init_data, np.ceil(init_data.shape[1] * factor).astype(int),axis=1)
    xdawn_estimator = XdawnCovariances()
    xdawn_estimator.fit(resampled_init, init_label_data)
    expected_output = xdawn_estimator.transform(output1)
    assert (res == expected_output).all()
    del KernelExecutionUnitTest_Object
    
    raw_data = np.random.randn(50,26)
    init_data = np.random.randn(50,26)
    init_labels_data = np.random.randint(0,2, (50,))
    KernelExecutionUnitTest_Object = Misc5PipelineUnitTest()
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
    
    KernelExecutionUnitTest_Object = Misc7PipelineUnitTest()
    res = KernelExecutionUnitTest_Object.TestMisc7PipelineExecution(raw_data, init_data, init_labels_data)
    # manual calculation
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
    
    KernelExecutionUnitTest_Object = Misc8PipelineUnitTest()
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

test_execute()