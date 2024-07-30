import mindpype as mp
import numpy as np
from pyriemann.estimation import XdawnCovariances

class MiscPipelineUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestMiscPipelineExecution(self, input_data, init_data):    
        init_data = mp.Tensor.create_from_data(self.__session, init_data)
        
        inTensor = mp.Tensor.create_from_data(self.__session, input_data)
        # outTensor = mp.Scalar.create_from_value(self.__session,-1)
        outTensor = mp.Tensor.create(self.__session, shape=(inTensor.shape))
        
        virtual_tensors = [
            mp.Tensor.create_virtual(self.__session),
            mp.Tensor.create_virtual(self.__session),
            mp.Tensor.create_virtual(self.__session),
            mp.Tensor.create_virtual(self.__session)
        ]
        node1 = mp.kernels.PadKernel.add_to_graph(self.__graph, inTensor, virtual_tensors[0], pad_width=1, init_input=init_data)
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
        
    
def test_execute():
    np.random.seed(44)
    raw_data = np.random.randn(10,10,10)
    init_data = np.random.randn(10,10,10)
    init_label_data = np.concatenate((np.zeros((5,)), np.ones((5,))))

    # KernelExecutionUnitTest_Object = MiscPipelineUnitTest()
    # res = KernelExecutionUnitTest_Object.TestMiscPipelineExecution(raw_data, init_data)
    
    # expected_output = 1
    # assert(res == expected_output).all()

    # del KernelExecutionUnitTest_Object
    
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
test_execute()