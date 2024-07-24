import mindpype as mp
import sys, os
import numpy as np
import mne

class CSPKernelCreationUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestCSPKernelCreation(self):
        inTensor = mp.Tensor.create(self.__session, (50,26,26))
        outTensor = mp.Tensor.create(self.__session,(50,26,26))
        initialization_tensor = mp.Tensor.create(self.__session, (50,26,26))
        labels = mp.Tensor.create(self.__session, (50,))
        node = mp.kernels.CommonSpatialPatternKernel.add_to_graph(self.__graph,inTensor,outTensor,initialization_tensor,labels)
        return node.mp_type
    
class CSPKernelExecutionUnitTest:

    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestCSPKernelExecution(self):
        np.random.seed(44)
        raw_data = np.random.randn(50,26,26)
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session,(50,4,26))
        init_data = np.random.randn(50,26,26)
        initialization_tensor = mp.Tensor.create_from_data(self.__session, init_data)
        init_labels_data = np.random.randint(0,2, (50,))
        labels = mp.Tensor.create_from_data(self.__session, init_labels_data)
        node = mp.kernels.CommonSpatialPatternKernel.add_to_graph(self.__graph,inTensor,outTensor,initialization_tensor,labels)
        
        sys.stdout = open(os.devnull, 'w')
        self.__graph.verify()
        self.__graph.initialize(initialization_tensor, labels)
        self.__graph.execute()
        sys.stdout = sys.__stdout__

        return (init_data, init_labels_data, raw_data, outTensor.data)

def test_create():
    KernelUnitTest_Object = CSPKernelCreationUnitTest()
    assert KernelUnitTest_Object.TestCSPKernelCreation() == mp.MPEnums.NODE
    del KernelUnitTest_Object


def test_execute():
    KernelExecutionUnitTest_Object = CSPKernelExecutionUnitTest()
    res = KernelExecutionUnitTest_Object.TestCSPKernelExecution()
    csp = mne.decoding.CSP(transform_into='csp_space')
    csp.fit(res[0],res[1])
    expected_output = csp.transform(res[2])
    assert (res[3] == expected_output).all()
    del KernelExecutionUnitTest_Object
