import mindpype as mp
import sys, os
import numpy as np
from scipy import signal

class FilterKernelCreationUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestFilterKernelCreation(self):
        inTensor = mp.Tensor.create(self.__session, (2,2))
        outTensor = mp.Tensor.create(self.__session, (2,2))
        
        Fs = 128
        l_freq = 1
        h_freq = 40
        Nc = 14
        Ns = int(Fs * 10)
        f = mp.Filter.create_fir(self.__session, Fs, l_freq, h_freq, method='fir', phase='minimum')
        
        node = mp.kernels.FilterKernel.add_to_graph(self.__graph,inTensor,f,outTensor,axis=0)
        return node.mp_type
    
class FilterKernelExecutionUnitTest:

    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestFilterKernelExecution(self):
        np.random.seed(44)
        raw_data = np.random.randint(-10,10, size=(2,2))
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, (2,2))
        
        Fs = 128
        l_freq = 1
        h_freq = 40
        Nc = 14
        Ns = int(Fs * 10)
        f = mp.Filter.create_fir(self.__session, Fs, l_freq, h_freq, method='fir', phase='minimum')
        
        tensor_test_node = mp.kernels.FilterKernel.add_to_graph(self.__graph,inTensor,f,outTensor,axis=0)

        sys.stdout = open(os.devnull, 'w')
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        sys.stdout = sys.__stdout__

        return (raw_data, f, outTensor.data)

def test_create():
    KernelUnitTest_Object = FilterKernelCreationUnitTest()
    assert KernelUnitTest_Object.TestFilterKernelCreation() == mp.MPEnums.NODE
    del KernelUnitTest_Object

def test_execute():
    KernelExecutionUnitTest_Object = FilterKernelExecutionUnitTest()
    res = KernelExecutionUnitTest_Object.TestFilterKernelExecution()
    
    expected_output = signal.lfilter(res[1].coeffs['fir'],[1],res[0],axis=0)
    assert (res[2] == expected_output).all()
    del KernelExecutionUnitTest_Object