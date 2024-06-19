import mindpype as mp
import sys, os
import numpy as np
from scipy import signal

# class BaFilterKernelCreationUnitTest:
#     def __init__(self):
#         self.__session = mp.Session.create()
#         self.__graph = mp.Graph.create(self.__session)

#     def TestBaFilterKernelCreation(self):
#         inTensor = mp.Tensor.create(self.__session, (2,2))
#         outTensor = mp.Tensor.create(self.__session, (2,2))
        
#         Fs = 128
#         l_freq = 1
#         h_freq = 40
#         Nc = 14
#         Ns = int(Fs * 10)
#         f = mp.Filter.create_bessel(self.__session, Fs, l_freq, h_freq, method='ba', phase='minimum')
        
#         node = mp.kernels.FilterKernel.add_to_graph(self.__graph,inTensor,f,outTensor,axis=0)
#         return node.mp_type
    
# class BaFilterKernelExecutionUnitTest:

#     def __init__(self):
#         self.__session = mp.Session.create()
#         self.__graph = mp.Graph.create(self.__session)

#     def TestBaFilterKernelExecution(self):
#         np.random.seed(44)
#         raw_data = np.random.randint(-10,10, size=(2,2))
#         inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
#         outTensor = mp.Tensor.create(self.__session, (2,2))
        
#         Fs = 128
#         l_freq = 1
#         h_freq = 40
#         Nc = 14
#         Ns = int(Fs * 10)
#         f = mp.Filter.create_bessel(self.__session, Fs, l_freq, h_freq, method='ba', phase='minimum')
        
#         tensor_test_node = mp.kernels.FilterKernel.add_to_graph(self.__graph,inTensor,f,outTensor,axis=0)

#         sys.stdout = open(os.devnull, 'w')
#         self.__graph.verify()
#         self.__graph.initialize()
#         self.__graph.execute()
#         sys.stdout = sys.__stdout__

#         return (raw_data, f, outTensor.data)

class FirFilterKernelCreationUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestFirFilterKernelCreation(self):
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
    
class FirFilterKernelExecutionUnitTest:

    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestFirFilterKernelExecution(self):
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

class SosFilterKernelCreationUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestSosFilterKernelCreation(self):
        inTensor = mp.Tensor.create(self.__session, (2,2))
        outTensor = mp.Tensor.create(self.__session, (2,2))
        
        Fs = 250
        order = 4
        bandpass = (8,35) # in Hz
        f = mp.Filter.create_butter(self.__session,order,bandpass,btype='bandpass',fs=Fs,implementation='sos')
        
        node = mp.kernels.FilterKernel.add_to_graph(self.__graph,inTensor,f,outTensor,axis=0)
        return node.mp_type
    
class SosFilterKernelExecutionUnitTest:

    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestSosFilterKernelExecution(self):
        np.random.seed(44)
        raw_data = np.random.randint(-10,10, size=(2,2))
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, (2,2))
        
        Fs = 250
        order = 4
        bandpass = (8,35) # in Hz
        f = mp.Filter.create_butter(self.__session,order,bandpass,btype='bandpass',fs=Fs,implementation='sos')
        tensor_test_node = mp.kernels.FilterKernel.add_to_graph(self.__graph,inTensor,f,outTensor,axis=0)

        sys.stdout = open(os.devnull, 'w')
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        sys.stdout = sys.__stdout__

        return (raw_data, f, outTensor.data)
    
class SosFiltFiltKernelCreationUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestSosFiltFiltKernelCreation(self):
        inTensor = mp.Tensor.create(self.__session, (2,2))
        outTensor = mp.Tensor.create(self.__session, (2,2))
        
        Fs = 250
        order = 4
        bandpass = (8,35) # in Hz
        f = mp.Filter.create_butter(self.__session,order,bandpass,btype='bandpass',fs=Fs,implementation='sos')
        
        node = mp.kernels.FiltFiltKernel.add_to_graph(self.__graph,inTensor,f,outTensor,axis=0)
        return node.mp_type
    
class SosFiltFiltKernelExecutionUnitTest:

    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestSosFiltFiltKernelExecution(self):
        np.random.seed(44)
        raw_data = np.random.randint(-10,10, size=(2,2))
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, (2,2))
        
        Fs = 250
        order = 4
        bandpass = (8,35) # in Hz
        f = mp.Filter.create_butter(self.__session,order,bandpass,btype='bandpass',fs=Fs,implementation='sos')
        
        tensor_test_node = mp.kernels.FiltFiltKernel.add_to_graph(self.__graph,inTensor,f,outTensor,axis=0)

        sys.stdout = open(os.devnull, 'w')
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        sys.stdout = sys.__stdout__

        return (raw_data, f, outTensor.data)


def test_create():
    # KernelUnitTest_Object = BaFilterKernelCreationUnitTest()
    # assert KernelUnitTest_Object.TestBaFilterKernelCreation() == mp.MPEnums.NODE
    # del KernelUnitTest_Object
    
    KernelUnitTest_Object = FirFilterKernelCreationUnitTest()
    assert KernelUnitTest_Object.TestFirFilterKernelCreation() == mp.MPEnums.NODE
    del KernelUnitTest_Object
    
    KernelUnitTest_Object = SosFilterKernelCreationUnitTest()
    assert KernelUnitTest_Object.TestSosFilterKernelCreation() == mp.MPEnums.NODE
    del KernelUnitTest_Object
    
    KernelUnitTest_Object = SosFiltFiltKernelCreationUnitTest()
    assert KernelUnitTest_Object.TestSosFiltFiltKernelCreation() == mp.MPEnums.NODE
    del KernelUnitTest_Object

def test_execute():
    # KernelExecutionUnitTest_Object = BaFilterKernelExecutionUnitTest()
    # res = KernelExecutionUnitTest_Object.TestBaFilterKernelExecution()
    # expected_output = signal.lfilter(res[1].coeffs['b'],res[1].coeffs['a'],res[0],axis=0)
    # assert (res[2] == expected_output).all()
    # del KernelExecutionUnitTest_Object
    
    KernelExecutionUnitTest_Object = FirFilterKernelExecutionUnitTest()
    res = KernelExecutionUnitTest_Object.TestFirFilterKernelExecution()
    expected_output = signal.lfilter(res[1].coeffs['fir'],[1],res[0],axis=0)
    assert (res[2] == expected_output).all()
    del KernelExecutionUnitTest_Object
    
    KernelExecutionUnitTest_Object = SosFilterKernelExecutionUnitTest()
    res = KernelExecutionUnitTest_Object.TestSosFilterKernelExecution()
    sos = signal.butter(4,(8,35),btype='bandpass',output='sos',fs=250)
    filtered_data = signal.sosfilt(sos,res[0],axis=0)
    assert (res[2] == filtered_data).all()
    del KernelExecutionUnitTest_Object
    
    # KernelExecutionUnitTest_Object = SosFiltFiltKernelExecutionUnitTest()
    # res = KernelExecutionUnitTest_Object.TestSosFiltFiltKernelExecution()
    # expected_output = signal.sosfiltfilt(res[1].coeffs['sos'], res[0],axis=0)
    # assert (res[2] == expected_output).all()
    # del KernelExecutionUnitTest_Object
   