import mindpype as mp
import numpy as np
from scipy import signal
import pytest

class BaFilterKernelUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestBaFilterKernelExecution(self, raw_data):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, raw_data.shape)
        
        Fs = 250
        order = 4
        bandpass = (8,35) # in Hz
        f = mp.Filter.create_butter(self.__session,order,bandpass,btype='bandpass',fs=Fs,implementation='ba')
        
        tensor_test_node = mp.kernels.FilterKernel.add_to_graph(self.__graph,inTensor,f,outTensor,axis=0)
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        return (f, outTensor.data)

class FirFilterKernelUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestFirFilterKernelExecution(self, raw_data):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, raw_data.shape)
        
        Fs = 128
        l_freq = 1
        h_freq = 40
        Nc = 14
        Ns = int(Fs * 10)
        f = mp.Filter.create_fir(self.__session, Fs, l_freq, h_freq, method='fir', phase='minimum')
        
        tensor_test_node = mp.kernels.FilterKernel.add_to_graph(self.__graph,inTensor,f,outTensor,axis=0)
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        return (f, outTensor.data)

class SosFilterKernelUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestSosFilterKernelExecution(self, raw_data):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, raw_data.shape)
        
        Fs = 250
        order = 4
        bandpass = (8,35) # in Hz
        f = mp.Filter.create_butter(self.__session,order,bandpass,btype='bandpass',fs=Fs,implementation='sos')
        tensor_test_node = mp.kernels.FilterKernel.add_to_graph(self.__graph,inTensor,f,outTensor,axis=0)
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        return outTensor.data


class BaFiltFiltKernelUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestBaFiltFiltKernelExecution(self, raw_data):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, raw_data.shape)
        
        Fs = 250
        order = 4
        bandpass = (8,35) # in Hz
        f = mp.Filter.create_butter(self.__session,order,bandpass,btype='bandpass',fs=Fs,implementation='ba')
        
        tensor_test_node = mp.kernels.FiltFiltKernel.add_to_graph(self.__graph,inTensor,f,outTensor,axis=0)
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        return (f, outTensor.data)
    
class FirFiltFiltKernelUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestFirFilterKernelExecution(self, raw_data):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, raw_data.shape)
        
        Fs = 128
        l_freq = 1
        h_freq = 40
        Nc = 14
        Ns = int(Fs * 10)
        f = mp.Filter.create_fir(self.__session, Fs, l_freq, h_freq, method='fir', phase='minimum')
        
        tensor_test_node = mp.kernels.FiltFiltKernel.add_to_graph(self.__graph,inTensor,f,outTensor,axis=0)
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        return (f, outTensor.data)
    
class SosFiltFiltKernelUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestSosFiltFiltKernelExecution(self, raw_data):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, raw_data.shape)
        
        Fs = 250
        order = 4
        bandpass = (8,35) # in Hz
        f = mp.Filter.create_butter(self.__session,order,bandpass,btype='bandpass',fs=Fs,implementation='sos')
        
        tensor_test_node = mp.kernels.FiltFiltKernel.add_to_graph(self.__graph,inTensor,f,outTensor,axis=0)
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        return (f, outTensor.data)

class ChebyFilterUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestCheby1FilterExecution(self, raw_data):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, raw_data.shape)

        order = 4
        rp = 3
        Fs=250
        bandpass = (8,35) # in Hz
        filt1 = mp.Filter.create_cheby1(self.__session,order,rp,bandpass,btype='bandpass',implementation='ba',fs=Fs)
        filt2 = mp.Filter.create_cheby1(self.__session,order,rp,bandpass,btype='bandpass',implementation='sos',fs=Fs)
        
        virt = [
            mp.Tensor.create_virtual(self.__session)
        ]
        
        tensor_test_node = mp.kernels.FilterKernel.add_to_graph(self.__graph,inTensor,filt1,virt[0])
        tensor_test_node2 = mp.kernels.FilterKernel.add_to_graph(self.__graph,virt[0],filt2,outTensor)
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        return outTensor.data

    def TestCheby2FilterExecution(self, raw_data):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, raw_data.shape)

        order = 4
        rp = 3
        rs = 3
        Fs=250
        bandpass = (8,35) # in Hz
        filt1 = mp.Filter.create_cheby2(self.__session,order,rp,bandpass,btype='bandpass',implementation='ba',fs=Fs)
        filt2 = mp.Filter.create_cheby2(self.__session,order,rs,bandpass,btype='bandpass',implementation='sos',fs=Fs)
        
        virt = [
            mp.Tensor.create_virtual(self.__session)
        ]
        
        tensor_test_node = mp.kernels.FilterKernel.add_to_graph(self.__graph,inTensor,filt1,virt[0])
        tensor_test_node2 = mp.kernels.FilterKernel.add_to_graph(self.__graph,virt[0],filt2,outTensor)
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        return outTensor.data

def test_execute():
    np.random.seed(44)
    raw_data = np.random.randint(-10,10, size=(30,30))
    KernelExecutionUnitTest_Object = BaFilterKernelUnitTest()
    res = KernelExecutionUnitTest_Object.TestBaFilterKernelExecution(raw_data)
    expected_output = signal.lfilter(res[0].coeffs['b'],res[0].coeffs['a'],raw_data,axis=0)
    assert (res[1] == expected_output).all()
    del KernelExecutionUnitTest_Object
    
    KernelExecutionUnitTest_Object = FirFilterKernelUnitTest()
    res = KernelExecutionUnitTest_Object.TestFirFilterKernelExecution(raw_data)
    expected_output = signal.lfilter(res[0].coeffs['fir'],[1],raw_data,axis=0)
    assert (res[1] == expected_output).all()
    del KernelExecutionUnitTest_Object
    
    KernelExecutionUnitTest_Object = SosFilterKernelUnitTest()
    res = KernelExecutionUnitTest_Object.TestSosFilterKernelExecution(raw_data)
    sos = signal.butter(4,(8,35),btype='bandpass',output='sos',fs=250)
    filtered_data = signal.sosfilt(sos,raw_data,axis=0)
    assert (res == filtered_data).all()
    del KernelExecutionUnitTest_Object
    
    KernelExecutionUnitTest_Object = BaFiltFiltKernelUnitTest()
    res = KernelExecutionUnitTest_Object.TestBaFiltFiltKernelExecution(raw_data)
    expected_output = signal.filtfilt(res[0].coeffs['b'],res[0].coeffs['a'],raw_data,axis=0)
    assert (res[1] == expected_output).all()
    del KernelExecutionUnitTest_Object
    
    KernelExecutionUnitTest_Object = FirFiltFiltKernelUnitTest()
    with pytest.raises(TypeError):
        res = KernelExecutionUnitTest_Object.TestFirFilterKernelExecution(raw_data)
    del KernelExecutionUnitTest_Object
    
    KernelExecutionUnitTest_Object = SosFiltFiltKernelUnitTest()
    res = KernelExecutionUnitTest_Object.TestSosFiltFiltKernelExecution(raw_data)
    expected_output = signal.sosfiltfilt(res[0].coeffs['sos'], raw_data,axis=0)
    assert (res[1] == expected_output).all()
    del KernelExecutionUnitTest_Object
    
    order = 4
    Fs = 250
    rp = 3
    rs=3
    bandpass = (8,35)
    
    KernelExecutionUnitTest_Object = ChebyFilterUnitTest()
    res = KernelExecutionUnitTest_Object.TestCheby1FilterExecution(raw_data)
    b, a = signal.cheby1(order,rp,bandpass,btype='bandpass',output='ba',fs=Fs)
    filtered_data = signal.lfilter(b,a,raw_data,axis=1)
    
    sos = signal.cheby1(order,rp,bandpass,btype='bandpass',output='sos',fs=Fs)
    filtered_data = signal.sosfilt(sos, filtered_data)
    assert (res == filtered_data).all()
    
    res = KernelExecutionUnitTest_Object.TestCheby2FilterExecution(raw_data)
    b, a = signal.cheby2(order,rs,bandpass,btype='bandpass',output='ba',fs=Fs)
    filtered_data = signal.lfilter(b,a,raw_data,axis=1)
    
    sos = signal.cheby2(order,rs,bandpass,btype='bandpass',output='sos',fs=Fs)
    filtered_data = signal.sosfilt(sos, filtered_data)
    
    assert (res == filtered_data).all()
    
    del KernelExecutionUnitTest_Object
   