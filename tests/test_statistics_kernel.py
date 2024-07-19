import mindpype as mp
import numpy as np
from scipy.stats import norm, chi2, kurtosis, skew, zscore

class CDFKernelUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestCDFKernelExecution(self, raw_data):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, raw_data.shape)
        tensor_test_node = mp.kernels.CDFKernel.add_to_graph(self.__graph,inTensor,outTensor,dist='chi2',df=55)

        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()

        return outTensor.data

class CovarianceKernelUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestCovarianceKernelExecution(self, raw_data):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, raw_data.shape)
        regularization = 0
        tensor_test_node = mp.kernels.CovarianceKernel.add_to_graph(self.__graph,inTensor,outTensor, regularization)

        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()

        return outTensor.data
    
class MaxKernelUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestMaxKernelExecution(self, raw_data):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, (1,))
        tensor_test_node = mp.kernels.MaxKernel.add_to_graph(self.__graph,inTensor,outTensor)

        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()

        return outTensor.data
    
class MinKernelUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestMinKernelExecution(self, raw_data):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, (1,))
        tensor_test_node = mp.kernels.MinKernel.add_to_graph(self.__graph,inTensor,outTensor)

        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        
        return outTensor.data
    
class MeanKernelUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestMeanKernelExecution(self, raw_data):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, (1,))
        tensor_test_node = mp.kernels.MeanKernel.add_to_graph(self.__graph,inTensor,outTensor)

        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()

        return outTensor.data
    
class StdKernelUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestStdKernelExecution(self, raw_data):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, (1,))
        tensor_test_node = mp.kernels.StdKernel.add_to_graph(self.__graph,inTensor,outTensor, ddof=0)

        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()

        return outTensor.data
    
class VarKernelUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestVarKernelExecution(self, raw_data):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, (1,))
        tensor_test_node = mp.kernels.VarKernel.add_to_graph(self.__graph,inTensor,outTensor)

        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()

        return outTensor.data
    
class KurtosisKernelUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestKurtosisKernelExecution(self, raw_data):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, (1,))
        tensor_test_node = mp.kernels.KurtosisKernel.add_to_graph(self.__graph,inTensor,outTensor)

        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()

        return outTensor.data
    
class SkewnessKernelUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestSkewnessKernelExecution(self, raw_data):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, (1,))
        tensor_test_node = mp.kernels.SkewnessKernel.add_to_graph(self.__graph,inTensor,outTensor)
        
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()

        return outTensor.data
    
class ZScoreKernelUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestZScoreKernelExecution(self, raw_data, init_data):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, raw_data.shape)
        initTensor = mp.Tensor.create_from_data(self.__session, init_data)
        tensor_test_node = mp.kernels.ZScoreKernel.add_to_graph(self.__graph,inTensor,outTensor,initTensor),

        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()

        return outTensor.data

def test_execute():
    np.random.seed(44)
    raw_data = np.random.randn(2,2)
    KernelExecutionUnitTest_Object = CDFKernelUnitTest()      
    res = KernelExecutionUnitTest_Object.TestCDFKernelExecution(raw_data)
    expected_output = chi2.cdf(raw_data, 55)
    assert (res == expected_output).all()
    del KernelExecutionUnitTest_Object
    
    KernelExecutionUnitTest_Object = CovarianceKernelUnitTest()
    res = KernelExecutionUnitTest_Object.TestCovarianceKernelExecution(raw_data)
    expected_output = np.cov(raw_data)
    assert (res == expected_output).all()
    del KernelExecutionUnitTest_Object
    
    KernelExecutionUnitTest_Object = MaxKernelUnitTest()
    res = KernelExecutionUnitTest_Object.TestMaxKernelExecution(raw_data)
    expected_output = np.max(raw_data)
    assert (res == expected_output).all()
    del KernelExecutionUnitTest_Object
    
    KernelExecutionUnitTest_Object = MinKernelUnitTest()
    res = KernelExecutionUnitTest_Object.TestMinKernelExecution(raw_data)
    expected_output = np.min(raw_data)
    assert (res == expected_output).all()
    del KernelExecutionUnitTest_Object
    
    KernelExecutionUnitTest_Object = MeanKernelUnitTest()
    res = KernelExecutionUnitTest_Object.TestMeanKernelExecution(raw_data)
    expected_output = np.mean(raw_data)
    assert (res == expected_output).all()
    del KernelExecutionUnitTest_Object
    
    KernelExecutionUnitTest_Object = StdKernelUnitTest()
    res = KernelExecutionUnitTest_Object.TestStdKernelExecution(raw_data)
    expected_output = np.std(raw_data)
    assert (res == expected_output).all()
    del KernelExecutionUnitTest_Object
    
    KernelExecutionUnitTest_Object = VarKernelUnitTest()
    res = KernelExecutionUnitTest_Object.TestVarKernelExecution(raw_data)
    expected_output = np.var(raw_data)
    assert (res == expected_output).all()
    del KernelExecutionUnitTest_Object
    
    KernelExecutionUnitTest_Object = KurtosisKernelUnitTest()
    res = KernelExecutionUnitTest_Object.TestKurtosisKernelExecution(raw_data)
    expected_output = kurtosis(raw_data, axis=None)
    assert (res == expected_output).all()
    del KernelExecutionUnitTest_Object
    
    KernelExecutionUnitTest_Object = SkewnessKernelUnitTest()
    res = KernelExecutionUnitTest_Object.TestSkewnessKernelExecution(raw_data)
    expected_output = skew(raw_data, axis=None)
    assert (res == expected_output).all()
    del KernelExecutionUnitTest_Object
    
    # KernelExecutionUnitTest_Object = ZScoreKernelUnitTest()
    # np.random.seed(44)
    # raw_data = np.random.randn(2,)
    # init_data = data = np.random.randn(2,)
    # res = KernelExecutionUnitTest_Object.TestZScoreKernelExecution(raw_data, init_data)
    # expected_output = (raw_data - np.mean(init_data))/np.std(init_data)
    # assert (res == expected_output).all()
    # del KernelExecutionUnitTest_Object
      
    