import mindpype as mp
import numpy as np
from scipy.stats import norm, chi2, kurtosis, skew, zscore

class CDFKernelUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestCDFKernelExecution(self, raw_data):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, (2,2))
        tensor_test_node = mp.kernels.CDFKernel.add_to_graph(self.__graph,inTensor,outTensor,dist='chi2',df=55)

        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()

        return (inTensor.data, outTensor.data)

class CovarianceKernelUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestCovarianceKernelExecution(self, raw_data):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, (2,2))
        regularization = 0
        tensor_test_node = mp.kernels.CovarianceKernel.add_to_graph(self.__graph,inTensor,outTensor, regularization)

        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()

        return (inTensor.data, outTensor.data)
    
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

        return (inTensor.data, outTensor.data)
    
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
        
        return (inTensor.data, outTensor.data)
    
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

        return (inTensor.data, outTensor.data)
    
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

        return (inTensor.data, outTensor.data)
    
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

        return (inTensor.data, outTensor.data)
    
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

        return (inTensor.data, outTensor.data)
    
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

        return (inTensor.data, outTensor.data)
    
class ZScoreKernelUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestZScoreKernelExecution(self, raw_data, init_data):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, (2,))
        initTensor = mp.Tensor.create_from_data(self.__session, init_data)
        tensor_test_node = mp.kernels.ZScoreKernel.add_to_graph(self.__graph,inTensor,outTensor,initTensor),

        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()

        return (inTensor.data, init_data, outTensor.data)

def test_execute():
    KernelExecutionUnitTest_Object = CDFKernelUnitTest()      
    np.random.seed(44)
    raw_data = np.random.randn(2,2)
    res = KernelExecutionUnitTest_Object.TestCDFKernelExecution(raw_data)
    expected_output = chi2.cdf(res[0], 55)
    assert (res[1] == expected_output).all()
    del KernelExecutionUnitTest_Object
    
    KernelExecutionUnitTest_Object = CovarianceKernelUnitTest()
    np.random.seed(44)
    raw_data = np.random.randn(2,2)
    res = KernelExecutionUnitTest_Object.TestCovarianceKernelExecution(raw_data)
    expected_output = np.cov(res[0])
    assert (res[1] == expected_output).all()
    del KernelExecutionUnitTest_Object
    
    KernelExecutionUnitTest_Object = MaxKernelUnitTest()
    np.random.seed(44)
    raw_data = np.random.randn(2,2)
    res = KernelExecutionUnitTest_Object.TestMaxKernelExecution(raw_data)
    expected_output = np.max(res[0])
    assert (res[1] == expected_output).all()
    del KernelExecutionUnitTest_Object
    
    KernelExecutionUnitTest_Object = MinKernelUnitTest()
    np.random.seed(44)
    raw_data = np.random.randn(2,2)
    res = KernelExecutionUnitTest_Object.TestMinKernelExecution(raw_data)
    expected_output = np.min(res[0])
    assert (res[1] == expected_output).all()
    del KernelExecutionUnitTest_Object
    
    KernelExecutionUnitTest_Object = MeanKernelUnitTest()
    np.random.seed(44)
    raw_data = np.random.randn(2,2)
    res = KernelExecutionUnitTest_Object.TestMeanKernelExecution(raw_data)
    expected_output = np.mean(res[0])
    assert (res[1] == expected_output).all()
    del KernelExecutionUnitTest_Object
    
    KernelExecutionUnitTest_Object = StdKernelUnitTest()
    np.random.seed(44)
    raw_data = np.random.randn(2,2)
    res = KernelExecutionUnitTest_Object.TestStdKernelExecution(raw_data)
    expected_output = np.std(res[0])
    assert (res[1] == expected_output).all()
    del KernelExecutionUnitTest_Object
    
    KernelExecutionUnitTest_Object = VarKernelUnitTest()
    np.random.seed(44)
    raw_data = np.random.randn(2,2)
    res = KernelExecutionUnitTest_Object.TestVarKernelExecution(raw_data)
    expected_output = np.var(res[0])
    assert (res[1] == expected_output).all()
    del KernelExecutionUnitTest_Object
    
    KernelExecutionUnitTest_Object = KurtosisKernelUnitTest()
    np.random.seed(44)
    raw_data = np.random.randn(2,2)
    res = KernelExecutionUnitTest_Object.TestKurtosisKernelExecution(raw_data)
    expected_output = kurtosis(res[0], axis=None)
    assert (res[1] == expected_output).all()
    del KernelExecutionUnitTest_Object
    
    KernelExecutionUnitTest_Object = SkewnessKernelUnitTest()
    np.random.seed(44)
    raw_data = np.random.randn(2,2)
    res = KernelExecutionUnitTest_Object.TestSkewnessKernelExecution(raw_data)
    expected_output = skew(res[0], axis=None)
    assert (res[1] == expected_output).all()
    del KernelExecutionUnitTest_Object
    
    # KernelExecutionUnitTest_Object = ZScoreKernelExecutionUnitTest()
    # np.random.seed(44)
    # raw_data = np.random.randn(2,)
    # init_data = data = np.random.randint(-10,10, size=(2,))
    # res = KernelExecutionUnitTest_Object.TestZScoreKernelExecution(raw_data, init_data)
    # expected_output = (res[0] - np.mean(res[1]))/np.std(res[1])
    # assert (res[2] == expected_output).all()
    # del KernelExecutionUnitTest_Object
      
    