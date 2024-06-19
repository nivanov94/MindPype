import mindpype as mp
import sys, os
import numpy as np
from scipy.stats import norm, chi2, kurtosis, skew

class MaxKernelCreationUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestMaxKernelCreation(self):
        inTensor = mp.Tensor.create(self.__session, (2,2))
        outTensor = mp.Tensor.create(self.__session, (1,))
        node = mp.kernels.MaxKernel.add_to_graph(self.__graph,inTensor,outTensor)
        return node.mp_type
    
class MaxKernelExecutionUnitTest:

    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestMaxKernelExecution(self):
        np.random.seed(44)
        raw_data = np.random.randint(-10,10, size=(2,2))
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, (1,))
        tensor_test_node = mp.kernels.MaxKernel.add_to_graph(self.__graph,inTensor,outTensor)

        sys.stdout = open(os.devnull, 'w')
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        sys.stdout = sys.__stdout__

        return (inTensor.data, outTensor.data)
    
class MinKernelCreationUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestMinKernelCreation(self):
        inTensor = mp.Tensor.create(self.__session, (2,2))
        outTensor = mp.Tensor.create(self.__session, (1,))
        node = mp.kernels.MinKernel.add_to_graph(self.__graph,inTensor,outTensor)
        return node.mp_type
    
class MinKernelExecutionUnitTest:

    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestMinKernelExecution(self):
        np.random.seed(44)
        raw_data = np.random.randint(-10,10, size=(2,2))
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, (1,))
        tensor_test_node = mp.kernels.MinKernel.add_to_graph(self.__graph,inTensor,outTensor)

        sys.stdout = open(os.devnull, 'w')
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        sys.stdout = sys.__stdout__

        return (inTensor.data, outTensor.data)
    
class MeanKernelCreationUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestMeanKernelCreation(self):
        inTensor = mp.Tensor.create(self.__session, (2,2))
        outTensor = mp.Tensor.create(self.__session, (1,))
        node = mp.kernels.MeanKernel.add_to_graph(self.__graph,inTensor,outTensor)
        return node.mp_type
    
class MeanKernelExecutionUnitTest:

    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestMeanKernelExecution(self):
        np.random.seed(44)
        raw_data = np.random.randint(-10,10, size=(2,2))
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, (1,))
        tensor_test_node = mp.kernels.MeanKernel.add_to_graph(self.__graph,inTensor,outTensor)

        sys.stdout = open(os.devnull, 'w')
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        sys.stdout = sys.__stdout__

        return (inTensor.data, outTensor.data)
    
class StdKernelCreationUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestStdKernelCreation(self):
        inTensor = mp.Tensor.create(self.__session, (2,2))
        outTensor = mp.Tensor.create(self.__session, (1,))
        node = mp.kernels.StdKernel.add_to_graph(self.__graph,inTensor,outTensor,ddof=0)
        return node.mp_type
    
class StdKernelExecutionUnitTest:

    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestStdKernelExecution(self):
        np.random.seed(44)
        raw_data = np.random.randint(-10,10, size=(2,2))
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, (1,))
        tensor_test_node = mp.kernels.StdKernel.add_to_graph(self.__graph,inTensor,outTensor, ddof=0)

        sys.stdout = open(os.devnull, 'w')
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        sys.stdout = sys.__stdout__

        return (inTensor.data, outTensor.data)
    
class VarKernelCreationUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestVarKernelCreation(self):
        inTensor = mp.Tensor.create(self.__session, (2,2))
        outTensor = mp.Tensor.create(self.__session, (1,))
        node = mp.kernels.VarKernel.add_to_graph(self.__graph,inTensor,outTensor)
        return node.mp_type
    
class VarKernelExecutionUnitTest:

    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestVarKernelExecution(self):
        np.random.seed(44)
        raw_data = np.random.randint(-10,10, size=(2,2))
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, (1,))
        tensor_test_node = mp.kernels.VarKernel.add_to_graph(self.__graph,inTensor,outTensor)

        sys.stdout = open(os.devnull, 'w')
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        sys.stdout = sys.__stdout__

        return (inTensor.data, outTensor.data)
    
class KurtosisKernelCreationUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestKurtosisKernelCreation(self):
        inTensor = mp.Tensor.create(self.__session, (2,2))
        outTensor = mp.Tensor.create(self.__session, (1,))
        node = mp.kernels.KurtosisKernel.add_to_graph(self.__graph,inTensor,outTensor)
        return node.mp_type
    
class KurtosisKernelExecutionUnitTest:

    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestKurtosisKernelExecution(self):
        np.random.seed(44)
        raw_data = np.random.randint(-10,10, size=(2,2))
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, (1,))
        tensor_test_node = mp.kernels.KurtosisKernel.add_to_graph(self.__graph,inTensor,outTensor)

        sys.stdout = open(os.devnull, 'w')
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        sys.stdout = sys.__stdout__

        return (inTensor.data, outTensor.data)
    
class SkewnessKernelCreationUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestSkewnessKernelCreation(self):
        inTensor = mp.Tensor.create(self.__session, (2,2))
        outTensor = mp.Tensor.create(self.__session, (1,))
        node = mp.kernels.SkewnessKernel.add_to_graph(self.__graph,inTensor,outTensor)
        return node.mp_type
    
class SkewnessKernelExecutionUnitTest:

    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestSkewnessKernelExecution(self):
        np.random.seed(44)
        raw_data = np.random.randint(-10,10, size=(2,2))
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, (1,))
        tensor_test_node = mp.kernels.SkewnessKernel.add_to_graph(self.__graph,inTensor,outTensor)

        sys.stdout = open(os.devnull, 'w')
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        sys.stdout = sys.__stdout__

        return (inTensor.data, outTensor.data)
    
class ZScoreKernelCreationUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestZScoreKernelCreation(self):
        inTensor = mp.Tensor.create(self.__session, (2,2,2))
        outTensor = mp.Tensor.create(self.__session, (1,))
        initTensor = mp.Tensor.create(self.__session,(2,2,2))
        node = mp.kernels.ZScoreKernel.add_to_graph(self.__graph,inTensor,outTensor,initTensor)
        return node.mp_type
    
class ZScoreKernelExecutionUnitTest:

    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestZScoreKernelExecution(self):
        np.random.seed(44)
        raw_data = np.random.randint(-10,10, size=(2,2,2))
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, (1,))
        init_data = data = np.random.randint(-10,10, size=(2,2,2))
        initTensor = mp.Tensor.create_from_data(self.__session, init_data)
        tensor_test_node = mp.kernels.ZScoreKernel.add_to_graph(self.__graph,inTensor,outTensor,initTensor),

        sys.stdout = open(os.devnull, 'w')
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        sys.stdout = sys.__stdout__

        return (inTensor.data, outTensor.data)


def test_create():
    KernelUnitTest_Object = MaxKernelCreationUnitTest()
    assert KernelUnitTest_Object.TestMaxKernelCreation() == mp.MPEnums.NODE
    del KernelUnitTest_Object
    
    KernelUnitTest_Object = MinKernelCreationUnitTest()
    assert KernelUnitTest_Object.TestMinKernelCreation() == mp.MPEnums.NODE
    del KernelUnitTest_Object
    
    KernelUnitTest_Object = MeanKernelCreationUnitTest()
    assert KernelUnitTest_Object.TestMeanKernelCreation() == mp.MPEnums.NODE
    del KernelUnitTest_Object
    
    KernelUnitTest_Object = StdKernelCreationUnitTest()
    assert KernelUnitTest_Object.TestStdKernelCreation() == mp.MPEnums.NODE
    del KernelUnitTest_Object
    
    KernelUnitTest_Object = VarKernelCreationUnitTest()
    assert KernelUnitTest_Object.TestVarKernelCreation() == mp.MPEnums.NODE
    del KernelUnitTest_Object
    
    KernelUnitTest_Object = KurtosisKernelCreationUnitTest()
    assert KernelUnitTest_Object.TestKurtosisKernelCreation() == mp.MPEnums.NODE
    del KernelUnitTest_Object
    
    KernelUnitTest_Object = SkewnessKernelCreationUnitTest()
    assert KernelUnitTest_Object.TestSkewnessKernelCreation() == mp.MPEnums.NODE
    del KernelUnitTest_Object
    
    KernelUnitTest_Object = ZScoreKernelCreationUnitTest()
    assert KernelUnitTest_Object.TestZScoreKernelCreation() == mp.MPEnums.NODE
    del KernelUnitTest_Object


def test_execute():
    KernelExecutionUnitTest_Object = MaxKernelExecutionUnitTest()
    res = KernelExecutionUnitTest_Object.TestMaxKernelExecution()
    expected_output = np.max(res[0])
    assert (res[1] == expected_output).all()
    del KernelExecutionUnitTest_Object
    
    KernelExecutionUnitTest_Object = MinKernelExecutionUnitTest()
    res = KernelExecutionUnitTest_Object.TestMinKernelExecution()
    expected_output = np.min(res[0])
    assert (res[1] == expected_output).all()
    del KernelExecutionUnitTest_Object
    
    KernelExecutionUnitTest_Object = MeanKernelExecutionUnitTest()
    res = KernelExecutionUnitTest_Object.TestMeanKernelExecution()
    expected_output = np.mean(res[0])
    assert (res[1] == expected_output).all()
    del KernelExecutionUnitTest_Object
    
    # KernelExecutionUnitTest_Object = StdKernelExecutionUnitTest()
    # res = KernelExecutionUnitTest_Object.TestStdKernelExecution()
    # expected_output = np.std(res[0])
    # assert (res[1] == expected_output).all()
    # del KernelExecutionUnitTest_Object
    
    KernelExecutionUnitTest_Object = VarKernelExecutionUnitTest()
    res = KernelExecutionUnitTest_Object.TestVarKernelExecution()
    expected_output = np.var(res[0])
    assert (res[1] == expected_output).all()
    del KernelExecutionUnitTest_Object
    
    # KernelExecutionUnitTest_Object = KurtosisKernelExecutionUnitTest()
    # res = KernelExecutionUnitTest_Object.TestKurtosisKernelExecution()
    # expected_output = kurtosis(res[0], axis=None)
    # assert (res[1] == expected_output).all()
    # del KernelExecutionUnitTest_Object
    
    KernelExecutionUnitTest_Object = SkewnessKernelExecutionUnitTest()
    res = KernelExecutionUnitTest_Object.TestSkewnessKernelExecution()
    expected_output = skew(res[0], axis=None)
    assert (res[1] == expected_output).all()
    del KernelExecutionUnitTest_Object
    
    # KernelExecutionUnitTest_Object = ZScoreKernelExecutionUnitTest()
    # res = KernelExecutionUnitTest_Object.TestZScoreKernelExecution()
    # expected_output = skew(res[0], axis=None)
    # assert (res[1] == expected_output).all()
    # del KernelExecutionUnitTest_Object
      
    