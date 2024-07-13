import mindpype as mp
import numpy as np

class RunningAverageKernelUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestRunningAverageKernelExecution(self, raw_data, init_data, running_average_len):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, (2,2))
        init_tensor = mp.Tensor.create_from_data(self.__session, init_data)
        tensor_test_node = mp.kernels.RunningAverageKernel.add_to_graph(self.__graph, inTensor, outTensor, running_average_len, init_input=init_tensor)

        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()

        return outTensor.data

def test_execute():
    np.random.seed(44)
    init_data = np.random.randint(-10,10, size=(2,2,2))
    raw_data = np.random.randint(-10,10, size=(2,2))
    KernelExecutionUnitTest_Object = RunningAverageKernelUnitTest()
    res = KernelExecutionUnitTest_Object.TestRunningAverageKernelExecution(raw_data, init_data, running_average_len=10)

    running_average = np.mean(np.concatenate((np.expand_dims(raw_data,axis=0), init_data)), axis=0)
  
    assert (res == running_average).all()
    del KernelExecutionUnitTest_Object
