# import mindpype as mp
# import numpy as np

# class RunningAverageKernelUnitTest:
#     def __init__(self):
#         self.__session = mp.Session.create()
#         self.__graph = mp.Graph.create(self.__session)

#     def TestRunningAverageKernelExecution(self, raw_data, running_average):
#         inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
#         outTensor = mp.Tensor.create(self.__session, (2,2))
#         tensor_test_node = mp.kernels.RunningAverageKernel.add_to_graph(self.__graph,inTensor,outTensor,running_average)

#         self.__graph.verify()
#         self.__graph.initialize()
#         self.__graph.execute()

#         return outTensor.data

# def test_execute():
#     np.random.seed(44)
#     raw_data = np.random.randint(-10,10, size=(2,2))
#     KernelExecutionUnitTest_Object = RunningAverageKernelUnitTest()
#     res1 = KernelExecutionUnitTest_Object.TestRunningAverageKernelExecution(raw_data, running_average=1)
#     res2 = KernelExecutionUnitTest_Object.TestRunningAverageKernelExecution(raw_data, running_average=2)

#     running_average = np.mean([raw_data, raw_data], axis=0)
    
#     assert (res2 == running_average).all()
#     del KernelExecutionUnitTest_Object
    
# test_execute()