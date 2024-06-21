# import mindpype as mp
# import sys, os
# import numpy as np

# class RunningAverageKernelCreationUnitTest:
#     def __init__(self):
#         self.__session = mp.Session.create()
#         self.__graph = mp.Graph.create(self.__session)

#     def TestRunningAverageKernelCreation(self):
#         inTensor = mp.Tensor.create(self.__session, (2,2))
#         outTensor = mp.Tensor.create(self.__session, (2,2))
#         running_average = 1
#         node = mp.kernels.TransposeKernel.add_to_graph(self.__graph,inTensor,outTensor,running_average)
#         return node.mp_type
    
# class RunningAverageKernelExecutionUnitTest:

#     def __init__(self):
#         self.__session = mp.Session.create()
#         self.__graph = mp.Graph.create(self.__session)

#     def TestRunningAverageKernelExecution(self):
#         np.random.seed(44)
#         raw_data = np.random.randint(-10,10, size=(2,2))
#         inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
#         outTensor = mp.Tensor.create(self.__session, (2,2))
#         running_average = 1
#         tensor_test_node = mp.kernels.RunningAverageKernel.add_to_graph(self.__graph,inTensor,outTensor,running_average)

#         sys.stdout = open(os.devnull, 'w')
#         self.__graph.verify()
#         self.__graph.initialize()
#         self.__graph.execute()
#         sys.stdout = sys.__stdout__

#         return (raw_data, outTensor.data)

# def test_create():
#     KernelUnitTest_Object = RunningAverageKernelCreationUnitTest()
#     assert KernelUnitTest_Object.TestRunningAverageKernelCreation() == mp.MPEnums.NODE
#     del KernelUnitTest_Object

# def test_execute():
#     KernelExecutionUnitTest_Object = RunningAverageKernelExecutionUnitTest()
#     res = KernelExecutionUnitTest_Object.TestRunningAverageKernelExecution()
    
#     # Calculate row averages
#     row_averages = np.mean(res[0], axis=1)
#     # Calculate column averages
#     column_averages = np.mean(res[0], axis=0)
#     # Create the running average matrix
#     running_average = np.outer(row_averages, column_averages)
    
#     assert (res[1] == running_average).all()
#     del KernelExecutionUnitTest_Object