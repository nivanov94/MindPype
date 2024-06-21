# import mindpype as mp
# import sys, os
# import numpy as np
# from pyriemann.utils.distance import distance_riemann

# class RiemannDistanceKernelCreationUnitTest:
#     def __init__(self):
#         self.__session = mp.Session.create()
#         self.__graph = mp.Graph.create(self.__session)

#     def TestRiemannDistanceKernelCreation(self):
#         inTensor1 = mp.Tensor.create(self.__session, (1,1))
#         inTensor2 = mp.Tensor.create(self.__session, (1,1))
#         outTensor = mp.Tensor.create(self.__session, (1,1))
#         node = mp.kernels.RiemannDistanceKernel.add_to_graph(self.__graph,inTensor1,inTensor2,outTensor)
#         return node.mp_type
    
# class RiemannDistanceKernelExecutionUnitTest:

#     def __init__(self):
#         self.__session = mp.Session.create()
#         self.__graph = mp.Graph.create(self.__session)

#     def TestRiemannDistanceKernelExecution(self):
#         np.random.seed(44)
#         raw_data1 = np.random.randint(-10,10, size=(1,1))
#         raw_data2 = np.random.randint(-10,10, size=(1,1))
#         inTensor1 = mp.Tensor.create_from_data(self.__session, raw_data1)
#         inTensor2 = mp.Tensor.create_from_data(self.__session, raw_data2)
#         outTensor = mp.Tensor.create(self.__session, (1,1))
#         tensor_test_node = mp.kernels.RiemannDistanceKernel.add_to_graph(self.__graph,inTensor1,inTensor2,outTensor)

#         sys.stdout = open(os.devnull, 'w')
#         self.__graph.verify()
#         self.__graph.initialize()
#         self.__graph.execute()
#         sys.stdout = sys.__stdout__

#         return (raw_data1, raw_data2, outTensor.data)

# def test_create():
#     KernelUnitTest_Object = RiemannDistanceKernelCreationUnitTest()
#     assert KernelUnitTest_Object.TestRiemannDistanceKernelCreation() == mp.MPEnums.NODE
#     del KernelUnitTest_Object


# def test_execute():
#     KernelExecutionUnitTest_Object = RiemannDistanceKernelExecutionUnitTest()
#     res = KernelExecutionUnitTest_Object.TestRiemannDistanceKernelExecution()
#     assert (res[2] == distance_riemann(res[0],res[1])).all()
#     del KernelExecutionUnitTest_Object