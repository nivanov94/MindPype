# import mindpype as mp
# import sys, os
# import numpy as np
# from pyriemann.tangentspace import TangentSpace

# class TangentSpaceKernelCreationUnitTest:
#     def __init__(self):
#         self.__session = mp.Session.create()
#         self.__graph = mp.Graph.create(self.__session)

#     def TestTangentSpaceKernelCreation(self):
#         inTensor = mp.Tensor.create(self.__session, (2,2,2))
#         outTensor = mp.Tensor.create(self.__session, (2,2))
#         node = mp.kernels.TangentSpaceKernel.add_to_graph(self.__graph,inTensor,outTensor)
#         return node.mp_type
    
# class TangentSpaceKernelExecutionUnitTest:

#     def __init__(self):
#         self.__session = mp.Session.create()
#         self.__graph = mp.Graph.create(self.__session)

#     def TangentSpaceKernelExecution(self):
#         np.random.seed(44)
#         raw_data = np.random.randint(-10,10, size=(2,2,2))
#         inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
#         outTensor = mp.Tensor.create(self.__session, (2,2))
    
#         initialization_data = np.random.randint(-10,10, size=(2,2,2))
#         init_in = mp.Tensor.create_from_data(self.__session, initialization_data)
#         init_out = mp.Tensor.create(self.__session, (2,2))
#         tensor_test_node = mp.kernels.TangentSpaceKernel.add_to_graph(self.__graph,inTensor, outTensor, initialization_data=init_in)

#         sys.stdout = open(os.devnull, 'w')
#         self.__graph.verify()
#         self.__graph.initialize(init_in, init_out)
#         self.__graph.execute()
#         sys.stdout = sys.__stdout__

#         return (inTensor.data, init_in.data, outTensor.data)

# def test_create():
#     KernelUnitTest_Object = TangentSpaceKernelCreationUnitTest()
#     assert KernelUnitTest_Object.TestTangentSpaceKernelCreation() == mp.MPEnums.NODE
#     del KernelUnitTest_Object


# def test_execute():
#     KernelExecutionUnitTest_Object = TangentSpaceKernelExecutionUnitTest()
#     res = KernelExecutionUnitTest_Object.TangentSpaceKernelExecution()

#     tangent_space = TangentSpace()
#     tangent_space.fit(res[1])
#     output = tangent_space.transform(res[0])
#     assert (res[2] == output).all()

#     del KernelExecutionUnitTest_Object