# import mindpype as mp
# import sys, os
# import numpy as np

# class FeatureSelectionKernelCreationUnitTest:
#     def __init__(self):
#         self.__session = mp.Session.create()
#         self.__graph = mp.Graph.create(self.__session)

#     def TestFeatureSelectionKernelCreation(self):
#         inTensor = mp.Tensor.create(self.__session, (10,10))
#         outTensor = mp.Tensor.create(self.__session,(10,10))
#         initialization_tensor = mp.Tensor.create(self.__session, (10,10))
#         labels = mp.Tensor.create(self.__session, (10,))
#         node = mp.kernels.FeatureSelectionKernel.add_to_graph(self.__graph,inTensor,outTensor,k=10,init_inputs=initialization_tensor,labels=labels)
#         return node.mp_type
    
# class FeatureSelectionKernelExecutionUnitTest:

#     def __init__(self):
#         self.__session = mp.Session.create()
#         self.__graph = mp.Graph.create(self.__session)

#     def TestFeatureSelectionKernelExecution(self):
#         np.random.seed(44)
#         raw_data = np.random.randint(-10,10, size=(10,10))
#         inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
#         outTensor = mp.Tensor.create(self.__session,(10,10))
#         init_data = np.random.randint(-10,10, size=(10,10))
#         initialization_tensor = mp.Tensor.create_from_data(self.__session, init_data)
#         labels_data = np.random.randint(0, 4, size=(10,))
#         labels = mp.Tensor.create_from_data(self.__session, labels_data)
#         node = mp.kernels.FeatureSelectionKernel.add_to_graph(self.__graph,inTensor,outTensor,k=10,init_inputs=initialization_tensor,labels=labels)
        
#         sys.stdout = open(os.devnull, 'w')
#         self.__graph.verify()
#         self.__graph.initialize(initialization_tensor, labels)
#         self.__graph.execute()
#         sys.stdout = sys.__stdout__

#         return (raw_data, init_data, outTensor.data)

# def test_create():
#     KernelUnitTest_Object = FeatureSelectionKernelCreationUnitTest()
#     assert KernelUnitTest_Object.TestFeatureSelectionKernelCreation() == mp.MPEnums.NODE
#     del KernelUnitTest_Object


# def test_execute():
#     KernelExecutionUnitTest_Object = FeatureSelectionKernelExecutionUnitTest()
#     res = KernelExecutionUnitTest_Object.TestFeatureSelectionKernelExecution()
#     # expected_output = (res[0] - np.mean(res[1], axis=1)) / np.std(res[1], axis=1)
#     # assert (res[2] == expected_output).all()
#     del KernelExecutionUnitTest_Object