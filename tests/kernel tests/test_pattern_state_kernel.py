# import mindpype as mp
# import numpy as np
# import pyriemann as py

# class PatternStateKernelTest:
#     def __init__(self):
#         self.__session = mp.Session.create()
#         self.__graph = mp.Graph.create(self.__session)

#     def TestKernelCreation(self, raw_data, init_data):
#         inA = mp.Tensor.create_from_data(self.__session, raw_data)
#         outA = mp.Scalar.create(self.__session, int)
#         outB = mp.Tensor.create(self.__session, (2,2))
#         node = mp.kernels.PatternStateKernel.add_to_graph(self.__graph, inA, outA, outB, init_data=init_data)

#         self.__graph.verify()
#         self.__graph.initialize()
#         self.__graph.execute()

# def test_execute():
#     t = PatternStateKernelTest()
#     raw_data = np.ones((2,2)) 
#     init_data = py.datasets.make_matrices(20, 10, 'spd')
#     t.TestKernelCreation(raw_data, init_data)

# test_execute()
