# import mindpype as mp
# import numpy as np

# class PatternStateKernelTest:
#     def __init__(self):
#         self.__session = mp.Session.create()
#         self.__graph = mp.Graph.create(self.__session)

#     def TestKernelCreation(self, raw_data):
#         inA = mp.Tensor.create_from_data(self.__session, raw_data)
#         outA = mp.Scalar.create(self.__session, int)
#         outB = mp.Tensor.create(self.__session, (1,4))
#         node = mp.kernels.PatternStateKernel.add_to_graph(self.__graph, inA, outA, outB)

#         self.__graph.verify()
#         self.__graph.initialize()
#         self.__graph.execute()

# def test_execute():
#     t = PatternStateKernelTest()
#     raw_data = np.ones((1,1))
#     t.TestKernelCreation(raw_data)

# test_execute()
