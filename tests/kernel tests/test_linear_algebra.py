import mindpype as mp
import numpy as np

class LinearAlgebraUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def Test(self):
        inA = mp.Tensor.create_from_data(self.__session, [[1,2], [3,4]])
        inB = mp.Tensor.create_from_data(self.__session, [[2,2], [2,2]])
        out = mp.Tensor.create(self.__session, (2,2))

        node = mp.kernels.MatrixMultKernel.add_to_graph(self.__graph, inA, inB, out)

        in_data = [mp.Tensor.create_from_data(self.__session, [1,1,1,1]), mp.Tensor.create_from_data(self.__session, [2,2,2,2]), mp.Tensor.create_from_data(self.__session, [3,3,3,3])]
        labels = mp.Tensor.create_from_data(self.__session, [1,2,3])
        # labels.data = np.array([1,2,3,4])

        # node = mp.kernels.MatrixMultKernel.add_to_graph(self.__graph, inA, inB, out, init_inputs=in_data, init_labels=labels)

        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()


    def TestComputeOutputSize(self):
        inA_inner_2 = mp.Tensor.create_from_data(self.__session, [[1,2], [1,2]])
        inB_inner_2 = mp.Tensor.create_from_data(self.__session, [[1,2,3], [1,2,3]])

        inA_inner_3 = mp.Tensor.create_from_data(self.__session, [[[1,2], [1,2]], [[1,2], [1,2]]])
        inB_inner_3 = mp.Tensor.create_from_data(self.__session, [[[1,2], [1,2]], [[1,2], [1,2]]])

        out1 = mp.Tensor.create(self.__session, (2, 3))

        n1 = mp.kernels.MatrixMultKernel.add_to_graph(self.__graph, inA_inner_2, inB_inner_2, out1)
        n2 = mp.kernels.MatrixMultKernel.add_to_graph(self.__graph, inA_inner_3, inB_inner_2)
        n3 = mp.kernels.MatrixMultKernel.add_to_graph(self.__graph, inA_inner_2, inB_inner_3)
        n4 = mp.kernels.MatrixMultKernel.add_to_graph(self.__graph, inA_inner_3, inB_inner_3)

        self.__graph.initialize()


def test_execute():
    test = LinearAlgebraUnitTest()
    
    test.Test()