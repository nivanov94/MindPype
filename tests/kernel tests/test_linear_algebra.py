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

        self.__graph.verify()

def test_execute():
    test = LinearAlgebraUnitTest()
    
    test.Test()