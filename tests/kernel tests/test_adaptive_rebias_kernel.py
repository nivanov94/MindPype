import mindpype as mp
import numpy as np

class AdaptiveRebiasKernelTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestKernelCreation(self, raw_data, labels):
        inA = mp.Tensor.create_from_data(self.__session, raw_data)
        outA = mp.Tensor.create(self.__session, inA.shape)
        
        node = mp.kernels.AdaptiveRebiasKernel.add_to_graph(self.__graph, inA, outA, raw_data, labels) 

        self.__graph.initialize()
        self.__graph.verify()
        self.__graph.execute()

def test_execute():
    raw_data = np.ones(2)  # [1,1]
    labels = np.ones(2)
    test = AdaptiveRebiasKernelTest()
    test.TestKernelCreation(raw_data, labels)

test_execute()