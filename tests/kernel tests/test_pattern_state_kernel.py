import mindpype as mp
import numpy as np

class PatternStateKernelTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestKernelCreation(self, raw_data):
        inA = mp.Tensor.create_from_data(self.__session, raw_data)
