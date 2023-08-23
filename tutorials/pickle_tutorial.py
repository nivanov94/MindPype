"""
This file was used to confirm that a generic pipeline could be created and saved to a pickle file.

"""

from bcipy import bcipy
import numpy as np
import pickle

class Pipeline():
    def __init__(self):

        self.s = bcipy.Session.create()
        self.g = bcipy.Graph.create(self.s)

        data_in = np.asarray([[1,2,3],[-1,-2,-3]])
        t_in1 = bcipy.Tensor.create_from_data(self.s, (2,3), data_in)
        t_in2 = bcipy.Tensor.create_from_data(self.s, (2,3), np.abs(data_in))
        t_out = bcipy.Tensor.create(self.s, (2,3))

        t_virt = bcipy.Tensor.create_virtual(self.s)

        bcipy.kernels.AbsoluteKernel.add_absolute_node(self.g, t_in1, t_virt)
        bcipy.kernels.EqualKernel.add_equal_node(self.g, t_in2, t_virt, t_out)

        sts = self.g.verify()

        if sts != bcipy.core.BcipEnums.SUCCESS:
            print('failed D=')

        self.s = self.s.save_session("C:/Users/lioa/Documents/bcipy_venv/bcipy/scrap/empty.pkl")
        with open("C:/Users/lioa/Documents/bcipy_venv/bcipy/scrap/empty.pkl", 'rb') as pickle_file:
            self.s = pickle.load(pickle_file) 

    def initialize(self):

        print('initializing')
        sts = self.g.initialize()
        print('initialized')
        self.s = self.s.save_session("C:/Users/lioa/Documents/bcipy_venv/bcipy/scrap/empty.pkl")
        
        with open("C:/Users/lioa/Documents/bcipy_venv/bcipy/scrap/empty.pkl", 'rb') as pickle_file:
            self.s = pickle.load(pickle_file)

    def execute(self):
        sts = self.g.execute()

        if sts != bcipy.core.BcipEnums.SUCCESS:
            print('failed D=')
        else:
            print('passed =D')


if __name__ == '__main__':
    p = Pipeline()
    save_state = p.s.save_session("C:/Users/lioa/Documents/bcipy_venv/bcipy/scrap/empty.pkl")
    with open("C:/Users/lioa/Documents/bcipy_venv/bcipy/scrap/empty.pkl", 'rb') as pickle_file:
        p.s = pickle.load(pickle_file)
    
    p.initialize()
    p.execute()