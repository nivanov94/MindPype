import os
os.sys.path.append(os.path.dirname(os.path.abspath('.')))
import numpy as np
from mindpype import mindpype as mp

s = mp.Session.create()
g = mp.Graph.create(s)

data_in = np.asarray([[1,2,3],[-1,-2,-3]])
t_in1 = mp.Tensor.create_from_data(s, data_in)
t_in2 = mp.Tensor.create_from_data(s, np.abs(data_in))
t_out = mp.Tensor.create(s, (2,3))

t_virt = mp.Tensor.create_virtual(s)

mp.kernels.EqualKernel.add_to_graph(g, t_in2, t_virt, t_out)
mp.kernels.AbsoluteKernel.add_to_graph(g, t_in1, t_virt)

g.execute()
print(t_out.data)
