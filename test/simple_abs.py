import bcipy
import numpy as np

s = bcipy.Session()
g = bcipy.Graph(s)

data_in = np.asarray([[1,2,3],[-1,-2,-3]])
t_in = bcipy.Tensor.create_from_data(s, (2,3), data_in)
t_out = bcipy.Tensor.create(s, (2,3))

bcipy.kernels.AbsoluteKernel.add_absolute_node(g, t_in, t_out)

sts = g.verify()

if sts != bcipy.core.BcipEnums.SUCCESS:
    print('failed D=')

sts = g.execute()

if sts != bcipy.core.BcipEnums.SUCCESS:
    print('failed D=')
else:
    print(t_out.data)
    print('passed =D')
