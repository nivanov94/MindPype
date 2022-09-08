import bcipy
import numpy as np

s = bcipy.Session()
g = bcipy.Graph(s)

data_in = np.asarray([[1,2,3],[-1,-2,-3]])
t_in1 = bcipy.Tensor.create_from_data(s, (2,3), data_in)
t_in2 = bcipy.Tensor.create_from_data(s, (2,3), np.abs(data_in))
t_out = bcipy.Tensor.create(s, (2,3))

t_virt = bcipy.Tensor.create_virtual(s)

bcipy.kernels.EqualKernel.add_equal_node(g, t_in2, t_virt, t_out)
bcipy.kernels.AbsoluteKernel.add_absolute_node(g, t_in1, t_virt)

sts = g.verify()

if sts != bcipy.core.BcipEnums.SUCCESS:
    print('failed D=')

sts = g.execute()

if sts != bcipy.core.BcipEnums.SUCCESS:
    print('failed D=')
else:
    print(t_out.data)
    print('passed =D')
