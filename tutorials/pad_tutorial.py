from bcipy import bcipy
import pkgutil, numpy as np

def pad_test():
    session = bcipy.Session.create()
    graph = bcipy.Graph.create(session)

    input_tensor = bcipy.Tensor.create_from_data(session, shape=(1,), data=np.ones(1))
    output_tensor = bcipy.Tensor.create_virtual(session, shape=())

    node = bcipy.kernels.PadKernel.add_pad_node(graph, input_tensor, output_tensor, pad_width = [1], mode = 'constant', constant_values = 0)
    
    graph.verify()
    graph.initialize()
    graph.execute()

    print(output_tensor.data)

pad_test()