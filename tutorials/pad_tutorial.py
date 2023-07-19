import bcipy.bcipy as bcipy
import pkgutil

def pad_test():
    session = bcipy.Session.create()
    graph = bcipy.Graph.create(session)

    input_tensor = bcipy.Tensor.create_from_data(session, shape=(1, 1), data=[[1]])
    output_tensor = bcipy.Tensor.create_virtual(session, shape=())

    bcipy.kernels.PadKernel.add_pad_kernel(graph, input_tensor, output_tensor, pad_width = 1, mode = 'constant', constant_values = 0)
    
    graph.verify()
    graph.initialize()
    graph.execute(poll_volatile_sources=False, push_volatile_outputs=False)

    print(output_tensor.data)

pad_test()