import bcipy.bcipy as bcipy
import pkgutil, numpy as np

def baseline_test():
    session = bcipy.Session.create()
    graph = bcipy.Graph.create(session)

    input_tensor = bcipy.Tensor.create_from_data(session, shape=(10, 10), data=np.random.rand(10, 10))
    output_tensor = bcipy.Tensor.create_virtual(session, shape=())

    bcipy.kernels.BaselineCorrectionKernel.add_baseline_node(graph, input_tensor, output_tensor, baseline_period = [0,1])
    
    graph.verify()
    graph.initialize()
    graph.execute(poll_volatile_sources=False, push_volatile_outputs=False)
    print(input_tensor.data)
    print(output_tensor.data)

    

baseline_test()