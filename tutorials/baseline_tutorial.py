import mindpype as mp
import pkgutil, numpy as np

def baseline_test():
    session = mp.Session.create()
    graph = mp.Graph.create(session)

    input_tensor = mp.Tensor.create_from_data(session, shape=(10, 10), data=np.random.rand(10, 10))
    output_tensor = mp.Tensor.create(session, shape=(10, 10))

    mp.kernels.BaselineCorrectionKernel.add_baseline_node(graph, input_tensor, output_tensor, baseline_period = [0,10])
    
    graph.verify()
    graph.initialize()
    graph.execute()
    print(input_tensor.data)
    print(output_tensor.data)

    

baseline_test()