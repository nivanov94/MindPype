import mindpype as mp
import numpy as np


def XDF_test(file, tasks, channels, start, samples):
    
    # Create a session
    session = mp.Session.create()
    
    # Create two graphs, one to test the epoched data and one to test the continuous data
    graph = mp.Graph.create(session)
    graph_cont = mp.Graph.create(session)

    # Create the XDF objects for the epoched and continuous data using the factory methods
    xdf_object = mp.source.BcipXDF.create_epoched(session, file, tasks, channels, start, samples)
    xdf_object_continuous = mp.source.BcipXDF.create_continuous(session, file, tasks, channels, start, samples)
    
    # Create the input tensors for the epoched and continuous data
    t_in = mp.Tensor.create_from_handle(session, (len(channels), samples), xdf_object)
    t_in_cont = mp.Tensor.create_from_handle(session, (len(channels), samples), xdf_object_continuous)

    # Create an input tensor for the second input to the addition kernel
    t_in_2 = mp.Tensor.create_from_data(session, shape=t_in.shape, data=np.ones(t_in.shape))

    # Create the output tensors for the epoched and continuous data
    t_out = mp.Tensor.create(session, shape=t_in.shape)
    t_out_cont = mp.Tensor.create(session, shape=t_in.shape)
    
    # Add the addition kernel to each graph
    mp.kernels.AdditionKernel.add_addition_node(graph, t_in, t_in_2, t_out)
    mp.kernels.AdditionKernel.add_addition_node(graph_cont, t_in_cont, t_in_2, t_out_cont)

    # Verify and initialize the graphs
    graph.verify() 
    graph_cont.verify()
    
    graph.initialize() 
    graph_cont.initialize()

    # Execute the graphs
    for i in range(9):
        graph.execute()
        graph_cont.execute()  

        # Check that the output tensors are equal to the sum of the input tensors
        print(np.array_equal(t_out.data, t_in.data + t_in_2.data))
        print(np.array_equal(t_out_cont.data, t_in_cont.data + t_in_2.data)) 


def main():    
    channels = [i for i in range(3,10)]
    tasks = ('flash', 'target')
    trial_data = XDF_test(['/path/to/mindset/data'], tasks, channels, -.2, 500)

main()