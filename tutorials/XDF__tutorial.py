from bcipy import bcipy
import numpy as np


def XDF_test(file, tasks, channels, start, samples):
    
    # Create a session
    session = bcipy.Session.create()
    
    # Create two graphs, one to test the epoched data and one to test the continuous data
    graph = bcipy.Graph.create(session)
    graph_cont = bcipy.Graph.create(session)

    # Create the XDF objects for the epoched and continuous data using the factory methods
    xdf_object = bcipy.source.BcipXDF.create_epoched(session, file, tasks, channels, start, samples)
    xdf_object_continuous = bcipy.source.BcipXDF.create_continuous(session, file, tasks, channels, start, samples)
    
    # Create the input tensors for the epoched and continuous data
    t_in = bcipy.Tensor.create_from_handle(session, (len(channels), samples), xdf_object)
    t_in_cont = bcipy.Tensor.create_from_handle(session, (len(channels), samples), xdf_object_continuous)

    # Create an input tensor for the second input to the addition kernel
    t_in_2 = bcipy.Tensor.create_from_data(session, shape=t_in.shape, data=np.ones(t_in.shape))

    # Create the output tensors for the epoched and continuous data
    t_out = bcipy.Tensor.create(session, shape=t_in.shape)
    t_out_cont = bcipy.Tensor.create(session, shape=t_in.shape)
    
    # Add the addition kernel to each graph
    Add = bcipy.kernels.AdditionKernel.add_addition_node(graph, t_in, t_in_2, t_out)
    Add_cont = bcipy.kernels.AdditionKernel.add_addition_node(graph_cont, t_in_cont, t_in_2, t_out_cont)

    # Verify and initialize the graphs
    sts1 = graph.verify() and graph_cont.verify()
    
    if sts1 != bcipy.BcipEnums.SUCCESS:
        print(sts1)
        print("Test Failed D=")
        return sts1

    sts2 = graph.initialize() and graph_cont.initialize()

    if sts2 != bcipy.BcipEnums.SUCCESS:
        print(sts2)
        print("Test Failed D=")
        return sts2
    
    # Execute the graphs
    i = 0

    sts = bcipy.BcipEnums.SUCCESS
    sts_cont = bcipy.BcipEnums.SUCCESS
    while i < 10 and sts == bcipy.BcipEnums.SUCCESS and sts_cont == bcipy.BcipEnums.SUCCESS:
    
        sts = graph.execute()
        sts_cont = graph_cont.execute()  

        # Check that the output tensors are equal to the sum of the input tensors
        print(np.array_equal(t_out.data, t_in.data + t_in_2.data))
        print(np.array_equal(t_out_cont.data, t_in_cont.data + t_in_2.data)) 
        i+=1 


def main():    
    channels = [i for i in range(1)]
    tasks = ('flash', 'target')
    trial_data = XDF_test(['C:/Users/lioa/Documents/Mindset P300 Code for Aaron/sub-P001_ses-S001_task-vP300+2x2_run-003.xdf'], tasks, channels, -.2, 500)

main()