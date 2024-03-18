# -*- coding: utf-8 -*-
"""
Created on Tues July 26 16:12:30 2022

@author: aaronlio
"""

# Create a simple graph for testing
import mindpype as mp

def main():
    # create a session
    sess = mp.Session.create()

    # Create a graph that will be used to collect training data and labels
    training_graph = mp.Graph.create(sess)

    # Create a graph that will be used to run online trials
    online_graph = mp.Graph.create(sess)

    # Constants
    training_trials = 6

    Fs = 128
    resample_fs = 50

    # create a filter
    order = 4
    bandpass = (1,25) # in Hz
    f = mp.Filter.create_butter(sess,order,bandpass,btype='bandpass',fs=Fs,implementation='sos')

    # Channels to use
    channels = tuple([_ for _ in range(3,17)])

    # Data sources from LSL
    LSL_data_src = mp.source.InputLSLStream.create_marker_coupled_data_stream(sess,
                                                                             "type='EEG'",
                                                                             channels,
                                                                             relative_start=-0.4,
                                                                             marker_fmt='(^SPACE pressed$)|(^RSHIFT pressed$)')
    LSL_data_out = mp.source.OutputLSLStream.create_outlet(sess, 'outlet', 'type="Markers"', 1, channel_format='float32')

    # Data input tensors connected to LSL data sources
    online_input_data = mp.Tensor.create_from_handle(sess, (len(channels), 700), LSL_data_src)
    training_input_data = mp.Tensor.create_from_handle(sess, (len(channels), 700), LSL_data_src)

    # Data output tensors connected to LSL data sources
    online_output_data = mp.Tensor.create_for_volatile_output(sess, (1,2), LSL_data_out)

    # Initialization data circle buffers; the training graph will enqueue the training data to these buffers with each trial
    training_data = {'data'   : mp.CircleBuffer.create(sess, training_trials, mp.Tensor.create(sess, (len(channels), 700))),
                     'labels' : mp.CircleBuffer.create(sess, training_trials, mp.Scalar.create(sess, int))}

    # output classifier label
    pred_label = mp.Scalar.create(sess, int)

    # virtual tensors to connect the nodes in the online graph
    t_virt = [mp.Tensor.create_virtual(sess), # output of filter, input to resample
              mp.Tensor.create_virtual(sess), # output of resample, input to extract
              mp.Tensor.create_virtual(sess), # output of extract, input to xdawn
              mp.Tensor.create_virtual(sess), # output of xdawn, input to tangent space
              mp.Tensor.create_virtual(sess)] # output of tangent space, input to classifier

    classifier = mp.Classifier.create_logistic_regression(sess)

    # extraction indices
    start_time = 0.2
    end_time = 1.2
    extract_indices = [
        slice(None), # all channels
        slice(int(start_time*resample_fs), int(end_time*resample_fs)) # central 1s
    ]


    # add the enqueue node to the training graph, will automatically enqueue the data from the lsl
    mp.kernels.EnqueueKernel.add_to_graph(training_graph, training_input_data, training_data['data'])

    # online graph nodes
    mp.kernels.FiltFiltKernel.add_to_graph(online_graph, online_input_data, f, t_virt[0])
    mp.kernels.ResampleKernel.add_to_graph(online_graph, t_virt[0], resample_fs/Fs, t_virt[1])
    mp.kernels.ExtractKernel.add_to_graph(online_graph, t_virt[1], extract_indices, t_virt[2])
    mp.kernels.XDawnCovarianceKernel.add_to_graph(online_graph, t_virt[2],t_virt[3])
    mp.kernels.TangentSpaceKernel.add_to_graph(online_graph, t_virt[3], t_virt[4])
    mp.kernels.ClassifierKernel.add_to_graph(online_graph, t_virt[4], classifier, pred_label, online_output_data)

    online_graph.set_default_initialization_data(training_data['data'], training_data['labels'])

    # verify the training graph (i.e. schedule the nodes)
    training_graph.verify()

    # verify the online graph (i.e. schedule the nodes)
    online_graph.verify()

    # initialize the training graph
    training_graph.initialize()

    # Execute the training graph
    for t_num in range(training_trials):
        try:
            training_graph.execute()
            # Get the most recent marker and add the equivalent label to the training labels circle buffer.
            last_marker = LSL_data_src.last_marker()
            training_data['labels'].enqueue(mp.Scalar.create_from_value(sess, 1) if last_marker == 'SPACE pressed' else mp.Scalar.create_from_value(sess, 0))

            print(f"Training Trial {t_num+1} Complete")
        except:
            print("Training graph error...")

    # initialize the online graph with the collected data
    online_graph.initialize()

    # Execute the online graph
    Ntrials = 10
    for t_num in range(Ntrials):

        try:
            sts = online_graph.execute()
            y_bar = online_output_data.data
            print(f"\tTrial {t_num+1}: Max Probability = {y_bar}")
        except:
            print(f"Trial {t_num+1} raised error, status code: {sts}")



if __name__ == "__main__":
    main()
