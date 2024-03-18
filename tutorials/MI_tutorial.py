import mindpype as mp
import numpy as np
import pylsl
import argparse

def main():

    parser = argparse.ArgumentParser(prog='Accompanying processing back-end to work with BCI Rocket',
                                     description='Gets data from LSL for processing and sends predicted labels to LSL')
    parser.add_argument('--tasks', nargs=3, type=str, required=False, default=['task1','task2','task3'])
    parser.add_argument('--fs', nargs=1, type=int, required=False, default=[250])
    parser.add_argument('--lsl', nargs=1, type=bool, required=False, default=False)
    args = parser.parse_args()

    ch_map =  {'FCz' : 0,
               'Fz'  : 1,
               'F3'  : 2,
               'F7'  : 3,
               'FC3' : 4,
               'T7'  : 5,
               'C5'  : 6,
               'C3'  : 7,
               'C1'  : 8,
               'Cz'  : 9,
               'CP3' : 10,
               'CPz' : 11,
               'P7'  : 12,
               'P5'  : 13,
               'P3'  : 14,
               'P1'  : 15,
               'Pz'  : 16,
               'PO3' : 17,
               'Oz'  : 18,
               'PO4' : 19,
               'P8'  : 20,
               'P6'  : 21,
               'P4'  : 22,
               'P2'  : 23,
               'CP4' : 24,
               'T8'  : 25,
               'C6'  : 26,
               'C4'  : 27,
               'C2'  : 28,
               'FC4' : 29,
               'F4'  : 30,
               'F8'  : 31}

    sel_chs = ('FCz', 'Fz', 'F3', 'F7', 'FC3', 'T7', 'C5', 'C3', 'C1',
               'Cz', 'CP3', 'CPz', 'P7', 'P5', 'P3', 'P1', 'Pz', 'PO3',
               'Oz', 'PO4', 'P8', 'P6', 'P4', 'P2', 'CP4', 'T8', 'C6',
               'C4', 'C2', 'FC4', 'F4', 'F8'
              )

    channels = [ch_map[ch] for ch in sel_chs]

    selected_tasks = tuple(args.tasks)
    print(selected_tasks)

    Fs = args.fs[0]

    crop_indices = slice(Fs, 4*Fs) # extract the central 3 seconds of the trial
    Nc = len(channels)
    Ns = len(crop_indices)

    ## create mindpype session
    s = mp.Session()


    ## create offline and online graphs
    if args.lsl:
        eeg_src = mp.source.LSLStream.create_marker_coupled_data_stream(s,
                                    "type='EEG' and channel_count=32",
                                    channels=channels,
                                    marker=True,
                                    marker_fmt='label_\d_name_*',
                                    marker_pred="name='bci_rocket_marker'")
        t_trial = mp.Tensor.create_from_handle(s, (Nc, Ns+2*Fs), eeg_src)
    else:
        t_trial = mp.Tensor.create(s, (Nc, Ns+2*Fs))

    t_filtered = mp.Tensor.create_virtual(s)
    t_cropped = mp.Tensor.create(s,(Nc,Ns))
    t_csp = mp.Tensor.create_virtual(s)
    t_var = mp.Tensor.create_virtual(s)
    t_log = mp.Tensor.create_virtual(s)
    t_cov = mp.Tensor.create_virtual(s)

    s_true = mp.Scalar.create(s,int)
    s_pred = mp.Scalar.create(s,int)
    s_artifact = mp.Scalar.create(s, int)

    f_bp_filt = mp.Filter.create_butter(s, 4, (8,30), btype='bandpass', implementation='sos', fs=Fs)
    c_lda = mp.Classifier.create_LDA(s,solver='lsqr',shrinkage='auto')

    # create the circle buffer to contain the data
    template_tensor = mp.Tensor.create(s, (Nc, Ns+2*Fs))
    template_scalar = mp.Scalar.create(s, int)


    max_buffer_length = 60 * len(selected_tasks) # 60 from the present session

    cb_prev_trials = mp.CircleBuffer.create(s, max_buffer_length, template_tensor)
    cb_labels = mp.CircleBuffer.create(s, max_buffer_length, template_scalar)

    # offline
    offline_graph = mp.Graph.create(s)
    mp.kernels.EnqueueKernel.add_enqueue_node(offline_graph,
                                              t_trial,
                                              cb_prev_trials)

    mp.kernels.EnqueueKernel.add_enqueue_node(offline_graph,
                                              s_true,
                                              cb_labels)

    # online
    online_graph = mp.Graph.create(s)
    mp.kernels.FiltFiltKernel.add_filtfilt_node(online_graph,
                                                   t_trial,
                                                   f_bp_filt,
                                                   t_filtered,
                                                   axis=1)

    mp.kernels.ExtractKernel.add_extract_node(online_graph,
                                                 t_filtered,
                                                 [slice(None),crop_indices],
                                                 t_cropped)

    mp.kernels.EnqueueKernel.add_enqueue_node(online_graph,
                                                 t_trial,
                                                 cb_prev_trials,
                                                 s_artifact)

    mp.kernels.EnqueueKernel.add_enqueue_node(online_graph,
                                                 s_true,
                                                 cb_labels,
                                                 s_artifact)

    mp.kernels.CommonSpatialPatternKernel.add_uninitialized_CSP_node(online_graph,
                                                                        t_cropped,
                                                                        t_csp,
                                                                        num_filts=4, # number of filters
                                                                        Ncls=3) # number of classes

    mp.kernels.VarKernel.add_var_node(online_graph, t_csp, t_var, 1, 1)
    mp.kernels.LogKernel.add_log_node(online_graph, t_var, t_log)

    mp.kernels.ClassifierKernel.add_classifier_node(online_graph,
                                                       t_log,
                                                       c_lda,
                                                       s_pred,
                                                       num_classes=3)

    mp.kernels.CovarianceKernel.add_covariance_node(online_graph, t_cropped, t_cov)
    mp.kernels.RiemannPotatoKernel.add_riemann_potato_node(online_graph,
                                                            t_cov, s_artifact)



    graphs = [offline_graph, online_graph]

    # add default initialization data to the online graph
    online_graph.set_default_init_data(cb_prev_trials, cb_labels)

    # verify graphs
    for g in graphs:
        g.verify()

    # create lsl inlet and outlet to communicate with BCI rocket
    if args.lsl:
        lsl_marker_inlet = pylsl.StreamInlet(pylsl.resolve_byprop('type', 'Markers')[0]) # todo verify there will only be one marker stream
        outlet_info = pylsl.StreamInfo('Marker-PredictedLabel', 'Markers', channel_format='string')
        lsl_marker_outlet = pylsl.StreamOutlet(outlet_info)

    for i_b in range(4): # four blocks total
        print(f"Block: {i_b+1}")
        if i_b < 2:
            active_graph = offline_graph
        else:
            active_graph = online_graph

        if not args.lsl:
            # generate random labels sequence
            block_labels = np.concatenate((np.zeros((15,)), np.ones((15,)), 2*np.ones((15,))))
            np.random.shuffle(block_labels)

        # initialize the active graph
        active_graph.initialize()

        for i_t in range(45): # 45 trials per block
            print(f"\ttrial: {i_t+1}")

            if args.lsl:
                # wait for lsl marker
                true_label = -1
                print("Waiting for marker...")
                while true_label == -1:
                    inlet_marker, _ = lsl_marker_inlet.pull_sample(timeout=0.1)

                    if inlet_marker != None and inlet_marker[0].find("cue_") != -1 and inlet_marker[0] != 'cue_rest':
                        start_index = inlet_marker[0].find("label_") + len("label_")
                        end_index = inlet_marker[0].find("_name")
                        label_str = inlet_marker[0][start_index:end_index]
                        true_label = int(label_str)
            else:
                true_label = block_labels[i_t]
                t_trial.assign_random_data(vmin=-10,vmax=10)


            # set the true label scalar
            s_true.data = true_label

            # process trial
            active_graph.execute(label=true_label)

            # create the outlet marker
            if i_b > 1:
                # check if data was clean
                if s_artifact.data == 1: #clean trial
                    outlet_marker = f"Block:{i_b+1}_Trial:{i_t+1}_Pred:{s_pred.data}"
                else:
                    outlet_marker = f"Block:{i_b+1}_Trial:{i_t+1}_Pred:{-2}"


                # push predicated label to marker outlet
                print(outlet_marker)
                if args.lsl:
                    lsl_marker_outlet.push_sample([outlet_marker])


    input("Session complete. Please Enter to terminate program...")

if __name__ == "__main__":
    main()
