import bcipy
import pylsl
import pyxdf
import numpy as np
import argparse



from scipy import signal



def main():
    
    parser = argparse.ArgumentParser(prog='Accompanying processing back-end to work with BCI Rocket',
                                     description='Gets data from LSL for processing and sends predicted labels to LSL')
    parser.add_argument('--tasks', nargs=3, type=str, required=False, default=['task1','task2','task3'])
    parser.add_argument('--fs', nargs=1, type=int, required=False, default=[250])
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
    
    sel_chs = ('FCz',
               'Fz',
               #'F3',
               #'F7',
               'FC3',
               #'T7',
               'C5',
               'C3',
               'C1',
               'Cz',
               'CP3',
               'CPz',
               'P7',
               'P5',
               'P3',
               'P1',
               'Pz',
               'PO3',
               'Oz',
               'PO4',
               'P8',
               'P6',
               'P4',
               'P2',
               'CP4',
               #'T8',
               'C6',
               'C4',
               'C2',
               'FC4',
               #'F4',
               #'F8'
              )
    
    channels = [ch_map[ch] for ch in sel_chs]

    selected_tasks = tuple(args.tasks)
    print(selected_tasks)

        
    Fs = args.fs[0]
    #print(Fs)
    
    crop_indices = [_ for _ in range(Fs,4*Fs)]
    Nc = len(channels)
    Ns = len(crop_indices)
    
    ## create bcipy session
    s = bcipy.Session()
    
    
    ## create offline and online graphs

    print("Creating LSL input")
    eeg_src = bcipy.source.LSLStream(s, "type='EEG' and channel_count=32",
                                  channels=channels,
                                  marker=True,
                                  marker_fmt='label_\d_name_*',
                                  marker_pred="name='bci_rocket_marker'")
    print("Done creating LSL input.")
    t_trial = bcipy.Tensor.create_from_handle(s, (Nc, Ns+2*Fs), eeg_src)
    t_file_trial = bcipy.Tensor.create(s, (Nc, Ns+2*Fs))

    t_filtered = bcipy.Tensor.create_virtual(s)
    t_cropped = bcipy.Tensor.create(s,(Nc,Ns))
    t_csp = bcipy.Tensor.create_virtual(s)
    t_var = bcipy.Tensor.create_virtual(s)
    t_log = bcipy.Tensor.create_virtual(s)
    t_cov = bcipy.Tensor.create_virtual(s)
    
    s_true = bcipy.Scalar.create(s,int)
    s_pred = bcipy.Scalar.create(s,int)
    s_artifact = bcipy.Scalar.create(s, int)
    
    f_bp_filt = bcipy.Filter.create_butter(s, 4, (8,30), btype='bandpass', implementation='sos', fs=Fs)
    c_lda = bcipy.Classifier.create_LDA(s,solver='lsqr',shrinkage='auto')
    
    # create the circle buffer to contain the data
    template_tensor = bcipy.Tensor.create(s, (Nc, Ns))
    template_scalar = bcipy.Scalar.create(s, int)

    
    max_buffer_length = 60 * len(selected_tasks) # 60 from the present session

    cb_prev_trials = bcipy.CircleBuffer.create(s, max_buffer_length, template_tensor)
    cb_labels = bcipy.CircleBuffer.create(s, max_buffer_length, template_scalar)
    
    # offline
    offline_graph = bcipy.Graph.create(s)
    
    bcipy.kernels.FiltFiltKernel.add_filtfilt_node(offline_graph, 
                                                   t_trial,
                                                   f_bp_filt,
                                                   t_filtered,
                                                   axis=1)
    
    bcipy.kernels.ExtractKernel.add_extract_node(offline_graph,
                                                 t_filtered,
                                                 (":",crop_indices),
                                                 t_cropped)
    
    bcipy.kernels.EnqueueKernel.add_enqueue_node(offline_graph,
                                                 t_cropped,
                                                 cb_prev_trials)
    
    bcipy.kernels.EnqueueKernel.add_enqueue_node(offline_graph,
                                                 s_true,
                                                 cb_labels)
    
    # online
    online_graph = bcipy.Graph.create(s)
    bcipy.kernels.FiltFiltKernel.add_filtfilt_node(online_graph, 
                                                   t_trial,
                                                   f_bp_filt,
                                                   t_filtered,
                                                   axis=1)
    
    bcipy.kernels.ExtractKernel.add_extract_node(online_graph,
                                                 t_filtered,
                                                 (":",crop_indices),
                                                 t_cropped)
    
    bcipy.kernels.EnqueueKernel.add_enqueue_node(online_graph,
                                                 t_cropped,
                                                 cb_prev_trials,
                                                 s_artifact)
    
    bcipy.kernels.EnqueueKernel.add_enqueue_node(online_graph,
                                                 s_true,
                                                 cb_labels,
                                                 s_artifact)
    
    bcipy.kernels.CommonSpatialPatternKernel.add_uninitialized_CSP_node(online_graph,
                                                                        t_cropped,
                                                                        t_csp,
                                                                        cb_prev_trials,
                                                                        cb_labels,
                                                                        4, # number of filters
                                                                        3) # number of classes
    
    bcipy.kernels.VarKernel.add_var_node(online_graph, t_csp, t_var, 1, 1)
    bcipy.kernels.LogKernel.add_log_node(online_graph, t_var, t_log)
    
    bcipy.kernels.ClassifierKernel.add_classifier_node(online_graph,
                                                       t_log,
                                                       c_lda,
                                                       s_pred)
    
    bcipy.kernels.CovarianceKernel.add_covariance_node(online_graph, t_cropped, t_cov)
    bcipy.kernels.RiemannPotatoKernel.add_riemann_potato_node(online_graph,
                                                              t_cov, 
                                                              cb_prev_trials,
                                                              s_artifact)
    
    
    
    graphs = [offline_graph, online_graph]

    # verify graphs
    for g in graphs:
        sts = g.verify()
        if sts != bcipy.BcipEnums.SUCCESS:
            raise Exception(sts)
    

    # create lsl inlet and outlet to communicate with BCI rocket
    lsl_marker_inlet = pylsl.StreamInlet(pylsl.resolve_byprop('type', 'Markers')[0]) # todo verify there will only be one marker stream
    outlet_info = pylsl.StreamInfo('Marker-PredictedLabel', 'Markers', channel_format='string')
    lsl_marker_outlet = pylsl.StreamOutlet(outlet_info)
    
    for i_b in range(4): # four blocks total
        print(f"Block: {i_b+1}")    
        if i_b < 2:
            active_graph = offline_graph
        else:
            active_graph = online_graph
            
        # initialize the active graph
        sts = active_graph.initialize()
        if sts != bcipy.BcipEnums.SUCCESS:
            raise Exception(sts)
            
        for i_t in range(45): # 45 trials per block
            print(f"\ttrial: {i_t+1}")
            
            
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
            
    
            # set the true label scalar
            s_true.data = true_label
        
            # process trial
            sts = active_graph.execute(label=true_label)
            if sts != bcipy.BcipEnums.SUCCESS:
                raise Exception(sts)
           
                
            # create the outlet marker
            if i_b > 1:
                # check if data was clean
                if s_artifact.data == 1: #clean trial
                    outlet_marker = f"Block:{i_b+1}_Trial:{i_t+1}_Pred:{s_pred.data}"
                else:
                    outlet_marker = f"Block:{i_b+1}_Trial:{i_t+1}_Pred:{-2}"
        
                
                # push predicated label to marker outlet
                lsl_marker_outlet.push_sample([outlet_marker])


    input("Session complete. Please Enter to terminate program...")
                
if __name__ == "__main__":
    main()
