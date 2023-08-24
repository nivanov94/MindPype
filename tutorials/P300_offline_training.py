# -*- coding: utf-8 -*-
"""
Created on Tues July 26 16:12:30 2022
@author: aaronlio
"""

# Create a simple graph for testing
import bcipy.bcipy as bcipy
import numpy as np
from datetime import datetime
import json, pickle
import matplotlib.pyplot as plt
from copy import deepcopy
def plot_func(data, name, tnum, Fs):
    x = np.arange(0, 1.4, 1/Fs)
    fig = plt.figure()
    plt.plot(data.data[0,:])
    plt.savefig(f'./images/{name}_{tnum}.png')
    plt.close(fig)


def plot_func2(data, name, tnum, Fs):
    x = np.arange(0, 1.4, 1/Fs)
    for i in range(data.shape[0]):
        fig = plt.figure()
        plt.plot(data.data[i,:])
        plt.savefig(f'./images/{name}_{tnum}_{i}.png')
        plt.close(fig)
    

def main(file):
    # create a session
    sess = bcipy.Session.create()
    online_graph = bcipy.Graph.create(sess)

    # Cosntants
    Fs = 128
    trial_len = 1.4
    resample_fs = 50

    # create a filter
    f = bcipy.Filter.create_fir(sess, fs=Fs, low_freq=1, high_freq=25, method='fir', fir_design='firwin', phase='minimum')
    channels = tuple([_ for _ in range(3,17)])

    # Data sources from LSL
    LSL_data_src = bcipy.source.InputLSLStream.create_marker_coupled_data_stream(sess, "type='EEG'", channels, relative_start=-0.2, marker_fmt='.*flash', marker_pred="type='Marker'") # type: ignore
    
    # training data sources from XDF file
    offline_data_src = bcipy.source.BcipXDF.create_class_separated(sess, file, ['flash'], channels=channels, relative_start=-0.2, Ns = np.ceil(Fs)) 

    #offline_data_src.trial_data['EEG']['time_series']['flash'] = offline_data_src.trial_data['EEG']['time_series']['flash']
    xdf_tensor = bcipy.containers.Tensor.create_from_data(sess, offline_data_src.trial_data['EEG']['time_series']['flash'].shape, offline_data_src.trial_data['EEG']['time_series']['flash'])
    
    init_d = pickle.load(open(r'C:\Users\lioa\Documents\mindset_refactor\output_list.pkl', 'rb'))[0]

    xdf_tensor = bcipy.containers.Tensor.create_from_data(sess, init_d.shape, init_d)

    # Create input tensors
    online_input_data = bcipy.Tensor.create_from_handle(sess, (len(channels), Fs), LSL_data_src)
    #offline_input_data = bcipy.Tensor.create_from_handle(sess, (len(channels), 180), offline_data_src)

    # Create circle buffer to store true labels
    labels = bcipy.CircleBuffer.create(sess, len(offline_data_src.trial_data['Markers']['time_series']), bcipy.Scalar.create(sess, int))

    ls = offline_data_src.trial_data['Markers']['time_series']
    target_pos = None
    task_series_list = []
    # Convert string markers to integer labels
    for i in range(len(ls)):
        scalar = bcipy.Scalar.create(sess, int)
        if list(json.loads(ls[i][0]).keys())[0] == 'target':
            target_pos = list(json.loads(ls[i][0]).values())[0]
        elif list(json.loads(ls[i][0]).keys())[0] == 'flash' and target_pos != None:
            if list(json.loads(ls[i][0]).values())[0][0] == target_pos:
                scalar.data = 1
                task_series_list.append(1)
            else:
                scalar.data = 0
                task_series_list.append(0)
            
            labels.enqueue(scalar)

    # Remove all markers that are not 'flash'
    i = 0
    l = len(ls)
    
    offline_data_src.trial_data['Markers']['time_series'] = np.squeeze(offline_data_src.trial_data['Markers']['time_series'])
    while i < l:
        if list(json.loads(offline_data_src.trial_data['Markers']['time_series'][i]).keys())[0] != 'flash':
            offline_data_src.trial_data['Markers']['time_series'] = np.delete(offline_data_src.trial_data['Markers']['time_series'], [i], axis=0)
            offline_data_src.trial_data['Markers']['time_stamps'] = np.delete(offline_data_src.trial_data['Markers']['time_stamps'], [i], axis=0)
            l -= 1
        else:
            i += 1

    # Convert flash markers to target/non-target labels
    for i in range(len(offline_data_src.trial_data['Markers']['time_series'])):    
        if task_series_list[i] == 1:
            offline_data_src.trial_data['Markers']['time_series'][i] = json.dumps({'target': 1})
        elif task_series_list[i] == 0:
            offline_data_src.trial_data['Markers']['time_series'][i] = json.dumps({'non-target': 0})

    # online graph data containers (i.e. graph edges)
    pred_probs = bcipy.Tensor.create_virtual(sess) # output of classifier, input to label
    pred_label = bcipy.Tensor.create_virtual(sess) 

    t_virt = [bcipy.Tensor.create_virtual(sess), # output of filter, input to resample
              bcipy.Tensor.create_virtual(sess), # output of resample, input to extract
              bcipy.Tensor.create_virtual(sess), # output of extract, input to xdawn
              bcipy.Tensor.create_virtual(sess), # output of xdawn, input to tangent space
              bcipy.Tensor.create_virtual(sess),  # output of tangent space, input to classifier
              bcipy.Tensor.create_virtual(sess),
              bcipy.Tensor.create_virtual(sess),
              bcipy.Tensor.create_virtual(sess)]
    
    start_time = 0
    end_time = 1
        
    #extract_indices = [":", [_ for _ in range(int(start_time*Fs + len(f.coeffs['fir']))-1,int(np.ceil(end_time*Fs + len(f.coeffs['fir']))))]]# All epochs, all channels, start_time to end_time
    
    #start_time2 = 0.2
    #end_time2 = 1.2

    #extract_indices2 = [":", [_ for _ in range(int(start_time2*Fs),int(np.ceil(end_time2*Fs)))]]# All epochs, all channels, start_time to end_time

    classifier = bcipy.Classifier.create_logistic_regression(sess)
    #bcipy.kernels.FiltFiltKernel.add_filtfilt_node(online_graph, online_input_data, f1, t_virt[0], axis=1)
    #node_1 = bcipy.kernels.PadKernel.add_pad_node(online_graph, online_input_data, t_virt[0], pad_width=((0,0), (len(f.coeffs['fir']), len(f.coeffs['fir']))), mode='edge')
    node_2 = bcipy.kernels.FilterKernel.add_filter_node(online_graph, online_input_data, f, t_virt[1], axis=1)

    #node_3 = bcipy.kernels.ExtractKernel.add_extract_node(online_graph, t_virt[1], extract_indices, t_virt[2])
    node_4 = bcipy.kernels.BaselineCorrectionKernel.add_baseline_node(online_graph, t_virt[1], t_virt[4], baseline_period=[0*Fs, 0.2*Fs])
    #node_5 = bcipy.kernels.ExtractKernel.add_extract_node(online_graph, t_virt[3], extract_indices2, t_virt[4])
    node_6 = bcipy.kernels.ResampleKernel.add_resample_node(online_graph, t_virt[4], resample_fs/Fs, t_virt[5], axis=1)
    node_7 = bcipy.kernels.XDawnCovarianceKernel.add_xdawn_covariance_node(online_graph, t_virt[5], t_virt[6], num_filters=4, estimator="lwf", xdawn_estimator="lwf")
    node_8 = bcipy.kernels.TangentSpaceKernel.add_tangent_space_node(online_graph, t_virt[6], t_virt[7], metric="riemann")
    node_9 = bcipy.kernels.ClassifierKernel.add_classifier_node(online_graph, t_virt[7], classifier , pred_label, pred_probs)

    if online_graph.verify() != bcipy.BcipEnums.SUCCESS:
        print("Test Failed D=")
        return bcipy.BcipEnums.INVALID_GRAPH

    # initialize the classifiers (i.e., train the classifier)
    if online_graph.initialize(xdf_tensor, labels) != bcipy.BcipEnums.SUCCESS:
        
        print("Init Failed D=")
        return bcipy.BcipEnums.INITIALIZATION_FAILURE

    in1 = node_2.kernel.init_inputs[0].data
    out3 = node_4.kernel.init_outputs[0].data
    print(out3.shape)
    out4 = node_4.kernel.init_outputs[0].data
    out6 = node_6.kernel.init_outputs[0].data
    P_7 = node_7.kernel.init_outputs[0].data
    ref_8 = node_8.kernel._tangent_space.reference_
    out9 = node_9.kernel.init_outputs[1].data

    initialize_outputs = [deepcopy(in1), deepcopy(out3), deepcopy(out4), deepcopy(out6), deepcopy(P_7), deepcopy(ref_8), deepcopy(out9)]
    pickle.dump(initialize_outputs, open("new_outputs.pkl", "wb"))

    """x = np.arange(0, 8.015625, 1/Fs)
    fig = plt.figure() 
    plt.plot(x, node_2.kernel.init_outputs[0].data[0,0,:])
    plt.savefig('filtered.png')
    plt.close(fig)

    x = np.arange(0, 1.4, 1/Fs)
    fig2 = plt.figure() 
    plt.plot(x, node_4.kernel.init_outputs[0].data[0,0,:])
    plt.savefig('baseline.png')
    plt.close(fig2)

    x = np.arange(0, 1, 1/resample_fs)
    fig3 = plt.figure()
    plt.plot(x, node_6.kernel.init_outputs[0].data[0,0,:])
    plt.savefig('resampled.png')
    plt.close(fig3)"""

    #create .log file in scrap folder to write timestamps
    

    
    
    # Run the online trials
    for t_num in range(100):
        sts = online_graph.execute()
        #print(f._coeffs['fir'])
        #print(sum(f._coeffs['fir']))
        #print(online_input_data.shape)
        #plot_func(online_input_data, 'raw_data', t_num, 128)
        #plot_func(t_virt[2], 'filtered_data', t_num, 128)
        #plot_func(t_virt[4], 'baseline_data', t_num, 128)
        #plot_func(t_virt[5], 'resampled_data', t_num, 50)
        #file = open("bcipy/scrap/timestamps.log", "a", encoding="utf-8")
        #file.write("start_time: " + str(datetime.utcnow()) + "\n")
        #file.write(f"marker timestamps: {LSL_data_src.marker_timestamps[-1]} \n")
        #file.write("first data timestamp: " + str(LSL_data_src.first_data_timestamp) + "\n")
        #file.write(f"First 10 Data samples from channel 1: {online_input_data.data[0,0:10]}" + "\n")
        #file.write(f"Output probabilities: {pred_probs.data}" + "\n")
        #file.close()
        in1 = node_1.kernel.inputs[0].data
        out3 = node_3.kernel.outputs[0].data
        out4 = node_4.kernel.outputs[0].data
        out6 = node_6.kernel.outputs[0].data
        P_7 = node_7.kernel.outputs[0].data
        ref_8 = node_8.kernel.outputs[0].data
        #out9 = node_9.kernel._classifier._classifier.coef_[0]

        list_of_outputs = [in1, out3, out6, P_7, ref_8, out9]
        pickle.dump(list_of_outputs, open(f"bcipy/scrap/outputs{t_num}.pkl", "wb"))
        

        if sts == bcipy.BcipEnums.SUCCESS:
            # print the value of the most recent trial
            print(f"\t{datetime.utcnow()}: Probabilities = {pred_probs.data}")
        else:
            print(f"Trial {t_num+1} raised error, status code: {sts}")
    
    
    print("Test Passed =D")

if __name__ == "__main__":
    #files = ["C:/Users/lioa/Documents/Mindset_Data/data/sub-P003/sourcedata/sub-P003_ses-S002_task-vP300+2x2_run-001.xdf",
    #         "C:/Users/lioa/Documents/Mindset_Data/data/sub-P003/sourcedata/sub-P003_ses-S002_task-vP300+2x2_run-002.xdf"]
    files = [r'c:\Users\lioa\Documents\Mindset_Data\data\sub-P004\sourcedata\sub-P004_ses-S001_task-vP300+2x2_run-001.xdf']
    #files = ["C:/Users/student_admin.PRISMLAB/Documents/Mindset_Data/data/sub-P002/sourcedata/sub-P002_ses-S001_task-vP300+2x2_run-001.xdf",
    #         "C:/Users/student_admin.PRISMLAB/Documents/Mindset_Data/data/sub-P002/sourcedata/sub-P002_ses-S001_task-vP300+2x2_run-002.xdf",
    #         "C:/Users/student_admin.PRISMLAB/Documents/Mindset_Data/data/sub-P002/sourcedata/sub-P002_ses-S001_task-vP300+2x2_run-003.xdf",
    #         "C:/Users/student_admin.PRISMLAB/Documents/Mindset_Data/data/sub-P002/sourcedata/sub-P002_ses-S001_task-vP300+2x2_run-004.xdf"]
    main(files)

