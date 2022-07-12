# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 10:17:16 2020

@author: Nick
"""

from classes import session, block, tensor, source, scalar, array, circle_buffer, bcip_enums
import classes.filter as bcip_filter
import kernels
import numpy as np
from scipy import io
from os import path


def main():
    
    trial_len = 2
    Fs = 1000
    Nclasses = 3
    window_len = 20
    
    trial_info = io.loadmat(path.join("D:/School/GRAD/Thesis/Pilot Work/debugging_data/feb25","trial_labels.mat"))
    accepted_trials = trial_info['artifact_labels'][0]
    labels = trial_info['total_sess_labels'][0]
    
    regs = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,'auto']
    r_accuracy = {}
    
    cropped_indices = (
        tuple([_ for _ in range(250,1750)]),
        ":")
    
    for r in regs:
        
    
        s = session.Session.create()
    
        label_map = {0:'class1_trials',1:"class2_trials",2:"class3_trials"}
        dims = (tuple([_ for _ in range(trial_len*Fs)]),
            (6,7,9,11,12,13,14,16,17))
        src = source.BcipMatFile.create(s,'class_trials.mat',
                                    "D:/School/GRAD/Thesis/Pilot Work/debugging_data/feb25",
                                    label_map,dims)
    
        # tensors
        raw_data = tensor.Tensor.create_from_handle(s,
                                                (len(dims[0]),len(dims[1])),
                                                src)
        cov = tensor.Tensor.create(s,(len(dims[1]),len(dims[1])))
        lpfilt_data = tensor.Tensor.create_virtual(s)
        bpfilt_data = tensor.Tensor.create_virtual(s)
        cropped_data = tensor.Tensor.create_virtual(s)
    
        triclass_training_labels = tensor.Tensor.create_from_data(s,(Nclasses*window_len,),
                                                     np.concatenate((np.zeros(window_len,),
                                                                     np.ones(window_len,),
                                                                     2*np.ones(window_len,)),
                                                                    axis=None))
        
        stage1_training_labels = tensor.Tensor.create_from_data(s,(Nclasses*window_len,),
                                                     np.concatenate((np.zeros(2*window_len,),
                                                                     np.ones(window_len,)),
                                                                    axis=None))

        stage2_training_labels = tensor.Tensor.create_from_data(s,((Nclasses-1)*window_len,),
                                                     np.concatenate((np.zeros(window_len,),
                                                                     np.ones(window_len,)),
                                                                    axis=None))

        # scalars
        tric_y_bar = scalar.Scalar.create(s,int)
        s1_y_bar = scalar.Scalar.create(s,int)
        s2_y_bar = scalar.Scalar.create(s,int)
    
        # arrays
        prev_covs = array.Array.create(s,Nclasses,
                                   circle_buffer.CircleBuffer.create(s,
                                                                     window_len,
                                                                     cov))
        prev_mvmt_covs = array.Array.create(s,Nclasses-1,
                                   circle_buffer.CircleBuffer.create(s,
                                                                     window_len,
                                                                     cov))
    
        # filter
        lp_filt = bcip_filter.Filter.create_butter(s,4,30,'low','sos',Fs)
        hp_filt = bcip_filter.Filter.create_butter(s,4,8,'high','sos',Fs)
    
        init_blocks = window_len // 4
        fb_blocks = 30-init_blocks
    
        for i in range(init_blocks):
            # create block
            b = block.Block.create(s,Nclasses,(4,4,4))
            kernels.FiltFiltKernel.add_filtfilt_node(b.trial_processing_graph,
                                                          raw_data,
                                                          lp_filt,
                                                          lpfilt_data)
            kernels.FiltFiltKernel.add_filtfilt_node(b.trial_processing_graph,
                                                 lpfilt_data,
                                                 hp_filt,
                                                 bpfilt_data)
            kernels.ExtractKernel.add_extract_node(b.trial_processing_graph,
                                               bpfilt_data,
                                               cropped_indices,
                                               cropped_data)
        
            kernels.CovarianceKernel.add_covariance_node(b.trial_processing_graph,
                                                                cropped_data,
                                                                cov,
                                                                0.05)
    
        for i in range(fb_blocks):
            b = block.Block.create(s,Nclasses,(4,4,4))
            kernels.FiltFiltKernel.add_filtfilt_node(b.trial_processing_graph,
                                                          raw_data,
                                                          lp_filt,
                                                          lpfilt_data)
            kernels.FiltFiltKernel.add_filtfilt_node(b.trial_processing_graph,
                                                 lpfilt_data,
                                                 hp_filt,
                                                 bpfilt_data)
            kernels.ExtractKernel.add_extract_node(b.trial_processing_graph,
                                               bpfilt_data,
                                               cropped_indices,
                                               cropped_data)
        
            kernels.CovarianceKernel.add_covariance_node(b.trial_processing_graph,
                                                                cropped_data,
                                                                cov,
                                                                0.05)
            kernels.riemann_ts_rLDA_classifier.RiemannTangentSpacerLDAClassifierKernel.add_untrained_riemann_tangent_space_rLDA_node(
                b.trial_processing_graph,cov,tric_y_bar,prev_covs,triclass_training_labels,r,'lsqr')
            
            kernels.riemann_ts_rLDA_classifier.RiemannTangentSpacerLDAClassifierKernel.add_untrained_riemann_tangent_space_rLDA_node(
                b.trial_processing_graph,cov,s1_y_bar,prev_covs,stage1_training_labels,r,'lsqr')
            
            kernels.riemann_ts_rLDA_classifier.RiemannTangentSpacerLDAClassifierKernel.add_untrained_riemann_tangent_space_rLDA_node(
                b.trial_processing_graph,cov,s2_y_bar,prev_mvmt_covs,stage2_training_labels,r,'lsqr')
    
        sts = s.verify()
        if sts != bcip_enums.BcipEnums.SUCCESS:
            print("verification error...")
            return sts
    
    
        tric_y_bars = []
        twos_y_bars = []
        session_trial_cnt = 0
        block_num = 1
        while s.remaining_blocks != 0:
            block_trial_cnt= [0,0,0]
            s.start_block()
            while sum(block_trial_cnt) < 12:
            
                label = labels[session_trial_cnt]
                if label < 0:
                    label = 0
                
                sts = s.execute_trial(label)
                if sts != bcip_enums.BcipEnums.SUCCESS:
                    print("EXE ERROR")
                if accepted_trials[session_trial_cnt] == 0:
                    print("rejected trial")
                    s.reject_trial()
                else:
                    if block_num > init_blocks:
                        # get the predicted label
                        tric_y_bars.append(tric_y_bar.data)
                        print("3 class predicted label: {}".format(tric_y_bar.data))
                        
                        if s1_y_bar.data == 1:
                            twos_y_bars.append(2)
                            print("Two stage predicted label: 2")
                        else:
                            twos_y_bars.append(s2_y_bar.data)
                            print("Two stage predicted label: {}".format(s2_y_bar.data))
                    
                    block_trial_cnt[label] += 1
                    print(block_trial_cnt)
                
                    # add the trial to the appropriate queue
                    prev_covs.get_element(label).enqueue(cov)
                    if label == 0 or label == 1:
                        prev_mvmt_covs.get_element(label).enqueue(cov)
                
                session_trial_cnt += 1
        
            print("Finished Block {}".format(block_num))
            s.close_block()
            block_num += 1
        
        tric_y_bars = np.asarray(tric_y_bars)
        twos_y_bars = np.asarray(twos_y_bars)
        
        y = [l for l, t in zip(labels, accepted_trials) if t == 1]
        y = np.asarray(y[Nclasses*window_len:])
    
        r_accuracy[r] = {'3c' : (sum(y == tric_y_bars),sum(y == tric_y_bars)/y.shape[0]),
                         '2s' : (sum(y == twos_y_bars),sum(y == tric_y_bars)/y.shape[0])}
    
        #print(r_accuracy)
    print(r_accuracy)
        
#    # Riemann potato field artifact filtering
#    pops = {'filt' : bcip_filter.create_butter(4,(1,20),'bandpass','sos',Fs),
#            'elec_groups' : [(i,i+1) for i in range(0,len(dims[1]),2)],
#            'tr_data' : [circle_buffer.CircleBuffer.create(s,36,tensor.Tensor.create(s,(2,2)))],
#            'labels'  : [scalar.Scalar.create(s,int) for _ in range(0,len(dims[1]),2)],
#            'scores'  : [tensor.Tensor.create(s,(1,1)) for _ in range(0,len(dims[1]),2)]}
#    
#    gen_groups = {'filt' : bcip_filter.create_butter(4,(8,30),'bandstop','sos',Fs),
#            'elec_groups' : [(i,i+1,i+2,i+3,i+4) for i in range(0,len(dims[1]),5)],
#            'tr_data' : [circle_buffer.CircleBuffer.create(s,36,tensor.Tensor.create(s,(5,5)))],
#            'labels'  : [scalar.Scalar.create(s,int) for _ in range(0,len(dims[1]),2)],
#            'scores'  : [tensor.Tensor.create(s,(1,1)) for _ in range(0,len(dims[1]),2)]}
#    
#    occ_group = {'filt' : bcip_filter.create_butter(4,(1,20),'bandpass','sos',Fs),
#            'elec_groups' : [(0,1,2)],
#            'tr_data' : [circle_buffer.CircleBuffer.create(s,36,tensor.Tensor.create(s,(3,3)))],
#            'labels'  : [scalar.Scalar.create(s,int)],
#            'scores'  : [tensor.Tensor.create(s,(1,1))]}
#
#    face_emg_group = {'filt' : bcip_filter.create_butter(4,(55,95),'bandpass','sos',Fs),
#            'elec_groups' : [(0,1,2)],
#            'tr_data' : [circle_buffer.CircleBuffer.create(s,36,tensor.Tensor.create(s,(3,3)))],
#            'labels'  : [scalar.Scalar.create(s,int)],
#            'scores'  : [tensor.Tensor.create(s,(1,1))]}
#
#    neck_emg_group = {'filt' : bcip_filter.create_butter(4,(55,95),'bandpass','sos',Fs),
#            'elec_groups' : [(17,18,19)],
#            'tr_data' : [circle_buffer.CircleBuffer.create(s,36,tensor.Tensor.create(s,(3,3)))],
#            'labels'  : [scalar.Scalar.create(s,int)],
#            'scores'  : [tensor.Tensor.create(s,(1,1))]}
    
    
    
    
    
if __name__ == "__main__":
    main()