# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 13:47:46 2020

@author: Nick
"""

from classes import session, block, tensor, source, scalar, array, circle_buffer, bcip_enums
import classes.filter as bcip_filter
import kernels
import numpy as np
from scipy import io
from os import path


def print_stats(y,y_bar,name):
    correct = np.sum(y == y_bar)
    total = y.shape[0]
    
    print("Classifier: ",name)
    print("Accuracy = ", correct / total)
    print("Correct (total): ", correct, " (", total, ")")
    print("\n\n")

def main():
    
    trial_len = 2
    Fs = 1000
    Nclasses = 3
    window_len = 60
    
    trial_info = io.loadmat(path.join("D:/School/GRAD/Thesis/Pilot Work/debugging_data/feb25","trial_labels.mat"))
    accepted_trials = trial_info['artifact_labels'][0]
    labels = trial_info['total_sess_labels'][0]
        
    
    s = session.Session.create()
    
    label_map = {0:'class1_trials',1:"class2_trials",2:"class3_trials"}
    dims = (tuple([_ for _ in range(trial_len*Fs)]),
            (6,7,9,11,12,13,14,16,17))
    src = source.BcipMatFile.create(s,'class_trials.mat',
                                    "D:/School/GRAD/Thesis/Pilot Work/debugging_data/feb25",
                                    label_map,dims)
    
    cropped_indices = (
        tuple([_ for _ in range(250,1750)]),
        ":")
    
    # tensors
    raw_data = tensor.Tensor.create_from_handle(s,
                                                (len(dims[0]),len(dims[1])),
                                                src)
    cov = tensor.Tensor.create(s,(len(dims[1]),len(dims[1])))
    lpfilt_data = tensor.Tensor.create_virtual(s)
    bpfilt_data = tensor.Tensor.create_virtual(s)
    cropped_data = tensor.Tensor.create_virtual(s)
    d2m = tensor.Tensor.create_virtual(s)
    clsf_feats = tensor.Tensor.create_virtual(s)
    
    norm_d2m_training_data = tensor.Tensor.create(s,(3*window_len,3))
    d2m_training_data = tensor.Tensor.create(s,(3*window_len,3))
    training_labels = tensor.Tensor.create_from_data(s,(Nclasses*window_len,),
                                                     np.concatenate((np.zeros(window_len,),
                                                                     np.ones(window_len,),
                                                                     2*np.ones(window_len,)),
                                                                    axis=None))
        
    mean_covs = tensor.Tensor.create(s,(3,len(dims[1]),len(dims[1])))
    
    
    covconcat1 = tensor.Tensor.create_virtual(s)
    covconcat2 = tensor.Tensor.create_virtual(s)


    trialconcat1 = tensor.Tensor.create_virtual(s)
    trialconcat2 = tensor.Tensor.create_virtual(s)

    # scalars
    linear_svm_label = scalar.Scalar.create(s,int)
    rbf_svm_label = scalar.Scalar.create(s,int)
    lda_label = scalar.Scalar.create(s,int)
    mdm_label = scalar.Scalar.create(s,int)
    rts_label = scalar.Scalar.create(s,int)
    
    
    # arrays
    prev_covs = array.Array.create(s,Nclasses,
                                   circle_buffer.CircleBuffer.create(s,
                                                                     window_len,
                                                                     cov))
    prev_trials = array.Array.create(s,Nclasses,
                                     circle_buffer.CircleBuffer.create(s,
                                                                    window_len,
                                                                    tensor.Tensor.create(s,(len(cropped_indices[0]),len(dims[1])))))
    
    means_arr = array.Array.create(s,Nclasses,cov)
    
    stacked_class_covs = array.Array.create(s,Nclasses,
                                              tensor.Tensor.create(s,(window_len,len(dims[1]),len(dims[1]))))
    
    dim_sz = (len(cropped_indices[0]),len(dims[1]))
    stacked_class_trials = array.Array.create(s,Nclasses,
                                              tensor.Tensor.create(s,(window_len,) + dim_sz))

    
    # filter
    lp_filt = bcip_filter.Filter.create_butter(s,4,30,'low','sos',Fs)
    hp_filt = bcip_filter.Filter.create_butter(s,4,8,'high','sos',Fs)
    
    # CSP num filts
    csp_m = [1,2,3,4]
    
    csp_labels = np.concatenate((np.zeros((window_len,)),
                                 np.ones((window_len,)),
                                 2*np.ones((window_len,))),
                                axis=0)
    
    csp_filtered_training_data = [[] for _ in csp_m]
    csp_training_data_var = []
    csp_lda_training_data = []
    csp_filtered_data = [[] for _ in csp_m]
    csp_log_var = []
    csp_var = []
    csp_label = []
    csp_initialization_labels = []
    csp_concat1 = []
    csp_concat2 = []
    csp_train_concat1 = []
    csp_train_concat2 = []
    for i_m in range(len(csp_m)):
        csp_training_data_var.append(tensor.Tensor.create_virtual(s))
        csp_lda_training_data.append(tensor.Tensor.create(s,(Nclasses*window_len,Nclasses*2*csp_m[i_m])))
        csp_var.append(tensor.Tensor.create_virtual(s))
        csp_log_var.append(tensor.Tensor.create_virtual(s))
        csp_label.append(scalar.Scalar.create(s,int))
        
        csp_initialization_labels.append(tensor.Tensor.create_from_data(s,csp_labels.shape,csp_labels==i_m))
        
        for concat_tensor in (csp_concat1,csp_concat2,csp_train_concat1,csp_train_concat2):
            concat_tensor.append(tensor.Tensor.create_virtual(s))
            
        # outputs for each of the csp nodes (need 3 for 3 class)
        for i_c in range(Nclasses):
            csp_filtered_data[i_m].append(tensor.Tensor.create_virtual(s))
            csp_filtered_training_data[i_m].append(tensor.Tensor.create_virtual(s))
    
    
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

    
    for j in range(Nclasses):
        # stack class covs into a tensor
        kernels.StackKernel.add_stack_node(b.postprocessing_graph,
                                               prev_covs.get_element(j),
                                               stacked_class_covs.get_element(j))
        # calculate class mean
        kernels.RiemannMeanKernel.add_riemann_mean_node(b.postprocessing_graph,
                                                        stacked_class_covs.get_element(j),
                                                        means_arr.get_element(j))
            
    kernels.ConcatenationKernel.add_concatenation_node(b.postprocessing_graph,
                                                       stacked_class_covs.get_element(0),
                                                       stacked_class_covs.get_element(1),
                                                       covconcat1)
    kernels.ConcatenationKernel.add_concatenation_node(b.postprocessing_graph,
                                                       stacked_class_covs.get_element(2),
                                                       covconcat1,
                                                       covconcat2)
        
    kernels.RiemannDistanceKernel.add_riemann_distance_node(b.postprocessing_graph,
                                                            covconcat2,
                                                            mean_covs,
                                                            d2m_training_data)
    
    kernels.StackKernel.add_stack_node(b.postprocessing_graph,
                                         means_arr,
                                         mean_covs)
    
    
    # CSP preprocessing
    for j in range(Nclasses):
        kernels.StackKernel.add_stack_node(b.postprocessing_graph,
                                                prev_trials.get_element(j),
                                                stacked_class_trials.get_element(j))
            
    kernels.ConcatenationKernel.add_concatenation_node(b.postprocessing_graph,
                                                            stacked_class_trials.get_element(1),
                                                            stacked_class_trials.get_element(2),
                                                            trialconcat1)
    kernels.ConcatenationKernel.add_concatenation_node(b.postprocessing_graph,
                                                            stacked_class_trials.get_element(0),
                                                            trialconcat1,
                                                            trialconcat2)
    
    
    ## FEEDBACK BLOCKS ##
    for i in range(fb_blocks):
        b = block.Block.create(s,Nclasses,(4,4,4))
        
        
        # CSP preprocessing
        for j in range(Nclasses):
            kernels.StackKernel.add_stack_node(b.postprocessing_graph,
                                                prev_trials.get_element(j),
                                                stacked_class_trials.get_element(j))
            
        kernels.ConcatenationKernel.add_concatenation_node(b.postprocessing_graph,
                                                            stacked_class_trials.get_element(1),
                                                            stacked_class_trials.get_element(2),
                                                            trialconcat1)
        kernels.ConcatenationKernel.add_concatenation_node(b.postprocessing_graph,
                                                            stacked_class_trials.get_element(0),
                                                            trialconcat1,
                                                            trialconcat2)
        
        
        for i_m in range(len(csp_m)):
            for i_c in range(Nclasses):
                kernels.CommonSpatialPatternKernel.add_uninitialized_CSP_node(b.preprocessing_graph,
                                                                          trialconcat2,
                                                                          csp_filtered_training_data[i_m][i_c],
                                                                          trialconcat2,
                                                                          csp_initialization_labels[i_c],
                                                                          2*csp_m[i_m])
                
            kernels.ConcatenationKernel.add_concatenation_node(b.preprocessing_graph,
                                                               csp_filtered_training_data[i_m][0],
                                                               csp_filtered_training_data[i_m][1],
                                                               csp_train_concat1[i_m],
                                                               2)
            
            kernels.ConcatenationKernel.add_concatenation_node(b.preprocessing_graph,
                                                               csp_train_concat1[i_m],
                                                               csp_filtered_training_data[i_m][2],
                                                               csp_train_concat2[i_m],
                                                               2)
            
                
            kernels.VarKernel.add_var_node(b.preprocessing_graph,
                                            csp_train_concat2[i_m],
                                            csp_training_data_var[i_m],
                                            1,1,False)
            
            kernels.LogKernel.add_log_node(b.preprocessing_graph,
                                            csp_training_data_var[i_m],
                                            csp_lda_training_data[i_m])
        
            # CSP trial processing
            for i_c in range(Nclasses):
                kernels.CommonSpatialPatternKernel.add_uninitialized_CSP_node(b.trial_processing_graph,
                                                                          cropped_data,
                                                                          csp_filtered_data[i_m][i_c],
                                                                          trialconcat2,
                                                                          csp_initialization_labels[i_c],
                                                                          2*csp_m[i_m])
                
            kernels.ConcatenationKernel.add_concatenation_node(b.trial_processing_graph,
                                                               csp_filtered_data[i_m][0],
                                                               csp_filtered_data[i_m][1],
                                                               csp_concat1[i_m],
                                                               1)
            
            kernels.ConcatenationKernel.add_concatenation_node(b.trial_processing_graph,
                                                               csp_concat1[i_m],
                                                               csp_filtered_data[i_m][2],
                                                               csp_concat2[i_m],
                                                               1)
            
            kernels.VarKernel.add_var_node(b.trial_processing_graph,
                                            csp_concat2[i_m],
                                            csp_var[i_m],
                                            0,1,True)
            
            kernels.LogKernel.add_log_node(b.trial_processing_graph,
                                            csp_var[i_m],
                                            csp_log_var[i_m])
            
            kernels.LDAClassifierKernel.add_untrained_LDA_node(b.trial_processing_graph,
                                                                csp_log_var[i_m],
                                                                csp_label[i_m],
                                                                csp_lda_training_data[i_m], 
                                                                training_labels)
        
        
        for j in range(Nclasses):
            # stack class covs into a tensor
            kernels.StackKernel.add_stack_node(b.postprocessing_graph,
                                               prev_covs.get_element(j),
                                               stacked_class_covs.get_element(j))
            # calculate class mean
            kernels.RiemannMeanKernel.add_riemann_mean_node(b.postprocessing_graph,
                                                            stacked_class_covs.get_element(j),
                                                            means_arr.get_element(j))
            
        kernels.ConcatenationKernel.add_concatenation_node(b.postprocessing_graph,
                                                           stacked_class_covs.get_element(1),
                                                           stacked_class_covs.get_element(2),
                                                           covconcat1)
        kernels.ConcatenationKernel.add_concatenation_node(b.postprocessing_graph,
                                                           stacked_class_covs.get_element(0),
                                                           covconcat1,
                                                           covconcat2)
        
        kernels.RiemannDistanceKernel.add_riemann_distance_node(b.postprocessing_graph,
                                                                covconcat2,
                                                                mean_covs,
                                                                d2m_training_data)
        
        kernels.FeatureNormalizationKernel.add_feature_normalization_node(b.preprocessing_graph,
                                                                          d2m_training_data,
                                                                          norm_d2m_training_data,
                                                                          d2m_training_data)
        
        
        
        kernels.StackKernel.add_stack_node(b.postprocessing_graph,
                                           means_arr,
                                           mean_covs)
        
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
        
        kernels.SVMClassifierKernel.add_untrained_SVM_node(b.trial_processing_graph,
                                                               clsf_feats,
                                                               linear_svm_label,
                                                               norm_d2m_training_data,
                                                               training_labels,
                                                               kernel='linear')
        
        kernels.SVMClassifierKernel.add_untrained_SVM_node(b.trial_processing_graph,
                                                               clsf_feats,
                                                               rbf_svm_label,
                                                               norm_d2m_training_data,
                                                               training_labels)
        
        kernels.LDAClassifierKernel.add_untrained_LDA_node(b.trial_processing_graph,
                                                               clsf_feats,
                                                               lda_label,
                                                               norm_d2m_training_data,
                                                               training_labels,
                                                               shrinkage='auto')
        
        kernels.RiemannMDMClassifierKernel.add_untrained_riemann_MDM_node(b.trial_processing_graph,
                                                                          cov,
                                                                          mdm_label,
                                                                          prev_covs,
                                                                          training_labels)
        
        kernels.RiemannTangentSpacerLDAClassifierKernel.add_untrained_riemann_tangent_space_rLDA_node(b.trial_processing_graph,
                                                                                                      cov,
                                                                                                      rts_label,
                                                                                                      prev_covs,
                                                                                                      training_labels,
                                                                                                      shrinkage=0.3)
        
        kernels.RiemannDistanceKernel.add_riemann_distance_node(b.trial_processing_graph,
                                                                cov,
                                                                mean_covs,
                                                                d2m)
        
        kernels.FeatureNormalizationKernel.add_feature_normalization_node(b.trial_processing_graph,
                                                                          d2m,
                                                                          clsf_feats,
                                                                          d2m_training_data)
        

    
    sts = s.verify()
    if sts != bcip_enums.BcipEnums.SUCCESS:
        print("verification error...")
        return sts
    
    
    linear_svm_labels = []
    rbf_svm_labels = []
    lda_labels = []
    mdm_labels = []
    rts_labels = []
    csp_labels = [[] for m in csp_m]
    
    session_trial_cnt = 0
    block_num = 1
    while s.remaining_blocks != 0:
        block_trial_cnt= [0,0,0]
        sts = s.start_block()
        if sts != bcip_enums.BcipEnums.SUCCESS:
            print("start block failed with code: ", str(sts))
            return sts
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
                    mdm_labels.append(mdm_label.data)
                    rts_labels.append(rts_label.data)
                    lda_labels.append(lda_label.data)
                    linear_svm_labels.append(linear_svm_label.data)
                    rbf_svm_labels.append(rbf_svm_label.data)
                    for i_m in range(len(csp_m)):
                        csp_labels[i_m].append(csp_label[i_m].data)
                    
                block_trial_cnt[label] += 1
                print(block_trial_cnt)
                
                # add the trial to the appropriate queue
                prev_covs.get_element(label).enqueue(cov)
                prev_trials.get_element(label).enqueue(cropped_data)

                
            session_trial_cnt += 1
        
        print("Finished Block {}".format(block_num))
        s.close_block()
        block_num += 1
        
    mdm_labels = np.asarray(mdm_labels)
    rts_labels = np.asarray(rts_labels)
    lda_labels = np.asarray(lda_labels)
    linear_svm_labels = np.asarray(linear_svm_labels)
    rbf_svm_labels = np.asarray(rbf_svm_labels)
    for i_m in range(len(csp_m)):
        csp_labels[i_m] = np.asarray(csp_labels[i_m])
    
    
    y = [l for l, t in zip(labels, accepted_trials) if t == 1]
    y = np.asarray(y[Nclasses*window_len:])    
    
    print_stats(y,mdm_labels,'mdm')
    print_stats(y,rts_labels,'rts')
    print_stats(y,lda_labels,'lda')
    print_stats(y,linear_svm_labels,'linear svm')
    print_stats(y,rbf_svm_labels,'rbf svm')
    for i_m in range(len(csp_m)):
        print_stats(y,csp_labels[i_m],'CSP-LDA m=' + str(csp_m[i_m]))
if __name__ == "__main__":
    main()