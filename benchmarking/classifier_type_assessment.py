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
    
    # tensors
    raw_data = tensor.Tensor.create_from_handle(s,
                                                (len(dims[0]),len(dims[1])),
                                                src)
    cov = tensor.Tensor.create(s,(len(dims[1]),len(dims[1])))
    filt_data = tensor.Tensor.create_virtual(s)
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
    
    
    concat1 = tensor.Tensor.create_virtual(s)
    concat2 = tensor.Tensor.create_virtual(s)


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
    means_arr = array.Array.create(s,Nclasses,cov)
    
    stacked_class_trials = array.Array.create(s,Nclasses,
                                              tensor.Tensor.create(s,(window_len,len(dims[1]),len(dims[1]))))

    
    # filter
    trial_filt = bcip_filter.Filter.create_butter(s,4,(8,30),'bandpass','sos',Fs)
    
    init_blocks = window_len // 4
    fb_blocks = 30-init_blocks
    
    for i in range(init_blocks):
        # create block
        b = block.Block.create(s,Nclasses,(4,4,4))
        kernels.FiltFiltKernel.add_filtfilt_node(b.trial_processing_graph,
                                                          raw_data,
                                                          trial_filt,
                                                          filt_data)
        kernels.CovarianceKernel.add_covariance_node(b.trial_processing_graph,
                                                                filt_data,
                                                                cov,
                                                                0.05)

    
    for j in range(Nclasses):
        # stack class covs into a tensor
        kernels.StackKernel.add_stack_node(b.postprocessing_graph,
                                               prev_covs.get_element(j),
                                               stacked_class_trials.get_element(j))
        # calculate class mean
        kernels.RiemannMeanKernel.add_riemann_mean_node(b.postprocessing_graph,
                                                        stacked_class_trials.get_element(j),
                                                        means_arr.get_element(j))
            
    kernels.ConcatenationKernel.add_concatenation_node(b.postprocessing_graph,
                                                       stacked_class_trials.get_element(0),
                                                       stacked_class_trials.get_element(1),
                                                       concat1)
    kernels.ConcatenationKernel.add_concatenation_node(b.postprocessing_graph,
                                                       stacked_class_trials.get_element(2),
                                                       concat1,
                                                       concat2)
        
    kernels.RiemannDistanceKernel.add_riemann_distance_node(b.postprocessing_graph,
                                                            concat2,
                                                            mean_covs,
                                                            d2m_training_data)
    
    kernels.StackKernel.add_stack_node(b.postprocessing_graph,
                                         means_arr,
                                         mean_covs)
    
    for i in range(fb_blocks):
        b = block.Block.create(s,Nclasses,(4,4,4))
        
        for j in range(Nclasses):
            # stack class covs into a tensor
            kernels.StackKernel.add_stack_node(b.postprocessing_graph,
                                               prev_covs.get_element(j),
                                               stacked_class_trials.get_element(j))
            # calculate class mean
            kernels.RiemannMeanKernel.add_riemann_mean_node(b.postprocessing_graph,
                                                            stacked_class_trials.get_element(j),
                                                            means_arr.get_element(j))
            
        kernels.ConcatenationKernel.add_concatenation_node(b.postprocessing_graph,
                                                           stacked_class_trials.get_element(0),
                                                           stacked_class_trials.get_element(1),
                                                           concat1)
        kernels.ConcatenationKernel.add_concatenation_node(b.postprocessing_graph,
                                                           stacked_class_trials.get_element(2),
                                                           concat1,
                                                           concat2)
        
        kernels.RiemannDistanceKernel.add_riemann_distance_node(b.postprocessing_graph,
                                                                concat2,
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
                                                          trial_filt,
                                                          filt_data)
        kernels.CovarianceKernel.add_covariance_node(b.trial_processing_graph,
                                                                filt_data,
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
                    mdm_labels.append(mdm_label.data)
                    rts_labels.append(rts_label.data)
                    lda_labels.append(lda_label.data)
                    linear_svm_labels.append(linear_svm_label.data)
                    rbf_svm_labels.append(rbf_svm_label.data)
                    
                block_trial_cnt[label] += 1
                print(block_trial_cnt)
                
                # add the trial to the appropriate queue
                prev_covs.get_element(label).enqueue(cov)

                
            session_trial_cnt += 1
        
        print("Finished Block {}".format(block_num))
        s.close_block()
        block_num += 1
        
    mdm_labels = np.asarray(mdm_labels)
    rts_labels = np.asarray(rts_labels)
    lda_labels = np.asarray(lda_labels)
    linear_svm_labels = np.asarray(linear_svm_labels)
    rbf_svm_labels = np.asarray(rbf_svm_labels)
        
    y = [l for l, t in zip(labels, accepted_trials) if t == 1]
    y = np.asarray(y[Nclasses*window_len:])    
    
    print_stats(y,mdm_labels,'mdm')
    print_stats(y,rts_labels,'rts')
    print_stats(y,lda_labels,'lda')
    print_stats(y,linear_svm_labels,'linear svm')
    print_stats(y,rbf_svm_labels,'rbf svm')
if __name__ == "__main__":
    main()