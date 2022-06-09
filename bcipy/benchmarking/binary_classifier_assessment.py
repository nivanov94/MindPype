# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 16:47:56 2020

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
    Nclasses = 2
    window_len = 60
    pairs = ((0,1),(0,2),(1,2))
    
    trial_info = io.loadmat(path.join("D:/School/GRAD/Thesis/Pilot Work/debugging_data/feb25","trial_labels.mat"))
    accepted_trials = trial_info['artifact_labels'][0]
    labels = trial_info['total_sess_labels'][0]
    
    regs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,'auto']
    accuracy = {}
    
    for pair in pairs:
        accuracy[pair] = {}
        for r in regs:
        
    
            s = session.Session.create()
    
            label_map = {0:'class{}_trials'.format(pair[0]+1),1:"class{}_trials".format(pair[1]+1)}
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
    
            training_labels = tensor.Tensor.create_from_data(s,(Nclasses*window_len,),
                                                     np.concatenate((0*np.ones(window_len,),
                                                                     1*np.ones(window_len,)),
                                                                    axis=None))

            # scalars
            y_bar = scalar.Scalar.create(s,int)
    
            # arrays
            prev_covs = array.Array.create(s,Nclasses,
                                   circle_buffer.CircleBuffer.create(s,
                                                                     window_len,
                                                                     cov))
    
            # filter
            trial_filt = bcip_filter.Filter.create_butter(s,4,(8,30),'bandpass','sos',Fs)
    
            init_blocks = window_len // 4
            fb_blocks = 30-init_blocks
    
            for i in range(init_blocks):
                # create block
                b = block.Block.create(s,Nclasses,(4,4))
                kernels.filtfilt.FiltFiltKernel.add_filtfilt_node(b.trial_processing_graph,
                                                          raw_data,
                                                          trial_filt,
                                                          filt_data)
                kernels.covariance.CovarianceKernel.add_covariance_node(b.trial_processing_graph,
                                                                filt_data,
                                                                cov,
                                                                0.05)
    
            for i in range(fb_blocks):
                b = block.Block.create(s,Nclasses,(4,4))
                kernels.filtfilt.FiltFiltKernel.add_filtfilt_node(b.trial_processing_graph,
                                                          raw_data,
                                                          trial_filt,
                                                          filt_data)
                kernels.covariance.CovarianceKernel.add_covariance_node(b.trial_processing_graph,
                                                                filt_data,
                                                                cov,
                                                                0.05)
                kernels.riemann_ts_rLDA_classifier.RiemannTangentSpacerLDAClassifierKernel.add_untrained_riemann_tangent_space_rLDA_node(
                        b.trial_processing_graph,cov,y_bar,prev_covs,training_labels,r,'lsqr')
            
    
            sts = s.verify()
            if sts != bcip_enums.BcipEnums.SUCCESS:
                print("verification error...")
                return sts
    
    
            y_bars = []
            y = []
            session_trial_cnt = 0
            block_num = 1
            while s.remaining_blocks != 0:
                block_trial_cnt= [0,0]
                s.start_block()
                while sum(block_trial_cnt) < 8:
            
                    label = labels[session_trial_cnt]
                    
                    if label in pair:
                        sts = s.execute_trial(pair.index(label))
                        if sts != bcip_enums.BcipEnums.SUCCESS:
                            print("EXE ERROR")
                        if accepted_trials[session_trial_cnt] == 0:
                            print("rejected trial")
                            s.reject_trial()
                        else:
                            if block_num > init_blocks:
                                # get the predicted label
                                y_bars.append(y_bar.data)
                                print("3 class predicted label: {}".format(y_bar.data))
                        
                    
                            block_trial_cnt[pair.index(label)] += 1
                            print(block_trial_cnt)
                            y.append(pair.index(label))
                
                            # add the trial to the appropriate queue
                            prev_covs.get_element(pair.index(label)).enqueue(cov)

                
                    session_trial_cnt += 1
        
                print("Finished Block {}".format(block_num))
                s.close_block()
                block_num += 1
        
            y_bars = np.asarray(y_bars)
        
            y = np.asarray(y[Nclasses*window_len:])
    
            accuracy[pair][r] = sum(y == y_bars)
    
        
    print(accuracy)
    
    
if __name__ == "__main__":
    main()