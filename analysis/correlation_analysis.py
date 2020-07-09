# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 13:41:31 2020

@author: Nick
"""

from metric_analysis import MetricAnalyzer
from clsf_analysis import ClsfStatAnalyzer
import numpy as np
from scipy.stats import linregress
import csv



class MetricClsfCorrelAnalyzer:
    
    def __init__(self,files):
        self.clsf_analyzers = [ClsfStatAnalyzer(filepair[1]) for filepair in files]
        self.metric_analyzers = [MetricAnalyzer(filepair[0]) for filepair in files]
    
    # def ext_metric_clsf_data_pairs(self):
    #     # extract metric data
    #     for m_analyzer in self.metric_analyzers:
    #         m_analyzer.ext_eval_metric_stats()
        
    #     # extract classifier accuracy data
    #     for c_analyzer in self.clsf_analyzers:
    #         c_analyzer.ext_test_stats()
        
        
    #     self.distinc = {}
    #     self.intersp = {}
    #     self.con_sum = {}
    #     self.con_ind = {}
        
    #     for i in range(len(self.participants)):
    #         m_analyzer = self.metric_analyzers[i]
    #         c_analyzer = self.clsf_analyzers[i]
    #         p, d = self.participants[i]
            
    #         for m_set in m_analyzer.eval_stats:
    #             m_hp = dict(m_set[0]) # make a copy to modify
    #             m_channels = m_hp.pop('channels')
    #             m_classes = m_hp.pop('classes')
                
    #             # convert hyp_set to a str, add to dict if missing
    #             m_hyp_str = str(m_hp)
    #             for target in (self.distinc,self.intersp,self.con_sum,self.con_ind):
    #                 if not (m_hyp_str in target):
    #                     target[m_hyp_str] = {}
                
    #             info = {'participant' : p,
    #                     'dataset' : d,
    #                     'classes' : m_classes,
    #                     'channels' : m_channels}
                
    #             for c_set in c_analyzer.eval_stats:
    #                 c_hp = dict(c_set[0])
    #                 c_channels = c_hp.pop('channels')
    #                 c_classes = c_hp.pop('classes')
                    
    #                 if (c_channels != m_channels) or (c_classes != m_classes):
    #                     # hyperparams not compatible, continue
    #                     continue
                    
    #                 # add pair to dicts
    #                 c_hyp_str = str(c_hp)
    #                 for target in (self.distinc,self.intersp,self.con_sum,self.con_ind):
    #                     if not (c_hyp_str in target[m_hyp_str]):
    #                         target[m_hyp_str][c_hyp_str] = []
                    
                    
    #                 self.distinc[m_hyp_str][c_hyp_str].append({
    #                     'info'      : info,
    #                     'metric'    : m_set[1],
    #                     'clsf_acc'  : c_set[2]})
                    
    #                 self.intersp[m_hyp_str][c_hyp_str].append({
    #                     'info'      : info,
    #                     'metric'    : m_set[2],
    #                     'clsf_acc'  : c_set[2]})
                    
    #                 self.con_sum[m_hyp_str][c_hyp_str].append({
    #                     'info'      : info,
    #                     'metric'    : m_set[3],
    #                     'clsf_acc'  : c_set[2]})
                
    #                 self.con_ind[m_hyp_str][c_hyp_str].append({
    #                     'info'      : info,
    #                     'metric'    : m_set[4],
    #                     'clsf_acc'  : c_set[2]})
        

    # def metric_clsf_acc_correl(self,metric,window,channels):
        
    #     if metric == 'distinct':
    #         m_dict = self.distinc
    #     elif metric == 'intersp':
    #         m_dict = self.intersp
    #     elif metric == 'consist':
    #         m_dict = self.con_sum
            
    #     if window == 'first':
    #         w_index = 0
    #     elif window == 'last':
    #         w_index = 1
    #     elif window == 'best':
    #         w_index = 2
    #     else:
    #         w_index = 3
        
    #     correl_pairs = {}
    #     correl = {}
        
    #     for m_hp in m_dict:
    #         correl_pairs[m_hp] = {}
    #         correl[m_hp] = {}
    #         for c_hp in m_dict[m_hp]:
    #             correl_pairs[m_hp][c_hp] = []
    #             correl[m_hp][c_hp] = None
                
    #             data = m_dict[m_hp][c_hp]
                
    #             for d in data:
    #                 d_dataset = d['info']['dataset']
    #                 d_channels = convert_numerical_channels(d_dataset,
    #                                                         d['info']['channels'])
                    
    #                 if d_channels != channels:
    #                     # not channel set of interest, continue
    #                     continue
                    
    #                 pair = (d['metric'][w_index],d['clsf_acc']['accuracy'])
                    
    #                 correl_pairs[m_hp][c_hp].append(pair)
        
                
    def metric_acc_correl(self,metric_name,window_desc,clsf_stat,class_sets,channels):
        Np = len(self.metric_analyzers)
        
        metric_data = {}
        metric_info = {}
        clsf_data = {}
        clsf_info = {}
        
        for i_p in range(Np):
            
            # get metric data
            metrics, hyp_sets, info = self.metric_analyzers[i_p].extract_eval_metric_stats(metric_name,
                                                                                           window_desc,
                                                                                           class_sets,
                                                                                           channels)
            
            Nm = len(hyp_sets)
            for i_m in range(Nm):
                hyp_str = str(hyp_sets[i_m])
                
                if hyp_str in metric_data:
                    metric_data[hyp_str].append(metrics[i_m])
                else:
                    metric_data[hyp_str] = [metrics[i_m]]
            
                if hyp_str in metric_info:
                    metric_info[hyp_str].append(info[i_m])
                else:
                    metric_info[hyp_str] = [info[i_m]]
            
            
            # get the classifier accuracies
            accs, hyp_sets, info = self.clsf_analyzers[i_p].extract_test_stats(clsf_stat,
                                                                               class_sets,
                                                                               channels)
            
            Nca = len(hyp_sets)
            for i_ca in range(Nca):
                hyp_str = str(hyp_sets[i_ca])
                
                if hyp_str in clsf_data:
                    clsf_data[hyp_str].append(accs[i_ca])
                else:
                    clsf_data[hyp_str] = [accs[i_ca]]
                
                if hyp_str in clsf_info:
                    clsf_info[hyp_str].append(info[i_ca])
                else:
                    clsf_info[hyp_str] = [info[i_ca]]
            
        
        # iterate over all pairs of hyperparams and calculate correlation
        Ni = len(metric_data)
        Nj = len(clsf_data)
        correl = np.zeros((Ni,Nj))
        lines = np.zeros((Ni,Nj,2))
        m_hps = [m_hp for m_hp in metric_data]
        c_hps = [c_hp for c_hp in clsf_data]
        pairs = [[None for _ in range(Nj)] for _ in range(Ni)]
        for i in range(Ni):
            m_hp = m_hps[i]
            metrics = np.asarray(metric_data[m_hp])
            
            for j in range(Nj):
                c_hp = c_hps[j]
                accs = np.asarray(clsf_data[c_hp])
                
                correl_mat = np.corrcoef(metrics,accs)
                correl[i,j] = correl_mat[0,1]
                m,b,r,p,_ = linregress(metrics,accs)
                
                lines[i,j,0] = m
                lines[i,j,1] = b
                
                pairs[i][j] = (metric_info[m_hp],clsf_info[c_hp],metric_data[m_hp],clsf_data[c_hp])
        
        return correl,lines,pairs

    def generate_csv(self,outfilename,freq,interspread_ref,eval_set_sz,
                     win_sz,win_type,step_sz,channel_set):
        
        with open(outfilename,'w',newline='') as fp:
            csvfile = csv.writer(fp)
            
            # header = ['participant','dataset','class.set',
            #           'num.classes','channel.set',
            #           'best.distinct','worst.distinct',
            #           'first.distinct','last.distinct','best.interspread',
            #           'worst.interspread','first.interspread',
            #           'last.interspread','best.consistency.sum',
            #           'worst.consistency.sum','first.consistency.sum',
            #           'last.consistency.sum','best.consistency.c1',
            #           'worst.consistency.c1','first.consistency.c1',
            #           'last.consistency.c1','best.consistency.c2',
            #           'worst.consistency.c2','first.consistency.c2',
            #           'last.consistency.c2','best.consistency.c3',
            #           'worst.consistency.c3','first.consistency.c3',
            #           'last.consistency.c3','train.acc','test.acc',
            #           'c1.precision','c1.recall','c1.f1',
            #           'c2.precision','c2.recall','c2.f1',
            #           'c3.precision','c3.recall','c3.f1',
            #           'weighted.precision','weighted.recall',
            #           'weighted.f1']
            header = ['participant','dataset','class.set',
                      'num.classes','channel.set','eval.trials',
                      'best.distinct','worst.distinct',
                      'first.distinct','last.distinct','mean.distinct',
                      'best.interspread',
                      'worst.interspread','first.interspread',
                      'last.interspread','mean.interspread',
                      'best.con',
                      'worst.con','first.con',
                      'last.con','mean.con',
                      'train.acc','test.acc','cross.entropy',
                      'weighted.precision','weighted.recall',
                      'weighted.f1']
            
            csvfile.writerow(header)
            
            for m_da, c_da in zip(self.metric_analyzers,self.clsf_analyzers):
                mid_info = m_da.extract_id()
                cid_info = c_da.extract_id()
                
                if mid_info != cid_info:
                    print(mid_info)
                    print(cid_info)
                    raise Exception("ID mismatch")
                
                
                m_data = m_da.extract_data(freq,interspread_ref,eval_set_sz,
                                           win_sz,win_type,step_sz)
                
                c_data = c_da.extract_test_data(channel_set)
                
                # make sure the set of keys for both sets are the same
                m_keys = set(m_data.keys())
                c_keys = set(c_data.keys())
                print(m_keys)
                print(c_keys)
                if len(m_keys.symmetric_difference(c_keys)) != 0:
                    raise Exception("Channel, class set mismatch")
                
                for key in m_data:
                    line = mid_info + key + m_data[key] + c_data[key]
                    csvfile.writerow(line)
                
            