# -*- coding: utf-8 -*-
"""
Created on Tues July 26 16:12:30 2022
@author: aaronlio
"""

# Create a simple graph for testing
from bcipy import bcipy
import numpy as np
import pylsl
import json
import re

class MindsetPipeline:
    def __init__(self, file):
        self.sess = bcipy.Session.create()
        self.online_graph = bcipy.Graph.create(self.sess)
        self.offline_trials = 500
        self.online_trials = 500

        self.Fs = 500
        self.trial_len = 1.0
        self.tasks = ('flash', 'target')
        self.resample_fs = 50

        #outlet_info = pylsl.StreamInfo('Marker-PredictedLabel', 'Markers', channel_format='string')
        #lsl_marker_outlet = pylsl.StreamOutlet(outlet_info)

        # create a filter

        self.order = 4
        self.bandpass = (1,25) # in Hz
        self.f = bcipy.Filter.create_butter(self.sess,self.order,self.bandpass,btype='bandpass',fs=self.Fs,implementation='sos')

        self.channels = tuple([_ for _ in range(0,32)])

        # Data sources from LSL

        self.LSL_data_src = bcipy.source.InputLSLStream.create_marker_coupled_data_stream(self.sess, "type='EEG'", self.channels, relative_start=-0.4, marker_fmt='^SPACE pressed$')

        LSL_data_out = bcipy.source.OutputLSLStream.create_outlet(self.sess, 'outlet', 'type="Markers"', 1, channel_format='float32')
        self.online_output_data = bcipy.Tensor.create_for_volatile_output(self.sess, (2,), LSL_data_out)



        # training data sources from xdf file
        self.offline_data_src = bcipy.source.BcipXDF.create_epoched(self.sess,
                file, 
                self.tasks, channels=self.channels, relative_start=0, Ns=self.Fs*self.trial_len)
        
        self.task_series = self.offline_data_src.data[1]['time_series']
        
        labels = bcipy.CircleBuffer.create(self.sess, len(task), bcipy.Scalar.create(self.sess, int))
        
        target = None
        for task in self.task_series:
            scalar = bcipy.Scalar.create(self.sess, int)
            if json.loads(task[0]).keys()[0] == 'target':
                target = json.loads(task[0]).values()[0]
            elif json.loads(task[0]).keys()[0] == 'flash' and target != None:
                if json.loads(task[0]).values()[0] == target:
                    scalar.data = 1
                else:
                    scalar.data = 0
                
                labels.enqueue(scalar)

        self.online_input_data = bcipy.Tensor.create_from_handle(self.sess, (len(self.channels), 700), self.LSL_data_src)
        self.offline_input_data = bcipy.Tensor.create_from_handle(self.sess, (len(self.channels), 700), self.offline_data_src)

        # Data input tensors
        self.label_input = bcipy.Scalar.create(self.sess, int)
        self.training_data = {'data'   : bcipy.CircleBuffer.create(self.sess, self.offline_trials, bcipy.Tensor(self.sess, (self.offline_input_data.shape[0],self.resample_fs),None,False,None)),
                            'labels' : labels}
        

        # online graph data containers (i.e. graph edges)
        pred_label = bcipy.Scalar.create_from_value(self.sess,-1) 

        t_virt = [bcipy.Tensor.create_virtual(self.sess), # output of filter, input to resample
                bcipy.Tensor.create_virtual(self.sess), # output of resample, input to extract
                bcipy.Tensor.create_virtual(self.sess), # output of extract, input to xdawn
                bcipy.Tensor.create_virtual(self.sess), # output of xdawn, input to tangent space
                bcipy.Tensor.create_virtual(self.sess)] # output of tangent space, input to classifier

        classifier = bcipy.Classifier.create_logistic_regression(self.sess)

        # extraction indices - TODO Ask Jason about filter-epoch execution order during online
        start_time = 0.2
        end_time = 1.2
        extract_indices = [":", # all channels
                        [_ for _ in range(int(start_time*self.resample_fs),int(end_time*self.resample_fs))] # central 1s
                        ]


        # online graph nodes 
        bcipy.kernels.FilterKernel.add_filter_node(self.online_graph, self.online_input_data, self.f, t_virt[0])
        bcipy.kernels.ResampleKernel.add_resample_node(self.online_graph, t_virt[0], self.resample_fs/self.Fs, t_virt[1])
        bcipy.kernels.ExtractKernel.add_extract_node(self.online_graph, t_virt[1], extract_indices, t_virt[2])
        bcipy.kernels.XDawnCovarianceKernel.add_xdawn_covariance_node(self.online_graph, t_virt[2], 
                                                        t_virt[3], self.training_data['data'], 
                                                        self.training_data['labels'])
        bcipy.kernels.TangentSpaceKernel.add_tangent_space_node(self.online_graph, t_virt[3], t_virt[4])
        bcipy.kernels.ClassifierKernel.add_classifier_node(self.online_graph, t_virt[4], classifier, pred_label, self.online_output_data)


        verify_sts = self.online_graph.verify()

        if verify_sts != bcipy.BcipEnums.SUCCESS:
            print("Test Failed D=")
            return verify_sts

        return bcipy.BcipEnums.SUCCESS
    
    def train(self):
        
        # initialize the classifiers (i.e., train the classifier)
        init_sts = self.online_graph.initialize()

        if init_sts != bcipy.BcipEnums.SUCCESS:
            print("Init Failed D=")
            return init_sts

        return bcipy.BcipEnums.SUCCESS
    
    def execute(self):
        sts = bcipy.BcipEnums.SUCCESS
        online_trials = 100


        for t_num in range(online_trials):
            true_label = -1
            print("waiting for marker...")
            while true_label == -1:

                sts = self.online_graph.execute()
            if sts == bcipy.BcipEnums.SUCCESS:
                # print the value of the most recent trial
                y_bar = self.online_output_data.data
                print(f"\tTrial {t_num+1}: Max Probability = {max(y_bar)}")
            else:
                print(f"Trial {t_num+1} raised error, status code: {sts}")
                break

        print("Test Passed =D")
        

