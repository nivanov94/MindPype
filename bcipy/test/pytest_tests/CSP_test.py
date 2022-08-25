# -*- coding: utf-8 -*-
"""
Created on Tues Aug 9th 14:04:39 2022

@author: aaronlio
"""
# For debugging 
import sys, os

from django.test import TestCase
sys.path.insert(0, os.getcwd())

# Create a simple graph for testing
from classes.classifier import Classifier
from classes.session import Session
from classes.tensor import Tensor
from classes.scalar import Scalar
from classes.filter import Filter
from classes.bcip_enums import BcipEnums
from classes.graph import Graph
from classes.source import BcipClassSeparatedMat

from kernels.csp import CommonSpatialPatternKernel
from kernels.filter_ import FilterKernel
from kernels.classifier_ import ClassifierKernel
from kernels.covariance import CovarianceKernel
from kernels.riemann_mdm_classifier_kernel import RiemannMDMClassifierKernel

import scipy.io as sio

import numpy as np
from random import shuffle
import pyriemann
import scipy

def CSP_lib(file_data, init_data, init_labels):
    # create a session
    session = Session.create()
    trial_graph = Graph.create(session)

    X = Tensor.create_from_data(session,np.shape(init_data), init_data)
    y = Tensor.create_from_data(session,np.shape(init_labels),init_labels)


    input_data = BcipClassSeparatedMat.create_class_separated(session, 2, 500, 0, 4000, 0, 'input_data', 'input_labels', 'test_data\input_data.mat', 'test_data\input_labels.mat')
    input_data = Tensor.create_from_handle(session, (12, 500), input_data)

    t_out = Tensor.create_virtual(session)
    #t_virt = Tensor.create_virtual(session)
    print(f"input data_point: {input_data.data[0,0]}")
    
    # add the nodes
    CommonSpatialPatternKernel.add_uninitialized_CSP_node(trial_graph,input_data, t_out, X, y, 2)
    
    # verify the session (i.e. schedule the nodes)
    verify = trial_graph.verify()

    if verify != BcipEnums.SUCCESS:
        print(verify)
        print("Test Failed D=")
        return verify
    
    start = trial_graph.initialize()
    #input_data = trial_graph._nodes[0].kernel._outputA.data

    if start != BcipEnums.SUCCESS:
        print(start)
        print("Test Failed D=")
        return start
    
    # RUN!
    trial_seq = [0]*4 + [1]*4

    t_num = 0
    sts = BcipEnums.SUCCESS
    output_array = []
    while t_num < 8 and sts == BcipEnums.SUCCESS:
        print(f"t_num {t_num}, length of trials: {len(trial_seq)}")
        y = trial_seq[t_num]
        sts = trial_graph.execute(y)
        print(f"input data_point: {input_data.data[0,0]}")
                                
        if sts != BcipEnums.SUCCESS:
            print(f"Trial {t_num+1} raised error, status code: {sts}")
            break
        
        output_array.append(t_out.data)   
        t_num += 1
     
    return output_array

def CSP_colab(trial_data, init_data, labels):
    trial_data = loadmat("test_data\input_data.mat")['input_data'][:, :500]
    print(f"trial data_point: {trial_data[0,0]}")
    
    E = init_data.T
    E = np.moveaxis(E, (0,1,2),(2,1,0) )

    C = pyriemann.utils.covariance.covariances(E)
    #C = np.stack([Ci/np.trace(Ci) for Ci in C]) # normalize by the trace
    
    # compute average covariance for each class
    C1_bar = np.mean(C[0:60,:,:], axis=0)
    C2_bar = np.mean(C[60:,:,:], axis=0)
    # Sum the averages 
    Ctot_bar = C1_bar + C2_bar

    # Step 1 - compute the whitening transform
    l, U = np.linalg.eig(Ctot_bar)
    P = np.matmul(np.diag(l**(-1/2)), U.T)

    C_tot_white = np.matmul(P,np.matmul(Ctot_bar,P.T))

    # lets apply the whitening transform
    E_white = np.matmul(P, E)

    # apply the whitening transform to both class covariance matrices
    C1_bar_white = np.matmul(P,np.matmul(C1_bar,P.T))
    C2_bar_white = np.matmul(P,np.matmul(C2_bar,P.T))

    l, V = scipy.linalg.eigh(C1_bar_white, C_tot_white)

    # sort the eigenvalues and eigenvectors in order
    ix = np.flip(np.argsort(l)) 

    l = l[ix]
    V = V[:,ix]

    Phi = V[:,(0,-1)]

    # Rotate the filters back into the channel space
    W = np.matmul(P.T,Phi)

    E_csp = np.matmul(Phi.T, E_white)

    E_trial_CSP = np.matmul(W.T, trial_data)
    print(f"E_Trial_CSP.shape: {E_trial_CSP.shape} = W.T.shape: {W.T.shape} x trial_data.shape: {trial_data.shape}")
        
    return W, E_trial_CSP, C

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    
    from: `StackOverflow <http://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries>`_
    '''
    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], sio.matlab.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

def main():
    file_data = loadmat("test_data\input_data.mat")['input_data']
    print(file_data[0,0])
    labels = loadmat("test_data\input_labels.mat")['input_labels']
    init_data = sio.loadmat('test_data\init_data.mat')['init_data']
    init_labels = sio.loadmat('test_data\init_labels.mat')['labels']

    print(CSP_lib(file_data, init_data, init_labels)[0][0][0], CSP_colab(file_data, init_data, init_labels)[1][0][0])

def test():
    file_data = loadmat("test_data\input_data.mat")['input_data']
    print(file_data[0,0])
    labels = loadmat("test_data\input_labels.mat")['input_labels']
    init_data = sio.loadmat('test_data\init_data.mat')['init_data']
    init_labels = sio.loadmat('test_data\init_labels.mat')['labels']
    assert CSP_lib(file_data, init_data, init_labels)[0][0][0] == CSP_colab(file_data, init_data, init_labels)[1][0][0]

main()
