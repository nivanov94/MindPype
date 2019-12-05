# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 16:12:30 2019

@author: ivanovn
"""

# Create a simple graph for testing


from classes.session import Session
from classes.tensor import Tensor
from classes.scalar import Scalar
from classes.filter import Filter
from classes.block import Block
from classes.bcip_enums import BcipEnums

from kernels.filter_kernel import FilterKernel
from kernels.covariance import CovarianceKernel
from kernels.riemann_mdm_classifier_kernel import RiemannMDMClassifierKernel

import numpy as np



def main():
    # create a session
    s = Session.create()

    # add a block and some tensors
    b = Block.create(s,4,3)

    # initialize the classifier
    # fake data for training
    raw_training_data = np.random.normal(loc=0.0,scale=1.0,size=(180,250,12))
    training_data = np.zeros((180,12,12))
    for i in range(180):
        training_data[i,:,:] = np.cov(raw_training_data[i,:,:],rowvar=False)
        
    labels = np.asarray([0]*60 + [1]*60 + [2]*60)
    X = Tensor.createFromData(s,training_data.shape,training_data)
    y = Tensor.createFromData(s,labels.shape,labels)

    input_data = np.random.randn(500,12)
    t_in = Tensor.createFromData(s,(500,12),input_data)
    s_out = Scalar.createFromValue(s,-1)
    t_virt = [Tensor.createVirtual(s), \
              Tensor.createVirtual(s)]
    
    # create a filter
    order = 4
    bandpass = (8,35) # in Hz
    fs = 250
    f = Filter.createButter(s,order,bandpass,btype='bandpass',fs=fs,implementation='sos')

    # add the nodes
    CovarianceKernel.addCovarianceNode(b.getTrialProcessGraph(),t_virt[0],t_virt[1])
    FilterKernel.addFilterNode(b.getTrialProcessGraph(),t_in,f,t_virt[0])
    RiemannMDMClassifierKernel.addUntrainedRiemannMDMKernel(b.getTrialProcessGraph(),
                                                            t_virt[1],
                                                            s_out,X,y)

    # verify the session (i.e. schedule the nodes)
    sts = s.verify()

    if sts != BcipEnums.SUCCESS:
        print(sts)
        print("Test Failed D=")
        return sts
    

    sts = s.startBlock()
    if sts != BcipEnums.SUCCESS:
        print(sts)
        print("Test Failed D=")
        return sts
    
    # RUN!
    sts = s.executeTrial(0)
    
    print(s_out.getData())
    
    print("Test Passed =D")


if __name__ == "__main__":
    main()