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
    CovarianceKernel.addCovarianceNode(b,t_virt[0],t_virt[1])
    FilterKernel.addFilterNode(b,t_in,f,t_virt[0])
    classifier = RiemannMDMClassifierKernel.addRiemannMDMKernel(b,t_virt[1],s_out)

    # verify the session (i.e. schedule the nodes)
    sts = s.verify()

    if sts != BcipEnums.SUCCESS:
        print(sts)
        print("Test Failed D=")
        return sts
    
    # initialize the classifier
    # fake data
    raw_training_data = np.random.normal(loc=0.0,scale=1.0,size=(180,250,12))
    training_data = np.zeros((180,12,12))
    for i in range(180):
        training_data[i,:,:] = np.cov(raw_training_data[i,:,:],rowvar=False)
        
    labels = np.asarray([0]*60 + [1]*60 + [2]*60)
    X = Tensor.createFromData(s,training_data.shape,training_data)
    y = Tensor.createFromData(s,labels.shape,labels)
    
    sts = classifier.kernel.trainClassifier(X,y)
    
    if sts != BcipEnums.SUCCESS:
        print(sts)
        print("Test Failed D=")
        return sts
    
    # RUN!
    sts = s.execute(0)
    
    print(s_out.getData())
    
    print("Test Passed =D")


if __name__ == "__main__":
    main()