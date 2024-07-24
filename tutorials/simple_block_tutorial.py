# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 15:37:58 2019

full block test

"""
import mindpype as mp
from random import shuffle

def main():
    # create a session
    s = mp.Session.create()
    g = mp.Graph.create(s)

    # create some random data for traininging the classifier
    X = mp.Tensor.create(s,(100,32,32))
    y = mp.Tensor.create(s,(100,))
    X.assign_random_data(covariance=True)
    y.assign_random_data(whole_numbers=True,vmax=2)


    # create the input data tensor
    eeg = mp.Tensor.create(s, (32, 750))

    # create virtual tensor (filtered data & covariance matrix)
    t_virt = [mp.Tensor.create_virtual(s),
              mp.Tensor.create_virtual(s)]

    # create the output label
    label = mp.Scalar.create_from_value(s,-1)


    # create a filter object
    order = 4
    bandpass = (8,35) # in Hz
    fs = 250
    f = mp.Filter.create_butter(s,order,bandpass,btype='bandpass',fs=fs,implementation='sos')


    # add the nodes to the block
    mp.kernels.CovarianceKernel.add_to_graph(g,t_virt[0],t_virt[1])

    mp.kernels.FiltFiltKernel.add_to_graph(g,eeg,f,t_virt[0])

    mp.kernels.RiemannMDMClassifierKernel.add_to_graph(g,
                                                       t_virt[1],
                                                       label,3,X,y)

    # verify the session (i.e. schedule the nodes)
    s.verify()

    # initialize the session
    s.initialize()

    trial_seq = [0]*4 + [1]*4 + [2]*4
    shuffle(trial_seq)

    # RUN!
    correct_labels = 0
    for t_num in range(len(trial_seq)):
        print(f"\nRunning trial {t_num+1} of {len(trial_seq)}")
        y = trial_seq[t_num]

        # set random input data for now
        eeg.assign_random_data()

        g.execute(label=y)
        y_bar = label.data
        print("Trial {}: Label = {}, Predicted label = {}".format(t_num+1,y,y_bar))

        if y == y_bar:
            correct_labels += 1

    print("\nAccuracy = {:.2f}%.".format(100 * correct_labels/len(trial_seq)))


if __name__ == "__main__":
    main()
