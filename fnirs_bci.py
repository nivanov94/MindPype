import mindpype as mp
import numpy as np

def create_graph(session):
    """
    Create a graph for the mindpype pipeline
    """
    Fs = 50 # sampling frequency
    Nc = 46 # number of channels
    Ns = Fs * 15 # number of samples per trial

    epoch_len = Fs * 4 # number of samples per epoch
    epoch_stride = Fs * 0.1

    graph = mp.Graph(session)

    # create input and output edges
    data_src = mp.source.create_marker_coupled_data_stream(session,
                                                           "type='fnirs'",
                                                           marker_fmt='marker_template')
    data_in = mp.Tensor.create_from_handle(session,
                                           (Nc, Ns),
                                           data_src)

    cls_pred = mp.Tensor.create(session,(1,))

    # create the filter obj
    mp_filt = mp.Filter.create_butter(session,
                                      4,
                                      (0.01, 0.8),
                                      'bandpass',
                                      'sos',
                                      Fs)

    # create the classifier obj
    mp_clsf = mp.Classifier.create_LDA(session,shrinkage='auto',solver='lsqr')

    # create virtual edges
    v_tensors = [
        mp.Tensor.create_virtual(session), #  0 - filtered data
        mp.Tensor.create_virtual(session), #  1 - epoched data
        mp.Tensor.create_virtual(session), #  2 - mean
        mp.Tensor.create_virtual(session), #  3 - variance
        mp.Tensor.create_virtual(session), #  4 - kurtosis
        mp.Tensor.create_virtual(session), #  5 - skew
        mp.Tensor.create_virtual(session), #  6 - slope
        mp.Tensor.create_virtual(session), #  7 - mean+var
        mp.Tensor.create_virtual(session), #  8 - mean+var+kurt
        mp.Tensor.create_virtual(session), #  9 - mean+var+kurt+skew
        mp.Tensor.create_virtual(session), # 10 - mean+var+kurt+skew+slope
        mp.Tensor.create_virtual(session), # 11 - flattened feature vector
        mp.Tensor.create_virtual(session), # 12 - normalized feature vector
        mp.Tensor.create_virtual(session), # 13 - selected features
    ]

    # create the nodes

    # filter the data
    mp.FiltFiltKernel.add_filtfilt_node(graph,
                                        data_in,
                                        mp_filt,
                                        v_tensors[0],
                                        init_inputs=training_data,
                                        labels=training_labels)

    # epoch the data
    mp.EpochKernel.add_epoch_node(graph,
                                  v_tensors[0],
                                  v_tensors[1],
                                  epoch_len=epoch_len,
                                  epoch_stride=epoch_stride,
                                  axis=1)

    # compute features
    mp.MeanKernel.add_mean_node(graph,
                                v_tensors[1],
                                v_tensors[2],
                                axis=1,
                                keepdims=True)
    mp.VarKernel.add_var_node(graph,
                              v_tensors[1],
                              v_tensors[3],
                              axis=1,
                              keepdims=True)
    mp.KurtosisKernel.add_kurtosis_node(graph,
                                        v_tensors[1],
                                        v_tensors[4],
                                        axis=1,
                                        keepdims=True)
    mp.SkewKernel.add_skew_node(graph,
                                v_tensors[1],
                                v_tensors[5],
                                axis=1,
                                keepdims=True)
    mp.PolynomialFitKernel.add_polynomial_fit_node(graph,
                                                   v_tensors[1],
                                                   v_tensors[6],
                                                   Fs=Fs,
                                                   order=1)

    # concatenate the features
    mp.ConcatenateKernel.add_concatenate_node(graph,
                                              v_tensors[2],
                                              v_tensors[3],
                                              v_tensors[7]
                                              axis=2)
    mp.ConcatenateKernel.add_concatenate_node(graph,
                                              v_tensors[7],
                                              v_tensors[4],
                                              v_tensors[8]
                                              axis=2)
    mp.ConcatenateKernel.add_concatenate_node(graph,
                                              v_tensors[8],
                                              v_tensors[5],
                                              v_tensors[9]
                                              axis=2)
    mp.ConcatenateKernel.add_concatenate_node(graph,
                                              v_tensors[9],
                                              v_tensors[6],
                                              v_tensors[10]
                                              axis=2)

    # flatten the feature vector
    mp.ReshapeKernel.add_reshape_node(graph,
                                      v_tensors[10],
                                      v_tensors[11],
                                      (1,-1))

    # normalize the features
    mp.FeatureNormalizationKernel.add_feature_normalization_node(graph,
                                                                 v_tensors[11],
                                                                 v_tensors[12])

    # select features
    mp.FeatureSelectionKernel.add_feature_selection_node(graph,
                                                         v_tensors[12],
                                                         v_tensors[13],
                                                         k=100)

    # classify
    mp.ClassificationKernel.add_classification_node(graph,
                                                    v_tensors[13],
                                                    mp_clsf,
                                                    cls_pred)

    return graph

def main():
    # create a session
    session = mp.Session()

    # create a graph
    graph = create_graph(session)

    # verify the graph
    graph.verify()

    # initialize the graph
    graph.initialize()

    while True:
        graph.execute()