import mindpype as mp
import numpy as np
from glob import glob
from pyxdf import load_xdf
from sklearn.model_selection import train_test_split

print("starting script")
MODE = 'OFFLINE' # 'ONLINE' or 'OFFLINE'

def load_training_data():
    """
    Load training data
    """

    trial_len = 15 # trial length in seconds

    # load the data from the XDF files
    directory = r'P:\general_prism\Side Projects\NIRS BCI\Data\Dec4\sub-P001\sourcedata\*.xdf'
    streams = []
    for file in glob(directory):
        found_streams, _ = load_xdf(file)
        #streams.append(found_streams)
        trial_stream = {}
        for stream in found_streams:
            if stream['info']['type'][0] == 'NIRS':
                trial_stream['NIRS'] = stream
            if stream['info']['type'][0] == 'Marker':
                trial_stream['marker'] = stream
        streams.append(trial_stream)

    # filter for trials that are too short
    sfreq = int(streams[0]['NIRS']['info']['nominal_srate'][0])
    numCh = 46
    numTrials = len(streams)
    data = np.zeros([numTrials, trial_len*sfreq, numCh])
    labels = [None] * numTrials
    for i_t, trial in enumerate(streams):
        #print(i_t)
        marker_stream = trial['marker']
        nirs_stream = trial['NIRS']
        nirs_data = nirs_stream['time_series']
        nirs_ts = nirs_stream['time_stamps']
        trial_label = marker_stream['time_series'][-1]
        trial_start_ts = marker_stream['time_stamps'][-1]

        trial_data = nirs_data[nirs_ts > trial_start_ts, :] # get all data after the trial start marker
        trial_data = trial_data[:trial_len*sfreq, :] # get the first 15 seconds of data
        if trial_data.shape[0] == sfreq*trial_len:
            data[i_t,:,:] = trial_data
            labels[i_t] = trial_label[0]
        else:
            #print(f"Trial {i_t+1} too short")
            labels[i_t] = 'BAD'

    keep_trials = np.asarray([True if label!='BAD' else False for label in labels])
    data = data[keep_trials]
    labels = [label for label in labels if label!='BAD']
    labels = np.asarray([0 if 'Neutral' in label else 1 for label in labels])

    return data, labels


def create_graph(session, training_data, training_labels):
    """
    Create a graph for the mindpype pipeline
    """
    Fs = 50 # sampling frequency
    Nc = 46 # number of channels
    Ns = Fs * 15 # number of samples per trial

    epoch_len = int(Fs * 4) # number of samples per epoch
    epoch_stride = int(Fs * 0.1)

    graph = mp.Graph(session)

    # create input and output edges
    if MODE == 'ONLINE':
        data_src = mp.source.create_marker_coupled_data_stream(session,
                                                            "type='fnirs'",
                                                            marker_fmt='marker_template')
        data_in = mp.Tensor.create_from_handle(session,
                                            (Nc, Ns),
                                            data_src)
    else:
        data_in = mp.Tensor.create(session,
                                   (Nc, Ns))

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

    # create training data tensors
    training_data = mp.Tensor.create_from_data(session,training_data)
    training_labels = mp.Tensor.create_from_data(session,training_labels)

    # create the nodes

    # filter the data
    mp.kernels.FiltFiltKernel.add_filtfilt_node(graph,
                                                data_in,
                                                mp_filt,
                                                v_tensors[0],
                                                init_input=training_data,
                                                init_labels=training_labels)

    # epoch the data
    mp.kernels.EpochKernel.add_epoch_node(graph,
                                          v_tensors[0],
                                          v_tensors[1],
                                          epoch_len=epoch_len,
                                          epoch_stride=epoch_stride,
                                          axis=1)

    # compute features
    mp.kernels.MeanKernel.add_mean_node(graph,
                                        v_tensors[1],
                                        v_tensors[2],
                                        axis=2,
                                        keepdims=True)
    mp.kernels.VarKernel.add_var_node(graph,
                                      v_tensors[1],
                                      v_tensors[3],
                                      axis=2,
                                      keepdims=True)
    mp.kernels.KurtosisKernel.add_kurtosis_node(graph,
                                                v_tensors[1],
                                                v_tensors[4],
                                                axis=2,
                                                keepdims=True)
    mp.kernels.SkewnessKernel.add_skewness_node(graph,
                                                v_tensors[1],
                                                v_tensors[5],
                                                axis=2,
                                                keepdims=True)
    mp.kernels.SlopeKernel.add_slope_node(graph,
                                          v_tensors[1],
                                          v_tensors[6],
                                          Fs=Fs)

    # concatenation the features
    mp.kernels.ConcatenationKernel.add_concatenation_node(graph,
                                                      v_tensors[2],
                                                      v_tensors[3],
                                                      v_tensors[7],
                                                      axis=2)
    mp.kernels.ConcatenationKernel.add_concatenation_node(graph,
                                                      v_tensors[7],
                                                      v_tensors[4],
                                                      v_tensors[8],
                                                      axis=2)
    mp.kernels.ConcatenationKernel.add_concatenation_node(graph,
                                                      v_tensors[8],
                                                      v_tensors[5],
                                                      v_tensors[9],
                                                      axis=2)
    mp.kernels.ConcatenationKernel.add_concatenation_node(graph,
                                                      v_tensors[9],
                                                      v_tensors[6],
                                                      v_tensors[10],
                                                      axis=2)

    # flatten the feature vector
    mp.kernels.ReshapeKernel.add_reshape_node(graph,
                                              v_tensors[10],
                                              v_tensors[11],
                                              (-1,))

    # normalize the features
    mp.kernels.FeatureNormalizationKernel.add_feature_normalization_node(graph,
                                                                         v_tensors[11],
                                                                         v_tensors[12],
                                                                         axis=0)

    # select features
    mp.kernels.FeatureSelectionKernel.add_feature_selection_node(graph,
                                                                 v_tensors[12],
                                                                 v_tensors[13],
                                                                 k=100)

    # classify
    mp.kernels.ClassifierKernel.add_classifier_node(graph,
                                                    v_tensors[13],
                                                    mp_clsf,
                                                    cls_pred)

    return graph, data_in, cls_pred

def main():
    # load the data
    training_data, training_labels = load_training_data()
    training_data = np.transpose(training_data, (0,2,1))

    if MODE == 'OFFLINE':
        # split data into training and testing sets
        training_data, testing_data, training_labels, testing_labels = train_test_split(training_data,
                                                                                        training_labels,
                                                                                        test_size=0.25,
                                                                                        stratify=training_labels,
                                                                                        random_state=48)
    print("data loaded")
    # create a session
    session = mp.Session()

    # create a graph
    graph, data_in, clsf_pred = create_graph(session, training_data, training_labels)
    print("graph created")

    # verify the graph
    graph.verify()
    print("graph verified")

    # initialize the graph
    graph.initialize()

    if MODE == 'OFFLINE':
        predicitons = np.zeros_like(testing_labels)
        for i_t, trial in enumerate(testing_data):
            data_in.data = trial
            graph.execute(label=testing_labels[i_t])
            predicitons[i_t] = clsf_pred.data[0]
            print(f"Prediction: {predicitons[i_t]}, Actual: {testing_labels[i_t]}")
        print(f"Accuracy: {np.sum(predicitons==testing_labels)/len(testing_labels)}")

if __name__ == "__main__":
    main()
