

Basic Setup
===========

**The following guide will walkthrough the creation of a basic processing graph.**

Creating a Session
------------------
The usage of the BCIP library is based on sessions, that is, a particular processing session. A Session object must be instantiated before building or executing a processing graph. Sessions are instantiated using the following syntax:

>>> example_session = Session.create()

Creating a Graph
----------------

Once the session has been defined, the trial graph(s) to conduct the processing must be created. 

>>> trial_graph = Graph.create(example_session)

Creating input data
-------------------
Next, the location of your input data and training data should be provided. Trial Input data is most commonly 
provided with an LSL stream (for online processing), or a MAT file (for batch processing); 
while training data is most often provided either as a MAT file or the output of a execution (as a circle buffer/tensor object). 
It may be easiest to understand the function of the BCIPY library using a MAT File input for input and training data. There are 3
types of MAT file inputs to be used for trial inputs, depending on how the data is organized. The epoched mat input is for MAT data that is separated by trial,
the class separated mat input is for data that is separated by class, and the continuous data mat input is for continuous data (raw eeg input).
For this setup, we'll use continuous data for the input data, and a generic Tensor with mat data for the initialization data, 
and use the following code to create the input/training data objects: 

>>> input_data = BcipContinuousMat.create_continuous(session, 500, 0, 4000, 0, 'input_data', 'input_labels', 'test_data\input_data.mat', 'test_data\input_labels.mat')
>>> training_data = sio.loadmat('test_data\init_data.mat')['init_data']
>>> training_labels = sio.loadmat('test_data\init_labels.mat')['labels']
>>> #500 is the trial duration, 0 is the first sample of the data to be used, 4000 is the last sample, and 0 is the relative start

Converting data to BCIPy type
-----------------------------
In the below step, it is critical to remember that inputs and outputs to graph/node must be a BCIPy built-in type 
(Tensor, Array, or Scalar depending on the specific nodes being used), for both training data and trial data. Thus, we'll 
transform the input and training data to Tensors. For the input data, we'll create a Tensor from a "handle", which simply
means that it will be updated between trials from an external data source (the MAT file we imported), while the training
data is a Tensor we create from data and doesn't need to be modified/polled.

>>> input_data = Tensor.create_from_handle(session, (12, 500), input_data)
>>> training_data_tensor = Tensor.create_from_data(session,np.shape(init_data), init_data) 
>>> training_labels_tensor = Tensor.create_from_data(session,np.shape(init_labels),init_labels)

Defining outputs
----------------
Along with the input and training data, we also need to define an output data object (to store the result of the processing graph), and edge object (which connect nodes/processing elements together). For this example, our last kernel will be a classifier (and its output a scalar), so we can define an output with the following:

>>> output = Scalar.create_from_value(session,-1)

Creating Nodes
--------------
This example will contain 3 nodes (a temporal filter, a CSP node, and an LDA classifier), thus, we will need 2 edges (think data holder

>>> edges_to_connect_nodes = [Tensor.create_virtual(session), Tensor.create_virtual(session)]

Once the edges have been created, nodes (processing elements) can be added to the graph. Nodes are the objects that get added to the graph, but nodes contain kernels, which are objects that conduct the data processing. To create a filter node and add it to the graph, use the following syntax:

>>> filter_object = Filter.create_butter(session,4,(8,35),btype='bandpass',fs=250,implementation='sos')
>>> # see filter class for parameter explanation
>>> FilterKernel.add_filter_node(trial_graph, input_data, filter_object, edges_to_connect_nodes[0])
>>> # where input_data is the input to the node, filter_object is the specific filter that will conduct the processing, and edges_to_connect_nodes[0] is the Tensor where the output will be stored

In a similar fashion, we can add our CSP and LDA Classifier Nodes:

>>> CommonSpatialPatternKernel.add_uninitialized_CSP_node(trial_graph, edges_to_connect_nodes[0], edges_to_connect_nodes[1], training_data_tensor, training_labels_tensor, 2)
>>> lda_object = Classifier.create_LDA(session)
>>> ClassifierKernel.add_classifier_node(trial_graph, edges_to_connect_nodes[1], lda_object, output, None, None)

In the above line, we set the training_data/labels input parameters to None. This is an important feature to remember about the BCIPy library;
if training data is provided for some upstream node (in this graph, the CSP node), the library will automatically compute the training data for
downstream nodes. If we wanted, we could've passed different training data, but opted not to for this example.

Verifying a graph
-----------------
Once the graph is built, the next step is to have it verified. Verification will schedule nodes, ensure that inputs and outputs
are the correct shape and type, and will determine whether any nodes are missing training data. This process can be started with a single command:

>>> verification_status = trial_graph.verify()
>>> print(verification_status)

    SUCCESS

Initializing the graph
----------------------
The last step before we can use the graph is to have it initialized, which will initialize each node within the graph (train, compute filters, etc):
this process can also be started with one command:

>>> initialization_status = trial_graph.initialize()
>>> print(initialization_status)

    SUCCESS

Executing a graph
-----------------
At this stage, the graph is ready to process data. During execution, we have a number of options, so we must first ask ourselves a number of questions.

1. Do the trials you're executing have a defined structure?
    - If yes, it may be useful to define a trial_set before execution
    - If not, it is assumed that you'll be executing trials indefinitely (ie. P300 speller setup)

2. Do you know the class labels of the trials you're executing. For example, in some mental imagery experiments, we know what action will be completed before it occurs.
    - If yes, you will not be able to use continuous data for this. You should use class separated / epoched data instead. At that point, you'll be able to call execute like:

>>> execution_status = trial_graph.execute(class_label)

Execution with epochs, known labels
-----------------------------------
If you were to use epoched data and a trial set, we could define a trial set using standard Python syntax: 

>>> trial_seq = [0]*4 + [1]*4 #4 trials of class 0, 4 of class 1

Based on the provided data (4000 samples, each trial lasting 500 samples), we already know that only 8 trials can occur, so we can loop through the trial set like this:

>>> while t_num < 8 and sts == BcipEnums.SUCCESS:
>>>     y = trial_seq[t_num]
>>>     sts = trial_graph.execute(y)
>>>     if sts == BcipEnums.SUCCESS:
>>>         t_num += 1
>>>         y_bar = s_out.data
            print("Trial {}: Predicted label = {}".format(t_num+1,y_bar))
>>>     else:
>>>         print(f"Trial {t_num+1} raised error, status code: {sts}")
>>>         break

**Remember, this will only work with class separated/epoched data, not continuous mat data**


Execution with continuous data, unknown labels
----------------------------------------------
If we were to use continuous data and no class labels (ie. P300) we could loop through in a similar way, but execute trials without labels.
**Remember, this only works with continuous data, labels are required for epoched/class separated data**

Based on the input data, we still know only 8 trials will occur, so we could use the following code.

>>> t_num = 0
>>> while t_num < 8 and sts == BcipEnums.SUCCESS:
>>>     sts = trial_graph.execute()
>>>     if sts == BcipEnums.SUCCESS:
>>>         t_num += 1
            y_bar = s_out.data
            print("Trial {}: Predicted label = {}".format(t_num+1,y_bar))
>>>     else:
>>>         print(f"Trial {t_num+1} raised error, status code: {sts}")
>>>         break

This concludes the setup guide, please see the other documentation, or examples available in our github repo for more examples/test scripts.
