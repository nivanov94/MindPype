Getting Started with BCIPy
==========================

**BCIPy** is a Python-based software library designed for BCI researchers and developers that greatly simplifies the design and development of BCI data processing pipelines. 
Users can create and implement custom processing pipelines to use alongside new or existing BCI interfaces. 

Setup Guide
============
Installing *pip* (Python Package Manager) is the first step to using BCIPy. @NICK this is for you to describe :-}
Once Pip is installed, install BCIPy and its dependencies from Pypi


The following guide will walkthrough the creation of a basic processing graph.

The usage of the BCIP library is based on sessions, that is, a particular processing session. A Session object must be instantiated before building or executing a processing graph. Sessions are instantiated using the following syntax:

SYNTAX TO INSTANTIATE A SESSION OBJECT

The next step to building a fully functional BCIP processing session is to define the block, or set of trials, that will be executed within the session. Blocks are defined with the following parameters; the number of classes within the session, and the number of trials per class. Multiple blocks can be defined within the same session, often in the structure of an offline processing block (specifically for training data collection), and online processing block (for live execution for user testing, etc). A single block can be defined using the following syntax.

SYNTAX TO INSTANTIATE A BLOCK OBJECT

To create a two-block session, as described above, users may use the following code:

SYNTAX TO INSTATIATE TWO BLOCK OBJECTS

Once the blocks (trial set) to be executed within the session has been defined, the location of your input data and training data should be provided. Input data is most commonly provided with an LSL stream (for online processing), or a MAT file (for batch processing); while training data is most often provided either as a MAT file or the output of a previous block (as a circle buffer/tensor object). It may be easiest to understand the function of the BCIPY library using a MAT File input for input and training data, and use the following code to create the input/training data objects:

SYNTAX TO INSTANTIATE MAT File Data

Along with the input and training data, we also need to define an output data object (to store the result of the processing graph), and edge object (which connect nodes/processing elements together). For this example, our last kernel will be a classifier (and its output a scalar), so we can define an output with the following:

SYNTAX TO INSTANTIATE OUTPUT SCALAR OBJECT

Furthermore, this example will contain 3 nodes (a temporal filter, a CSP node, and an LDA classifier), thus, we will need 2 edges (think data holder

Following the creation of the data objects to be used within the session, the processing pipeline, or graph, that will process the data from each individual trial, can be created. To create a graph object, the following code can be used:

SYNTAX TO INSTANTIATE GRAPH OBJECT

Once the graph object has been created, nodes (processing elements) can be added to the graph. Nodes are the objects that get added to the graph, but nodes contain kernels, which are objects that conduct the data processing. To create a filter node and add it to the graph, use the following syntax:

SYNTAX TO INSTANTIATE FILTER KERNEL & NODE OBJECT
