"""
Created on Mon Dec  2 12:00:43 2019

graph.py - Defines the graph object

@author: ivanovn
"""

import logging
import warnings

from .core import MPBase, MPEnums
from .containers import Tensor
import sys

class Graph(MPBase):
    """
    This class represents the data processing flow graph, or processing pipelines.
    Individual nodes, or processing steps, are added to the graph to create the pipeline.

    Parameters
    ----------
    sess : Session Object
        Session where the graph will exist

    Attributes
    ----------
    _nodes : List of Node
        List of Node objects within the graph

    initialization_edges : List of Edge objects
        List of Edge objects used within the graph

    _verified : bool
        True is graph has been verified, false otherwise

    _sess : Session object
        Session where the Graph object exists

    _volatile_sources : List of Sources
        Data sources within this array will be polled/executed when the graph is executed.

    _volatile_outputs : List of data Outputs
        Data outputs within this array will push to external sources when the graph is executed.

    Examples
    --------


    """

    def __init__(self, sess):
        """
        Constructor for the Graph object
        """

        super().__init__(MPEnums.GRAPH, sess)

        # private attributes
        self._nodes = []
        self.initialization_edges = []
        self._verified = False
        self._initialized = False
        self._sess = sess
        self._volatile_sources = []
        self._volatile_outputs = []
        self._edges = {}

        self._default_init_required = False
        self._default_init_data = None
        self._default_init_labels = None

    def add_node(self, node):
        """
        Append a node object to the list of nodes

        Parameters
        ----------
        node : Node object
            Adds the specified Node object to the referenced graph

        Examples
        --------
        example_graph.add_node(example_node)

        """
        self._verified = False
        self._initialized = False
        self._nodes.append(node)

    def set_default_init_data(self, data, labels):
        """
        Add default initialization data to the graph

        Parameters
        ----------
        data : MindPype Tensor or Array
            Tensor or array containing the default initialization data

        labels : MindPype Tensor or Array
            Tensor or array containing the default initialization labels

        """
        self._verified = False
        self._initialized = False
        self._default_init_data = data
        self._default_init_labels = labels

    def verify(self):
        """
        Verify the processing graph is valid. This method orders the nodes
        for execution if the graph is valid
        """
        if self._verified:
            return

        # begin by scheduling the nodes in execution order
        self._schedule_nodes()

        # assign default initialization data to nodes that require it
        self._assign_default_init_data()

        # now all the nodes are in execution order create any necessary initialization edges
        self._insert_init_edges()

        # insert phony edges for verification
        self._insert_phony_edges()

        # set phony inputs with random data for validation
        self._init_phony_edges()

        # finally, validate each node
        self._validate_nodes()

        # delete phony inputs and outputs
        self._delete_phony_edges()

        # Done, all nodes scheduled and verified!
        self._verified = True

        # cleanup any data used within verification that are no longer needed
        self._edges = {} # clear edges for garbage collection
        self.session.free_unreferenced_data()

    def _schedule_nodes(self):
        """
        Place the nodes of the graph in execution order
        """
        # first we'll create a set of edges representing data within the graph
        self._edges = {} # keys: session_id of data obj, vals: edge object
        for n in self._nodes:
            # get a list of all the input objects to the node
            n_inputs = n.extract_inputs()
            n_outputs = n.extract_outputs()

            # add these inputs/outputs to edge objects
            for n_i in n_inputs:
                if not n_i.session_id in self._edges:
                    # no edge created for this input yet, so create a new one
                    self._edges[n_i.session_id] = Edge(n_i)
                    if n_i.volatile:
                        self._volatile_sources.append(n_i)
                # now add the node the edge's list of consumers
                self._edges[n_i.session_id].add_consumer(n)

            for n_o in n_outputs:
                if not n_o.session_id in self._edges:
                    # no edge created for this output yet, so create a new one
                    self._edges[n_o.session_id] = Edge(n_o)

                    # add the node as a producer
                    self._edges[n_o.session_id].add_producer(n)
                    if n_o.volatile_out:
                        self._volatile_outputs.append(n_o)
                else:
                    # edge already created, must check that it has no other
                    # producer
                    if len(self._edges[n_o.session_id].producers) != 0:
                        # this is an invalid graph, each data object can only
                        # have a single producer
                        raise Exception("Invalid graph, multiple nodes write to single data object")
                    else:
                        # add the producer to the edge
                        self._edges[n_o.session_id].add_producer(n)
                        if n_o.volatile_out and n_o not in self._volatile_outputs:
                            self._volatile_outputs.append(n_o)

        # now determine which edges are ready to be consumed
        consumable_edges = {}
        for e_key in self._edges:
            if len(self._edges[e_key].producers) == 0:
                # these edges have no producing nodes, so they are inputs to
                # the block and therefore can be consumed immediately
                consumable_edges[e_key] = self._edges[e_key]

        scheduled_nodes = 0
        total_nodes = len(self._nodes)

        while scheduled_nodes != total_nodes:
            nodes_added = 0
            # find the next node that has all its inputs ready to be consumed
            for node_index in range(scheduled_nodes, len(self._nodes)):
                n = self._nodes[node_index]
                n_inputs = n.extract_inputs()
                n_outputs = n.extract_outputs()
                consumable = True
                for n_i in n_inputs:
                    if not (n_i.session_id in consumable_edges):
                        # the inputs for this node must be produced by another
                        # node first, therefore this node cannot be scheduled
                        # yet
                        consumable = False

                if consumable:
                    # schedule this node
                    if scheduled_nodes != node_index:
                        # swap the nodes at these indices
                        tmp = self._nodes[scheduled_nodes]
                        self._nodes[scheduled_nodes] = self._nodes[node_index]
                        self._nodes[node_index] = tmp

                    # mark this node's outputs ready for consumption
                    for n_o in n_outputs:
                        consumable_edges[n_o.session_id] = self._edges[n_o.session_id]

                    nodes_added = nodes_added + 1

                    scheduled_nodes = scheduled_nodes + 1


            if nodes_added == 0:
                # invalid graph, cannot be scheduled
                raise Exception("Invalid graph, nodes cannot be scheduled, check connections between nodes.")

    def _insert_init_edges(self):
        """
        Insert initialization edges into the graph
        """
        init_required = False # flag to indicate if any nodes in the graph require initialization
        init_links_missing = False # flag to indicate if any initialization data will need to be propagated through the graph
        for n in self._nodes:
            # check for missing init data
            if n.kernel.init_style == MPEnums.INIT_FROM_DATA:
                init_required = True

                # check whether all init inputs have been provided by the user
                init_provided = True
                for n_ii in n.kernel.init_inputs:
                    if n_ii is None:
                        init_provided = False

                # if not provided, flag that graph will need initialization data propagated through the graph
                if not init_provided:
                    init_links_missing = True

        # fill in all init data links
        if init_required and init_links_missing:
            # use the existing Edge objects to create init connections mirroring the processing graph
            for e in self._edges:
                self._edges[e].insert_init_data()


    def _validate_nodes(self):
        """
        Validate each node within the graph individually
        """
        for n in self._nodes:
            try:
                n.verify()
            except Exception as e:
                raise type(e)(f"{str(e)} - Node: {n.kernel.name} failed verification").with_traceback(sys.exc_info()[2])


    def _assign_default_init_data(self):
        """
        If default init data exists, add it to any root nodes
        that do not have any init data
        """
        if self._default_init_data is None:
            return

        for n in self._nodes:
            n_inputs = n.extract_inputs()
            root_data_node = False

            # check whether this node ingests data from outside the graph
            for index, n_i in enumerate(n_inputs):
                if len(self._edges[n_i.session_id].producers) == 0:
                    root_data_node = True
                    init_data_input_index = index
                    break

            if root_data_node:
                # copy the default init data to the node's init input
                n.kernel.init_inputs[init_data_input_index] = self._default_init_data
                if self._default_init_labels is not None:
                    n.kernel.init_input_labels = self._default_init_labels

    def _insert_phony_edges(self):
        """
        Add phony edges to the graph to be used during verification
        """
        for e_id in self._edges:
            e = self._edges[e_id]

            # check if the data is virtual
            if not e.data.virtual:
                # if not virtual, create a phony edge
                e.add_phony_data()

            # check if the edge has non-virtual init data
            if e.init_data is not None and not e.init_data.virtual:
                e.add_phony_init_data()


    def _init_phony_edges(self):
        """
        Initialize phony edges with random data for validation
        """
        for eid in self._edges:
            self._edges[eid].initialize_phony_data()

    def _delete_phony_edges(self):
        """
        Remove references to any phony edges so the
        data will be freed during garbage collection
        """
        for eid in self._edges:
            self._edges[eid].delete_phony_data()

    def initialize(self, default_init_data = None, default_init_labels = None):
        """
        Initialize each node within the graph for trial execution

        Parameters
        ----------
        default_init_dataA : Tensor, default = None
            If the graph has no initialization data, this tensor will be used to initialize the graph
        default_init_labels : Tensor, default = None
            If the graph has no initialization labels, this tensor will be used to initialize the graph

        """
        if default_init_data is not None:
            self.set_default_init_data(default_init_data, default_init_labels)

        if not self._verified:
            self.verify()

        # execute initialization for each node in the graph
        for n in self._nodes:
            try:
                n.initialize()
            except Exception as e:
                raise type(e)(f"{str(e)} - Node: {n.kernel.name} failed initialization").with_traceback(sys.exc_info()[2])

        self._initialized = True
        self.session.free_unreferenced_data()


    def execute(self, label = None):
        """
        Execute the graph by iterating over all the nodes within the graph and executing each one

        Parameters

        ----------

        Label : int, default = None
            * If the trial label is known, it can be passed when a trial is executed. This is required for class-separated input data
            * If the trial label is not known, it will be polled from the data source

        """
        # first ensure the graph has been verified,
        # if not, verify and schedule the nodes
        if not self._verified:
            self.verify()

        if not self._initialized:
            self.initialize()

        # Check whether first node has volatile input
        # if so, poll the volatile data
        if len(self._volatile_sources) > 0:
            self.poll_volatile_sources(label)

        print("Executing trial with label: {}".format(label))

        # iterate over all the nodes and execute the kernel
        for n in self._nodes:
            try:
                n.kernel.execute()
            except Exception as e:
                raise type(e)(f"{str(e)} - Node: {n.kernel.name} failed execution").with_traceback(sys.exc_info()[2])

        if len(self._volatile_outputs) > 0:
            self.push_volatile_outputs(label)

    def poll_volatile_sources(self, label = None):
        """
        Poll data (update input data) from volatile sources within the graph.

        Parameters
        ----------
        label : int, default = None
            If the class label of the current trial is known, it can be passed to poll epoched data.

        Return
        ------
        None

        Example
        -------
        >>> example_graph.poll_volatile_data(0) # Polls next class 0 trial data
        """
        for datum in self._volatile_sources:
            datum.poll_volatile_data(label)

    def push_volatile_outputs(self, label=None):
        """
        Push data (update output data) to volatile outputs within the graph.

        Parameters
        ----------
        label : int, default = None
            If the class label of the current trial is known, it can be passed to poll epoched data.

        Return
        ------
        None

        Example
        -------
        >>> example_graph.poll_volatile_data(0) # Polls next class 0 trial data
        """
        for datum in self._volatile_outputs:
            datum.push_volatile_outputs(label=label)


    @classmethod
    def create(cls, sess):
        """
        Generic factory method for a graph
        """
        graph = cls(sess)
        sess.add_graph(graph)

        return graph


class Node(MPBase):
    """
    Generic node object containing a kernel function

    Parameters
    ----------
    graph : Graph object
        Graph where the Node object will exist
    kernel : Kernel Object
        Kernel object to be used for processing within the Node
    params : dict
        Dictionary of parameters outputted by kernel

    Attributes
    ----------
    _kernel : Kernel Object
        Kernel object to be used for processing within the Node
    _params : dict
        Dictionary of parameters outputted by kernel

    Examples
    --------
    >>> Node.create(example_graph, example_kernel, example_params)
    """

    def __init__(self, graph, kernel, params):
        sess = graph.session
        super().__init__(MPEnums.NODE, sess)

        self._kernel = kernel
        self._params = params

        self._graph = graph

    # API getters
    @property
    def kernel(self):
        return self._kernel

    def extract_inputs(self):
        """
        Return a list of all the node's inputs

        Parameters
        ----------
        None

        Return
        ------
        List of inputs for the Node : List of Nodes

        Examples
        --------

        >>> inputs = example_node.extract_inputs()
        >>> print(inputs)

            None

        """
        inputs = []
        for p in self._params:
            if p.direction != MPEnums.OUTPUT:
                inputs.append(p.data)

        return inputs

    def extract_outputs(self):
        """
        Return a list of all the node's outputs

        Parameters
        ----------
        None

        Return
        ------
        List of inputs for the Node : List of Nodes

        Examples
        --------

        >>> inputs = example_node.extract_outputs()
        >>> print(inputs)

            None
        """
        outputs = []
        for p in self._params:
            if p.direction == MPEnums.OUTPUT:
                outputs.append(p.data)

        return outputs

    def verify(self):
        """
        Verify the node is executable
        """
        return self.kernel.verify()

    def initialize(self):
        """
        Initialize the kernel function for execution
        """
        return self.kernel.initialize()

    def update_parameters(self, parameter, value):
        """
        Update the parameters of the node
        """

        self.kernel.update_parameters(parameter, value)

        self._graph._verified = False

    def add_initialization_data(self, init_data, init_labels=None):
        """
        Add initialization data to the node

        Parameters
        ----------
        init_data : list or tuple of MindPype data objects
            MindPype container containing the initialization data
        init_labels : MindPype data object containing initialization labels, default = None
            MindPype container containing the initialization labels

        """
        self.kernel.add_initialization_data(init_data, init_labels)
        self._graph.verified = False


class Edge:
    """
    Edge class used by MindPype block to schedule graphs. Each edge object
    represents a different MindPype data object and stores the nodes that produce
    and consume that data.

    Parameters
    ----------
    data : Data object
        The data to be stored within the Edge object

    Attributes
    ----------
    _producers : array of Node objects
        Node objects that will produce the data within the Edge object
    _consumers : array of Node objects
        Node objects that will consume the data within the Edge object

    Examples
    --------
    >>> Edge.create(example_data)

    """

    def __init__(self, data):
        """
        Constructor for Edge object
        """
        self._data = data
        self._producers = []
        self._consumers = []

        self._init_data = None
        self._init_labels = None
        self._phony_data = None
        self._phony_init_data = None
        self._phony_init_labels = None


    @property
    def producers(self):
        """
        Getter for producers property

        Return
        ------
        _producers : List of Node
            List of producers for the Edge object

        Examples
        --------
        >>> example_edge.producers
        """
        return self._producers

    @property
    def consumers(self):
        """
        Getter for consumers property

        Return
        ------
        List of consumers for the Edge object

        Return Type
        -----------
        List of Node objects

        Examples
        --------
        >>> print(example_edge.consumers)

            [example_consumer_node]

        """
        return self._consumers

    @property
    def data(self):
        """
        Getter for data property

        Return
        ------
        Data object stored within the Edge object

        Return Type
        -----------
        MindPype Data object
        """

        return self._data

    @property
    def init_data(self):
        """
        Getter for init_data property

        Return
        ------
        Data object stored within the Edge object

        Return Type
        -----------
        MindPype Data object
        """

        return self._init_data

    @property
    def init_labels(self):
        """
        Getter for init_labels property

        Return
        ------
        Data object stored within the Edge object

        Return Type
        -----------
        MindPype Data object
        """

        return self._init_labels

    def add_producer(self, producing_node):
        """
        Add a specified node as a producer to an Edge object

        .. note:: Adds producer in place, does not return a new Edge object

        Parameters
        ----------
        producing_node : Node object
            Node to be added as a producer to the referenced Edge object

        Examples
        --------
        example_edge.add_producer(example_producing_edge)

        """
        self.producers.append(producing_node)

    def add_consumer(self, consuming_node):
        """
        Add a specified node as a consumer to an Edge object

        .. note:: Adds consumer in place, does not return a new Edge object

        Parameters
        ----------
        consuming_node : Node object
            Node to be added as a consumer to the referenced Edge object

        Examples
        --------
        example_edge.add_consumer(example_consumer_edge)

        """
        self.consumers.append(consuming_node)

    def add_data(self, data):
        """
        Add specified data to an Edge object

        .. note:: Adds data object in place, does not return a new Edge object

        Parameters
        ----------
        data : Tensor, Scalar, Array, Python Built-in Data Types
            Data to be added to the referenced Edge object

        Examples
        --------
        example_edge.add_data(example_data)

        """
        self._data = data


    def insert_init_data(self):
        """
        Insert initialization data tensors into the graph that mirror the
        connections contained within the Edge object
        """
        # create a virtual tensor that will contain the initialization data
        self._init_data = Tensor.create_virtual(self.data.session)
        self._init_labels = Tensor.create_virtual(self.data.session)

        for p in self.producers:
            output_index = self._find_output_index(p)

            # assign the tensor to the producer's corresponding init output
            p.kernel.init_outputs[output_index] = self.init_data
            p.kernel.init_output_labels = self.init_labels

        for c in self.consumers:
            # find the index of the data from the consumer node (input index)
            input_index = self._find_input_index(c)

            # check whether this input has not already been assigned init data
            if c.kernel.init_inputs[input_index] is None:
                # If so, assign the tensor to the consumer's corresponding init input
                c.kernel.init_inputs[input_index] = self.init_data
                c.kernel.init_input_labels = self.init_labels
            else:
                # overwrite the edge's init data, we need this to create phony inputs later
                self._init_data = c.kernel.init_inputs[input_index]
                self._init_labels = c.kernel.init_input_labels


    def add_phony_data(self):
        """
        Add phony data to the edge and the
        nodes it is connected to
        """
        self._phony_data = self.data.make_copy()

        # get the producing node
        for p in self.producers:
            # find the index of the data from the producer node (output index)
            output_index = self._find_output_index(p)

            # assign the tensor to the producer's corresponding init output
            p.kernel.phony_outputs[output_index] = self._phony_data

        for c in self.consumers:
            # find the index of the data from the consumer node (input index)
            input_index = self._find_input_index(c)

            #  assign the tensor to the consumer's corresponding init input
            c.kernel.phony_inputs[input_index] = self._phony_data

    def add_phony_init_data(self):
        """
        Add phony init data to the edge and the
        nodes connected to it
        """
        self._phony_init_data = self.init_data.make_copy()
        if self.init_labels is not None:
            self._phony_init_labels = self.init_labels.make_copy()

        # get the producing node
        for p in self.producers:
            # find the index of the data from the producer node (output index)
            output_index = self._find_output_index(p)

            # assign the tensor to the producer's corresponding init output
            p.kernel.phony_init_outputs[output_index] = self._phony_init_data

        # get the consuming node
        for c in self.consumers:
            # find the index of the data from the consumer node (input index)
            input_index = self._find_input_index(c)

            # assign the tensor to the consumer's corresponding init input
            c.kernel.phony_init_inputs[input_index] = self._phony_init_data
            if self._phony_init_labels is not None:
                c.kernel.phony_init_input_labels = self._phony_init_labels


    def initialize_phony_data(self):
        """
        Assign random data to phony inputs
        """
        cov = self.is_covariance_input()
        if self._phony_data is not None:
            self._phony_data.assign_random_data(covariance=cov)

        if self._phony_init_data is not None:
            self._phony_init_data.assign_random_data(covariance=cov)

        if self._phony_init_labels is not None:
            self._phony_init_labels.assign_random_data(whole_numbers=True)

    def delete_phony_data(self):
        """
        Remove references to phony data so it can be freed
        during garbage collection
        """
        self._phony_data = None
        self._phony_init_data = None
        self._phony_init_labels = None

        # remove the references within the nodes
        for p in self.producers:
            # find the index of the data from the producer node (output index)
            output_index = self._find_output_index(p)

            # assign the tensor to the producer's corresponding init output
            if output_index in p.kernel.phony_outputs:
                p.kernel.phony_outputs[output_index] = None

        for c in self.consumers:
            # find the index of the data from the consumer node (input index)
            input_index = self._find_input_index(c)

            # assign the tensor to the consumer's corresponding init input
            if input_index in c.kernel.phony_inputs:
                c.kernel.phony_inputs[input_index] = None

            if input_index in c.kernel.phony_init_inputs:
                c.kernel.phony_init_inputs[input_index] = None

            c.kernel.phony_init_input_labels = None

    def is_covariance_input(self):
        """
        Check whether the data object contained within the edge is a covariance
        matrix

        Return
        ------
        bool : True if the data object is a covariance matrix, False otherwise
        """
        if len(self.consumers) == 0:
            return False

        # get one of the consumers of this edge
        consumer = self.consumers[0]

        # check whether this edge is a covariance input to the consumer
        return consumer.kernel.is_covariance_input(self.data)

    def _find_output_index(self, producer):
        """
        Find and return the numerical index of the producer's output that
        corresponds to this edge
        """
        # find the index of the data from the producer node (output index)
        for index, producer_output in enumerate(producer.kernel.outputs):
            if (producer_output is not None and
                producer_output.session_id == self.data.session_id):
                output_index = index
                break

        return output_index

    def _find_input_index(self, consumer):
        """
        Find and return the numerical index of the consumer's input that
        corresponds to this edge
        """
        # find the index of the data from the consumer node (input index)
        for index, consumer_input in enumerate(consumer.kernel.inputs):
            if (consumer_input is not None and
                consumer_input.session_id == self.data.session_id):
                input_index = index
                break

        return input_index




class Parameter:
    """
    Parameter class can be used to abstract data types as inputs and outputs
    to nodes.

    Parameters
    ----------
    data : any
        Reference to the data object represented by the parameter object
    direction : [MPEnums.INPUT, MPEnums.OUTPUT]
        Enum indicating whether this is an input-type or output-type parameter

    """

    def __init__(self, data, direction):
        """
        Constructor for Parameter object
        """
        self._data = data # reference of the data object represented by parameter
        self._direction = direction # enum indicating whether this is an input or output

    @property
    def direction(self):
        """
        Getter for direction property

        Return
        ------
        Enum indicating whether this is an input-type or output-type parameter

        Return Type
        -----------
        MPEnums.INPUT or MPEnums.OUTPUT
        """
        return self._direction

    @property
    def data(self):
        """
        Getter for data property

        Return
        ------
        Data object stored within the Parameter object

        Return Type
        ------------
        MindPype Data object

        """

        return self._data

