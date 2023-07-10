"""
Created on Mon Dec  2 12:00:43 2019

graph.py - Defines the graph object

@author: ivanovn
"""
from .core import BCIP, BcipEnums
from .containers import Tensor
import logging
import warnings

class Graph(BCIP):
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

        super().__init__(BcipEnums.GRAPH,sess)
        
        # private attributes
        self._nodes = []
        self.initialization_edges = []
        self._verified = False
        self._sess = sess
        self._volatile_sources = []
        self._volatile_outputs = []
        self._edges = {}

        self._initialization_links_created = False
    
    def add_node(self,node):
        """
        Append a node object to the list of nodes

        Parameters
        ----------
        node : Node object
            Adds the specified Node object to the referenced graph

        Examples
        --------
        example_graph.add_node(example_node)
        
        Return
        ------
        sts : BcipEnums Status Code
            Returns a status code indicating the success or failure of the operation
        """
        self._verified = False
        self._nodes.append(node)
        
    def verify(self):
        """
        Verify the processing graph is valid. This method orders the nodes
        for execution if the graph is valid

        Return
        ------
        BCIP Status Code

        Examples
        --------
        >>> status = example_graph.verify()
        >>> print(status)
            
            SUCCESS

        """
        if self._verified:
            return BcipEnums.SUCCESS
        
        # begin by scheduling the nodes in execution order
        
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
                if not (n_o.session_id in self._edges):
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
                        print("scheduling failed")
                        return BcipEnums.INVALID_GRAPH
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
            for node_index in range(scheduled_nodes,len(self._nodes)):
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
                    print(f"\t\t{n.kernel._name}Node Scheduled")

            

            if nodes_added == 0:
                # invalid graph, cannot be scheduled
                return BcipEnums.INVALID_GRAPH
        
        
        # now all the nodes are in execution order, validate each node
        missing_data = False # flag to indicate if any nodes are missing init data connections
        for n in self._nodes:
            valid = n.verify()
            if valid != BcipEnums.SUCCESS:
                print("Node {} failed verification".format(n.kernel.name))
                return valid           

            # check for missing init data
            if n._kernel.init_style == BcipEnums.INIT_FROM_DATA:
                if (n._kernel._init_inA == None or # TODO replace with a kernel method
                    (hasattr(n._kernel, "_init_inB") and n._kernel._init_inB == None)):
                        missing_data = True
                    

        # fill in all init data links
        if missing_data:
            for n in self._nodes:
                n_inputs = n.extract_inputs()
                n_outputs = n.extract_outputs()
                producers = []
                consumers = []
                for n_i in n_inputs:
                    if len(self._edges[n_i.session_id]._producers) > 0:
                        producers.append(self._edges[n_i.session_id]._producers[0])
                for n_o in n_outputs:
                    if len(self._edges[n_o.session_id]._consumers) > 0:
                        consumers.append(self._edges[n_o.session_id]._consumers[0])
                    
                if len(producers) == 0 and n._kernel._init_inA == None:
                    n._kernel._init_inA = Tensor.create_virtual(sess=self._sess)
                    n._kernel._init_labels_in = Tensor.create_virtual(sess=self._sess)
  
                if len(consumers) == 0 and n._kernel._init_outA == None:
                    n._kernel._init_outA = Tensor.create_virtual(sess=self._sess)
                    n._kernel._init_labels_out = Tensor.create_virtual(sess=self._sess)

                for p in producers:
                    if p._kernel._init_outA != None and n._kernel._init_inA == None:
                        n._kernel._init_inA = p._kernel._init_outA
                        n._kernel._init_labels_in = p._kernel._init_labels_out
                        
                    elif p._kernel._init_outA != None and (hasattr(n._kernel, "_init_inB") and n._kernel._init_inB == None):
                        n._kernel._init_inB = p._kernel._init_outA
                        n._kernel._init_labels_in = p._kernel._init_labels_out

                    elif p._kernel._init_outA == None and n._kernel._init_inA == None:
                        p._kernel._init_outA = Tensor.create_virtual(sess=self._sess)
                        p._kernel._init_labels_out = Tensor.create_virtual(sess=self._sess)
                        n._kernel._init_inA = p._kernel._init_outA
                        n._kernel._init_labels_in = p._kernel._init_labels_out
                        
                    elif p._kernel._init_outA == None and (hasattr(n._kernel, "_init_inB") and n._kernel._init_inB == None):
                        p._kernel._init_outA = Tensor.create_virtual(sess=self._sess)
                        p._kernel._init_labels_out = Tensor.create_virtual(sess=self._sess)
                        n._kernel._init_inB = p._kernel._init_outA
                        n._kernel._init_labels_in = p._kernel._init_labels_out

        
        for node in self._nodes:
            if (node._kernel._init_inA == None or node._kernel._init_labels_in == None or node._kernel._init_outA == None or \
                node._kernel._init_labels_out == None or (hasattr(node._kernel, "_init_inB") and node._kernel._init_inB == None)) and \
                    node._kernel._init_style == BcipEnums.INIT_FROM_DATA:
                self._initialization_links_created = False
                break
            else:
                self._initialization_links_created = True                                

        # Done, all nodes scheduled and verified!
        self._verified = True
        return BcipEnums.SUCCESS


    def initialize(self, default_init_dataA = None, default_init_labels = None,  default_init_dataB = None):
        """
        Initialize each node within the graph for trial execution

        Parameters
        ----------
        default_init_dataA : Tensor, default = None
            If the graph has no initialization data, this tensor will be used to initialize the graph
        default_init_labels : Tensor, default = None
            If the graph has no initialization labels, this tensor will be used to initialize the graph
        default_init_dataB : Tensor, default = None
            If the graph has no initialization data and a top-level node has two inputs, this tensor will be used to initialize the graph
        
        Return
        ------
        BCIP Status Code

        Examples
        --------
        >>> status = example_graph.initialize()
        >>> print(status)

            SUCCESS
        """
        # 1. Check whether nodes in the graph are missing initialization links (similar to verification)

        if not self._initialization_links_created and default_init_dataA == None:
            warnings.warn("No default initialization data provided, graph is not initialized correctly")
            return BcipEnums.INVALID_GRAPH

        if default_init_dataA != None:
            for n in self._nodes:
                n_inputs = n.extract_inputs()
                producers = []

                for n_i in n_inputs:
                    if len(self._edges[n_i.session_id]._producers) > 0:
                        producers.append(self._edges[n_i.session_id]._producers[0])
                
                if len(producers) == 0 and n._kernel._init_inA.shape == ():
                    n._kernel._init_inA.shape = default_init_dataA.shape
                    n._kernel._init_inA = default_init_dataA.data

                    n._kernel._init_labels_in.shape = default_init_labels.shape
                    n._kernel._init_labels_in.data = default_init_labels.data

                    if hasattr(n._kernel, "_init_inB"):
                        n._kernel._init_inB.shape = default_init_dataB.shape
                        n._kernel._init_inB.data = default_init_dataB.data
                break


        for n in self._nodes:
            sts = n.initialize()
                
            if sts != BcipEnums.SUCCESS:
                return sts
        
        return BcipEnums.SUCCESS

 
    def execute(self, label = None, poll_volatile_sources = True, push_volatile_outputs = True):
        """
        Execute the graph by iterating over all the nodes within the graph and executing each one

        Parameters
        
        ----------

        Label : int, default = None
            * If the trial label is known, it can be passed when a trial is executed. This is required for epoched input data
        
        poll_volatile_sources : bool, default = True
            * If true, volatile sources (ie. LSL input data), will be updated. If false, the input data will not be updated

        Return
        ------
        BCIP Status Code

        Examples
        -------
        >>> status = example_graph.execute(0, True)
        >>> print(status)

            SUCCESS
        """
        # first ensure the block's processing graph has been verified,
        # if not, verify and schedule the nodes
        if not self._verified:
            executable = self.verify()
            if executable != BcipEnums.SUCCESS:
                return executable

        if poll_volatile_sources:
            self.poll_volatile_sources(label)
            
        print("Executing trial with label: {}".format(label))
        
        # iterate over all the nodes and execute the kernel
        for n in self._nodes:
            sts = n.kernel.execute()
            if sts != BcipEnums.SUCCESS:
                logging.warning(f"Trial execution failed with status {sts} in kernel: {n.kernel.name}. This trial will be disregarded.", stacklevel=2)
                return sts

        if push_volatile_outputs:
            self.push_volatile_outputs(label)

        return BcipEnums.SUCCESS
    
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
    def create(cls,sess):
        """
        Generic factory method for a graph
        """
        graph = cls(sess)
        sess.add_graph(graph)
        
        return graph


class Node(BCIP):
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
    
    def __init__(self,graph,kernel,params):
        sess = graph.session
        super().__init__(BcipEnums.NODE,sess)
        
        self._kernel = kernel
        self._params = params
        
    
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
            if p.direction == BcipEnums.INPUT:
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
            if p.direction == BcipEnums.OUTPUT:
                outputs.append(p.data)
        
        return outputs
    
    def verify(self):
        """
        Verify the node is executable

        Parameters
        ----------
        None

        Return
        ------
        BCIP Status Code

        Examples
        --------
        >>> status = example_node.verify()
        >>> print(status)

            INVALID_PARAMETERS

        """
        return self.kernel.verify()
    
    def initialize(self):
        """
        Initialize the kernel function for execution
        
        Parameters
        ----------
        None

        Return
        ------
        BCIP Status Code

        Examples
        --------
        >>> status = example_node.initialize()
        >>> print(status)

            SUCCESS
        """
        return self.kernel.initialize()


class Edge:
    """
    Edge class used by BCIP block to schedule graphs. Each edge object
    represents a different BCIP data object and stores the nodes that produce
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
    
    def __init__(self,data):
        """
        Constructor for Edge object
        """
        self._data = data
        self._producers = []
        self._consumers = []
        
    
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
        BCIPy Data object
        """

        return self._data

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
        self.data = data


class Parameter:
    """
    Parameter class can be used to abstract data types as inputs and outputs 
    to nodes.

    Parameters
    ----------
    data : any
        Reference to the data object represented by the parameter object
    direction : [BcipEnums.INPUT, BcipEnums.OUTPUT]
        Enum indicating whether this is an input-type or output-type parameter

    Attributes
    ----------
    data : any
        Reference to the data object represented by the parameter object
    direction : [BcipEnums.INPUT, BcipEnums.OUTPUT]
        Enum indicating whether this is an input-type or output-type parameter
    """
    
    def __init__(self,data,direction):
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
        BcipEnums.INPUT or BcipEnums.OUTPUT
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
        BCIPy Data object

        """

        return self._data

