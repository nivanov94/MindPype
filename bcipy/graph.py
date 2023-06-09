"""
Created on Mon Dec  2 12:00:43 2019

graph.py - Defines the graph object

@author: ivanovn
"""
from .core import BCIP, BcipEnums
from .containers import Tensor
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
    _nodes : Array of Node objects
        List of Node objects within the graph
    
    initialization_edges : Array of Edge objects
        List of Edge objects used within the graph

    _verified : bool
        True is graph has been verified, false otherwise

    _sess : Session object
        Session where the Graph object exists

    _volatile_sources : Array of data Source objects
        Data sources within this array will be polled/executed when the graph is executed.

    _volatile_outputs : Array of data Output objects
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
        edges = {} # keys: session_id of data obj, vals: edge object
        for n in self._nodes:
            # get a list of all the input objects to the node
            n_inputs = n.extract_inputs()
            n_outputs = n.extract_outputs()
            
            # add these inputs/outputs to edge objects
            for n_i in n_inputs:
                if not n_i.session_id in edges:
                    # no edge created for this input yet, so create a new one
                    edges[n_i.session_id] = Edge(n_i)
                    if n_i.volatile:
                        self._volatile_sources.append(n_i)
                    if n_i.volatile_out:
                        self._volatile_outputs.append(n_i)
                # now add the node the edge's list of consumers
                edges[n_i.session_id].add_consumer(n)
                
            for n_o in n_outputs:
                if not (n_o.session_id in edges):
                    # no edge created for this output yet, so create a new one
                    edges[n_o.session_id] = Edge(n_o)
                    
                    # add the node as a producer
                    edges[n_o.session_id].add_producer(n)
                else:
                    # edge already created, must check that it has no other 
                    # producer
                    if len(edges[n_o.session_id].producers) != 0:
                        # this is an invalid graph, each data object can only
                        # have a single producer
                        print("scheduling failed")
                        return BcipEnums.INVALID_GRAPH
                    else:
                        # add the producer to the edge
                        edges[n_o.session_id].add_producer(n)
        
        # now determine which edges are ready to be consumed
        consumable_edges = {}
        for e_key in edges:
            if len(edges[e_key].producers) == 0:
                # these edges have no producing nodes, so they are inputs to 
                # the block and therefore can be consumed immediately
                consumable_edges[e_key] = edges[e_key]
        
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
                        consumable_edges[n_o.session_id] = edges[n_o.session_id]
                    
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

        # fill in missing init data links
        if missing_data:         
            for n in self._nodes:
                if n._kernel.init_style == BcipEnums.INIT_FROM_DATA:
                    if (n._kernel._init_inA == None or # TODO replace with a kernel method
                        (hasattr(n._kernel, "_init_inB") and n._kernel._init_inB == None)):

                        sts = self.fill_initialization_links(n, edges)
                        if sts != BcipEnums.SUCCESS:
                            return BcipEnums.INVALID_GRAPH
                                                    

        # Done, all nodes scheduled and verified!
        self._verified = True
        return BcipEnums.SUCCESS
    
    def fill_initialization_links(self, n, edges):
        """
        Connect initialization input of nodes missing init_data to the output of producer nodes. 
        Used recursively to ensure all nodes within a particular graph have the required initialization data.

        Parameters
        ----------
        n : Node Object
            Node object which will have its initialization data filled

        edges : list of Edge objects within a graph
            Edge object connections used to identify upstream/downstream nodes.

        Return
        ------
        BCIP status code
        """
        
        sts = BcipEnums.SUCCESS
        
        n_inputs = n.extract_inputs()
        if len(n_inputs) == 0 and n._kernel._init_inA == None:
            return BcipEnums.INVALID_GRAPH
        for n_i in n_inputs:
            # identify the up-stream node producing each input
            try:
                producer = edges[n_i.session_id].producers[0]
            except:
                continue # TODO resolve this??
            # TODO: need to more robustly identify the correct initialization output
            # may require modification of the node/parameter data structures.

            if producer._kernel._init_outA != None: # TODO init data may come from other node outputs
                if hasattr(n._kernel, "_init_inB") and n._kernel._init_inA != None:
                    if n._kernel._init_inB != None and n._kernel._init_inA != None:
                        return BcipEnums.INVALID_GRAPH
                    
                    n._kernel._init_inB = producer._kernel._init_outA

                    if n._kernel._init_labels_in == None:
                        n._kernel._init_labels_in = producer._kernel._init_labels_out

                if (not hasattr(n._kernel, "_init_inB")) and n._kernel._init_inA != None:
                    return BcipEnums.INVALID_GRAPH

                else:
                    n._kernel._init_inA = producer._kernel._init_outA
                    
                    if n._kernel._init_labels_in == None:
                        n._kernel._labels = producer._kernel._init_labels_out

            else:
                producer._kernel._init_outA = Tensor.create_virtual(sess=self._sess) # TODO determine if this would ever not be a tensor
                producer._kernel._init_labels_out = Tensor.create_virtual(sess=self._sess)
                
                if hasattr(n._kernel, "_init_inB") and n._kernel._init_inA != None:
                    if n._kernel._init_inB != None:
                        return BcipEnums.INVALID_GRAPH
                    n._kernel._init_inB = producer._kernel._init_outA

                    if n._kernel._init_labels_in == None:
                        n._kernel._init_labels_in = producer._kernel._init_labels_out

                else:
                    n._kernel._init_inA = producer._kernel._init_outA
                    
                    if n._kernel._init_labels_in == None:
                        n._kernel._init_labels_in = producer._kernel._init_labels_out

            # recurse process on up-stream nodes
            if producer._kernel._init_inA == None:
                if (producer._kernel._init_inA == None or # TODO kernel method
                    (hasattr(n._kernel, "_init_inB") and n._kernel._init_inB == None)):
                    sts = self.fill_initialization_links(producer, edges)
        
        return sts


    def initialize(self):
        """
        Initialize each node within the graph for trial execution

        Parameters
        ----------
        None

        Return
        ------
        BCIP Status Code

        Examples
        --------
        >>> status = example_graph.initialize()
        >>> print(status)

            SUCCESS
        """
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
                warnings.warn(f"Trial execution failed with status {sts} in kernel: {n.kernel.name}. This trial will be disregarded.", category=RuntimeWarning, stacklevel=2)
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
        Array
            List of inputs for the Node

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
        Array
            List of outputs for the Node

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
        _producers : array of Node
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

