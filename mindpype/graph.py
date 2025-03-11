from .core import MPBase, MPEnums
from .containers import Tensor
import sys
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, 
                             precision_score, recall_score,
                             log_loss)


class Graph(MPBase):
    """
    Represents a data processing directed acyclic graph in MindPype.

    A `Graph` object models a directed acyclic graph (DAG) used to define 
    data processing pipelines. Nodes in the graph represent individual 
    data transformations or operations, and edges represent the flow of 
    data between nodes. The `Graph` class provides methods for verifying 
    graph validity, scheduling nodes in execution order, initializing nodes, 
    and executing the pipeline.

    Parameters
    ----------
    sess : Session
        The session object in which the graph will exist and operate.

    
    Graph objects are used to represent the data processing flow graph, or
    processing pipelines. Graphs consist of nodes representing data 
    transformations and edges representing data ingested and produced by the 
    nodes. The graphs must be directed and acyclic. The graph object consists
    of methods to verify the validity of the graph and its nodes, schedule the
    nodes in execution order, initialize the nodes, and execute the nodes.

    Parameters
    ----------
    sess : Session Object
        Session where the graph will exist

    Attributes
    ----------
    nodes : List of Node
        List of `Node` objects within the graph. After verification,
        this list is ordered according to the node execution sequence.
    verified : bool
        Indicates whether the graph structure has been verified as valid.
        `True` if verified, `False` otherwise.
    initialized : bool
        Indicates whether the nodes in the graph have been initialized.
        `True` if initialized, `False` otherwise.

    Methods
    -------
    add_node(node)
        Append a node to the list of nodes within the graph
    verify()
        Verify the graph structure and schedule the nodes for execution
    initialize()
        Initialize the nodes within the graph for execution
    update()
        Update the nodes within the graph for trial execution
    execute(label=None)
        Execute the nodes within the graph
    cross_validate(target_validation_output, folds=5, shuffle=False, random_state=None, statistic='accuracy')
        Perform k-fold cross-validation on the graph or a portion of the graph
    create(sess)
        Factory method to create and register a graph within a session.

    Notes
    -----
    - The graph must always be a directed acyclic graph (DAG). Cyclic 
      graphs are not supported and will result in a verification failure.
    - Node scheduling ensures that each node's dependencies are executed 
      before the node itself, respecting data dependencies.
    - Graph execution is only possible after the graph has been verified 
      and initialized.
    """

    def __init__(self, sess):
        """Init."""
        super().__init__(MPEnums.GRAPH, sess)
        self.nodes = []
        self.verified = False
        self.initialized = False

        # private attributes
        self._volatile_sources = []  # data objects that need to be polled before execution
        self._volatile_outputs = []  # data objects that need to be pushed after execution
        self._edges = {}  # keys: session_id of data obj, vals: edge object - used for scheduling

    def add_node(self, node):
        """
        Add a node to the graph.

        This method appends a `Node` object to the list of nodes in the graph. 
        Adding a node marks the graph as unverified and uninitialized, 
        requiring re-verification and re-initialization before execution.

        Parameters
        ----------
        node : Node
            The `Node` object to be added to the graph.
        """
        # the graph has changed, so it needs to be re-verified 
        # and re-initialized
        self.verified = False
        self.initialized = False
        self.nodes.append(node)

    def verify(self):
        """
        Verify the validity of the graph and schedule nodes for execution.

        This method checks that the graph structure is valid and schedules the 
        nodes in an appropriate execution order based on their dependencies. 
        During verification, any necessary initialization edges between nodes 
        are also created. If the verification succeeds, the graph is marked as 
        verified, and the nodes are prepared for initialization or execution 
        (if initialization is not required).

        Raises
        ------
        ValueError
            If the graph is invalid or cannot be scheduled due to structural 
            issues (e.g., cyclic dependencies or missing connections).
        TypeError
            If any of the nodes in the graph contain invalid data inputs, such 
            as incompatible types or missing required inputs.
        """
        if self.verified:
            # if the graph has already been verified,
            # there is no need to verify again
            return

        # begin by scheduling the nodes in execution order
        self._schedule_nodes()

        # now all the nodes are in execution order create any
        # necessary initialization edges. The initialization
        # edges will mirror the processing graph.
        self._insert_init_edges()

        # insert verif edges for verification. These verif edges
        # will be used to validate the nodes within the graph and confirm
        # that the graph and its nodes can be executed without error
        self._insert_verif_edges()

        # set verif inputs with random data for validation
        self._init_verif_edges()

        # finally, validate each node
        self._validate_nodes()

        # delete verif inputs and outputs
        # as they are no longer needed post-node verification
        self._delete_verif_edges()

        # Done, all nodes scheduled and verified!
        self.verified = True

        # cleanup any data used within verification that are no longer needed
        self.session.free_unreferenced_data()

    def _schedule_nodes(self):
        """
        Places the nodes of the graph in execution order. 

        This method will determine a valid execution order for the nodes within
        the graph. The execution order is determined by the data dependencies
        of the nodes and their parameters. The method will also create a set
        of Edge objects that will be used to represent the data parameters
        within the graph. These edges will also be used to create any necessary
        initialization data connections between nodes.

        Raises
        ------
        ValueError
            If the graph is invalid and cannot be scheduled.
        """
        # first we'll create a set of edges representing data within the graph
        self._edges = {}  # keys: session_id of data obj, vals: edge object
        for n in self.nodes:
            # get a list of all the input objects to the node
            n_inputs = n.extract_inputs()
            n_outputs = n.extract_outputs()

            # add these inputs/outputs to edge objects
            for n_i in n_inputs:
                if n_i.session_id not in self._edges:
                    # no edge created for this input yet, so create a new one
                    self._edges[n_i.session_id] = Edge(n_i)

                    # if the data object is volatile, add it to the list of
                    # volatile sources
                    if n_i.volatile:
                        self._volatile_sources.append(n_i)

                # now add the node the edge's list of consumers
                self._edges[n_i.session_id].add_consumer(n)

            for n_o in n_outputs:
                if n_o.session_id not in self._edges:
                    # no edge created for this output yet, so create a new one
                    self._edges[n_o.session_id] = Edge(n_o)

                    # add the node as a producer
                    self._edges[n_o.session_id].add_producer(n)

                    # if the data object is volatile, add it to the list of
                    # volatile outputs
                    if n_o.volatile_out:
                        self._volatile_outputs.append(n_o)

                else:
                    # edge already created, must check that it has no other
                    # producer
                    if len(self._edges[n_o.session_id].producers) != 0:
                        # this is an invalid graph, each data object can only
                        # have a single producer
                        raise ValueError(
                            "Invalid graph, multiple " 
                            "nodes write to single data object"
                        )
                    else:
                        # add the producer to the edge
                        self._edges[n_o.session_id].add_producer(n)
                        if (n_o.volatile_out and
                                n_o not in self._volatile_outputs):
                            self._volatile_outputs.append(n_o)

        # now determine which edges are ready to be consumed
        consumable_edges = {}
        for e_key in self._edges:
            if len(self._edges[e_key].producers) == 0:
                # these edges have no producing nodes, so they are inputs to
                # the graph and therefore can be consumed immediately
                consumable_edges[e_key] = self._edges[e_key]

        scheduled_nodes = 0
        total_nodes = len(self.nodes)

        while scheduled_nodes != total_nodes:
            nodes_added = 0
            # find the next node that has all its inputs ready to be consumed
            for node_index in range(scheduled_nodes, len(self.nodes)):
                n = self.nodes[node_index]
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
                        self.nodes[scheduled_nodes] = self.nodes[node_index]
                        self.nodes[node_index] = tmp

                    # mark this node's outputs ready for consumption
                    for n_o in n_outputs:
                        consumable_edges[n_o.session_id] = self._edges[n_o.session_id]

                    nodes_added = nodes_added + 1
                    scheduled_nodes = scheduled_nodes + 1

            if nodes_added == 0:
                # No nodes were added to the schedule, this means that the
                # graph is invalid and cannot be scheduled
                raise ValueError(
                    "Invalid graph, nodes cannot be scheduled, "
                    "check connections between nodes."
                )

    def _insert_init_edges(self):
        """
        Insert initialization data edges into the graph to mirror the
        processing graph. 
        
        The inserted edges will be used to create the
        necessary initialization data connections between nodes. This
        method checks for any nodes that require initializtion data but
        have not been provided with it during node creation. For these nodes,
        the method will attempt to create the necessary initialization data
        edges to propagate initialization data provided to upstream nodes
        to the nodes that require it.
        """
        init_required = False  # flag if any nodes in the graph require init
        init_links_missing = False  # flag if any init data will need to propagate through graph
        for n in self.nodes:
            # check for missing init data
            if n.kernel.init_style == MPEnums.INIT_FROM_DATA:
                init_required = True

                # check whether all init inputs have been provided by the user
                init_provided = True
                for init_index in range(n.kernel.num_inputs):
                    n_ii = n.kernel.get_parameter(init_index, 'init', MPEnums.INPUT)
                    if n_ii is None:
                        init_provided = False
                        break  # 1 missing init triggers process of inserting init links

                # if not provided, flag that graph will need initialization
                # data propagated through the graph
                if not init_provided:
                    init_links_missing = True

        # fill in all init data links
        if init_required and init_links_missing:
            # use the existing Edge objects to create init connections
            # mirroring the processing graph
            for e in self._edges:
                self._edges[e].insert_init_data()

    def _validate_nodes(self):
        """
        Validate each node within the graph to ensure that the nodes
        can be executed without error. 
        
        This method will check that the
        nodes have all the necessary data inputs and parameters required
        for execution. If any node fails validation, the method will raise
        an exception with a message indicating the node that failed validation.

        Raises
        ------
        ValueError
            If any node within the graph fails validation.
        TypeError
            If any of the nodes within the graph contain invalid data inputs.
        """
        for n in self.nodes:
            try:
                n.verify()
            except Exception as e:
                additional_msg = f"Node: {n.kernel.name} failed verification. See traceback for details."
                if sys.version_info[:2] >= (3,11):
                    e.add_note(additional_msg)
                else:
                    # for older versions of Python, print a hint and raise the exception
                    # TODO may be useful to encapsulate these errors into a MindPype specific exception
                    pretty_msg = f"\n{'*'*len(additional_msg)}\n{additional_msg}\n{'*'*len(additional_msg)}\n"
                    print(pretty_msg)
                raise

    def _insert_verif_edges(self):
        """
        Add verif edges to the graph to be used during verification. 
        
        The inserted edges contain references to containers that will contain
        random data that will be used to attempt node execution during the 
        verification step. These edges are stored as distinct attributes 
        within the `Edge` objects used to define the data flow within the graph.
        """
        for e_id in self._edges:
            e = self._edges[e_id]

            # check if the data is virtual
            if not e.data.virtual:
                # if not virtual, create a verif edge
                e.add_verif_data()

            # check if the edge has non-virtual init data
            if e.init_data is not None and not e.init_data.virtual:
                e.add_verif_init_data()

    def _init_verif_edges(self):
        """
        Initialize verif edges with random data for validation
        """
        for eid in self._edges:
            self._edges[eid].initialize_verif_data()

    def _delete_verif_edges(self):
        """
        Remove references to any verif edges so the
        data will be freed during garbage collection
        """
        for eid in self._edges:
            self._edges[eid].delete_verif_data()

    def initialize(self):
        """
        Initialize the nodes within the graph for execution.

        This method initializes each node in the graph in the order determined during 
        the verification step. If a node fails to initialize, an exception is raised, 
        indicating the specific node that caused the failure. For nodes that require 
        initialization data but were not explicitly provided such inputs during creation, 
        the method supplies transformed initialization data from upstream nodes. If the 
        graph has not been verified, this method will automatically verify it before 
        proceeding with initialization.

        Raises
        ------
        ValueError
            If any node within the graph fails to initialize successfully.
        TypeError
            If any node contains invalid or incompatible initialization data inputs.

        Notes
        -----
        - Initialization ensures that all nodes are prepared to execute with the 
          necessary data dependencies resolved.
        - The graph must be verified before initialization; otherwise, verification 
          is performed as part of this method.
        """
        # if not verified, verify the graph first
        if not self.verified:
            self.verify()

        # execute initialization for each node in the graph
        for n in self.nodes:
            try:
                n.initialize()
            except Exception as e:
                additional_msg = f"Node: {n.kernel.name} failed initialization. See traceback for details."
                if sys.version_info[:2] >= (3,11):
                    e.add_note(additional_msg)
                else:
                    # for older versions of Python, print a hint and raise the exception
                    # TODO may be useful to encapsulate these errors into a MindPype specific exception
                    pretty_msg = f"\n{'*'*len(additional_msg)}\n{additional_msg}\n{'*'*len(additional_msg)}\n"
                    print(pretty_msg)
                raise

        self.initialized = True
        self.session.free_unreferenced_data()

    def update(self):
        """
        Update each node within the graph for trial execution. 
        
        This method
        is similar to initialization. It will update the nodes in the graph
        according to any update or partial re-initialization methods defined
        within the node's kernel class. The data within the graph nodes'
        intiialization inputs will be used to update the nodes.

        Raises
        ------
        ValueError
            If any node within the graph fails update.
        TypeError
            If any of the nodes within the graph contain invalid initialization
            data inputs.
        """
        if not self._verified:
            self.verify()

        # execute initialization for each node in the graph
        for n in self._nodes:
            try:
                n.update()
            except Exception as e:
                additional_msg = f"Node: {n.kernel.name} failed update. See traceback for details."
                if sys.version_info[:2] >= (3,11):
                    e.add_note(additional_msg)
                else:
                    # for older versions of Python, print a hint and raise the exception
                    # TODO may be useful to encapsulate these errors into a MindPype specific exception
                    pretty_msg = f"\n{'*'*len(additional_msg)}\n{additional_msg}\n{'*'*len(additional_msg)}\n"
                    print(pretty_msg)
                raise

        self.session.free_unreferenced_data()

    def execute(self, label=None):
        """
        Execute all nodes within the graph.

        This method executes the nodes in the graph in the order determined during 
        the verification step. If any node fails during execution, an exception 
        is raised, providing a message and traceback for the failing node. If the 
        graph has not been verified or initialized, it will be verified and/or 
        initialized automatically before execution. For graphs containing volatile 
        data sources or outputs, the method will poll these data sources before 
        execution and push volatile outputs after execution.

        Parameters
        ----------
        label : int, default=None
            The class label for the current trial, if known. This label is used 
            to poll and push epoched data, typically in scenarios involving an 
            external data source, such as a file.

        Raises
        ------
        ValueError
            If any node in the graph fails during execution.

        Notes
        -----
        - The execution order is determined during graph verification.
        - Volatile data sources are dynamic inputs that may change at runtime, 
          and volatile outputs are dynamic results produced by the graph.
        - This method handles dependencies, ensuring that all required steps are 
          executed in the correct order.
        """
        # first ensure the graph has been verified,
        # if not, verify and schedule the nodes
        if not self.verified:
            self.verify()

        if not self.initialized:
            self.initialize()

        # Check whether first node has volatile input
        # if so, poll the volatile data
        if len(self._volatile_sources) > 0:
            self._poll_volatile_sources(label)

        # iterate over all the nodes and execute the kernel
        for n in self.nodes:
            try:
                n.kernel.execute()
            except Exception as e:
                additional_msg = f"Node: {n.kernel.name} failed execution. See traceback for details."
                if sys.version_info[:2] >= (3,11):
                    e.add_note(additional_msg)
                else:
                    # for older versions of Python, print a hint and raise the exception
                    # TODO may be useful to encapsulate these errors into a MindPype specific exception
                    pretty_msg = f"\n{'*'*len(additional_msg)}\n{additional_msg}\n{'*'*len(additional_msg)}\n"
                    print(pretty_msg)
                raise

        # If there any volatile outputs, push the data
        if len(self._volatile_outputs) > 0:
            self.push_volatile_outputs(label)

    def _poll_volatile_sources(self, label=None):
        """
        Poll data from volatile sources within the graph. 
        
        This method will update the
        data within any graph edges that represent data from external sources
        (eg. files, LSL streams). This method will update the data attribute
        within any container objects that are ingested by nodes within the
        graph using the Source objects associated with those containers.

        Parameters
        ----------
        label : int, default = None
            If the class label of the current trial is known, it can be
            passed to poll epoched data. This is typically used when using
            a file as a external data source.

        Example
        -------
        >>> example_graph._poll_volatile_data(0) # Polls next class 0 trial data
        """
        for datum in self._volatile_sources:
            datum.poll_volatile_data(label)

    def _push_volatile_outputs(self, label=None):
        """
        Push data to volatile outputs within the graph. 
        
        This method will publish any
        data within volatile outputs to external sources (eg. files, LSL 
        streams). 

        Parameters
        ----------
        label : int, default = None
            If the class label of the current trial is known, it can be passed
            to poll epoched data.
        """
        for datum in self._volatile_outputs:
            datum.push_volatile_outputs(label=label)


    def cross_validate(self, target_validation_output, folds=5,
                       shuffle=False, random_state=None, statistic='accuracy'):
        """
        Perform k-fold cross-validation on the graph or a portion of the graph.

        This method first verifies and schedules the nodes in the graph 
        (if not already done), then initializes and executes the graph for each
        fold of the cross-validation. It returns the average score of the 
        specified statistic across all folds. The statistic can be any of the
        supported metrics for model evaluation, such as accuracy or F1 score.

        Parameters
        ----------
        target_validation_output : (Tensor, Scalar)
            The target validation output container, typically the result of a 
            classification node. This can be a `Tensor` or a `Scalar`.

        folds : int, optional, default=5
            The number of folds to use for the cross-validation. The data will 
            be split into this many parts, with each fold serving as the 
            validation set once.

        shuffle : bool, optional, default=False
            Whether to shuffle the data before splitting it into folds. Setting
            this to `True` can ensure that the splits are randomized for better
            generalization.

        random_state : int, optional, default=None
            The random state to use for shuffling the data. If `None`, the 
            random state is not set, leading to a different shuffling each 
            time.

        statistic : str, optional, default='accuracy'
            The evaluation metric to compute for each fold. Options include:
            'accuracy', 'f1', 'precision', 'recall', and 'cross_entropy'. 
            The method will return the average value of the specified 
            statistic.

        Returns
        -------
        float
            The average score of the specified statistic across all folds.

        Raises
        ------
        ValueError
            If the target validation output is not produced by a valid node in
            the graph or if the graph structure is invalid for 
            cross-validation.
        """
        # first ensure the graph has been verified,
        # if not, verify and schedule the nodes
        if not self.verified:
            self.verify()

        # find the subset of nodes that need to executed for cross validation
        cv_node_subset = []
        upstream_nodes = []

        # the first node is the node that produces the target validation output
        n = self._edges[target_validation_output.session_id].producers[0]
        if n is None:
            raise ValueError(
                "Invalid target validation output. The target "
                "must be produced by a node in the graph."
            )

        upstream_nodes.append(n) 
        subset_node_ids = set([n.session_id])  # ids of the nodes to run for CV
        init_data_nodes = []

        # now find all upstream nodes that are required for the cross validation
        while len(upstream_nodes):
            n = upstream_nodes.pop()

            # check if this node has initialization data
            init_provided = True
            for init_index in range(n.kernel.num_inputs):
                n_ii = n.kernel.get_parameter(init_index, 'init', MPEnums.INPUT)
                if n_ii.virtual:
                    init_provided = False

            if not init_provided:
                # add nodes that produce the current node's inputs
                # to the uptream nodes set
                for n_i in n.extract_inputs():
                    p = self._edges[n_i.session_id].producers[0]
                    # add this node if it has not been added yet
                    if p.session_id not in subset_node_ids:
                        upstream_nodes.append(p)
                        subset_node_ids.add(p.session_id)
            else:
                init_data_nodes.append(n)

            # add the current node to the cross validation subset
            cv_node_subset.insert(0, n)

        if len(init_data_nodes) != 1:
            # check that all these nodes are ingesting the same init data
            first_node_init_in = init_data_nodes[0].kernel.get_parameter(
                0, 'init', MPEnums.INPUT
            )

            for n in init_data_nodes:
                current_node_init_in = n.kernel.get_parameter(
                    0, 'init', MPEnums.INPUT
                )

                if current_node_init_in.session_id != first_node_init_in.session_id:
                    raise ValueError(
                        "Cross validation could not be performed. " 
                        "This may be because the target validation output " 
                        "is generated by a node that does not require " 
                        "initialization or because there are multiple " 
                        "nodes that require initialization data."
                    )

        # check the execution order of the subset of nodes
        node_execution_position = np.zeros((len(cv_node_subset),))
        for index, n in enumerate(cv_node_subset):
            for position, nn in enumerate(self._nodes):
                if nn.session_id == n.session_id:
                    node_execution_position[index] = position
                    break

        # sort the nodes by execution order
        subset_order = np.argsort(node_execution_position)
        cv_node_subset = [cv_node_subset[i] for i in subset_order]

        # verify that the the node with initialization data is the first node
        if init_data_nodes[0].session_id != cv_node_subset[0].session_id:
            raise ValueError(
                "Cross validation could not be performed. " 
                "Invalid graph structure"
            )

        # copy the initialization data object
        init_data = init_data_nodes[0].kernel.get_parameter(
            0, 'init', MPEnums.INPUT
        )
        init_labels = init_data_nodes[0].kernel.get_parameter(
            0, 'labels', MPEnums.INPUT
        )

        if init_data.mp_type != MPEnums.TENSOR:
            init_data = init_data.convert_to_tensor()

        if init_labels.mp_type != MPEnums.TENSOR:
            init_labels = init_labels.convert_to_tensor()


        # create the cross validation object
        skf = StratifiedKFold(n_splits=folds, shuffle=shuffle, random_state=random_state)
        mean_stat = 0
        for train_index, test_index in skf.split(init_data.data, init_labels.data):
            # create Tensors for the CV training and testing data
            train_data = Tensor.create_from_data(self.session, init_data.data[train_index])
            train_labels = Tensor.create_from_data(self.session, init_labels.data[train_index])
            test_data = Tensor.create_from_data(self.session, init_data.data[test_index])
            test_labels = Tensor.create_from_data(self.session, init_labels.data[test_index])

            # set the initialization data for the nodes
            for n in init_data_nodes:
                init_param_index = 0
                n.kernel.set_parameter(train_data, init_param_index, 'init', MPEnums.INPUT)
                n.kernel.set_parameter(train_labels, init_param_index, 'labels', MPEnums.INPUT)

            # initialize the subset of nodes
            for n in cv_node_subset:
                n.initialize()

            predictions = np.zeros((test_labels.shape[0],))

            # determine if the inputs are batched or individual samples
            if len(init_data_nodes[0].kernel.inputs[0].shape) == len(test_data.shape):
                batched = True
                if target_validation_output.mp_type == MPEnums.TENSOR:
                    Ngph_samples = target_validation_output.shape[0]
                elif target_validation_output.mp_type == MPEnums.SCALAR:
                    Ngph_samples = 1
                else:
                    Ngph_samples = target_validation_output.num_elements
                Ntest_samples = test_data.shape[0]
            else:
                batched = False

            if not batched:
                for i_t in range(test_labels.shape[0]):
                    predictions[i_t] = self._cv_execute_batch(
                        init_data_nodes, 
                        cv_node_subset, 
                        test_data.data[i_t],
                        target_validation_output
                    )
            else:
                if Ngph_samples == Ntest_samples:
                    # number of samples in the test data is the same as 
                    # the graph, so we can execute the graph in batch mode
                    predictions = self._cv_execute_batch(
                        init_data_nodes,
                        cv_node_subset,
                        test_data.data,
                        target_validation_output
                    )
                
                elif Ngph_samples < Ntest_samples:
                    # there are more samples in the test data than the
                    # graph can accomodate, so we need to execute the 
                    # graph multiple times
                    batches = Ntest_samples // Ngph_samples
                    offset_final_batch = False
                    if Ntest_samples % Ngph_samples != 0:
                        batches += 1
                        offset_final_batch = True

                    overlap_offset = 0
                    for i_b in range(batches):
                        if offset_final_batch and i_b == (batches - 1):
                            # in the final batch, reuse some samples from the
                            # previous batch
                            overlap_offset = Ntest_samples - i_b * Ngph_samples

                        start = i_b * Ngph_samples - overlap_offset
                        end = (i_b + 1) * Ngph_samples - overlap_offset

                        predictions[start:end] = self._cv_execute_batch(
                            init_data_nodes,
                            cv_node_subset,
                            test_data.data[start:end],
                            target_validation_output
                        )

                else:
                    # there are more samples in the graph than the test data
                    # so we need over-sample the test data to fill the graph
                    # input requirements
                    Noversamples = Ngph_samples // Ntest_samples
                    oversampled_data = np.zeros((Ngph_samples,) + test_data.shape[1:])
                    oversampled_data[:Noversamples*Ntest_samples] = np.tile(
                        test_data.data, 
                        (Noversamples,) + (1,) * (len(test_data.shape)-1)
                    )

                    if Ngph_samples % Ntest_samples != 0:
                        oversampled_data[Noversamples*Ntest_samples:] = test_data.data[:Ngph_samples - Noversamples*Ntest_samples]

                    oversampled_pred = self._cv_execute_batch(
                        init_data_nodes,
                        cv_node_subset,
                        oversampled_data,
                        target_validation_output
                    )

                    predictions = oversampled_pred[:Ntest_samples]

            # calculate the statistic
            target = test_labels.data
            if statistic == 'accuracy':
                stat = accuracy_score(target, predictions)
            elif statistic == 'f1':
                stat = f1_score(target, predictions)
            elif statistic == 'precision':
                stat = precision_score(target, predictions)
            elif statistic == 'recall':
                stat = recall_score(target, predictions)
            elif statistic == 'cross_entropy':
                stat = log_loss(target, predictions)

            mean_stat += stat

        # compute mean statistic across folds
        mean_stat /= folds

        # reset the initialization data for the nodes
        for n in init_data_nodes:
            init_param_index = 0
            n.kernel.set_parameter(init_data, init_param_index, 'init', MPEnums.INPUT)
            n.kernel.set_parameter(init_labels, init_param_index, 'labels', MPEnums.INPUT)

        # cleanup data objects
        del train_data, train_labels, test_data, test_labels
        self.session.free_unreferenced_data()

        return mean_stat

    def _cv_execute_batch(self, init_data_nodes, cv_node_subset, 
                          test_data, target_validation_output):
        """
        Execute the subset of nodes in the graph in batch mode for 
        cross-validation.

        This method exectutes a subset of nodes in the graph in batch mode
        for cross-validation. It is used to execute a batch of test inputs
        within a single fold. 

        Parameters
        ----------
        init_data_nodes : list of Node
            List of nodes that required initialization for the 
            cross-validation. These are the nodes that will ingest the test
            inputs.
        cv_node_subset : list of Node
            List of nodes to execute for the cross-validation.
        test_data : np.ndarray
            The test data to be ingested by the nodes.
        target_validation_output : (Tensor, Scalar)
            The target validation output container, typically the result of a
            classification node. This can be a `Tensor` or a `Scalar`.

        Returns
        -------
        np.ndarray
            The predictions of the target validation output.
        """
        # set the test data input for the ingestion nodes
        for n in init_data_nodes:
            n.kernel.inputs[0].data = test_data

        # execute the subset of nodes
        for n in cv_node_subset:
            n.kernel.execute()

        # get the output of the target validation node
        predictions = target_validation_output.data

        return predictions

    @classmethod
    def create(cls, sess):
        """
        Factory method to create and register a graph within a session.

        This method creates a new graph instance associated with the provided
        session and automatically adds the graph to the session for management.

        Parameters
        ----------
        sess : Session
            The session object where the graph will be registered and managed.

        Returns
        -------
        Graph
            A reference to the graph that was created and added to the session.
        """
        graph = cls(sess)
        sess.add_to_session(graph)

        return graph


class Node(MPBase):
    """
    Node objects are used to represent the processing steps within a MindPype
    graph. 
    
    Each node contains a kernel object that defines the processing
    steps to be executed on the node's inputs. The node object also contains
    a set of parameters that are used to store the input and output data
    objects for the node.

    Parameters
    ----------
    graph : Graph object
        The graph object that the node belongs to
    kernel : Kernel Object
        The kernel object that defines the processing steps for the node
    params : tuple of Parameters
        The parameters that define the input and output data objects for the
        node

    Attributes
    ----------
    graph : Graph object
        The graph object that the node belongs to
    kernel : Kernel object
        The kernel object that defines the processing steps for the node
    params : tuple of Parameters
        The parameters that define the input and output data objects for the
        node

    Methods
    -------
    extract_inputs()
        Return a list of all the node's inputs
    extract_outputs()
        Return a list of all the node's outputs
    verify()
        Verify the node is executable
    initialize()
        Initialize the kernel function for execution
    update()
        Update the kernel function for execution
    update_attributes(attribute, value)
        Update the parameters of the node
    add_initialization_data(init_data, init_labels=None)
        Add initialization data to the node
    """

    def __init__(self, graph, kernel, params):
        super().__init__(MPEnums.NODE, graph.session)

        self.graph = graph
        self.kernel = kernel
        self.params = params


    def extract_inputs(self):
        """
        Return a list of all the node's input parameters. 

        This mehod identifies all of the node's input parameters and returns
        them as a list.

        Return
        ------
        List of Nodes
            List of the node's input parameters.
        """
        inputs = []
        for p in self.params:
            if p.direction != MPEnums.OUTPUT:
                inputs.append(p.data)

        return inputs

    def extract_outputs(self):
        """
        Return a list of all the node's output parameters.

        This mehod identifies all of the node's output parameters and returns
        them as a list.

        Return
        ------
        List of Nodes
            List of the node's output parameters.
        """
        outputs = []
        for p in self.params:
            if p.direction == MPEnums.OUTPUT:
                outputs.append(p.data)

        return outputs

    def verify(self):
        """
        Verify the node is executable.

        This method verifies that the node is executable by checking that all
        of the node's parameters are valid.
        """
        self.kernel.verify()

    def initialize(self):
        """
        Initialize the kernel function for execution.

        This method initializes the kernel function for execution by calling
        the kernel's initialization method. The kernel will be initialized
        with either the initialization data explicitly provided to the node
        or with the transformed initialization data provided to the node by
        upstream nodes within its graph.
        """
        self.kernel.initialize()
    
    def update(self):
        """
        Update the kernel function for execution.

        This method updates the kernel function for execution by calling
        the kernel's update method. The kernel will be updated with either
        the initialization data explicitly provided to the node or with the
        transformed initialization data provided to the node by upstream nodes
        within its graph.
        """
        self.kernel.update()

    def update_attributes(self, attribute, value):
        """
        Update the parameters of the node.

        This method updates the parameters of the node by calling the kernel's
        update_attribute method. The method will update the specified attribute
        with the provided value.

        Parameters
        ----------
        attribute : str
            The attribute to be updated
        value : object
            The value to be assigned to the attribute
        
        Notes
        -----
        The graph will be marked as unverified after the attributes are 
        updated. The graph will need to be re-verified before execution.
        """

        self.kernel.update_attribute(attribute, value)
        self.graph._verified = False

    def add_initialization_data(self, init_data, init_labels=None):
        """
        Add initialization data to the node. 

        This method adds initialization data to the node by calling the 
        kernel's add_initialization_data method. 

        Parameters
        ----------
        init_data : list or tuple of MPBase
            MindPype container (Tensor, Array, etc.) containing the
            initialization data
        init_labels : list or tuple of MPBase, default = None
            MindPype container (Tensor, Array, etc.) containing the
            initialization labels. Can be omitted if the node does not
            require labels for initialization.

        Notes
        -----
        The graph will be marked as unverified after the attributes are 
        updated. The graph will need to be re-verified before execution.
        """
        self.kernel.add_initialization_data(init_data, init_labels)
        self.graph.verified = False

    def _update_initialization_data(self, init_data, init_labels=None):
        """
        Update the initialization data of the node.

        This method updates the initialization data of the node by calling the
        kernel's update_initialization_data method. 

        Parameters
        ----------
        init_data : list or tuple of MPBase
            MindPype container (Tensor, Array, etc.) containing the
            initialization data
        init_labels : list or tuple of MPBase, default = None
            MindPype container (Tensor, Array, etc.) containing the
            initialization labels. Can be omitted if the node does not
            require labels for initialization.

        Notes
        -----
        The graph will be marked as unverified after the attributes are 
        updated. The graph will need to be re-verified before execution.
        """
        self.kernel.remove_initialization_data()
        self.add_initialization_data(init_data, init_labels)
        self.session.free_unreferenced_data()


class Edge:
    """
    Edge objects are used to represent the data flow between nodes within a
    MindPype graph. 
    
    Each edge object contains references to the nodes that
    produce and consume the data within the edge. Edge objects are only
    created by the graph object during the scheduling process and should not 
    need to be used directly by the user.

    Parameters
    ----------
    data : (Tensor, Scalar, Array, or CircularBuffer)
        The data represented by the edge

    Attributes
    ----------
    producers : List of Node
        Nodes that will produce the data represented by the edge
    consumers : List of Node
        Nodes that will consume the data represented by the edge
    """
    def __init__(self, data):
        """Init."""
        self.data = data  # data that the edge represents within the graph
        self.producers = []  # nodes that produce the data, should only be 1
        self.consumers = []  # nodes that consume the data, can be multiple

        ## Initialization data attributes
        # These attributes are used for constructing data flow for 
        # initialization data that is propagated through the graph
        self.init_data = None
        self.init_labels = None

        ## Verification data attributes
        # These attributes are used for constructing data flow for
        # verification data that is propagated through the graph
        self.verif_data = None
        self.verif_init_data = None
        self.verif_init_labels = None

    def add_producer(self, producing_node):
        """
        Add a node as a producer to the edge.

        This method appends the specified node to the list of producers for
        the edge.

        Parameters
        ----------
        producing_node : Node
            Node to be added as a producer to the edge
        """
        self.producers.append(producing_node)

    def add_consumer(self, consuming_node):
        """
        Add a node as a consumer to the edge.

        This method appends the specified node to the list of consumers for
        the edge.

        Parameters
        ----------
        consuming_node : Node
            Node to be added as a consumer to the edge
        """
        self.consumers.append(consuming_node)

    def insert_init_data(self):
        """
        Insert initialization data tensors into the graph that mirror the
        connections represented by the edge.

        This method creates virtual tensors that will contain the 
        initialization data for the edge. The virtual tensors are then
        assigned to the corresponding input and output parameters of the
        nodes that produce and consume the data represented by the edge.
        """
        # create a virtual tensor that will contain the initialization data
        self.init_data = Tensor.create_virtual(self.data.session)
        self.init_labels = Tensor.create_virtual(self.data.session)

        for p in self.producers:
            # find the output index number of the edge's data within the 
            # producer node (output index)
            output_index = self.find_output_index(p)

            # assign the init_data tensor to the producer's corresponding
            # init output
            p.kernel.set_parameter(
                self.init_data, 
                output_index,
                'init',
                MPEnums.OUTPUT
            )
            p.kernel.set_parameter(
                self.init_labels, 
                output_index,
                'labels', 
                MPEnums.OUTPUT
            )

        for c in self.consumers:
            # find the index number of the edge's data within the consumer
            # node (input index)
            input_index = self.find_input_index(c)

            # check whether this input has not already been assigned init data
            if c.kernel.get_parameter(input_index, 'init', MPEnums.INPUT) is None:
                # If so, assign the tensor to the consumer's corresponding
                # init input
                c.kernel.set_parameter(
                    self.init_data, 
                    input_index,
                    'init', 
                    MPEnums.INPUT
                )
                c.kernel.set_parameter(
                    self.init_labels, 
                    input_index,
                    'labels', 
                    MPEnums.INPUT
                )
            else:
                # otherwise, overwrite the edge's init data attribute with
                # the existing init data. We will need to use this to
                # create verif inputs later
                self.init_data = c.kernel.get_parameter(
                    input_index, 
                    'init', 
                    MPEnums.INPUT
                )
                self.init_labels = c.kernel.get_parameter(
                    input_index,
                    'labels',
                    MPEnums.INPUT
                )

    def add_verif_data(self):
        """
        Insert verification data tensors into the graph that mirror the
        connections represented by the edge.

        This method creates virtual tensors that will contain the verification
        data for the edge. The virtual tensors are then assigned to the
        corresponding input and output parameters of the nodes that produce
        and consume the data represented by the edge. The verification data
        will be random data that is used to verify the execution of the graph.
        """
        self.verif_data = self.data.make_copy()

        # get the producing node
        for p in self.producers:
            # find the index of the data from the producer node (output index)
            output_index = self.find_output_index(p)

            # assign the tensor to the producer's corresponding init output
            p.kernel.set_parameter(
                self.verif_data, 
                output_index, 
                'verif', 
                MPEnums.OUTPUT, 
                add_if_missing=True
            )

        for c in self.consumers:
            # find the index of the data from the consumer node (input index)
            input_index = self.find_input_index(c)

            #  assign the tensor to the consumer's corresponding init input
            c.kernel.set_parameter(
                self.verif_data, 
                input_index, 
                'verif', 
                MPEnums.INPUT, 
                add_if_missing=True
            )

    def add_verif_init_data(self):
        """
        Insert verification initialization data tensors into the graph that
        mirror the connections represented by the edge.

        This method creates virtual tensors that will contain the verification
        initialization data for the edge. The virtual tensors are then assigned
        to the corresponding input and output parameters of the nodes that
        produce and consume the data represented by the edge. The verification
        initialization data will be random data that is used to verify the
        initialization of the graph.
        """
        self.verif_init_data = self.init_data.make_copy()
        if self.init_labels is not None:
            self.verif_init_labels = self.init_labels.make_copy()

        # get the producing node
        for p in self.producers:
            # find the index of the data from the producer node (output index)
            output_index = self.find_output_index(p)

            # assign the tensor to the producer's corresponding init output
            p.kernel.set_parameter(
                self.verif_init_data,
                output_index, 
                'verif_init',
                MPEnums.OUTPUT, 
                add_if_missing=True
            )

        # get the consuming node
        for c in self.consumers:
            # find the index of the data from the consumer node (input index)
            input_index = self.find_input_index(c)

            # assign the tensor to the consumer's corresponding init input
            c.kernel.set_parameter(
                self.verif_init_data,
                input_index, 
                'verif_init',
                MPEnums.INPUT,
                add_if_missing=True
            )
            c.kernel.set_parameter(
                self.verif_init_labels,
                input_index, 
                'verif_labels',
                MPEnums.INPUT,
                add_if_missing=True
            )

    def initialize_verif_data(self):
        """
        Assign random data to verif inputs.

        This method assigns random data to the verification data tensors
        that are used to verify the execution of the graph.
        """
        cov = self.is_covariance_input()
        if self.verif_data is not None:
            self.verif_data.assign_random_data(covariance=cov)

        if self.verif_init_data is not None:
            self.verif_init_data.assign_random_data(covariance=cov)

        if self.verif_init_labels is not None:
            self.verif_init_labels.assign_random_data(whole_numbers=True)

    def delete_verif_data(self):
        """
        Remove references to verif data so that it can be garbage collected.
        
        This method removes the references to the verification data tensors
        that were used to verify the execution of the graph. This is done to
        allow the tensors to be garbage collected. This should only be called
        after the graph verification process is complete.
        """
        self.verif_data = None
        self.verif_init_data = None
        self.verif_init_labels = None

        # remove the references within the nodes
        for p in self.producers:
            # find the index of the data from the producer node (output index)
            output_index = self.find_output_index(p)

            # assign the tensor to the producer's corresponding init output
            p.kernel.set_parameter(None, output_index, 'verif', MPEnums.OUTPUT)

        for c in self.consumers:
            # find the index of the data from the consumer node (input index)
            input_index = self.find_input_index(c)

            # assign the tensor to the consumer's corresponding init input
            c.kernel.set_parameter(None, input_index, 'verif', MPEnums.INPUT)
            c.kernel.set_parameter(None, input_index, 'verif_init', MPEnums.INPUT)
            c.kernel.set_parameter(None, input_index, 'verif_labels', MPEnums.INPUT)

    def is_covariance_input(self):
        """
        Check whether the data represented by the edge is a covariance matrix

        Return
        ------
        bool
            True if the data parameter is a covariance matrix, False otherwise
        """
        if len(self.consumers) == 0:
            return False

        # get one of the consumers of this edge
        consumer = self.consumers[0]

        # check whether this edge is a covariance input to the consumer
        return consumer.kernel.is_covariance_input(self.data)

    def find_output_index(self, producer):
        """
        Find and return the numerical index of the producer's output that
        produces the data represented by the edge.

        Parameters
        ----------
        producer: Node
            The node that produces the data represented by the edge.

        Returns
        -------
        int
            Index of the data from the producer node
        """
        # find the index of the data from the producer node (output index)
        return producer.kernel.find_param_index(self.data, MPEnums.OUTPUT)

    def find_input_index(self, consumer):
        """
        Find and return the numerical index of the consumer's input that
        consumes the data represented by the edge.

        Parameters
        ----------
        consumer: Node
            The node that consumes the data represented by the edge.

        Returns
        -------
        input_index: int
            index of the data from the consumer node.
        """
        # find the index of the data from the consumer node (input index)
        return consumer.kernel.find_param_index(self.data, MPEnums.INPUT)


class Parameter:
    """
    Parameter objects are used to represent the input and output data objects
    for nodes within a MindPype graph.

    This class provides a simple container for the data objects that
    are the connections between nodes in MindPype graphs.

    Parameters
    ----------
    data : MPBase
        A MindPype container (Tensor, Array, etc.) that is will contain the
        data produced or consumed by the node
    direction : (MPEnums.INPUT, MPEnums.OUTPUT, or MPEnums.INOUT)
        Enum indicating the direction of the parameter

    Attributes
    ----------
    data : MPBase
        The data object represented by the parameter
    direction : (MPEnums.INPUT, MPEnums.OUTPUT, or MPEnums.INOUT)
        Enum indicating the direction of the parameter
    """
    def __init__(self, data, direction):
        """Init. """
        self.data = data
        self.direction = direction
