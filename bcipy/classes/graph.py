# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 12:00:43 2019

graph.py - Defines the graph object

@author: ivanovn
"""
from bcip import BCIP
from bcip_enums import BcipEnums
from bcipy.classes.tensor import Tensor
from edge import Edge

class Graph(BCIP):
    """
    Data processing flow graph
    """
    
    def __init__(self, sess):
        """
        Create a new graph within an existing block
        """

        super().__init__(BcipEnums.GRAPH,sess)
        
        # private attributes
        self._nodes = []
        self.initialization_edges = []
        self._verified = False
        self._missing_data = False
        self._sess = sess
        
    
    def add_node(self,node):
        """
        Append a node object to the block's list of nodes
        """
        self._verified = False
        self._nodes.append(node)
        
        
    def verify(self):
        """
        Verify the processing graph is valid. This method orders the nodes
        for execution if the graph is valid
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
                if not (n_i.session_id in edges):
                    # no edge created for this input yet, so create a new one
                    edges[n_i.session_id] = Edge(n_i)
                
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
        for n in self._nodes:
            valid = n.verify()
            
            if hasattr(n._kernel, "initialization_data"):
                if n._kernel.initialization_data == None:
                    self._missing_data = True
                    
            if valid != BcipEnums.SUCCESS:
                print("Node {} failed verification".format(n.kernel.name))
                return valid
        
        if self._missing_data:         
            for n in self._nodes:
                if hasattr(n._kernel, "initialization_data"):
                    if n._kernel.initialization_data == None:
                        
                        sts = self.fill_initialization_links(n, edges)
                        if sts != BcipEnums.SUCCESS:
                            return BcipEnums.INVALID_GRAPH
                                                    

        # Done, all nodes scheduled and verified!
        self._verified = True
        return BcipEnums.SUCCESS
    
    def fill_initialization_links(self, n, edges):
        n_inputs = n.extract_inputs()
        if len(n_inputs) == 0 and n._kernel._init_inA == None:
            return BcipEnums.INVALID_GRAPH
        for n_i in n_inputs:
            # identify the up-stream node producing each input
            producer = edges[n_i.session_id].producers[0]

            # TODO: need to more robustly identify the correct initialization output
            # may require modification of the node/parameter data structures.

            if producer._kernel._init_outA != None: # TODO init data may come from other node outputs
                if hasattr(n._kernel, "_init_inB") and n._kernel._init_inA != None:
                    if n._kernel._init_inB != None and n._kernel._init_inA != None:
                        return BcipEnums.INVALID_GRAPH
                    
                    n._kernel._init_inB = producer._kernel._init_outA
                
                if (not hasattr(n._kernel, "_init_inB")) and n._kernel._init_inA != None:
                    return BcipEnums.INVALID_GRAPH

                else:
                    n._kernel._init_inA = producer._kernel._init_outA
            else:
                init_data = Tensor.create_virtual(sess=self._sess) # TODO determine if this would ever not be a tensor
                producer._kernel._init_outA = init_data
                if hasattr(n._kernel, "_init_inB") and n._kernel._init_inA != None:
                    if n._kernel._init_inB != None:
                        return BcipEnums.INVALID_GRAPH
                    n._kernel._init_inB = init_data
                else:
                    n._kernel._init_inA = init_data
            
             # recurse process on up-stream nodes
            if producer._kernel._init_inA == None:
                sts = self.fill_initialization_links(self, producer, edges)
                if sts != BcipEnums.SUCCESS:
                    return BcipEnums.INVALID_GRAPH
        
        return BcipEnums.SUCCESS


    def initialize(self):
        """
        Initialize each node within the graph for trial execution
        """
        for n in self._nodes:
            sts = n.initialize()
            
            if sts != BcipEnums.SUCCESS:
                return sts
        
        return BcipEnums.SUCCESS
    
    
    def execute(self):
        """
        Execute the graph
        """
        # first ensure the block's processing graph has been verified,
        # if not, verify and schedule the nodes
        if not self._verified:
            executable = self.verify()
            if executable != BcipEnums.SUCCESS:
                return executable
            
        # iterate over all the nodes and execute the kernel
        for n in self._nodes:
            sts = n.kernel.execute()
            if sts != BcipEnums.SUCCESS:
                print("Node {} failed with status {}".format(n.kernel.name,sts))
                return sts
        
        return BcipEnums.SUCCESS
            
    @classmethod
    def create(cls,sess):
        """
        Generic factory method for a graph
        """
        graph = cls(sess)
        sess.add_graph(graph)
        
        return graph
