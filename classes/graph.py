# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 12:00:43 2019

graph.py - Defines the graph object

@author: ivanovn
"""

from .bcip import BCIP
from .bcip_enums import BcipEnums
from .edge import Edge

class Graph(BCIP):
    """
    Data processing flow graph
    """
    
    def __init__(self,block):
        """
        Create a new graph within an existing block
        """
        sess = block.session
        super().__init__(BcipEnums.GRAPH,sess)
        
        # private attributes
        self._nodes = []
        self._verified = False
        
    
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
                        return BcipEnums.INVALID_BLOCK
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
                    
            if nodes_added == 0:
                # invalid graph, cannot be scheduled
                return BcipEnums.INVALID_BLOCK
        
        print("\t\tNodes Scheduled")
        # now all the nodes are in execution order, validate each node
        for n in self._nodes:
            valid = n.verify()
            #print(n.kernel)
            if valid != BcipEnums.SUCCESS:
                return valid
        
        # Done, all nodes scheduled and verified!
        self._verified = True
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
                return sts
        
        return BcipEnums.SUCCESS
            
    @classmethod
    def create(cls,block):
        """
        Generic factory method for a graph
        """
        return cls(block)