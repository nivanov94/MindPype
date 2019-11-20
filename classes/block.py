# -*- coding: utf-8 -*-
"""
Block.py - Defines the block class for BCIP

@author: ivanovn
"""

from bcip import BCIP
from bcip_types import BcipEnums
from session import Session
from edge import Edge

class Block(BCIP):
    """
    Defines a block within a BCIP session.
    """
    
    def __init__(self,sess,n_trials,n_classes):
        super().__init__(BcipEnums.BLOCK)
        
        self.sess = sess
        self.n_trials = n_trials
        self.n_classes = n_classes
        
        # private attributes
        self._nodes = []
        self._trials_executed = 0
        self._verified = False
        
        # add the block to the session
        sess.enqueueBlock(self)
        
    def _addNode(self,node):
        """
        Append a node object to the block's list of nodes
        """
        self._nodes.append(node)
    
    def execute(self):
        """
        Execute the block's processing graph. 
        
        Pre: Ensure the block's input data objects have been updated to 
             contain the correct trial's data
        
        Returns a status code (false for failure)
        """
        
        # TODO update return codes to be ENUM codes
        
        # first ensure the block's processing graph has been verified,
        # if not, verify and schedule the nodes
        if not self._verified:
            executable = self.verify()
            if not executable:
                return executable
            
        # iterate over all the nodes and execute the kernel
        for n in self._nodes:
            sts = n.execute()
            
            if not sts:
                # execute failed, exit...
                return sts
        
        self._trials_executed = self._trials_executed + 1
        return True
        
    def verify(self):
        """
        Verify the processing graph is valid. This method orders the nodes
        for execution if the graph is valid
        """
        if self._verified:
            return True
        
        # begin by scheduling the nodes in execution order
        
        # first we'll create a set of edges representing data within the graph
        edges = {} # keys: uid of data obj, vals: edge object
        for n in self._nodes:
            # get a list of all the input objects to the node
            n_inputs = n.getInputs()
            n_outputs = n.getOutputs()
            
            # add these inputs/outputs to edge objects
            for n_i in n_inputs:
                if not (n_i.uid in edges.keys):
                    # no edge created for this input yet, so create a new one
                    edges[n_i.uid] = Edge(n_i)
                
                # now add the node the edge's list of consumers
                edges[n_i.uid].addConsumer(n)
                
            for n_o in n_outputs:
                if not (n_o.uid in edges.keys):
                    # no edge created for this output yet, so create a new one
                    edges[n_o.uid] = Edge(n_o)
                    
                    # add the node as a producer
                    edges[n_o.uid].addProducer(n)
                else:
                    # edge already created, must check that it has no other 
                    # producer
                    if len(edges[n_o.uid].getProducers()) != 0:
                        # this is an invalid graph, each data object can only
                        # have a single producer
                        return False
                    else:
                        # add the producer to the edge
                        edges[n_o.uid].addProducer(n)
        
        # now determine which edges are ready to be consumed
        consumable_edges = {}
        for e_key in edges.keys:
            if len(edges[e_key].getProducers()) == 0:
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
                n_inputs = n.getInputs()
                consumable = True
                for n_i in n_inputs:
                    if not (n_i.uid in consumable_edges.keys):
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
                    for n_o in n.getOutputs():
                        consumable_edges[n_o.uid] = edges[n_o.uid]
                    
                    nodes_added = nodes_added + 1
                    scheduled_nodes = scheduled_nodes + 1
                    
                    
            if nodes_added == 0:
                # invalid graph, cannot be scheduled
                return False
        
        # now all the nodes are in execution order, validate each node
        for n in self._nodes:
            valid = n.verify()
            if not valid:
                return False
        
        # Done, all nodes scheduled and verified!
        return True