# -*- coding: utf-8 -*-
"""
Session.py - Defines the session class for BCIP

@author: ivanovn
"""

from .bcip import BCIP
from .bcip_enums import BcipEnums

class Session(BCIP):
    """
    Session objects contain all other BCIP objects instances within a data
    capture session.

    Parameters
    ----------
    None

    Attributes
    ----------
    _datum : dict
        - 


    Examples
    --------
    >>> from bcipy.classes import session as S
    >>> S.session.create()

    """
    
    def __init__(self):
        super().__init__(BcipEnums.SESSION,self)
        
        # define some private attributes
        # self._blocks = [] # queue of blocks to execute
        self._datum = {}
        self._misc_objs = {}
        self._ext_srcs = {}
        self._verified = False
        self._trials_elapsed = 0
        
        
        self.graphs = []

    # API Getters
    #@property
    #def current_block(self):
    #    return self._blocks[0]
    #
    #@property
    #def remaining_blocks(self):
    #    return len(self._blocks)
    
    def verify(self):
        """
        Ensure all graphs are valid.
        Execute this method prior data collection to mitigate potential
        crashes due to invalid processing graph construction.
        
        Parameters
        ----------
        None

        Returns
        -------
        BCIP Status Code

        Examples
        --------
        >>> status = session.verify()
        >>> print(status)

            SUCCESS
        """
        print("Verifying session...")
        graph_count = 1
        for graph in self.graphs:
            print("\tVerifying graph {} of {}".format(graph_count,len(self.graphs)))
            verified = graph.verify()
            
            if verified != BcipEnums.SUCCESS:
                self._verified = False
                return verified
            
            graph_count += 1
        
        self._verified = True
        return BcipEnums.SUCCESS
    
    def initialize_graph(self, graph):
        return graph.initialize()

    #def initialize_block(self):
    #    """
    #    Initialize the current block object for trial execution.
    #    """
    #    return self.current_block.initialize()
    
    def poll_volatile_channels(self,label=None):
        """
        Update the contents of all volatile data streams
        
        TODO - may need to add an input parameter with some timing information
        to indicate how each data object should be synced
        """
        for d in self._datum:
            if self._datum[d].volatile:
                self._datum[d].poll_volatile_data(label)
        
        
    #def close_block(self):
    #    """
    #    Run any postprocessing on the block and remove it from the session
    #    queue.
    #    """
    #    print("Closing block...")
    #    # get the current block
    #    b = self.current_block
    #
    #    # check if the block is finished
    #    if sum(b.remaining_trials()) != 0:
    #    # block not finished, don't close
    #        return BcipEnums.FAILURE
    #    
    #    # run postprocessing
    #    sts = b.close_block()
    #    if sts != BcipEnums.SUCCESS:
    #    #        return sts
    #
    #    # if everything executed nicely, remove the block from the session queue
    #    self.dequeue_block()
    #    return BcipEnums.SUCCESS    
        
    #def start_block(self, graph):
    #    """
    #    Initialize the block nodes and execute the preprocessing graph
    #    """
    #    print("Starting block...")
    #    if len(self._blocks) == 0:
    #        return BcipEnums.FAILURE
    #    
    #    b = self.current_block
    #    
    #    # make sure we're not in the middle of a block
    #    if b.remaining_trials() != b.n_class_trials:
    #        return BcipEnums.FAILURE
    #    
    #    return b.initialize(graph)
    
    def execute_trial(self,label,graph):
        """
        Execute a trial
        First updates all volatile input channels
        Then executes current block
        """
        print("Executing trial with label: {}".format(label))
        self.poll_volatile_channels(label)
        #sts = self.current_block.process_trial(label, graph)
        sts = graph.execute(label)
        return sts
    
    def reject_trial(self):
        """
        Reject the previous trial and reset the block trial counters
        """
        return self.current_block.reject_trial()
        
    def add_graph(self,graph):
        self._verified = False
        self.graphs.append(graph)

    #def enqueue_block(self,b):
    #    # block added, so make sure verified is false -> not needed, graphs verified now
    #    self._blocks.append(b)
    #
    #def dequeue_block(self):
    #    return self._blocks.pop(0)
    
    def add_data(self,data):
        self._datum[data.session_id] = data
        
    def add_misc_bcip_obj(self,obj):
        self._misc_objs[obj.session_id] = obj
        
    def add_ext_src(self,src):
        self._ext_srcs[src.session_id] = src
        
    def find_obj(self,id_num):
        """
        Search for and return a BCIP object within the session with a
        specific ID number
        """
        
        # check if the ID is the session itself
        if id_num == self.session_id:
            return self
        
        # check if its a block
        #for b in self._blocks:
        #    if id_num == b.session_id:
        #        return b
        
        # check if its a data obj
        if id_num in self._datum:
            return self._datum[id_num]
        
        # check if its a misc obj
        if id_num in self._misc_objs:
            return self._misc_objs[id_num]
        
        # check if its a external source
        if id_num in self._ext_srcs:
            return self._ext_srcs[id_num]
        
        # not found, return None type
        return None
    
    @classmethod
    def create(cls):
        return cls()