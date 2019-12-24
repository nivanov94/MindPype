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
    """
    
    def __init__(self):
        super().__init__(BcipEnums.SESSION,self)
        
        # define some private attributes
        self._blocks = [] # queue of blocks to execute
        self._datum = []
        self._misc_objs = []
        self._ext_srcs = []
        self._verified = False
        
    # API Getters
    @property
    def current_block(self):
        return self._blocks[0]
    
    @property
    def remaining_blocks(self):
        return len(self._blocks)
    
    def verify(self):
        """
        Ensure all blocks and their processing graphs are valid.
        Execute this method prior data collection to mitigate potential
        crashes due to invalid processing graph construction.
        
        Return true if the session has passed verification, false otherwise.
        """
        print("Verifying session...")
        b_count = 1
        for b in self._blocks:
            print("\tVerifying block {} of {}".format(b_count,len(self._blocks)))
            verified = b.verify()
            
            if verified != BcipEnums.SUCCESS:
                self._verified = False
                return verified
        
        self._verified = True
        return BcipEnums.SUCCESS
    
    def initialize_block(self):
        """
        Initialize the current block object for trial execution.
        """
        return self.current_block.initialize()
    
    def poll_volatile_channels(self,label):
        """
        Update the contents of all volatile data streams
        
        TODO - may need to add an input parameter with some timing information
        to indicate how each data object should be synced
        """
        for d in self._datum:
            if d.volatile:
                d.poll_volatile_data(label)
        
        
    def close_block(self):
        """
        Run any postprocessing on the block and remove it from the session
        queue.
        """
        
        # get the current block
        b = self.current_block
        
        # check if the block is finished
        if sum(b.remaining_trials()) != 0:
            # block not finished, don't close
            return BcipEnums.FAILURE
        
        # run postprocessing
        sts = b.post_process()
        if sts != BcipEnums.SUCCESS:
            return sts
        
        # if everything executed nicely, remove the block from the session queue
        self.dequeue_block()
        return BcipEnums.SUCCESS    
    
    def start_block(self):
        """
        Initialize the block nodes and execute the preprocessing graph
        """
        b = self.current_block
        
        # make sure we're not in the middle of a block
        if b.remaining_trials() != b.n_class_trials:
            return BcipEnums.FAILURE
        
        # initialize everything first
        sts = b.initialize()
        if sts != BcipEnums.SUCCESS:
            return sts
        
        # execute the preprocessing graph
        sts = b.pre_process()
        
        return sts
    
    def execute_trial(self,label):
        """
        Execute a trial
        First updates all volatile input channels
        Then executes current block
        """
        self.poll_volatile_channels(label)
        sts = self.current_block.process_trial(label)
        
        return sts
    
    def enqueue_block(self,b):
        # block added, so make sure verified is false
        self._verified = False
        self._blocks.append(b)
    
    def dequeue_block(self):
        return self._blocks.pop(0)
    
    def add_data(self,data):
        self._datum.append(data)
        
    def add_misc_bcip_obj(self,obj):
        self._misc_objs.append(obj)
        
    def add_ext_src(self,src):
        self._ext_srcs.append(src)
        
    def find_obj(self,id_num):
        """
        Search for and return a BCIP object within the session with a
        specific ID number
        """
        
        # check if the ID is the session itself
        if id_num == self.session_id:
            return self
        
        # check if its a block
        for b in self._blocks:
            if id_num == b.session_id:
                return b
        
        # check if its a data obj
        for d in self._datum:
            if id_num == d.session_id:
                return d
        
        # check if its a misc obj
        for o in self._misc_objs:
            if id_num == o.session_id:
                return o
        
        # check if its a external source
        for s in self._ext_srcs:
            if id_num == s.session_id:
                return s
        
        # not found, return None type
        return None
    @classmethod
    def create(cls):
        return cls()