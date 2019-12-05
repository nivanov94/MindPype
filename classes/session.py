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
        self._verified = False
    
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
    
    def initializeBlock(self):
        """
        Initialize the current block object for trial execution.
        """
        return self.getCurrentBlock().initialize()
    
    def pollVolatileChannels(self):
        """
        Update the contents of all volatile data streams
        
        TODO - may need to add an input parameter with some timing information
        to indicate how each data object should be synced
        """
        for d in self._datum:
            if d.isVolatile():
                d.pollInputStream()
        
        
    def closeBlock(self):
        """
        Run any postprocessing on the block and remove it from the session
        queue.
        """
        
        # get the current block
        b = self.getCurrentBlock()
        
        # check if the block is finished
        if b.trialsRemaining() != 0:
            # block not finished, don't close
            return BcipEnums.FAILURE
        
        # run postprocessing
        sts = b.postProcess()
        if sts != BcipEnums.SUCCESS:
            return sts
        
        # if everything executed nicely, remove the block from the session queue
        self.dequeueBlock()
        return BcipEnums.SUCCESS    
    
    def startBlock(self):
        """
        Initialize the block nodes and execute the preprocessing graph
        """
        b = self.getCurrentBlock()
        
        # make sure we're not in the middle of a block
        if b.trialsRemaining() != b.getNumberTrials():
            return BcipEnums.FAILURE
        
        # initialize everything first
        sts = b.initialize()
        if sts != BcipEnums.SUCCESS:
            return sts
        
        # execute the preprocessing graph
        sts = b.preProcess()
        
        return sts
    
    def executeTrial(self,label):
        """
        Execute a trial
        First updates all volatile input channels
        Then executes current block
        """
        self.pollVolatileChannels()
        b = self.getCurrentBlock()
        sts = b.processTrial(label)
        
        return sts
    
    def getBlocksRemaining(self):
        return len(self._blocks)
    
    def getCurrentBlock(self): #TODO Determine how this should be implemented
        return self._blocks[0]
    
    def enqueueBlock(self,b):
        # block added, so make sure verified is false
        self._verified = False
        self._blocks.append(b)
    
    def dequeueBlock(self):
        return self._blocks.pop(0)
    
    def addData(self,data):
        self._datum.append(data)
        
    def addMiscBcipObj(self,obj):
        self._misc_objs.append(obj)
        
    @classmethod
    def create(cls):
        return cls()