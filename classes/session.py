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
        super().__init__(BcipEnums.SESSION)
        
        # define some private attributes
        self._blocks = [] # queue of blocks to execute
        self._datum = []
        self._verified = False
    
    def verify(self):
        """
        Ensure all blocks and their processing graphs are valid.
        Execute this method prior data collection to mitigate potential
        crashes due to invalid processing graph construction.
        
        Return true if the session has passed verification, false otherwise.
        """
        
        for b in self._blocks:
            verified = b.verify()
            
            if verified != BcipEnums.SUCCESS:
                self._verified = False
                return verified
        
        self._verified = True
        return BcipEnums.SUCCESS
    
    def pollVolatileChannels(self):
        """
        Update the contents of all volatile data streams
        
        TODO - may need to add an input parameter with some timing information
        to indicate how each data object should be synced
        """
        for d in self._datum:
            if d.isVolatile():
                d.pollInputStream()
        
    def execute(self,label):
        """
        Execute a trial
        First updates all volatile input channels
        Then executes current block
        """
        self.pollVolatileChannels()
        b = self.getCurrentBlock()
        sts = b.execute()
        
        # check if the block is finished
        if b.trialsRemaining() == 0:
            # block finished, close it up and remove it from the session queue
            b.postProcess()
            self.dequeueBlock()
            
        return sts
    
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
        
    @classmethod
    def create(cls):
        return cls()