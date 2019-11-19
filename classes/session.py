# -*- coding: utf-8 -*-
"""
Session.py - Defines the session class for BCIP

@author: ivanovn
"""

from bcip import BCIP
from bcip_types import BcipEnums

class Session(BCIP):
    """
    Session objects contain all other BCIP objects instances within a data
    capture session.
    """
    
    def __init__(self,n_classes = 2):
        super().__init__(BcipEnums.SESSION)
        
        self.n_classes = n_classes
        
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
        verified = True
        for b in self._blocks:
            verified = b.verify()
            
            if verified == False:
                return verified
        
        return verified
    
    def pollVolatileChannels(self):
        """
        Update the contents of all volatile data streams
        
        TODO - may need to add an input parameter with some timing information
        to indicate how each data object should be synced
        """
        for d in self._datum:
            if d.isVolatile():
                d.pollInputStream()
        
    def execute(self):
        """
        Execute a trial
        First updates all volatile input channels
        Then executes current block
        """
        self.pollVolatileChannels()
        b = self.getCurrentBlock()
        b.execute()
    
    def getCurrentBlock(self): #TODO Determine how this should be implemented
        return self._blocks[0]
    
    def enqueueBlock(self,b):
        self._blocks.append(b)
    
    def dequeueBlock(self):
        return self._blocks.pop(0)