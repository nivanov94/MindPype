# -*- coding: utf-8 -*-
"""
Block.py - Defines the block class for BCIP

@author: ivanovn
"""

from .bcip import BCIP
from .bcip_enums import BcipEnums
from .graph import Graph

class Block(BCIP):
    """
    Defines a block within a BCIP session.
    """
    
    def __init__(self,sess,n_trials_per_class,n_classes):
        super().__init__(BcipEnums.BLOCK,sess)
        
        self.n_trials_per_class = n_trials_per_class
        self.n_classes = n_classes
        
        # private attributes
        self._trials_executed = [0] * n_classes
        self._verified = False
        
        # create the block's data processing graphs
        self._preprocessing_graph = Graph.create(self)
        self._postprocessing_graph = Graph.create(self)
        self._trial_processing_graph = Graph.create(self)
        
    
    def getNumberTrials(self):
        """
        Return the total number of trials to be executed within the block
        """
        return self.n_classes * self.n_trials_per_class
    
    def getRemainingTrials(self,label=None):
        """
        Get the number of trials remaining for each class
        """
        if label is None:
            return tuple([self.n_trials_per_class - n for n in self._trials_executed])
        else:
            return self.n_trials_per_class - self._trials_executed[label]
    
    def getTrialProcessGraph(self):
        """
        Return the trial processing graph
        """
        return self._trial_processing_graph
    
    def getPreProcessingGraph(self):
        """
        Return the block preprocessing graph
        """
        return self._preprocessing_graph
    
    def getPostProcessingGraph(self):
        """
        Return the block postprocessing graph
        """
        return self._postprocessing_graph
        
    def postProcess(self):
        """
        Perform any actions that need to be done at the end of the block
        and run the block close graph
        """
        
        # execute the closing block graph
        return self._postprocessing_graph.execute()
    
    def preProcess(self):
        """
        Initialize all block nodes and Execute the block setup graph
        """
        
        # set the internal state of the nodes
        sts = self.initialize()
        if sts != BcipEnums.SUCCESS:
            return sts
        
        # execute the preprocess graph
        return self._preprocessing_graph.execute()
    
    def trialsRemaining(self):
        """
        Calculate and return the total number of trials remaining in the block
        """
        return  sum(self.getRemainingTrials())
        
    
    def processTrial(self,label):
        """
        Execute the block's processing graph. 
        
        Pre: Ensure the block's input data objects have been updated to 
             contain the correct trial's data
        
        Returns a status code
        """
        
        if self._trials_executed[label] == self.n_trials_per_class:
            return BcipEnums.EXCEED_TRIAL_LIMIT
                
        sts = self._trial_processing_graph.execute()
        if sts != BcipEnums.SUCCESS:
            return sts
        
        self._trials_executed[label] = self._trials_executed[label] + 1
        return BcipEnums.SUCCESS
        
    def verify(self):
        """
        Verify each graph within the block
        """
        sts = self._preprocessing_graph.verify()
        
        if sts != BcipEnums.SUCCESS:
            return sts
        
        sts = self._trial_processing_graph.verify()
        
        if sts != BcipEnums.SUCCESS:
            return sts
        
        sts = self._postprocessing_graph.verify()
        
        if sts != BcipEnums.SUCCESS:
            return sts
        
        return BcipEnums.SUCCESS
    
    def initialize(self):
        """
        Initialize each graph within the block for trial execution
        """
        sts = self._preprocessing_graph.initialize()
        
        if sts != BcipEnums.SUCCESS:
            return sts
        
        sts = self._trial_processing_graph.initialize()
        
        if sts != BcipEnums.SUCCESS:
            return sts
        
        sts = self._postprocessing_graph.initialize()
        
        if sts != BcipEnums.SUCCESS:
            return sts
        
        return BcipEnums.SUCCESS
    
    @classmethod
    def create(cls,sess,n_trials_per_class,n_classes):
        b = cls(sess,n_trials_per_class,n_classes)
        
        # add the block to the session
        sess.enqueueBlock(b)
        
        return b
        