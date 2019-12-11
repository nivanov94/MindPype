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
    
    def __init__(self,sess,n_classes,n_class_trials):
        super().__init__(BcipEnums.BLOCK,sess)
        
        self._n_class_trials = n_class_trials
        self._n_classes = n_classes
        
        # create the block's data processing graphs
        self._preprocessing_graph = Graph.create(self)
        self._postprocessing_graph = Graph.create(self)
        self._trial_processing_graph = Graph.create(self)
        
        # private attributes
        self._trials_executed = [0] * n_classes
        self._verified = False
        
    
    # API Getters
    @property
    def n_classes(self):
        return self._n_classes
    
    @property
    def n_class_trials(self):
        return self._n_class_trials
    
    @property
    def preprocessing_graph(self):
        return self._preprocessing_graph
    
    @property
    def postprocessing_graph(self):
        return self._postprocessing_graph
    
    @property
    def trial_processing_graph(self):
        return self._trial_processing_graph
    
    
    def total_trials(self):
        """
        Return the total number of trials to be executed within the block
        """
        return sum(self.n_class_trials)
    
    def remaining_trials(self,label=None):
        """
        Get the number of trials remaining for a single class or all classes
        
        Returned as a tuple
        """
        if label is None:
            return tuple([self.n_class_trials[i] - self._trials_executed[i] 
                              for i in range(self.n_classes)])
        else:
            return (self.n_class_trials[label] - self._trials_executed[label],)
    
        
    def post_process(self):
        """
        Perform any actions that need to be done at the end of the block
        and run the block close graph
        """
        
        # execute the closing block graph
        return self.postprocessing_graph.execute()
    
    def pre_process(self):
        """
        Initialize all block nodes and Execute the block setup graph
        """
        
        # set the internal state of the nodes
        sts = self.initialize()
        if sts != BcipEnums.SUCCESS:
            return sts
        
        # execute the preprocess graph
        return self.preprocessing_graph.execute()
        
    
    def process_trial(self,label):
        """
        Execute the block's processing graph. 
        
        Pre: Ensure the block's input data objects have been updated to 
             contain the correct trial's data
        
        Returns a status code
        """
        
        if self.remaining_trials(label) == 0:
            return BcipEnums.EXCEED_TRIAL_LIMIT
                
        sts = self.trial_processing_graph.execute()
        if sts != BcipEnums.SUCCESS:
            return sts
        
        self._trials_executed[label] += 1
        return BcipEnums.SUCCESS
        
    def verify(self):
        """
        Verify each graph within the block
        """
        sts = self.preprocessing_graph.verify()
        
        if sts != BcipEnums.SUCCESS:
            return sts
        
        sts = self.trial_processing_graph.verify()
        
        if sts != BcipEnums.SUCCESS:
            return sts
        
        sts = self.postprocessing_graph.verify()
        
        if sts != BcipEnums.SUCCESS:
            return sts
        
        return BcipEnums.SUCCESS
    
    def initialize(self):
        """
        Initialize each graph within the block for trial execution
        """
        sts = self.preprocessing_graph.initialize()
        
        if sts != BcipEnums.SUCCESS:
            return sts
        
        sts = self.trial_processing_graph.initialize()
        
        if sts != BcipEnums.SUCCESS:
            return sts
        
        sts = self.postprocessing_graph.initialize()
        
        if sts != BcipEnums.SUCCESS:
            return sts
        
        return BcipEnums.SUCCESS
    
    @classmethod
    def create(cls,sess,n_classes,n_class_trials):
        b = cls(sess,n_classes,n_class_trials)
        
        # add the block to the session
        sess.enqueue_block(b)
        
        return b
        
