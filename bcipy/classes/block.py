"""
Block.py - Defines the block class for BCIP

@author: ivanovn
"""

from bcip import BCIP
from bcip_enums import BcipEnums
from graph import Graph

class Block(BCIP):
    """
    Defines a block within a BCIP session.
    """
    
    def __init__(self,sess,n_classes,n_class_trials):
        super().__init__(BcipEnums.BLOCK,sess)
        
        self._n_class_trials = tuple(n_class_trials)
        self._n_classes = n_classes
        self.sess = sess

        # create the block's data processing graphs
        #self._preprocessing_graph = Graph.create(self)
        #self._postprocessing_graph = Graph.create(self)
        #self._trial_processing_graph = Graph.create(self)
        
        # private attributes
        self._trials_executed = [0] * n_classes
        self._previous_trial_label = None
        self._verified = False
        
    
    # API Getters
    @property
    def n_classes(self):
        return self._n_classes
    
    @property
    def n_class_trials(self):
        return self._n_class_trials

    @property
    def graph(self, graph_name):
        return self.sess.graphs[graph_name]
    
    #@property
    #def preprocessing_graph(self):
        return self._preprocessing_graph
    
    #@property
    #def postprocessing_graph(self):
        return self._postprocessing_graph
    
    #@property
   # def trial_processing_graph(self):
        return self._trial_processing_graph
    


    @property
    def latest_trial_label(self):
        return self._previous_trial_label
    
    
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
    
        
    #def post_process(self):
        """
        Perform any actions that need to be done at the end of the block
        and run the block close graph
        """
        
        # execute the closing block graph
        return self.postprocessing_graph.execute()
    
    #def pre_process(self):
        """
        Execute the block setup graph
        """        
        # execute the preprocess graph
        return self.preprocessing_graph.execute()
        
    
    def process_trial(self,label,graph):
        """
        Execute the block's processing graph. 
        
        Pre: Ensure the block's input data objects have been updated to 
             contain the correct trial's data
        
        Returns a status code
        """
        
        if self.remaining_trials(label) == (0,):
            return BcipEnums.EXCEED_TRIAL_LIMIT
                
        sts = graph.execute()
        if sts != BcipEnums.SUCCESS:
            return sts
        
        self._previous_trial_label = label
        self._trials_executed[label] += 1
        return BcipEnums.SUCCESS
        
    
    def reject_trial(self):
        """
        Reject the previous trial by rewinding the trial counter
        """
        if self._previous_trial_label == None \
           or self._trials_executed[self._previous_trial_label] == 0:
            return BcipEnums.FAILURE
        
        self._trials_executed[self._previous_trial_label] -= 1
        
        return BcipEnums.SUCCESS
    
    @classmethod
    def create(cls,sess,n_classes,n_class_trials):
        b = cls(sess,n_classes,n_class_trials)
        
        # add the block to the session
        sess.enqueue_block(b)
        
        return b
        
