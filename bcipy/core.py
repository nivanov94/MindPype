from enum import IntEnum

class BCIP(object):
    """
    This is the base class for all objects used in the BCIP API.
    It serves to define some attributes that will be shared across all
    other objects.

    Arguments
    ---------
    bcip_type : Object type enum (int)
        Indicates what type of object is being created
    session : session object
        The session where the object will exist

    
    Attributes
    ----------
    _bcip_type : Object type enum (int)
        Indicates what type of object is being created
    _id : int
        Unique identifier for the object
    _session : session object
        The session where the object will exist

    Returns
    -------
    BCIP Object

    """
    def __init__(self,bcip_type,session):
        self._bcip_type = bcip_type
        self._id  = id(self)
        self._session = session
    
    # API getters
    @property
    def bcip_type(self):
        """
        Returns the type of object

        Arguments
        ---------
        None

        Returns
        -------
        bcip_type : Object type enum (int)
        """

        return self._bcip_type
    
    @property
    def session_id(self):
        """
        Returns the session id of the object

        Arguments
        ---------
        None

        Returns
        -------
        session_id : int
        """
        return self._id
    
    @property
    def session(self):
        """
        Returns the session object of the object

        Arguments
        ---------
        None

        Returns
        -------
        session : session object
        """

        return self._session


class BcipEnums(IntEnum):
    """
    Defines a class of enums used by BCIP
    """

    # Object Type Enums - Have a leading '1'
    BCIP          = 100
    SESSION       = 101
    GRAPH         = 102
    NODE          = 103
    KERNEL        = 104
    PARAMETER     = 105
    TENSOR        = 106
    SCALAR        = 107
    ARRAY         = 108
    CIRCLE_BUFFER = 109
    FILTER        = 110
    SRC           = 111
    CLASSIFIER    = 112
    
    # Status Codes - Leading '2'
    SUCCESS = 200
    FAILURE = 201
    INVALID_BLOCK = 202
    INVALID_NODE  = 203
    INVALID_PARAMETERS = 204
    EXCEED_TRIAL_LIMIT = 205
    NOT_SUPPORTED = 206
    INITIALIZATION_FAILURE = 207
    EXE_FAILURE_UNINITIALIZED = 208
    EXE_FAILURE = 209
    NOT_YET_IMPLEMENTED = 210
    INVALID_GRAPH = 211
    
    # Parameter Directions - Leading '3'
    INPUT  = 300
    OUTPUT = 301
    INOUT  = 302
    
    # Kernel Initialization types - leading '4'
    INIT_FROM_NONE = 400
    INIT_FROM_DATA = 401
    INIT_FROM_COPY = 402
    
    # Block graph identifiers - leading '5'
    ON_BEGIN = 500 # graph executes when block begins
    ON_CLOSE = 501 # graph executes when block ends
    ON_TRIAL = 502 # graph executes when a new trial is recorded
    
    def __str__(self):
        return self.name


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
        self._ext_out = {}
        
        
        self.graphs = []


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

    
    def poll_volatile_channels(self,label=None):
        """
        Update the contents of all volatile data streams
        
        TODO - may need to add an input parameter with some timing information
        to indicate how each data object should be synced
        """
        for d in self._datum:
            if self._datum[d].volatile:
                self._datum[d].poll_volatile_data(label)
        
    def push_volatile_outputs(self, label=None):
        """
        Push outputs to volatile sources
        """ 

        for d in self._datum:
            if self._datum[d].volatile_out:
                self._datum[d].push_volatile_outputs(label)
    
    def execute_trial(self,label,graph):
        """
        Execute a trial
        First updates all volatile input channels
        Then executes current block
        """
        print("Executing trial with label: {}".format(label))
        self.poll_volatile_channels(label)
        self.push_volatile_outputs(label)
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
    
    def add_data(self,data):
        self._datum[data.session_id] = data
        
    def add_misc_bcip_obj(self,obj):
        self._misc_objs[obj.session_id] = obj
        
    def add_ext_src(self,src):
        self._ext_srcs[src.session_id] = src
        
    def add_ext_out(self,src):
        self._ext_out[src.session_id] = src
    
    def find_obj(self,id_num):
        """
        Search for and return a BCIP object within the session with a
        specific ID number
        """
        
        # check if the ID is the session itself
        if id_num == self.session_id:
            return self
        
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
