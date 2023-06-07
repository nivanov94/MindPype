from enum import IntEnum

class BCIP(object):
    """
    This is the base class for all objects used in the BCIP API.
    It serves to define some attributes that will be shared across all
    other objects.

    Parameters
    ----------
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

    """
    def __init__(self,bcip_type,session):
        """
        Constructor for BCIP base class        
        """
        self._bcip_type = bcip_type
        self._id  = id(self)
        self._session = session
    
    # API getters
    @property
    def bcip_type(self):
        """
        Returns the type of object

        Return
        ------
        bcip_type : Object type enum (int)

        Return Type
        -----------
        BCIPEnum
        """

        return self._bcip_type
    
    @property
    def session_id(self):
        """
        Returns the session id of the object

        Return
        ------
        session_id
            ID of the session where the object exists

        Return Type
        -----------
        int
        """
        return self._id
    
    @property
    def session(self):
        """
        Returns the session object of the object

        Returns
        -------
        session
            Session where the object exists
        
        Return Type
        -----------
        BCIPy Session Object
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

    Attributes
    ----------
    _datum : dict
        Dictionary of all datum objects


    Examples
    --------
    >>> from bcipy.classes import session as S
    >>> S.session.create()

    """
    
    def __init__(self):
        """
        Constructor for Session class
        """
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
        """
        Initialize a graph

        Parameters
        ----------
        graph : Graph object
            Graph to initialize

        Returns
        -------
        BCIP Status Code

        Examples
        --------
        >>> status = session.initialize_graph(graph)
        >>> print(status)
        """

        return graph.initialize()

    
    def poll_volatile_channels(self,label=None):
        """
        Update the contents of all volatile data streams
        
        TODO - may need to add an input parameter with some timing information
        to indicate how each data object should be synced

        Parameters
        ----------
        label : str, optional
            Label for the current trial. The default is None.

        Examples
        --------
        >>> session.poll_volatile_channels()
        """
        for d in self._datum:
            if self._datum[d].volatile:
                self._datum[d].poll_volatile_data(label)
        
    def push_volatile_outputs(self, label=None):
        """
        Push outputs to volatile sources

        Parameters
        ----------
        label : str, optional
            Label for the current trial. The default is None.

        Examples
        --------
        >>> session.push_volatile_outputs()

        """ 

        for d in self._datum:
            if self._datum[d].volatile_out:
                self._datum[d].push_volatile_outputs(label)
    
    def execute_trial(self,label,graph):
        """
        Execute a trial
        First updates all volatile input channels
        Then executes current block

        Parameters
        ----------
        label : str
            Label for the current trial.
        graph : Graph object
            Graph to execute

        Returns
        -------
        BCIP Status Code

        Examples
        --------
        >>> status = session.execute_trial(label,graph)
        >>> print(status)
        """
        print("Executing trial with label: {}".format(label))
        self.poll_volatile_channels(label)
        self.push_volatile_outputs(label)
        sts = graph.execute(label)
        return sts
    
        
    def add_graph(self,graph):
        """
        Add a graph to the session

        Parameters
        ----------
        graph : Graph object
            Graph to add

        Examples
        --------
        >>> session.add_graph(graph)
        """
        self._verified = False
        self.graphs.append(graph)
    
    def add_data(self,data):
        """
        Add a data object to the session

        Parameters
        ----------
        data : BCIPy Data object
            Data object to add

        Examples
        --------
        >>> session.add_data(data)
        """
        self._datum[data.session_id] = data
        
    def add_misc_bcip_obj(self,obj):
        """
        Add a misc BCIP object to the session

        Parameters
        ----------
        obj : BCIP object
            BCIP object to add
        
        Examples
        --------
        >>> session.add_misc_bcip_obj(obj)
        """
        self._misc_objs[obj.session_id] = obj
        
    def add_ext_src(self,src):
        """
        Add an external source to the session

        Parameters
        ----------
        src : External Source object
            External source to add
        """
        self._ext_srcs[src.session_id] = src
        
    def add_ext_out(self,src):
        """
        Add an external outlet to the session

        Parameters
        ----------
        src : External outlet object
            External outlet to add
        """
        self._ext_out[src.session_id] = src
    
    def find_obj(self,id_num):
        """
        Search for and return a BCIP object within the session with a
        specific ID number

        Parameters
        ----------
        id_num : int
            ID number of the object to find

        Returns
        -------
        BCIP object
            BCIP object with the specified ID number

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
        """
        Create a new session object

        Returns
        -------
        Session object
            New session object

        """
        return cls()
