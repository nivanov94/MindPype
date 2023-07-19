""" 
.. sectionauthor:: Aaron Lio <aaron.lio@mail.utoronto.ca>

.. codeauthor:: Nicolas Ivanov <nicolas.ivanov@mail.utoronto.ca>
.. codeauthor:: Aaron Lio <aaron.lio@mail.utoronto.ca>
"""

from enum import IntEnum
import logging, pickle, copy

class BCIP(object):
    """
    This is the base class for all objects used in the BCIP API.
    It serves to define some attributes that will be shared across all
    other objects.

    Parameters
    ----------
    bcip_type : BcipEnums
        Indicates what type of object is being created
    session : Session
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
        bcip_type : BcipEnums
            Indicates what type of object the reference object is
        """

        return self._bcip_type
    
    @property
    def session_id(self):
        """
        Returns the session id of the object

        Return
        ------
        session_id : int
            ID of the session where the object exists
        """
        return self._id
    
    @property
    def session(self):
        """
        Returns the session object of the object

        Returns
        -------
        Current Session : Session
            Session where the object exists
        
        """

        return self._session


class BcipEnums(IntEnum):
    """
    Defines a class of enums used by BCIP

    The following enums are defined and available for use:

    .. list-table:: Object Type BcipEnums - Leading '1'
        :widths: 50 50
        :header-rows: 1
    
        * - Enum
          - Value
        * - BCIP
          - 100
        * - SESSION
          - 101
        * - GRAPH
          - 102
        * - NODE
          - 103
        * - KERNEL
          - 104
        * - PARAMETER
          - 105
        * - TENSOR
          - 106
        * - SCALAR
          - 107
        * - ARRAY
          - 108
        * - CIRCLE_BUFFER
          - 109
        * - FILTER
          - 110
        * - SRC
          - 111
        * - CLASSIFIER
          - 112
        
    .. list-table:: Status Codes - Leading '2'
        :widths: 50 50
        :header-rows: 1

        * - Enum
          - Value
        * - SUCCESS
          - 200
        * - FAILURE
          - 201
        * - INVALID_BLOCK
          - 202
        * - INVALID_NODE
          - 203
        * - INVALID_PARAMETERS
          - 204
        * - EXCEED_TRIAL_LIMIT
          - 205
        * - NOT_SUPPORTED
          - 206
        * - INITIALIZATION_FAILURE
          - 207
        * - EXE_FAILURE_UNINITIALIZED
          - 208
        * - EXE_FAILURE
          - 209
        * - NOT_YET_IMPLEMENTED
          - 210
        * - INVALID_GRAPH
          - 211

    .. list-table:: Parameter Directions - Leading '3'
        :widths: 50 50
        :header-rows: 1

        * - Enum
          - Value
        * - INPUT
          - 300
        * - OUTPUT
          - 301
        * - INOUT
          - 302

    .. list-table:: Kernel Initialization types - leading '4'
        :widths: 50 50
        :header-rows: 1

        * - Enum
          - Value
        * - INIT_FROM_NONE
          - 400
        * - INIT_FROM_DATA
          - 401
        * - INIT_FROM_COPY
          - 402

    .. list-table:: Block graph identifiers - leading '5'
        :widths: 50 50
        :header-rows: 1

        * - Enum
          - Value
        * - ON_BEGIN
          - 500
        * - ON_CLOSE
          - 501
        * - ON_TRIAL
          - 502

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

        # Configure logging for the session
        #logging.basicConfig(filename='../bcipy/logs/log.log', filemode='a', format='%(asctime)s: [%(pathname)s:%(funcName)s:%(lineno)d] - [%(levelname)s] - %(message)s ', level=logging.INFO, encoding='utf-8')
        #logging.info("Session created")

    def verify(self):
        """
        Ensure all graphs are valid.
        Execute this method prior data collection to mitigate potential
        crashes due to invalid processing graph construction.
        
        Returns
        -------
        BCIP Status Code : BcipEnums
            Status of graph verification

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
        #logging.info("Session verified successfully")
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
        BCIP Status Code : BcipEnums
            Status of graph initialization

        Examples
        --------
        >>> status = session.initialize_graph(graph)
        >>> print(status)
        """

        return graph.initialize()

    
    def poll_volatile_channels(self,label=None):
        """
        Update the contents of all volatile data streams
    

        Parameters
        ----------
        label : str, optional
            Label for the current trial. The default is None.

        Examples
        --------
        >>> session.poll_volatile_channels()

.. warning::
    For Developers: may need to add an input parameter with some timing information
    to indicate how each data object should be synced
        """
        for d in self._datum:
            if self._datum[d].volatile:
                self._datum[d].poll_volatile_data(label)
       
        #logging.info("Volatile channels polled")
        
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

        #logging.info("Volatile outputs pushed")
    
    def execute_trial(self,label,graph):
        """
        Execute a trial
        First updates all volatile input channels
        Then executes current block

        Parameters
        ----------
        label : str
            Label for the current trial.
        graph : Graph
            Graph to execute for the current trial.

        Returns
        -------
        Status Code : BcipEnums
            Status code for the execution of the trial.

        Examples
        --------
        >>> status = session.execute_trial(label,graph)
        >>> print(status)
        """
        print("Executing trial with label: {}".format(label))
        self.poll_volatile_channels(label)
        sts = graph.execute(label)
        self.push_volatile_outputs(label)
        self._trials_elapsed += 1
        #logging.info("Trial executed successfully")
        
        return sts
    
        
    def add_graph(self,graph):
        """
        Add a graph to the session

        Parameters
        ----------
        graph : Graph
            Graph to add to the session

        Examples
        --------
        >>> session.add_graph(graph)
        """
        self._verified = False
        self.graphs.append(graph)
        #logging.info("Graph added to session")
    
    def add_data(self,data):
        """
        Add a data object to the session

        Parameters
        ----------
        data : Tensor or Scalar or Array or CircleBuffer
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
        any BCIPy object : BCIP
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
        src : LSLStream, CircleBuffer, or other external source
            External source to add
        """
        self._ext_srcs[src.session_id] = src
        
    def add_ext_out(self,src):
        """
        Add an external outlet to the session

        Parameters
        ----------
        src : OutputLSLStream, CircleBuffer, or other external outlet
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
        BCIP object : BCIP
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
    

    def save_session(self: object, file: str) -> None:
        """
        Save the session object and all of its contents to a file

        Returns
        -------
        Pickle object : Pickle
            Pickle object containing the session and all of its contents

        """
        new_object = copy.deepcopy(self)
        
        sfile = open(file, 'wb')
        pickle.dump(new_object, sfile)
        sfile.close()
        
    @classmethod
    def create(cls):
        """
        Create a new session object

        Returns
        -------
        New session object : Session

        """
        return cls()
