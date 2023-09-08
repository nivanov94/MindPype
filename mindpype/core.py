""" 
.. sectionauthor:: Aaron Lio <aaron.lio@mail.utoronto.ca>

.. codeauthor:: Nicolas Ivanov <nicolas.ivanov@mail.utoronto.ca>
.. codeauthor:: Aaron Lio <aaron.lio@mail.utoronto.ca>
"""

from enum import IntEnum
import sys
import pickle, copy

class MPBase(object):
    """
    This is the base class for all objects used in the MindPype API.
    It serves to define some attributes that will be shared across all
    other objects.

    Parameters
    ----------
    mp_type : MPEnums
        Indicates what type of object is being created
    session : Session
        The session where the object will exist

    """
    def __init__(self,mp_type,session):
        """
        Constructor for MPBase base class        
        """
        self._mp_type = mp_type
        self._id  = id(self)
        self._session = session
    
    # API getters
    @property
    def mp_type(self):
        """
        Returns the type of object

        Return
        ------
        mp_type : MPEnums
            Indicates what type of object the reference object is
        """

        return self._mp_type
    
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


class MPEnums(IntEnum):
    """
    Defines a class of enums used by MindPype

    The following enums are defined and available for use:

    .. list-table:: Object Type MPEnums - Leading '1'
        :widths: 50 50
        :header-rows: 1
    
        * - Enum
          - Value
        * - MPBase
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
        
    .. list-table:: Status Codes - Leading '2' -- DEPRECIATED
        :widths: 50 50
        :header-rows: 1

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

    """

    # Object Type Enums - Have a leading '1'

    MPBase        = 100
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
    
    
    # Parameter Directions - Leading '3'

    INPUT  = 300
    OUTPUT = 301
    INOUT  = 302
    
    # Kernel Initialization types - leading '4'

    INIT_FROM_NONE = 400
    INIT_FROM_DATA = 401
    INIT_FROM_COPY = 402
    
    def __str__(self):
        return self.name


class Session(MPBase):
    """
    Session objects contain all other MindPype objects instances within a data
    capture session.

    Examples
    --------
    >>> from mindpype.classes import session as S
    >>> S.session.create()

    """
    
    def __init__(self):
        """
        Constructor for Session class
        """
        super().__init__(MPEnums.SESSION,self)
        
        # define some private attributes
        # self._blocks = [] # queue of blocks to execute
        self._datum = {}
        self._misc_objs = {}
        self._ext_srcs = {}
        self._verified = False
        self._initialized = False
        self._ext_out = {}
        self.graphs = {}

        # Configure logging for the session
        #logging.basicConfig(filename='../mindpype/logs/log.log', filemode='a', format='%(asctime)s: [%(pathname)s:%(funcName)s:%(lineno)d] - [%(levelname)s] - %(message)s ', level=logging.INFO, encoding='utf-8')
        #logging.info("Session created")

    def verify(self):
        """
        Ensure all graphs are valid.
        Execute this method prior data collection to mitigate potential
        crashes due to invalid processing graph construction.
        """
        print("Verifying session...")
        self._verified = False

        for i_g, g_id in enumerate(self.graphs):
            print("\tVerifying graph {} of {}".format(i_g+1,len(self.graphs)))
            try:
                self.graphs[g_id].verify()
            except Exception as e:
                raise type(e)(f"{str(e)}\n\tGraph {i_g} failed verification. Please check graph definition").with_traceback(sys.exc_info()[2])
            
        self._verified = True
        self.free_temp_data()

    def initialize(self):
        """
        Initialize all graphs in the session
        """
        print("Initializing graphs...")
        for i_g, g_id in enumerate(self.graphs):
            print("\tInitializing graph {} of {}".format(i_g+1,len(self.graphs)))
            try:
                self.graphs[g_id].initialize()
            except Exception as e:
                raise type(e)(f"{str(e)}\n\tGraph {i_g} failed initialization. Please check graph definition").with_traceback(sys.exc_info()[2])
            
        self._initialized = True
        self.free_temp_data()
        
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
        graph : Graph
            Graph to execute for the current trial.
        """
        print("Executing trial with label: {}".format(label))
        self.poll_volatile_channels(label)
        graph.execute(label)
        self.push_volatile_outputs(label)
        #logging.info("Trial executed successfully")
        
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
        
    def add_misc_mp_obj(self,obj):
        """
        Add a misc MindPype object to the session

        Parameters
        ----------
        any MindPype object : MPBase
            MindPype object to add
        
        Examples
        --------
        >>> session.add_misc_mp_obj(obj)
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
        Search for and return a MindPype object within the session with a
        specific ID number

        Parameters
        ----------
        id_num : int
            ID number of the object to find

        Returns
        -------
        MindPype object : MPBase
            MindPype object with the specified ID number

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
    

    def save_session(self: object, file: str, additional_params=None) -> None:
        """
        Save the session object and all of its contents to a file

        Parameters
        ----------
        file : str
            File path to save the session to
        additional_params : dict, optional
            Additional parameters to save with the session. The default is None.
        Returns
        -------
        Pickle object : Pickle
            Pickle object containing the session and all of its contents

        """
        new_object = copy.deepcopy(self)
        output = additional_params if additional_params else {}
        
        output['pipeline'] = new_object

        sfile = open(file, 'wb')
        pickle.dump(output, sfile)
        sfile.close()

        return output
    
    def free_unreferenced_data(self):
        """
        Free all data objects that are no longer in use
        """
        keys_to_remove = []
        for d in self._datum:
            if sys.getrefcount(self._datum[d]) == 2:
                keys_to_remove.append(d)

        for d in keys_to_remove:
            self._datum.pop(d)
        
    @classmethod
    def create(cls):
        """
        Create a new session object

        Returns
        -------
        New session object : Session

        """
        return cls()
