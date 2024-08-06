from enum import IntEnum
import sys
import pickle
import copy


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
    def __init__(self, mp_type, session):
        """
        Constructor for MPBase base class
        """
        self._mp_type = mp_type
        self._id = id(self)
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

    .. list-table:: Base Object Type MPEnums - Leading '1'
        :widths: 50 50
        :header-rows: 1

        * - Enum
          - Value
        * - MPBase
            - 10
        * - SESSION
            - 11
    
    .. list-table:: Graph Object Type Enums - Leading '2'
        :widths: 50 50
        :header-rows: 1
        * - Enum
            - Value
        * - GRAPH_ELEMENT
            - 20
        * - GRAPH
            - 21
        * - NODE
            - 22
        * - KERNEL
            - 23
        * - PARAMETER
            - 24
        
    .. list-table:: Data container Object Type Enums - Leading '3'
        :widths: 50 50
        :header-rows: 1
        * - Enum
            - Value
        * - CONTAINER
            - 30
        * - TENSOR
            - 31
        * - SCALAR
            - 32
        * - ARRAY
            - 33
        * - CIRCLE_BUFFER
            - 34
        
    .. list-table:: MISCELLANEOUS Object Type Enums - Leading '4'
        :widths: 50 50
        :header-rows: 1
        * - Enum
            - Value
        * - MISC
            - 40
        * - FILTER
            - 41
        * - CLASSIFIER
            - 42

    .. list-table:: External CONNECTIONS Object Type Enums - Leading '5'
        :widths: 50 50
        :header-rows: 1
        * - Enum
            - Value
        * - EXTERNAL_CONNECTION
            - 50
        * - SRC
            - 51
    
    .. list-table:: Parameter Directions - Leading '6'
        :widths: 50 50
        :header-rows: 1

        * - Enum
          - Value
        * - PARAMETER_DIRECTION
            - 60
        * - INPUT
            - 61
        * - OUTPUT
            - 62
        * - INOUT
            - 63

    .. list-table:: Kernel Initialization types - leading '7'
        :widths: 50 50
        :header-rows: 1

        * - Enum
          - Value
        * - INIT_TYPE
            - 70
        * - INIT_FROM_NONE
            - 71
        * - INIT_FROM_DATA
            - 72
        * - INIT_FROM_COPY
            - 73

    """

    # Base Object Type Enums - Have a leading '1'
    MPBase        = 10
    SESSION       = 11

    # Graph Object Type Enums - Leading '2'
    GRAPH_ELEMENT = 20
    GRAPH         = 21
    NODE          = 22
    KERNEL        = 23
    PARAMETER     = 24

    # Data container Object Type Enums - Leading '3'
    CONTAINER     = 30
    TENSOR        = 31
    SCALAR        = 32
    ARRAY         = 33
    CIRCLE_BUFFER = 34

    # MISCELLANEOUS Object Type Enums - Leading '4'
    MISC          = 40
    FILTER        = 41
    CLASSIFIER    = 42

    # External CONNECTIONS Object Type Enums - Leading '5'
    EXTERNAL_CONNECTION = 50
    SRC                 = 51

    # Parameter Directions - Leading '6'
    PARAMETER_DIRECTION = 60
    INPUT               = 61
    OUTPUT              = 62
    INOUT               = 63

    # Kernel Initialization types - leading '7'
    INIT_TYPE      = 70
    INIT_FROM_NONE = 71
    INIT_FROM_DATA = 72
    INIT_FROM_COPY = 73

    def __str__(self):
        return self.name
    
    @property
    def section_width(self):
        """
        Defines the width of the section of enums that
        the current enum belongs to
        """
        return 10


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
        super().__init__(MPEnums.SESSION, self)

        # define some private attributes
        # self._blocks = [] # queue of blocks to execute
        self._data = {}
        self._misc_objs = {}
        self._ext_srcs = {}
        self._verified = False
        self._initialized = False
        self._ext_out = {}
        self.graphs = {}


    def add_to_session(self, mp_obj):
        """
        Add a MindPype object to the session

        Parameters
        ----------
        mp_obj : MPBase
            MindPype object to add to the session
        """

        if not issubclass(type(mp_obj), MPBase):
            raise ValueError(f"Objects of type {type(mp_obj)} cannot be added to the session."
                             "Only MindPype objects inheriting from MPBase can be added to the session.")
        
        if mp_obj.mp_type == MPEnums.GRAPH:
            self.graphs[mp_obj.session_id] = mp_obj

        elif 0 < (mp_obj.mp_type - MPEnums.CONTAINER) < mp_obj.mp_type.section_width:
            self._data[mp_obj.session_id] = mp_obj

        elif 0 < (mp_obj.mp_type - MPEnums.MISC) < mp_obj.mp_type.section_width:
            self._misc_objs[mp_obj.session_id] = mp_obj

        elif 0 < (mp_obj.mp_type - MPEnums.EXTERNAL_CONNECTION) < mp_obj.mp_type.section_width:
            self._ext_srcs[mp_obj.session_id] = mp_obj

        else:
            raise ValueError(f"Objects of type {mp_obj.mp_type} cannot be directly added to the session.")


    def find_obj(self, id_num):
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
        if id_num in self._data:
            return self._data[id_num]

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
            Additional parameters to save with the session. The default
            is None.
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
        for d in self._data:
            if sys.getrefcount(self._data[d]) == 2:
                keys_to_remove.append(d)

        for d in keys_to_remove:
            self._data.pop(d)

    @classmethod
    def create(cls):
        """
        Create a new session object

        Returns
        -------
        New session object : Session

        """
        return cls()
