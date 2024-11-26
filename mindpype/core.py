from enum import IntEnum
import sys
import pickle
import copy


class MPBase(object):
    """
    Base class for all objects used in the MindPype API.

    This class provides foundational attributes and behavior that are 
    shared across all MindPype objects. It serves as the parent class 
    for other objects in the MindPype framework, ensuring consistency 
    and providing a common interface for session management.

    Parameters
    ----------
    mp_type : MPEnums
        The type of the object, represented as an enumeration value 
        from `MPEnums`. This indicates the category or role of the object 
        within the MindPype framework.
    session : Session
        The session object to which this instance belongs. Each MindPype 
        object must be associated with a session to enable proper management 
        and interaction within the API.

    Attributes
    ----------
    mp_type : MPEnums
        The type of the object, indicating its role or function within the 
        MindPype system.
    session_id : int
        A unique identifier for the object within the session. This ID is 
        automatically generated using Python's `id()` function.
    session : Session
        The session instance that owns this object, representing the context 
        in which the object operates.
    """
    def __init__(self, mp_type, session):
        """Init"""
        self.mp_type = mp_type
        self.session_id = id(self)
        self.session = session

        
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
    Session objects are the top-level entities in the MindPype API.

    A `Session` serves as a container for all other MindPype objects, including
    data, graphs, miscellaneous objects, and external sources.

    Methods
    -------
    add_to_session(mp_obj)
        Add a MindPype object to the session, categorizing it based on its type.
    find_obj(id_num)
        Search for and return a MindPype object within the session by its 
        unique ID number.
    free_unreferenced_data()
        Free all data objects within the session that are no longer in use.
    create()
        Create and return a new `Session` instance.
    """

    def __init__(self):
        """ Init."""
        super().__init__(MPEnums.SESSION, self)

        # define private attributes
        self._data = {}  # container objects, eg. tensors, scalars, etc.
        self._misc_objs = {}  # miscellaneous objects, eg. filters, classifiers, etc.
        self._ext_srcs = {}  # external sources, eg. LSL streams
        self._ext_out = {}  # external outputs, eg. LSL streams
        self._graphs = {}  # graph objects


    def add_to_session(self, mp_obj):
        """
        Add a MindPype object to the session.

        This method adds a `MindPype` object to the appropriate category within 
        the session based on its type (`mp_type`). The object must inherit 
        from the `MPBase` class to be compatible with the session. If the object 
        type is invalid or unsupported, a `ValueError` is raised.

        Parameters
        ----------
        mp_obj : MPBase
            A MindPype object to add to the session. This object must inherit 
            from `MPBase` and have a valid `mp_type` attribute.

        Raises
        ------
        ValueError
            If the object does not inherit from `MPBase` or its `mp_type` is not 
            supported for addition to the session.
        """

        if not issubclass(type(mp_obj), MPBase):
            raise ValueError(
                f"Objects of type {type(mp_obj)} cannot be added to the "
                "session. Only MindPype objects inheriting from MPBase can be "
                "added to the session."
            )
        
        if mp_obj.mp_type == MPEnums.GRAPH:
            self._graphs[mp_obj.session_id] = mp_obj

        elif 0 < (mp_obj.mp_type - MPEnums.CONTAINER) < mp_obj.mp_type.section_width:
            self._data[mp_obj.session_id] = mp_obj

        elif 0 < (mp_obj.mp_type - MPEnums.MISC) < mp_obj.mp_type.section_width:
            self._misc_objs[mp_obj.session_id] = mp_obj

        elif 0 < (mp_obj.mp_type - MPEnums.EXTERNAL_CONNECTION) < mp_obj.mp_type.section_width:
            self._ext_srcs[mp_obj.session_id] = mp_obj

        else:
            raise ValueError(
                f"Objects of type {mp_obj.mp_type} cannot be "
                "directly added to the session."
            )


    def find_obj(self, id_num):
        """
        Search for and return a MindPype object within the session by its
        unique ID number

        Parameters
        ----------
        id_num : int
            ID number of the object to find.

        Returns
        -------
        MPBase or None
            The `MindPype` object with the specified ID number, or `None` if
            no object with that ID number is found.
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

    def free_unreferenced_data(self):
        """
        Free all data objects within the session that are no longer in use.

        This method iterates through the data stored within the session 
        and identifies objects that are no longer referenced by 
        anything outside of the session (determined by a reference count of 2). 
        These objects are then removed from the session, freeing up memory.

        Notes
        -----
        The reference count of 2 indicates that the object is referenced only by 
        the `_data` dictionary itself and the `getrefcount` call. Therefore, it 
        is safe to assume that such objects are no longer in use elsewhere.

        References are managed internally using `sys.getrefcount`.

        Examples
        --------
        >>> session = Session()
        >>> some_scalar = Scalar.create_from_value(session, 5)
        >>> scalar_id = some_scalar.session_id
        >>> session.find_obj(scalar_id) is None
        False
        >>> # do something with some_scalar ...
        >>> del some_scalar
        >>> session.free_unreferenced_data()
        >>> session.find_obj(scalar_id) is None
        True
        """
        # find all data objects that are no longer referenced
        keys_to_remove = []
        for d in self._data:
            if sys.getrefcount(self._data[d]) == 2:
                keys_to_remove.append(d)

        # remove them from the session so they will be garbage collected
        for d in keys_to_remove:
            self._data.pop(d)

    @classmethod
    def create(cls):
        """
        Factory method to create a new `Session` instance.

        Returns
        -------
        Session
            A reference to the new Session instance.
        """
        return cls()
