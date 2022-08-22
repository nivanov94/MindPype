# -*- coding: utf-8 -*-
"""
BCIP - Base class for all BCI PRISM lab API objects
"""

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

    """
    def __init__(self,bcip_type,session):
        self._bcip_type = bcip_type
        self._id  = id(self)
        self._session = session
    
    # API getters
    @property
    def bcip_type(self):
        return self._bcip_type
    
    @property
    def session_id(self):
        return self._id
    
    @property
    def session(self):
        return self._session
