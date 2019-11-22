# -*- coding: utf-8 -*-
"""
BCIP - Base class for all BCI PRISM lab API objects
"""

class BCIP(object):
    """
    This is the base class for all objects used in the BCIP API.
    It serves to define some attributes that will be shared across all
    other objects.
    """
    def __init__(self,type,session):
        self.type = type
        self.uid  = id(self)
        self.session = session
    
    def getID(self):
        return self.uid
    
    def getType(self):
        return self.type
    
    def getSession(self):
        return self.session
