# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 09:58:54 2019

BcipEnums - Define a class of enums used by BCIP

@author: ivanovn
"""

from enum import Enum

class BcipEnums(Enum):
    # Object Type Enums - Have a leading '1'
    BCIP    = 100
    SESSION = 101
    BLOCK   = 102
    NODE    = 103
    KERNEL  = 104
    PARAMETER = 105
    TENSOR  = 106
    SCALAR  = 107
    ARRAY   = 108
    
    # Status Codes - Leading '2'
    SUCCESS = 200
    FAILURE = 201
    INVALID_BLOCK = 202
    INVALID_NODE  = 203
    INVALID_PARAMETERS = 204
    EXCEED_TRIAL_LIMIT = 205
    
    # Parameter Directions - Leading '3'
    INPUT  = 300
    OUTPUT = 301
    INOUT  = 302
    
    def __str__(self):
        return self.name
    
    