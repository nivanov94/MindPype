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
    TENSOR  = 104
    SCALAR  = 105
    ARRAY   = 106
    
    # Status Codes - Leading '2'
    SUCCESS = 200
    FAILURE = 201
    INVALID_BLOCK = 202
    INVALID_NODE  = 203
    
    
    