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
    GRAPH   = 103
    NODE    = 104
    KERNEL  = 105
    PARAMETER = 106
    TENSOR  = 107
    SCALAR  = 108
    ARRAY   = 109
    FILTER  = 110
    
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
    
    