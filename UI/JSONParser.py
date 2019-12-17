"""
Created on Tue Dec 17 13:22:29 2019

@author: ivanovn
"""

import json
from classes.bcip_enums import BcipEnums
from classes.session import Session
from classes.block import Block
from classes.tensor import Tensor
from classes.scalar import Scalar
from classes.array import Array
from classes.circle_buffer import CircleBuffer
from classes.filter import Filter
from classes.source import BcipMatFile, LSLStream

import kernels

def _create_session(create_attrs):
    """
    Create a session obj given some creation attributes
    """
    return Session.create()

def _create_block(session, create_attrs):
    """
    Create a block obj given some creation attributes
    """
    n_classes = create_attrs['n_classes']
    n_class_trials = create_attrs['n_class_trials']
    
    b = Block.create(session,n_classes,n_class_trials)

    # add the preprocessing nodes
    for node in create_attrs['pre']:
        if 

    return b

def parse(bcip_ext_req,sess_hash):
    """
    (dict,dict) -> dict
    
    Parse a dictionary outlining requests from an external client to
    execute on sessions running on the server.
    
    Returns a dictionary containing the return values from the API
    """
    
    for req_key in ('cmd_type','sess_id','params'):
        if not req_key in bcip_ext_req:
            return_packet = json.dumps({"sts" : BcipEnums.FAILURE})
            return return_packet
    
    if len(sess_hash) == 0 and bcip_ext_req['sess_id'] != "null":
        return_packet = json.dumps({"sts" : BcipEnums.FAILURE})
        return return_packet
    
    if bcip_ext_req['sess_id'] != 'null' and not bcip_ext_req['sess_id'] in sess_hash:
        return_packet = json.dumps({"sts" : BcipEnums.FAILURE})
        return return_packet
    
    return_packet = {'sts' : BcipEnums.SUCCESS}
    
    sess_id = bcip_ext_req['sess_id']
    
    if bcip_ext_req['cmd_type'] == 'create':
        create_attrs = bcip_ext_req['params']['attrs']
        if bcip_ext_req['params']['obj'] == 'session':
            s = _create_session(create_attrs)
            if s == None:
                return_packet['sts'] = BcipEnums.FAILURE
            else:
                return_packet['id'] = max(sess_hash.keys())+1 if len(sess_hash) != 0 else 0
        elif bcip_ext_req['params']['obj'] == 'block':
            b = _create_block(sess_hash[sess_id], create_attrs)
            if b == None:
                return_packet['sts'] = BcipEnums.FAILURE
            else:
                return_packet['id'] = b.session_id
            
            
            