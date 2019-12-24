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


# this is kinda gross, but works for now
kernel_LUT = {
        "addition"       : kernels.addition.AdditionKernel,
        "concatenation"  : kernels.concatenation.ConcatenationKernel,
        "covariance"     : kernels.covariance.CovarianceKernel,
        "diffusion_map"  : kernels.diffusion_map.DiffusionMapKernel,
        "division"       : kernels.divsion.DivisionKernel,
        "enqueue"        : kernels.enqueue.EnqueueKernel,
        "equal"          : kernels.equal.EqualKernel,
        "extract"        : kernels.extract.ExtractKernel,
        "filter"         : kernels.filter_.FilterKernel,
        "filtfilt"       : kernels.filtfilt.FiltFiltKernel,
        "greater"        : kernels.greater.GreaterKernel,
        "less"           : kernels.less.LessKernel,
        "logical_and"    : kernels.logical_and.AndKernel,
        "logical_not"    : kernels.logical_not.NotKernel,
        "logical_or"     : kernels.logical_or.OrKernel,
        "logical_xor"    : kernels.logical_xor.XorKernel,
        "multiplication" : kernels.multiplication.MultiplicationKernel,
        "reduced_sum"    : kernels.reduced_sum.ReducedSumKernel,
        "riemann_mdm"    : kernels.riemann_mdm_classifier_kernel.RiemannMDMClassifierKernel,
        "riemann_mean"   : kernels.riemann_mean.RiemannMeanKernel,
        "set"            : kernels.set_data.SetKernel,
        "stack"          : kernels.stack.StackKernel,
        "subtraction"    : kernels.subtraction.SubtractionKernel,
        "transpose"      : kernels.transpose.TransposeKernel
    }

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

    # add the preprocessing, trial_processing, and postprocessing
    for graph in ['pre','trial_processing','post']:
        for node in create_attrs[graph]:
            if not node['name'] in kernel_LUT:
                return
            kern = kernel_LUT[node['name']]
        
            if not 'creation_method' in node.keys():
                create_method_name = 'create'
            else:
                create_method_name = node['creation_method']
        
            if hasattr(kern,create_method_name):
                create_method = getattr(kern,create_method_name)
                if not callable(create_method):
                    return
            else:
                return
        
            if graph == 'pre':
                g = b.preprocessing_graph
            elif graph == 'trial_processing':
                g = b.trial_processing_graph
            else:
                g = b.postprocessing_graph
            
            create_params = [g]
        
            # iterate through the params, find the objects and call the creation method
            for param in node['params']:
                if isinstance(param,str) and param[:3] == "id_":
                    # lookup the id within the session and add
                    create_params.append(session.find_obj(int(param[3:])))
                else:
                    # no lookup, add as literal
                    create_params.append(param)
            
            # create the node
            create_method(*create_params)

    return b

def _create_obj(session,obj,create_attrs):
    """
    Create a BCIP object according to the create_attrs
    """
    create_method_name = create_attrs['creation_method'] if 'creation_method' \
                                                            in create_attrs \
                                                       else 'create'
    
    if hasattr(obj,create_method_name):
        create_method = getattr(obj,create_method_name)
        if not callable(create_method):
            return
    else:
        return
    
    # look up any parameters
    create_params = [session]
    
    if 'params' in create_attrs:
        for param in create_attrs['params']:
            if isinstance(param,str) and param[:3] == "id_":
                create_params.append(session.find_obj(int(param[3:])))
            else:
                create_params.append(param)
    
    return create_method(*create_params)
    

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
                interface_sess_id = max(sess_hash.keys())+1 if len(sess_hash) != 0 else 0
                return_packet['id'] = interface_sess_id
                sess_hash[interface_sess_id] = s
        elif bcip_ext_req['params']['obj'] == 'block':
            b = _create_block(sess_hash[sess_id], create_attrs)
            if b == None:
                return_packet['sts'] = BcipEnums.FAILURE
            else:
                return_packet['id'] = b.session_id
        elif bcip_ext_req['params']['obj'] == 'tensor':
            t = _create_obj(sess_hash[sess_id],Tensor,create_attrs)
            if t == None:
                return_packet['sts'] = BcipEnums.FAILURE
            else:
                return_packet['id'] = t.session_id
        elif bcip_ext_req['params']['obj'] == 'scalar':
            s = _create_obj(sess_hash[sess_id],Scalar,create_attrs)
            if s == None:
                return_packet['sts'] = BcipEnums.FAILURE
            else:
                return_packet['id'] = s.session_id
        elif bcip_ext_req['params']['obj'] == 'array':
            a = _create_obj(sess_hash[sess_id],Array,create_attrs)
            if a == None:
                return_packet['sts'] = BcipEnums.FAILURE
            else:
                return_packet['id'] = a.session_id
        elif bcip_ext_req['params']['obj'] == 'circle_buffer':
            cb = _create_obj(sess_hash[sess_id],CircleBuffer,create_attrs)
            if cb == None:
                return_packet['sts'] = BcipEnums.FAILURE
            else:
                return_packet['id'] = cb.session_id
        elif bcip_ext_req['params']['obj'] == 'filter':
            f = _create_obj(sess_hash[sess_id],Filter,create_attrs)
            if f == None:
                return_packet['sts'] = BcipEnums.FAILURE
            else:
                return_packet['id'] = f.session_id
        elif bcip_ext_req['params']['obj'] == 'lsl_stream':
            lsl = _create_obj(sess_hash[sess_id],LSLStream,create_attrs)
            if lsl == None:
                return_packet['sts'] = BcipEnums.FAILURE
            else:
                return_packet['id'] = lsl.session_id
        elif bcip_ext_req['params']['obj'] == 'mat_file':
            f = _create_obj(sess_hash[sess_id],BcipMatFile,create_attrs)
            if f == None:
                return_packet['sts'] = BcipEnums.FAILURE
            else:
                return_packet['id'] = f.session_id
    elif bcip_ext_req['cmd_type'] == 'execute':
        session = sess_hash[sess_id]
        obj = session.find_obj(bcip_ext_req['params']['obj']) \
                            if 'obj' in bcip_ext_req['params'] else session
        
        execution_method_name = bcip_ext_req['params']['func']
        
        if hasattr(obj,execution_method_name):
            execution_method = getattr(obj,execution_method_name)
            if not callable(execution_method):
                return_packet['sts'] = BcipEnums.FAILURE
        else:
            return_packet['sts'] = BcipEnums.FAILURE
        
        exe_attrs = bcip_ext_req['params']['attrs']
        exe_params = []
        for param in exe_attrs:
            if isinstance(param,str) and param[:3] == "id_":
                exe_params.append(session.find_obj(int(param[3:])))
            else:
                exe_params.append(param)
        
        return_packet['sts'] = execution_method(*exe_params)
    elif bcip_ext_req['cmd_type'] == 'req_data':
        session = sess_hash[sess_id]
        obj = session.find_obj(bcip_ext_req['params']['obj']) \
                            if 'obj' in bcip_ext_req['params'] else session
        
        prop = bcip_ext_req['params']['property']
        
        if hasattr(obj,prop):
            return_packet['data'] = getattr(obj,prop)
        else:
            return_packet['sts'] = BcipEnums.FAILURE
    
    return json.dumps(return_packet)
            
            
            