"""
Created on Tue Dec 17 13:22:29 2019

@author: ivanovn
"""
import numpy as np
import json
import os
from classes.bcip import BCIP
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


# this is not ideal, but works for now
kernel_LUT = {
        "absolute"         : kernels.absolute.AbsoluteKernel,
        "addition"         : kernels.addition.AdditionKernel,
        "cdf"              : kernels.cdf.CDFKernel,
        "combine_pvalues"  : kernels.combine_pvalues.CombinePValuesKernel,
        "concatenation"    : kernels.concatenation.ConcatenationKernel,
        "covariance"       : kernels.covariance.CovarianceKernel,
        "cumulative_rmean" : kernels.cumulative_riemann_mean.CumulativeRiemannMeanKernel,
        "diffusion_map"    : kernels.diffusion_map.DiffusionMapKernel,
        "division"         : kernels.divsion.DivisionKernel,
        "enqueue"          : kernels.enqueue.EnqueueKernel,
        "equal"            : kernels.equal.EqualKernel,
        "extract"          : kernels.extract.ExtractKernel,
        "filter"           : kernels.filter_.FilterKernel,
        "filtfilt"         : kernels.filtfilt.FiltFiltKernel,
        "greater"          : kernels.greater.GreaterKernel,
        "less"             : kernels.less.LessKernel,
        "logical_and"      : kernels.logical_and.AndKernel,
        "logical_not"      : kernels.logical_not.NotKernel,
        "logical_or"       : kernels.logical_or.OrKernel,
        "logical_xor"      : kernels.logical_xor.XorKernel,
        "max"              : kernels.max_.MaxKernel,
        "mean"             : kernels.mean.MeanKernel,
        "min"              : kernels.min_.MinKernel,
        "multiplication"   : kernels.multiplication.MultiplicationKernel,
        "reduced_sum"      : kernels.reduced_sum.ReducedSumKernel,
        "riemann_mdm"      : kernels.riemann_mdm_classifier_kernel.RiemannMDMClassifierKernel,
        "riemann_rLDA"     : kernels.riemann_ts_rLDA_classifier.RiemannTangentSpacerLDAClassifierKernel,
        "riemann_distance" : kernels.riemann_distance.RiemannDistanceKernel,
        "riemann_mean"     : kernels.riemann_mean.RiemannMeanKernel,
        "riemann_potato"   : kernels.riemann_potato.RiemannPotatoKernel,
        "set"              : kernels.set_data.SetKernel,
        "stack"            : kernels.stack.StackKernel,
        "std"              : kernels.std.StdKernel,
        "subtraction"      : kernels.subtraction.SubtractionKernel,
        "threshold"        : kernels.threshold.ThresholdKernel,
        "transpose"        : kernels.transpose.TransposeKernel,
        "zscore"           : kernels.zscore.ZScoreKernel
    }

def _repl_nulls(l):
    '''
    find and replace all strings containing the word null in a list and with
    the Python None type
    '''
    while 'null' in l:
        i = l.index('null')
        l[i] = None

def _create_session(create_method_name, api_params):
    """
    Create a session obj given some creation attributes
    """
    return Session.create()

def _create_block(session, create_method_name, api_params, other_attrs):
    """
    Create a block obj given some creation attributes
    """
    n_classes = api_params[0]
    n_class_trials = api_params[1]
    
    b = Block.create(session,n_classes,n_class_trials)

    # add the preprocessing, trial_processing, and postprocessing
    for graph in ['pre','trial_processing','post']:
        for node in other_attrs[graph]:
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
            for param in node['api_params']:
                if isinstance(param,str) and param[:3] == "id_":
                    # lookup the id within the session and add
                    obj = session.find_obj(int(param[3:]))
                    if obj == None:
                        return
                    create_params.append(obj)
                else:
                    # no lookup, add as literal
                    create_params.append(param)
            
            # create the node
            create_method(*create_params)

    return b

def _create_obj(session, obj, create_method_name, api_params):
    """
    Create a BCIP object according to the create_attrs
    """
    
    if hasattr(obj,create_method_name):
        create_method = getattr(obj,create_method_name)
        if not callable(create_method):
            return
    else:
        return
    
    # look up any parameters
    create_params = [session]
    

    for param in api_params:
        if isinstance(param,str) and param[:3] == "id_":
            create_params.append(session.find_obj(int(param[3:])))
        else:
            create_params.append(param)
            
            # hack to create 1D tensors, 1D tuples are lost in JSON transmission
            if obj == Tensor and param == 1:
                create_params[-1] = (1,)
    
    print("Creating {} object...".format(obj.__name__))
    return create_method(*create_params)

def _extract_nested_data(bcip_obj):
    """
    Recursively extract Tensor data within a BCIP array or array-of-arrays
    """
    if not isinstance(bcip_obj,Array):
        return np.array(())
    
    X = np.array(())
    for i in range(bcip_obj.capacity):
        e = bcip_obj.get_element(i)
        if isinstance(e,Tensor):
            elem_data = np.expand_dims(e.data,0) # add dimension so we can append below
        elif isinstance(e,Scalar):
            elem_data = np.expand_dims(np.array(e.data),0)
        else:
            elem_data = _extract_nested_data(e)
        
        if X.shape == (0,):
            X = elem_data
        else:
            X = np.append(X,elem_data,axis=0)
    
    return X

def _save_obj_data(obj,params):
    # extract the new data to save
    if isinstance(obj,Tensor):
        new_data = obj.data
    elif isinstance(obj,Scalar):
        new_data = np.array(obj.data)
    elif isinstance(obj,Array):
        new_data = _extract_nested_data(obj)
    else:
        return BcipEnums.FAILURE
    
    new_data_rank = len(new_data.shape)
    
    f = params['filename']
    p = params['path']
    if not os.path.exists(p):
        return BcipEnums.FAILURE
    
    if os.path.isfile(os.path.join(p,f)):
        # read the file and then append the new data
        with np.load(os.path.join(p,f)) as data:
            prev_data = data['arr_0']
            prev_data_rank = len(prev_data.shape)
            
            if new_data_rank == prev_data_rank:
                file_data = np.append(np.expand_dims(prev_data,0),
                                      np.expand_dims(new_data,0),
                                      axis=0)
            elif prev_data_rank == (new_data_rank+1):
                file_data = np.append(prev_data,
                                      np.expand_dims(new_data,0),
                                      axis=0)
            else:
                return BcipEnums.FAILURE
    else:
        # create the file with the new data
        file_data = new_data
    
    np.savez_compressed(os.path.join(p,f),file_data)
    return BcipEnums.SUCCESS

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
        # extract the creation method
        creation_method = 'create' # default
        if 'creation_method' in bcip_ext_req['params']:
            creation_method = bcip_ext_req['params']['creation_method']
        
        # extract the api_params
        api_params = []
        if 'api_params' in bcip_ext_req['params']:
            api_params = bcip_ext_req['params']['api_params']
            if 'null' in api_params:
                _repl_nulls(api_params)
        
        
        # determine the object to create and call the creation function
        if bcip_ext_req['params']['obj'] == 'session':
            s = _create_session(creation_method, api_params)
            if s == None:
                return_packet['sts'] = BcipEnums.FAILURE
            else:
                interface_sess_id = max(sess_hash.keys())+1 if len(sess_hash) != 0 else 0
                return_packet['id'] = s.session_id
                return_packet['hash_id'] = interface_sess_id
                sess_hash[interface_sess_id] = s
        elif bcip_ext_req['params']['obj'] == 'block':
            other_attrs = {}
            if 'other_attrs' in bcip_ext_req['params']:
                other_attrs = bcip_ext_req['params']['other_attrs']
            b = _create_block(sess_hash[sess_id], creation_method, api_params, other_attrs)
            if b == None:
                return_packet['sts'] = BcipEnums.FAILURE
            else:
                return_packet['id'] = b.session_id
        elif bcip_ext_req['params']['obj'] == 'tensor':
            t = _create_obj(sess_hash[sess_id],Tensor, creation_method, api_params)
            if t == None:
                return_packet['sts'] = BcipEnums.FAILURE
            else:
                return_packet['id'] = t.session_id
        elif bcip_ext_req['params']['obj'] == 'scalar':
            s = _create_obj(sess_hash[sess_id],Scalar, creation_method, api_params)
            if s == None:
                return_packet['sts'] = BcipEnums.FAILURE
            else:
                return_packet['id'] = s.session_id
        elif bcip_ext_req['params']['obj'] == 'array':
            a = _create_obj(sess_hash[sess_id],Array, creation_method, api_params)
            if a == None:
                return_packet['sts'] = BcipEnums.FAILURE
            else:
                return_packet['id'] = a.session_id
        elif bcip_ext_req['params']['obj'] == 'circle_buffer':
            cb = _create_obj(sess_hash[sess_id],CircleBuffer, creation_method, api_params)
            if cb == None:
                return_packet['sts'] = BcipEnums.FAILURE
            else:
                return_packet['id'] = cb.session_id
        elif bcip_ext_req['params']['obj'] == 'filter':
            f = _create_obj(sess_hash[sess_id],Filter, creation_method, api_params)
            if f == None:
                return_packet['sts'] = BcipEnums.FAILURE
            else:
                return_packet['id'] = f.session_id
        elif bcip_ext_req['params']['obj'] == 'lsl_stream':
            lsl = _create_obj(sess_hash[sess_id],LSLStream, creation_method, api_params)
            if lsl == None:
                return_packet['sts'] = BcipEnums.FAILURE
            else:
                return_packet['id'] = lsl.session_id
        elif bcip_ext_req['params']['obj'] == 'mat_file':
            f = _create_obj(sess_hash[sess_id],BcipMatFile, creation_method, api_params)
            if f == None:
                return_packet['sts'] = BcipEnums.FAILURE
            else:
                return_packet['id'] = f.session_id
                
    elif bcip_ext_req['cmd_type'] == 'execute':
        session = sess_hash[sess_id]
        obj = session.find_obj(bcip_ext_req['params']['obj']) \
                            if 'obj' in bcip_ext_req['params'] else session
        
        execution_method_name = bcip_ext_req['params']['method']

        if hasattr(obj,execution_method_name):
            execution_method = getattr(obj,execution_method_name)
            if not callable(execution_method):
                return_packet['sts'] = BcipEnums.FAILURE
                return json.dumps(return_packet)
        else:
            return_packet['sts'] = BcipEnums.FAILURE
            return json.dumps(return_packet)
        
        
        api_params = []
        if 'api_params' in bcip_ext_req['params']:
            api_params = bcip_ext_req['params']['api_params']
            if 'null' in api_params:
                _repl_nulls(api_params)
            
        exe_params = []
        for param in api_params:
            if isinstance(param,str) and param[:3] == "id_":
                exe_params.append(session.find_obj(int(param[3:])))
            else:
                exe_params.append(param)

        method_return = execution_method(*exe_params)
        
        if isinstance(method_return,BcipEnums):
            return_packet['sts'] = method_return
        elif isinstance(method_return,BCIP):
            return_packet['sts'] = BcipEnums.SUCCESS
            return_packet['obj'] = method_return.session_id
            return_packet['type'] = str(type(method_return))
        else:
            return_packet['sts'] = BcipEnums.FAILURE
        
    elif bcip_ext_req['cmd_type'] == 'req_data':
        session = sess_hash[sess_id]
        obj = session.find_obj(bcip_ext_req['params']['obj']) \
                            if 'obj' in bcip_ext_req['params'] else session
        
        prop = bcip_ext_req['params']['property']
        
        if hasattr(obj,prop):
            data = getattr(obj,prop)
            if isinstance(data,np.ndarray):
                data = data.tolist()
            
            if isinstance(data,BCIP):
                return_packet['sts'] = BcipEnums.SUCCESS
                return_packet['data'] = data.session_id
                return_packet['type'] = str(type(data))
            else:
                return_packet['data'] = data
                return_packet['sts'] = BcipEnums.SUCCESS
        else:
            return_packet['sts'] = BcipEnums.FAILURE
            
    elif bcip_ext_req['cmd_type'] == 'save_data':
        session = sess_hash[sess_id]
        obj = session.find_obj(bcip_ext_req['params']['obj']) \
                            if 'obj' in bcip_ext_req['params'] else session
        
        try:
            return_packet['sts'] = _save_obj_data(obj,bcip_ext_req['params'])
        except:
            return_packet['sts'] = BcipEnums.FAILURE
    
    return json.dumps(return_packet)
            
            
            
