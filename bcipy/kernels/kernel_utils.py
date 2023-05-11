from ..core import BcipEnums
import numpy as np

def extract_nested_data(bcip_obj):
    """
    Recursively extract Tensor data within a BCIP array or array-of-arrays
    """
    if (bcip_obj._bcip_type != BcipEnums.ARRAY and 
        bcip_obj._bcip_type != BcipEnums.CIRCLE_BUFFER):
        return np.array(())
    
    X = np.array(())
    if bcip_obj._bcip_type == BcipEnums.ARRAY:    
        num_elements = bcip_obj.capacity
    else:
        num_elements = bcip_obj.num_elements

    
    for i in range(num_elements):
        
        if bcip_obj._bcip_type == BcipEnums.ARRAY:
            e = bcip_obj.get_element(i)
        else:
            e = bcip_obj.get_queued_element(i)
            
            
        if e._bcip_type == BcipEnums.TENSOR or e._bcip_type == BcipEnums.SCALAR:
            if e._bcip_type == BcipEnums.SCALAR:
                elem_data = np.asarray((e.data,))
            else:
                elem_data = e.data
                
            elem_data = np.expand_dims(elem_data,0) # add dimension so we can append below
        else:
            elem_data = extract_nested_data(e)
        
        if X.shape == (0,):
            X = elem_data
        else:
            X = np.append(X,elem_data,axis=0)
    
    return X