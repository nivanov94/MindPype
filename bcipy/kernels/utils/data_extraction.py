from ...classes.array import Array
from ...classes.tensor import Tensor


import numpy as np

def extract_nested_data(bcip_obj):
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
        else:
            elem_data = extract_nested_data(e)
        
        if X.shape == (0,):
            X = elem_data
        else:
            X = np.append(X,elem_data,axis=0)
    
    return X