from ..core import MPEnums
import numpy as np

def extract_nested_data(mp_obj):
    """
    Recursively extract Tensor data within a MindPype array or array-of-arrays

    Parameters
    ----------
    mp_obj : Array or array-of-arrays
        The input to be extracted from
    
    Returns
    -------
    X : np array
        The extracted data as a numpy array
    """
    if (mp_obj.mp_type != MPEnums.ARRAY and 
        mp_obj.mp_type != MPEnums.CIRCLE_BUFFER):
        return np.array(())
    
    X = np.array(())
    if mp_obj.mp_type == MPEnums.ARRAY:    
        num_elements = mp_obj.capacity
    else:
        num_elements = mp_obj.num_elements

    
    for i in range(num_elements):
        
        if mp_obj.mp_type == MPEnums.ARRAY:
            e = mp_obj.get_element(i)
        else:
            e = mp_obj.get_queued_element(i)
            
            
        if e.mp_type == MPEnums.TENSOR or e.mp_type == MPEnums.SCALAR:
            if e.mp_type == MPEnums.SCALAR:
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

def extract_init_inputs(init_in):
    """
    Extracts the initialization parameters from a potentially nested data structure
    
    Parameters
    ----------
    init_in : Object
        The input to be extracted from

    Returns
    -------
    init_input_data : np array
        The initialization inputs as a numpy array
    """
    if init_in.mp_type == MPEnums.TENSOR: 
            init_input_data = init_in.data
    else:
        try:
            # extract the data from a potentially nested array of tensors
            init_input_data = extract_nested_data(init_in)
        except Exception as e:
            e.add_note("Unable to extract initialization data from input")
            raise
        
    return init_input_data
