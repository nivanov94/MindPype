"""
Created on Tue Nov 19 16:08:19 2019

Array.py - Defines class of array objects that contain other BCIP objects

@author: ivanovn
"""

from .bcip import BCIP
from .bcip_enums import BcipEnums
from .tensor import Tensor
from .scalar import Scalar

import numpy as np

class Array(BCIP):
    """
    Array containing instances of other BCIP classes. Each array can only hold one type of BCIP class.

    Parameters
    ----------
    sess : Session object
        - Session where the Array object will exist
    
    capacity : int
        - Maximum number of elements to be stored within the array (for allocation purposes)
    
    element_template : any
        - The template BCIP element to populate the array (see examples)

    Attributes
    ----------

    Examples
    --------
    - Creating An Array
        example = Array.create(example_session, example_capacity, Tensor.create(example_session, input_data.shape))
    
    Return
    ======
    Array Object
    
    Notes
    -----
    - A single array object should only contain one BCIP/data object type.

    
    """
    
    def __init__(self,sess,capacity,element_template):
        super().__init__(BcipEnums.ARRAY,sess)
        
        self._virtual = False # no virtual arrays for now
        self._volatile = False # no volatile arrays for now...
        
        self._capacity = capacity
        
        self._elements = [None] * capacity
        
        for i in range(capacity):
            self._elements[i] = element_template.make_copy()
        
    
    # Returns an element at a particular index
    def get_element(self,index):
        """
        Returns the element at a specific index within an array object.

        Parameters
        ----------
        index : int
            - Index is the position within the array with the element will be returned. Index should be 0 <= Index < Capacity

        Return
        ------
        any : Data object at index index 
        
        Examples
        --------
        example_element = example_array.get_element(0)


        """
        if index >= self.capacity or index < 0:
            return
        
        return self._elements[index]
    
    # Changes the element at a particular index to a specified value
    def set_element(self,index,element):

        """
        Changes the element at a particular index to a specified value

        Parameters
        ----------
        index : int
            - Index in the array where the element will changed. 0 <= Index < capacity

        element : any
            - specified value which will be set at index index

        Examples
        --------
        >>> example_array.set_element(0, 12) # changes 0th element to 12 
        >>> print(example_array.get_element(0), example_array.get_element(1))
                (12, 5)
            

        Notes
        -----
        element must be the same type as the other elements within the array. 
        """

        if index >= self.capacity or index < 0:
            return BcipEnums.FAILURE
        
        element.copy_to(self._elements[index])
        return BcipEnums.SUCCESS
    
    # User Facing Getters
    @property
    def capacity(self):
        return self._capacity
    
    @property
    def num_elements(self):
        # this property is included to allow for seamless abstraction with 
        # circle buffer property
        return self.capacity
    
    @property
    def virtual(self):
        return self._virtual
    
    @property
    def volatile(self):
        return self._volatile
    
    @capacity.setter
    def capacity(self,capacity):
        if self.virtual:
            self._capacity = capacity
            self._elements = [None] * capacity
            
            
    def make_copy(self):
        """
        Create and return a deep copy of the array
        The copied array will maintain references to the same objects.
        If a copy of these is also desired, they will need to be copied
        separately.

        Parameters
        ----------
        None

        Examples
        --------
        new_array = old_array.make_copy()
        """
        cpy = Array(self.session,
                    self.capacity,
                    self.get_element(0))
        
        for e in range(self.capacity):
            cpy.set_element(e,self.get_element(e))
            
        # add the copy to the session
        self.session.add_data(cpy)
            
        return cpy
    
    def copy_to(self,dest_array):
        """
        Copy all the attributes of the array to another array. Note
        these will reference the same objects within the element list

        Parameters 
        ----------
        dest_array : Array object
            - Array object where the attributes with the referenced array will copied to

        Examples
        --------
        old_array.copy_to(copy_of_old_array) 

        """
        dest_array.capacity = self.capacity
        for i in range(self.capacity):
            sts = dest_array.set_element(i,self.get_element(i))
            
            if sts != BcipEnums.SUCCESS:
                return sts
        
        return BcipEnums.SUCCESS

    def to_tensor(self):
        """
        Stack the elements of the array into a Tensor object.
        """
        element = self.get_element(0)

        if not (element._bcip_type == BcipEnums.TENSOR or
                (element._bcip_type == BcipEnums.SCALAR and element.data_type in Scalar.valid_numeric_types())):
            return None

        # extract elements and stack into numpy array
        elements = [self.get_element(i).data for i in range(self.capacity)]
        stacked_elements = np.stack(elements)
        
        if element._bcip_type == BcipEnums.TENSOR:
            shape = (self.capacity,) + element.shape
        else:
            shape = (self.capacity,)

        # create tensor
        return Tensor.create_from_data(self._session,shape,stacked_elements)
    
    # API constructor
    @classmethod
    def create(cls,sess,capacity,element_template):

        """
        Factory method to create array object

         Parameters
        ----------
        sess : Session object
            - Session where the Array object will exist
        capacity : int
            - Maximum number of elements to be stored within the array (for allocation purposes)
        element_template : any
            - The template BCIP element to populate the array (see examples)

        """

        a = cls(sess,capacity,element_template)
        
        # add the array to the session
        sess.add_data(a)
        return a
