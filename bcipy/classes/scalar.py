# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 15:47:58 2019

Scalar.py - Define the Scalar class for BCIP

@author: ivanovn
"""

from .bcip import BCIP
from .bcip_enums import BcipEnums

import numpy as np

class Scalar(BCIP):

    """
    BCIP Data type defining scalar-type data. The valid data types are int, float, complex, str, and bool.

    Parameters
    ----------
    sess : Session Object
        - Session where the Scalar object will exist

    value_type : one of [int, float, complex, str, bool]
        - Indicates the type of data represented by the Scalar

    val : value of type int, float, complex, str, or bool
        - Data value represented by the Scalar object

    is_virtual : bool
        - If true, the Scalar object is virtual, non-virtual otherwise

    ext_src : LSL data source input object, MAT data source, or None
        - External data source represented by the scalar; this data will be polled/updated when trials are executed.
        - If the data does not represent an external data source, set ext_src to None

    Attributes
    ----------
    _data_type : one of [int, float, complex, str, bool]
        - Indicates the type of data represented by the Scalar

    data : value of type int, float, complex, str, or bool
        - Data value represented by the Scalar object

    is_virtual : bool
        - If true, the Scalar object is virtual, non-virtual otherwise

    _ext_src : LSL data source input object, MAT data source, or None
        - External data source represented by the scalar; this data will be polled/updated when trials are executed.
        - If the data does not represent an external data source, set ext_src to None

    _volatile : bool
        - True if source is volatile (needs to be updated/polled between trials), false otherwise

    Examples
    --------
    >>> example_scalar = Scalar.create_from_value(example_session, 5)

    """

    _valid_types = [int, float, complex, str, bool]
    
    def __init__(self,sess,value_type,val,is_virtual,ext_src):
        super().__init__(BcipEnums.SCALAR,sess)
        self._data_type = value_type

        self._ext_src = ext_src
        
        if val == None:
            if value_type == int:
                val = 0
            elif value_type == float:
                val = 0.0
            elif value_type == complex:
                val = complex(0,0)
            elif value_type == str:
                val = ""
            elif value_type == bool:
                val = False
        
        
        self.data = val
        
        self._virtual = is_virtual        
        if ext_src is None:
            self._volatile = False
        else:
            self._volatile = True
            
    # API Getters
    @property
    def volatile(self):
        return self._volatile
    
    @property
    def virtual(self):
        return self._virtual
    
    @property
    def data(self):
        return self._data
    
    @property
    def data_type(self):
        return self._data_type
    
    @property
    def ext_src(self):
        return self._ext_src
    
    
    # API Setters
    @data.setter
    def data(self,data):
        """
        Set the referenced Scalar's data field to the specified data

        Parameters
        ----------
        data : Python built-in data or numpy array
            - Data to be represented by the Scalar object
            - Data must be scalar (1x1 numerical, single string/bool, etc)

        Return
        ------
        None

        Examples
        --------
        empty_scalar.data(5)
        """

        # if the data passed in is a numpy array, check if its a single value
        if type(data).__module__ == np.__name__:    
            if type(data) == np.ndarray and data.shape == (1,):
                # convert from the np type to native python type
                data = data[0]
        
            if isinstance(data, np.integer):
                data = int(data)
            elif isinstance(data, np.float):
                data = float(data)
            elif isinstance(data,np.complex):
                data = complex(data)
        
        if type(data) == self.data_type:
            self._data = data
        else:
            raise ValueError(("BCIP Scalar contains data of type {}. Cannot" +\
                              " set data to type {}").format(self.data_type,
                                                             type(data))) 
    
    def make_copy(self):
        """
        Produce and return a deep copy of the scalar

        Parameters
        ----------
        None

        Return
        ------
        int
            - Deep copy of referenced parameter

        Examples
        --------
        >>> new_scalar = example_scalar.make_copy()
        >>> print(new_scalar.data)

            12
        """
        cpy = Scalar(self.session,
                     self.data_type,
                     self.data,
                     self.virtual,
                     self.ext_src)
        
        # add the copy to the session
        sess = self.session
        sess.add_data(cpy)
        
        return cpy
    
    def copy_to(self,dest_scalar):
        """
        Copy all the elements of the scalar to another scalar

        Parameters
        ----------
        dest_scalar : Scalar Object
            - Scalar object which will represent the copy of the referenced Scalar's elements

        Return
        ------
        None

        Examples
        --------
        example_scalar.copy_to(copy_of_example_scalar)
        """
        dest_scalar.data = self.data
        
        # for now, don't copy the type, virtual and ext_src attributes because these
        # should really be set during creation not later
    
    
    def poll_volatile_data(self,label=None):
        """
        Polling/Updating volatile data within the scalar object

        Parameters
        ----------
        Label : int, default = None
            - Class label corresponding to class data to poll. This is required for epoched data but should be set to None for LSL data

        Return
        ------
        BCIP Status Code

        Examples
        --------
        >>> status = example_scalar.poll_data()
        >>> print(status)

            SUCCESS
        """
        
        # check if the data is actually volatile, if not just return
        if not self.is_voltatile:
            return BcipEnums.SUCCESS
        
        self.data = self.ext_src.poll_data(label)
        
        return BcipEnums.SUCCESS
        
    
    @classmethod
    def valid_numeric_types(cls):
        """
        Valid numeric types for a BCIP Scalar object
        
        Return
        ------
        [int, float, complex]
        """
        return [int,float,complex]
    
    # Factory Methods
    @classmethod
    def create(cls,sess,data_type):
        """
        Initialize a non-virtual, non-volatile Scalar object with an empty data field and add it to the session

        Parameters
        ----------
        sess : Session Object
            - Session where the Scalar object will exist

        data_type : Python built-in data type (int, float, complex, str, or bool)
            - Data type of data represented by Scalar object

        Return
        ------
        BCIP Scalar object

        Examples
        --------


        """
        if isinstance(data_type,str):
            dtypes = {'int':int, 'float':float,'complex':complex,
                      'str':str,'bool':bool}
            if data_type in dtypes:
                data_type = dtypes[data_type]
        
        if not (data_type in Scalar._valid_types):
            return
        s = cls(sess,data_type,None,False,None)
        
        sess.add_data(s)
        return s
    
    @classmethod
    def create_virtual(cls,sess,data_type):

        """
        Initialize a virtual, non-volatile Scalar object with an empty data field and add it to the session

        Parameters
        ----------
        sess : Session Object
            - Session where the Scalar object will exist

        data_type : Python built-in data type (int, float, complex, str, or bool)
            - Data type of data represented by Scalar object

        Return
        ------
        BCIP Scalar object

        Examples
        --------
        """

        if isinstance(data_type,str):
            dtypes = {'int':int, 'float':float,'complex':complex,
                      'str':str,'bool':bool}
            if data_type in dtypes:
                data_type = dtypes[data_type]
        
        if not (data_type in Scalar._valid_types):
            return
        
        if not (data_type in Scalar._valid_types):
            return
        s = cls(sess,data_type,None,True,None)
        
        # add the scalar to the session
        sess.add_data(s)
        return s
    
    @classmethod
    def create_from_value(cls,sess,value):

        """
        Initialize a non-virtual, non-volatile Scalar object with specified data and add it to the session

        Parameters
        ----------
        sess : Session Object
            - Session where the Scalar object will exist

        value : Value of type int, float, complex, str, or bool
            - Data represented by Scalar object

        Return
        ------
        BCIP Scalar object

        Examples
        --------
        """

        data_type = type(value)
        if not (data_type in Scalar._valid_types):
            return
        
        s = cls(sess,data_type,value,False,None)
        
        # add the scalar to the session
        sess.add_data(s)
        return s
    
    @classmethod
    def create_from_handle(cls,sess,data_type,src):

        """
        Initialize a non-virtual, volatile Scalar object with an empty data field and add it to the session

        Parameters
        ----------
        sess : Session Object
            - Session where the Scalar object will exist

        data_type : Python built-in data type (int, float, complex, str, or bool)
            - Data type of data represented by Scalar object

        src : Data Source object
            - Data source object (LSL, continuousMat, or epochedMat) from which to poll data

        Return
        ------
        BCIP Scalar object

        Examples
        --------
        """

        if not (data_type in Scalar._valid_types):
            return
        s = cls(sess,data_type,None,False,src)
        
        # add the scalar to the session
        sess.add_data(s)
        return s
    