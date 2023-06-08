"""
Defines data container classes for BCIPy. These classes are used to represent data in the BCIPy framework.

@author: Nicolas Ivanov, Aaron Lio
"""

from .core import BCIP, BcipEnums

import numpy as np

class Scalar(BCIP):

    """
    BCIP Data type defining scalar-type data. The valid data types are int, float, complex, str, and bool.

    Parameters
    ----------
    sess : Session Object
        Session where the Scalar object will exist
    value_type : one of [int, float, complex, str, bool]
        Indicates the type of data represented by the Scalar
    val : value of type int, float, complex, str, or bool
        Data value represented by the Scalar object
    is_virtual : bool
        If true, the Scalar object is virtual, non-virtual otherwise
    ext_src : LSL data source input object, MAT data source, or None
        External data source represented by the scalar; this data will be polled/updated when trials are executed.
        If the data does not represent an external data source, set ext_src to None

    Attributes
    ----------
    _data_type : one of [int, float, complex, str, bool]
        Indicates the type of data represented by the Scalar
    data : value of type int, float, complex, str, or bool
        Data value represented by the Scalar object
    is_virtual : bool
        If true, the Scalar object is virtual, non-virtual otherwise
    _ext_src : LSL data source input object, MAT data source, or None
        External data source represented by the scalar; this data will be polled/updated when trials are executed.
        If the data does not represent an external data source, set ext_src to None
    _volatile : bool
        True if source is volatile (needs to be updated/polled between trials), false otherwise

    Examples
    --------
    >>> example_scalar = Scalar.create_from_value(example_session, 5)

    """

    _valid_types = [int, float, complex, str, bool]
    
    def __init__(self,sess,value_type,val,is_virtual,ext_src, ext_out=None):
        """
        Constructor for Scalar object
        """
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
        
        self._ext_out = ext_out
        self.data = val
        
        self._virtual = is_virtual        
        if ext_src is None:
            self._volatile = False
        else:
            self._volatile = True

        if ext_out is None:
            self._volatile_out = False
        else:
            self._volatile_out = True
            
    # API Getters
    @property
    def volatile(self):
        """
        Getter for volatile attribute

        Return
        ------
        bool
            True if source is volatile (needs to be updated/polled between trials), false otherwise
        """
        return self._volatile

    @property
    def volatile_out(self):
        """
        Getter for volatile_out attribute
        
        Return
        ------
        bool
            True if output is volatile (needs to be updated/pushed between trials), false otherwise
        """
        return self._volatile_out
    
    @property
    def virtual(self):
        """
        Getter for virtual attribute

        Return
        ------
        bool
            True if the Scalar object is virtual, non-virtual otherwise
        """
        return self._virtual
    
    @property
    def data(self):
        """
        Getter for scalar data attribute
        
        Return
        ------
        Data value represented by the Scalar object
        
        Return Type
        -----------
        int, float, complex, str, or bool
        """
        return self._data
    
    @property
    def data_type(self):
        """
        Getter for data_type attribute

        Return
        ------
        one of [int, float, complex, str, bool]
            Indicates the type of data represented by the Scalar
        
        Return Type
        -----------
        type
        """
        return self._data_type
    
    @property
    def ext_src(self):
        """
        Getter for ext_src attribute

        Return
        ------
        External data source represented by the scalar; this data will be polled/updated when trials are executed.
            If the data does not represent an external data source, ext_src is None

        Return Type
        -----------
        LSL data source input object, XDF data source, MAT data source, or None

        """

        return self._ext_src
    
    @property
    def ext_out(self):
        """
        Getter for ext_out attribute

        Return
        ------
        External data output represented by the scalar; this data will be pushed when trials are executed.
            If the data does not represent an external data source, ext_out is None

        Return Type
        -----------
        LSL data source output object, XDF data source, MAT data source, or None

        """
        return self._ext_out

    # API Setters
    @data.setter
    def data(self,data):
        """
        Set the referenced Scalar's data field to the specified data

        Parameters
        ----------
        data : Python built-in data or numpy array
            Data to be represented by the Scalar object
            Data must be scalar (1x1 numerical, single string/bool, etc)

        Return
        ------
        None

        Examples
        --------
        >>> empty_scalar.data = 5
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

        Return
        ------
        Deep copy of referenced parameter

        Return Type
        -----------
        BCIPy Scalar object

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
                     self.ext_src,
                     self.ext_out)
        
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
            Scalar object which will represent the copy of the referenced Scalar's elements

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
            Class label corresponding to class data to poll. This is required for epoched data but should be set to None for LSL data

        Return
        ------
        BcipEnums.SUCCESS if successful, BcipEnums.FAILURE otherwise

        Return Type
        -----------
        BcipEnums

        Examples
        --------
        >>> status = example_scalar.poll_data()
        >>> print(status)

            BcipEnums.SUCCESS
        """
        
        # check if the data is actually volatile, if not just return
        if not self.volatile:
            return BcipEnums.SUCCESS
        
        self.data = self.ext_src.poll_data(label)
        
        return BcipEnums.SUCCESS
        
    def push_volatile_outputs(self, label=None):

        if not self.volatile_out:
            return BcipEnums.SUCCESS
        
        self.ext_out.push_data(self.data, label)

        return BcipEnums.SUCCESS
    
    @classmethod
    def valid_numeric_types(cls):
        """
        Valid numeric types for a BCIP Scalar object
        
        Return
        ------
        [int, float, complex]
        """
        return [int,float,complex, str, bool]
    
    # Factory Methods
    @classmethod
    def create(cls,sess,data_type):
        """
        Initialize a non-virtual, non-volatile Scalar object with an empty data field and add it to the session

        Parameters
        ----------
        sess : Session Object
            Session where the Scalar object will exist
        data_type : Python built-in data type (int, float, complex, str, or bool)
            Data type of data represented by Scalar object

        Return
        ------
        BCIPy Scalar object

        Examples
        --------
        >>> new_scalar = Scalar.create(sess,int)
        >>> new_scalar.data = 5

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
            Session where the Scalar object will exist
        data_type : Python built-in data type (int, float, complex, str, or bool)
            Data type of data represented by Scalar object

        Return
        ------
        BCIPy Scalar object

        Examples
        --------
        >>> new_scalar = Scalar.create_virtual(sess,int)
        >>> new_scalar.data = 5
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
            Session where the Scalar object will exist
        value : Value of type int, float, complex, str, or bool
            Data represented by Scalar object

        Return
        ------
        BCIPy Scalar object

        Examples
        --------
        >>> new_scalar = Scalar.create_from_value(sess,5)
        >>> print(new_scalar.data)

            5
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
            Session where the Scalar object will exist
        data_type : Python built-in data type (int, float, complex, str, or bool)
            Data type of data represented by Scalar object
        src : Data Source object
            Data source object (LSL, continuousMat, or epochedMat) from which to poll data

        Return
        ------
        BCIP Scalar object

        Examples
        --------
        >>> new_scalar = Scalar.create_from_handle(sess,int,src)
        """

        if not (data_type in Scalar._valid_types):
            return
        s = cls(sess,data_type,None,False,src)
        
        # add the scalar to the session
        sess.add_data(s)
        return s




class Tensor(BCIP):
    """

    Tensor (or n-dimensional matrices), are defined by the tensor class. 
    BCIP tensors can either be volatile (are updated/change each trial, generally reserved for tensors containing current trial data), virtual (empty, dimensionless tensor object). Like scalars and array, tensors can be created from data, copied from a different variable, or created virtually, so they don’t initially contain a value. 
    Each of the scalars, tensors and array data containers also have an external source (_ext_src) attribute, which indicates, if necessary, the source from which the data is being pulled from. This is especially important if trial/training data is loaded into a tensor each trial from an LSL stream or MAT file.

    Parameters
    ----------
    sess : Session object
        Session where Tensor will exist  
    shape : shape_like
        Shape of the Tensor
    data : ndarray
        Data to be stored within the array
    is_virtual : bool
        If False, the Tensor is non-virtual, if True, the Tensor is virtual
    ext_src : BCIPy input Source
        Data source the tensor pulls data from (only applies to Tensors created from a handle)
    ext_out : BCIPy output Source
        Data source the tensor pushes data to (only applies to Tensors created from a handle)
    """
    
    def __init__(self,sess,shape,data,is_virtual,ext_src, ext_out = None):
        """
        Constructor for Tensor class
        """
        super().__init__(BcipEnums.TENSOR,sess)
        self._shape = tuple(shape)
        self._virtual = is_virtual
        self._ext_src = ext_src
        self._ext_out = ext_out
        
        if not (data is None):
            self.data = data
        else:
            self.data = np.zeros(shape)
        
        if ext_src is None:
            self._volatile = False
        else:
            self._volatile = True
            
        if ext_out is None:
            self._volatile_out = False
        else:
            self._volatile_out = True
    
    # API Getters
    @property
    def data(self):
        """
        Getter for Tensor data

        Returns
        -------
        Data stored in Tensor : ndarray

        Examples
        --------
        >>> print(tensor.data)
        """
        return self._data
    
    @property
    def shape(self):
        return self._shape
    
    @property
    def virtual(self):
        return self._virtual
    
    @property
    def volatile(self):
        return self._volatile
    
    @property
    def volatile_out(self):
        return self._volatile_out

    @property
    def ext_src(self):
        return self._ext_src
    
    @property
    def ext_out(self):
        return self._ext_out
    
    #API setters
    @data.setter
    def data(self,data):
        """
        Set data of a Tensor. If the current shape of the Tensor is different from the shape of the data being inputted, you must first change 
        the shape of the Tensor before adding the data, or an error will be thrown

        Parameters
        ----------
        data : nd_array
            Data to have the Tensor data changed to
    
        Raises
        ------
        ValueError
            If the shape of the Tensor is different from the shape of the data being inputted

        Examples
        --------
        >>> tensor.data = np.array([1,2,3,4,5,6])
            
        """

        # special case where every dimension is a singleton
        if (np.prod(np.asarray(data.shape)) == 1 and 
            np.prod(np.asarray(self.shape)) == 1):
            while len(self.shape) > len(data.shape):
                data = np.expand_dims(data,axis=0)
            
            while len(self.shape) < len(data.shape):
                data = np.squeeze(data,axis=0)
        
        if self.shape == data.shape:
            self._data = data
        else:
            raise ValueError("Mismatched shape")
    
    @shape.setter
    def shape(self,shape):
        """
        Method to set the shape of a Tensor. Only applies to non-virtual tensors and sets all values in the modified tensor to 0.

        Parameters
        ----------
        shape : shape_like
            Shape to change the Tensor to

        Raises
        ------
        ValueError
            If the Tensor is non-virtual, the shape cannot be changed

        .. note:: This method is only applicable to virtual Tensors, and will throw an error if called on a non-virtual Tensor.

        Examples
        --------
        >>> t = Tensor.create_virtual((1,2,3))
        >>> t.shape = (3,2,1)
        """

        if self.virtual:
            self._shape = shape
            # when changing the shape write a zero tensor to data
            self.data = np.zeros(shape)
        else:
            raise ValueError("Cannot change shape of non-virtual tensor") 
            
    def make_copy(self):
        """
        Create and return a deep copy of the tensor

        Returns
        -------
        Tensor object
            Deep copy of the Tensor object

        Examples
        --------
        >>> t = Tensor.create_virtual((1,2,3))
        >>> t2 = t.make_copy()
        """
        #TODO determine what to do when copying virtual
        cpy = Tensor(self.session,
                     self.shape,
                     self.data,
                     self.virtual,
                     self.ext_src)
        
        # add the copy to the session
        sess = self.session
        sess.add_data(cpy)
        
        return cpy
    
    def copy_to(self,dest_tensor):
        """
        Copy the attributes of the tensor to another tensor object

        Parameters 
        ----------
        dest_tensor : Tensor object
            Tensor object where the attributes with the referenced Tensor will copied to

        Returns
        -------
        BcipEnums.SUCCESS : int
            If the copy was successful

        Examples
        --------
        >>> t = Tensor.create_virtual((1,2,3))
        >>> t2 = Tensor.create_virtual((1,3,3))
        >>> print(t.copy_to(t2))

            BcipEnums.SUCCESS
        """
        if dest_tensor.virtual:
            dest_tensor.shape = self.shape
        dest_tensor.data = self.data
        
        return BcipEnums.SUCCESS
        
        # Not copying virtual and ext_src attributes because these should 
        # only be set during creation and modifying could cause unintended
        # consequences
    
    def poll_volatile_data(self,label=None):
        """
        Pull data from external sources or BCIPy input data sources.
        """
        
        # check if the data is actually volatile, if not just return
        if not self.volatile:
            return BcipEnums.SUCCESS
        
        data = self.ext_src.poll_data(Ns=self.shape[1], label=label)
        try:
            # if we only pulled one trial, remove the first dimension
            data = np.squeeze(data)
        except ValueError:
            pass # just ignore the error for now
        
        # set the data 
        self.data = data
        
        return BcipEnums.SUCCESS
    
    def push_volatile_outputs(self):
        """
        Push data to external sources.
        """
        
        # check if the output is actually volatile, if not just return
        if not self.volatile_out:
            return BcipEnums.SUCCESS
        
        # push the data
        self.ext_out.push_data(self.data)

        return BcipEnums.SUCCESS
    
    # Factory Methods
    @classmethod
    def create(cls,sess,shape):
        """
        Factory Method to create a generic, non-virtual, Tensor object. The shape must be known to create this object 
        
        Parameters
        ----------
        sess : Session object
            Session where Tensor will exist
        shape : shape_like
            Shape of the Tensor
        
        """

        t = cls(sess,shape,None,False,None)
        
        # add the tensor to the session
        sess.add_data(t)
        return t
    
    @classmethod
    def create_virtual(cls,sess,shape=()):
        """
        Factory method to create a virtual Tensor

        Parameters
        ----------

        sess : Session object
            Session where Tensor will exist
        
        shape : shape_like, default = ()
            Shape of the Tensor, can be changed for virtual tensors
        """

        t = cls(sess,shape,None,True,None)
        
        # add the tensor to the session
        sess.add_data(t)
        return t
    
    @classmethod
    def create_from_data(cls,sess,shape,data):
        """
        Factory method to create a Tensor from data

        Parameters
        ----------

        sess : Session object
            Session where Tensor will exist
        
        shape : shape_like
            Shape of the Tensor
        
        data : ndarray
            Data to be stored within the array
            
        """

        if type(data) is list:
            data = np.asarray(data)
        
        # make sure data is valid
        if not cls.validate_data(shape,data):
            # data invalid!
            return 
        t = cls(sess,shape,data,False,None)
        
        # add the tensor to the session
        sess.add_data(t)
        return t
        
    @classmethod
    def create_from_handle(cls,sess,shape,src):
        """
        Factory method to create a Tensor from a handle/external source

        Parameters
        ----------

        sess : Session object
            Session where Tensor will exist
        
        shape : shape_like
            Shape of the Tensor
        
        ext_src : BCIPy input Source
            Data source the tensor pulls data from (only applies to Tensors created from a handle)
        
        """
        t = cls(sess,shape,None,False,src)
        
        # add the tensor to the session
        sess.add_data(t)
        return t
    
    @classmethod
    def create_for_volatile_output(cls, sess, shape, out):
        t = cls(sess,shape,None,False,None,out)
        sess.add_data(t)
        return t
    
    # utility static methods
    @staticmethod
    def validate_data(shape,data):
        """
        Method that returns True if  the data within the tensor is the right shape and is a numpy ndarray. False otherwise.
        """
        if data is None:
            return False
        
        if (not (type(data) is np.ndarray)) or (tuple(shape) != data.shape):
            return False
            
        return True
 

class Array(BCIP):
    """
    Array containing instances of other BCIP classes. Each array can only hold one type of BCIP class.

    Parameters
    ----------
    sess : Session object
        Session where the Array object will exist
    capacity : int
        Maximum number of elements to be stored within the array (for allocation purposes)
    element_template : any
        The template BCIP element to populate the array (see examples)

    Attributes
    ----------

    Examples
    --------
    >>> #Creating An Array
    >>> example = Array.create(example_session, example_capacity, Tensor.create(example_session, input_data.shape))
    
    Return
    ======
    Array Object
    
    .. note:: A single array object should only contain one BCIP/data object type.

    
    """
    
    def __init__(self,sess,capacity,element_template):
        super().__init__(BcipEnums.ARRAY,sess)
        
        self._virtual = False # no virtual arrays for now
        self._volatile = False # no volatile arrays for now...
        self._volatile_out = False # no volatile arrays for now...
        
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
            Index is the position within the array with the element will be returned. Index should be 0 <= Index < Capacity

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
            Index in the array where the element will changed. 0 <= Index < capacity

        element : any
            specified value which will be set at index index

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
    
    @property
    def volatile_out(self):
        return self._volatile_out
    
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
            Array object where the attributes with the referenced array will copied to

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
            Session where the Array object will exist
        capacity : int
            Maximum number of elements to be stored within the array (for allocation purposes)
        element_template : any
            The template BCIP element to populate the array (see examples)

        """

        a = cls(sess,capacity,element_template)
        
        # add the array to the session
        sess.add_data(a)
        return a


class CircleBuffer(Array):
    """
    A circular buffer/Array for BCIP/data objects. 

    Parameters
    ----------
    sess : Session object
        Session where the Array object will exist
    capacity : int
        Maximum number of elements to be stored within the array (for allocation purposes)
    element_template : any
        The template BCIP element to populate the array (see Array examples)

    Examples
    --------
    Creating a Circle Buffer
        example_buffer = CircleBuffer.create(example_session, 5, Tensor.create(example_session, example_input_shape))

    """
    
    def __init__(self,sess,capacity,element_template):
        super().__init__(sess,capacity,element_template)
        self._bcip_type = BcipEnums.CIRCLE_BUFFER # overwrite
        
        self._head = None
        self._tail = None
    
    @property
    def num_elements(self):
        """
        Return the number of elements currently in the buffer.

        Parameters
        ----------
        None

        Return
        ------
        int: Number of elements currently in the buffer

        Examples
        --------
        example_num_elements = example_buffer.num_elements()
        """
        if self.is_empty():
            return 0
        else:
            return ((self._tail - self._head) % self.capacity) + 1
    
    def is_empty(self):
        """
        Parameters
        ----------

        Return
        ------
        bool : True if circle buffer is empty, false otherwise


        Examples
        --------

        >>> is_empty = example_buffer.is_empty()
        >>> print(is_empty)
            
            True
        """

        if self._head == None and self._tail == None:
            return True
        else:
            return False
        
    def is_full(self):
        """
        Parameters
        ----------

        Return
        ------
        bool : True if circle buffer is empty, false otherwise


        Examples
        --------

        >>> is_empty = example_buffer.is_empty()
        >>> print(is_empty)
            
            True
        """

        if self._head == ((self._tail + 1) % self.capacity):
            return True
        else:
            return False
    
    
    def get_queued_element(self,index):
        """
        Returns the element at a specific index within an Circle Buffer object.

        Parameters
        ----------
        index : int
            Index is the position within the array with the element will be returned. Index should be 0 <= Index < Capacity

        Return
        ------
        any : Data object at index index 
        
        Examples
        --------
        example_element = example_circle_buffer.get_element(0)
        """
        if index > self.num_elements:
            return None
        
        abs_index = (index + self._head) % self.capacity
        
        return self.get_element(abs_index)
    
    def peek(self):
        """
        Returns the front element of a circle buffer

        Parameters
        ----------
        None

        Return
        ------
        any : Data object at first index
        
        Examples
        --------
        >>> example_element = example_circle_buffer.peek()
        >>> print(example_element)
            
            12


        """   

        if self.is_empty():
            return None
        
        return super(CircleBuffer,self).get_element(self._head)
    
    def enqueue(self,obj):
        if self.is_empty():
            self._head = 0
            self._tail = -1
            
        elif self.is_full():
            self._head = (self._head + 1) % self.capacity
        
        self._tail = (self._tail + 1) % self.capacity
        return super().set_element(self._tail,obj)
        
    def enqueue_chunk(self,cb):
        """
        enqueue a number of elements from another circle buffer into this
        circle buffer
        """
        
        while not cb.is_empty():
            element = cb.dequeue()
            sts = self.enqueue(element)
            
            if sts != BcipEnums.SUCCESS:
                return sts
        
        return BcipEnums.SUCCESS
            
    
    def dequeue(self):
        if self.is_empty():
            return None
        
        ret = super().get_element(self._head)
        
        if self._head == self._tail:
            self._head = None
            self._tail = None
        else:
            self._head = (self._head + 1) % self.capacity
        
        return ret
    
    def make_copy(self):
        """
        Create and return a deep copy of the array
        The copied array will maintain references to the same objects.
        If a copy of these is also desired, they will need to be copied
        separately.
        """
        cpy = CircleBuffer(self.session,
                           self.capacity,
                           self.get_element(0))
        
        for e in range(self.capacity):
            cpy.set_element(e,self.get_element(e))
        
        # TODO this should be handled using API methods instead
            # copy the head and tail as well
        cpy._tail = self._tail
        cpy._head = self._head
        
        # add the copy to the session
        self.session.add_data(cpy)
            
        return cpy
    
    def copy_to(self,dest_array):
        """
        Copy all the attributes of the array to another array. Note
        these will reference the same objects within the element list
        """
        dest_array.capacity = self.capacity
        for i in range(self.capacity):
            sts = dest_array.set_element(i,self.get_element(i))
            
            if sts != BcipEnums.SUCCESS:
                return sts
        
        if dest_array._bcip_type == BcipEnums.CIRCLE_BUFFER:
            # copy the head and tail as well
            dest_array._tail = self._tail
            dest_array._head = self._head
            
        return BcipEnums.SUCCESS
    
    def flush(self):
        """
        Empty the buffer of all elements

        Parameters
        ----------
        None

        Examples
        --------
        >>> status = example_buffer.flush()
        >>> print(status)

            SUCCESS

        Returns
        -------
        BCIP Status code

        """
        while not self.is_empty():
            self.dequeue()
        
        return BcipEnums.SUCCESS
    
    
    def to_tensor(self):
        """
        Copy the elements of the buffer to a Tensor and return the tensor

        """
        if self.is_empty():
            return Tensor.create(self.session, (0,))
        
        t = super().to_tensor()
        
        # remove data from tensor that is outside the range of the cb's head and tail
        if self._head < self._tail:
            valid_data = t.data[self._head:self._tail+1]
        else:
            valid_data = np.concatenate((t.data[self._head:], t.data[:self._tail+1]),
                                        axis=0)
        
        return Tensor.create_from_data(self.session, valid_data.shape, valid_data)
        
    @classmethod
    def create(cls,sess,capacity,element_template):
        cb = cls(sess,capacity,element_template)

        # add to the session
        sess.add_data(cb)
        
        return cb
