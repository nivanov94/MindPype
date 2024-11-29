from .core import MPBase, MPEnums
import numpy as np


class Scalar(MPBase):
    """
    Represents scalar data that can be ingested and produced by nodes
    within MindPype graphs. 

    A `Scalar` object contains a single value of a specified data type
    (int, float, complex, str, or bool). 

    Parameters
    ----------
    sess : Session 
        The session object in which the scalar will exist and operate.
    data_type : type
        The data type contained within the scalar
    value : (int, float, complex, str, or bool), optional
        Data value contained within the scalar object. If not specified,
        a default value will be assigned based on the data type.
    is_virtual : bool, default = False
        If True, the scalar is a virtual object, otherwise it is non-virtual.
    ext_src : Source, default = None
        A volatile external data source from which the scalar will pull data
        during graph execution. If the scalar does not represent an external
        data source, set ext_src to None.
    ext_out : Source, default = None
        A volatile external data source to which the scalar will push data
        during graph execution. If the scalar does not push data to an
        external source, set ext_out to None.

    Attributes
    ----------
    data_type : type
        The data type contained within the scalar.
    data : (int, float, complex, str, or bool)
        Data value contained within the scalar object.
    is_virtual : bool
        Indicates whether the scalar is a virtual object.
    ext_src : Source
        External data source from which the scalar will pull data during
        graph execution.
    ext_out : Source
        External data source to which the scalar will push data during
        graph execution.
    volatile : bool
        Indicates whether the scalar is a volatile object that needs to pull
        from or push to external sources during graph execution.

    Methods
    -------
    make_copy()
        Create and return a deep copy of the scalar.
    copy_to(dest_scalar)
        Copy all the attributes of the scalar to another scalar object.
    assign_random_data(whole_numbers=False, vmin=0, vmax=1, covariance=False)
        Assign random data to the scalar.
    poll_volatile_data(label=None)
        Pull data from external sources and assign it to the scalar's data
        attribute.
    push_volatile_outputs(label=None)
        Push the scalar's data to external sources.
    create(sess, data_type)
        Factory method to create a non-virtual, non-volatile scalar object.
    create_virtual(sess, data_type)
        Factory method to create a virtual scalar object.
    create_from_value(sess, value)
        Factory method to create a non-virtual, non-volatile scalar object
        with a specified data value.
    create_from_source(sess, data_type, src)
        Factory method to create a non-virtual, volatile scalar object with an
        external data source.

    Notes
    -----
    - Virtual scalars should only be used to define connections between nodes 
      in a graph. The data within virtual scalars may be modified by
      `Graph` or `Node` methods and they should not be used to store data that 
      is used outside of graph execution.
    - Volatile scalars cannot be virtual.
    """

    valid_types = [int, float, complex, str, bool]
    valid_numeric_types = [int, float, complex, bool]

    def __init__(
        self, sess, data_type, value=None, is_virtual=False, ext_src=None, ext_out=None
    ):
        """Init."""
        super().__init__(MPEnums.SCALAR, sess)

        if isinstance(data_type, str):
            dtypes = {"int": int, "float": float, "complex": complex, 
                      "str": str, "bool": bool}

            if data_type in dtypes:
                data_type = dtypes[data_type]
            else:
                raise ValueError(
                    f"{data_type} is an invalid data type for scalar. "
                    "Must be one of [int, float, complex, str, bool]."
                )

        self.data_type = data_type

        self.ext_src = ext_src

        if value is None:
            if data_type == int:
                val = 0
            elif data_type == float:
                val = 0.0
            elif data_type == complex:
                val = complex(0, 0)
            elif data_type == str:
                val = ""
            elif data_type == bool:
                val = False
            else:
                raise ValueError(
                    "Invalid data type for scalar, must be one of " 
                    "[int, float, complex, str, bool]"
                )

        self.ext_out = ext_out
        self.data = value

        self.virtual = is_virtual

        if ext_src is None:
            self.volatile = False
        else:
            self.volatile = True

        if ext_out is None:
            self.volatile_out = False
        else:
            self.volatile_out = True

        if self.volatile and self.virtual:
            raise ValueError("Volatile scalars cannot be virtual")

    @property
    def data(self):
        """
        Retrieve the scalar data value.

        This property provides access to the scalar data value stored within 
        the `Scalar` object. The data can represent various types, including 
        integers, floats, complex numbers, strings, or booleans.

        Returns
        -------
        int, float, complex, str, or bool
            The scalar data value encapsulated by the `Scalar` object.
        """
        return self._data

    @property
    def is_numeric(self):
        """
        Return whether the scalar's data type is numeric.

        This property checks whether the scalar's data type belongs to a set of 
        valid numeric types, such as integers, floats, or complex numbers.

        Returns
        -------
        bool
        True if the scalar's data type is numeric, False otherwise.
        """
        return self.data_type in self.valid_numeric_types

    @data.setter
    def data(self, data):
        """
        Set the scalar's data value.

        Parameters
        ----------
        data : int, float, complex, str, bool, or numpy.ndarray
            The value to be represented by the scalar object. 
            The input must be scalar (e.g., a single numerical value, 
            string, or boolean).

        Raises
        ------
        TypeError
            If the data type of the input does not match the scalar's 
            existing data type.
        """
        # Handle numpy array input
        if isinstance(data, np.ndarray):
            # Ensure the array contains a single value
            if data.size == 1:
                data = data.item()  # Extract the scalar value
            else:
                raise ValueError(
                    "The input numpy array must contain exactly one element."
                )

        # Convert numpy scalar types to native Python types
        if isinstance(data, (np.integer, np.int_)):
            data = int(data)
        elif isinstance(data, (np.floating, np.float_)):
            data = int(data) if data.is_integer() and self.data_type is int else float(data)
        elif isinstance(data, np.complexfloating):
            data = complex(data)

        # Validate data type and set the value
        if isinstance(data, self.data_type):
            self._data = data
        else:
            raise TypeError(
                f"MindPype Scalar expects data of type {self.data_type}. "
                f"Cannot set data to type {type(data)}."
            )

    def make_copy(self):
        """
        Create and return a deep copy of the scalar.

        This method creates a new `Scalar` object with the same attributes
        as the original object, including the data type, data value, and
        external data sources. The copied scalar is added to the same session
        as the original scalar.

        Returns
        -------
        Scalar
            A deep copy of the scalar.
        """
        # create a new scalar object
        cpy = Scalar(
            self.session,
            self.data_type,
            self.data,
            self.virtual,
            self.ext_src,
            self.ext_out
        )

        # add the copy to the session
        sess = self.session
        sess.add_to_session(cpy)

        return cpy

    def copy_to(self, dest_scalar):
        """
        Copy the data to another scalar object.
        
        This method copies the data value from the referenced scalar to the
        destination scalar object. Other attributes, such as the data type, 
        virtual status, and external data sources, are not copied as they
        should not be modified after the scalar is created.

        Parameters
        ----------
        dest_scalar : Scalar
            Scalar object to which the data value will be copied.

        Raises
        ------
        TypeError
            If the destination scalar's data type does not match the source
            scalar's data type.
        """
        # Validate the destination scalar's data type
        if dest_scalar.data_type != self.data_type:
            raise TypeError(
                f"Cannot copy data from scalar with data type {self.data_type} "
                f"to scalar with data type {dest_scalar.data_type}."
            )
        dest_scalar.data = self.data

    def assign_random_data(
            self, whole_numbers=False, vmin=0, vmax=1, covariance=False
    ):
        """
        Assign random data to the scalar. This is useful for testing and
        verification purposes.

        Parameters
        ----------
        whole_numbers: bool
            Assigns data that is only whole numbers if True
        vmin: int
            Lower limit for values in the random data
        vmax: int
            Upper limits for values in the random data
        covarinace: bool
            If True, assigns random covariance matrix

        """
        if self.data_type == int or whole_numbers:
            self._data = np.random.randint(vmin, vmax+1)
        elif self.data_type == float:
            vrange = vmax - vmin
            self.data = vrange * np.random.rand() + vmin
        elif self.data_type == complex:
            self.data = complex(np.random.rand(), np.random.rand())
        elif self.data_type == str:
            self.data = str(np.random.rand())
        elif self.data_type == bool:
            self.data = np.random.choice([True, False])

    def poll_volatile_data(self, label=None):
        """
        Poll data from external sources to update the scalar's data value.

        This method uses the scalar's external data source to pull data and
        update the scalar's data value. The label parameter may be used to
        specify a class label corresponding to the data to be polled.

        Parameters
        ----------
        Label : int, default=None
            Class label corresponding to class data to poll, if applicable.
            This is typically used when polling epoched data from a file
            source.
        """
        # check if the data is actually volatile
        if self.volatile:
            self.data = self.ext_src.poll_data(label)

    def push_volatile_outputs(self, label=None):
        """
        Push the scalar's data to external sources.

        This method uses the scalar's external output source (e.g., LSL stream)
        to push data during graph execution. The label parameter may be used
        to specify a class label corresponding to the data to be pushed.

        Parameters
        ----------
        Label : int, default=None
            Class label corresponding to class data to push, if applicable.
        """
        # check if the output is actually volatile
        if self.volatile_out:
            self.ext_out.push_data(self.data, label)

    # Factory Methods
    @classmethod
    def create(cls, sess, data_type):
        """
        Factory method to create a non-virtual, non-volatile Scalar and 
        register it within a session.

        This method creates a new `Scalar` object with a default data value
        based on the specified data type. The scalar is added to the specified
        session.

        Parameters
        ----------
        sess : Session
            Session where the scalar will exist.
        data_type : int, float, complex, str, or bool
            Data type of data represented by scalar.

        Return
        ------
        Scalar 
            A reference to the scalar that was created and added to the session.

        Raises
        ------
        ValueError
            If the specified data type is not supported by the Scalar class.
        """
        s = cls(sess, data_type, None, False, None)
        sess.add_to_session(s)
        return s

    @classmethod
    def create_virtual(cls, sess, data_type):
        """
        Factory method to create a virtual Scalar object.

        This method creates a new virtual `Scalar` object with the specified
        data type. The scalar is added to the specified session.

        Parameters
        ----------
        sess : Session
            Session where the scalar will exist.
        data_type : int, float, complex, str, or bool
            Data type of data represented by scalar.

        Return
        ------
        Scalar
            A reference to the virtual scalar that was created and added to the
            session.

        Raises
        ------
        ValueError
            If the specified data type is not supported by the Scalar class. 
        """
        s = cls(sess, data_type)
        sess.add_to_session(s)
        return s

    @classmethod
    def create_from_value(cls, sess, value):
        """
        Factory method to create a non-virtual, non-volatile Scalar object
        with a specified data value.

        This method creates a new `Scalar` object with the specified data value
        and adds it to the specified session. The scalar's data type is inferred
        from the input value.

        Parameters
        ----------
        sess : Session
            Session where the Scalar object will exist.
        value : int, float, complex, str, or bool
            Data value to be stored within the scalar.

        Return
        ------
        Scalar
            A reference to the scalar that was created and added to the session.
        
        Raises
        ------
        TypeError
            If the specified data type is not supported by the Scalar class.
        """
        data_type = type(value)
        if not (data_type in cls.valid_types):
            raise TypeError(
                f"Data type {data_type} is not supported by the Scalar class."
            )

        s = cls(sess, data_type, value)
        sess.add_to_session(s)
        return s

    @classmethod
    def create_from_source(cls, sess, data_type, src):
        """
        Factory method to create a non-virtual, volatile Scalar object with an
        external data source.

        This method creates a new `Scalar` object with an external data source
        and adds it to the specified session.

        Parameters
        ----------
        sess : Session
            Session where the Scalar object will exist.
        data_type : int, float, complex, str, or bool
            Data type of data represented by scalar.
        src : Source
            External data source from which to poll data

        Return
        ------
        Scalar
            A reference to the scalar that was created and added to the session.

        Raises
        ------
        ValueError
            If the specified data type is not supported by the Scalar class.
        """
        s = cls(sess, data_type, ext_src=src)
        sess.add_to_session(s)
        return s


class Tensor(MPBase):
    """
    Tensor (or n-dimensional matrices), are defined by the tensor class.
    MindPype tensors can either be volatile (are updated/change each trial,
    generally reserved for tensors containing current trial data), virtual
    (empty, dimensionless tensor object). Like scalars and array, tensors can
    be created from data, copied from a different variable, or created
    virtually, so they don’t initially contain a value.
    Each of the scalars, tensors and array data containers also have an
    external source (_ext_src) attribute, which indicates, if necessary,
    the source from which the data is being pulled from. This is
    especially important if trial/training data is loaded into a tensor
    each trial from an LSL stream or MAT file.

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
    ext_src : input Source
        Data source the tensor pulls data from (only applies to Tensors
        created from a handle)
    ext_out : output Source
        Data source the tensor pushes data to (only applies to Tensors created
        from a handle)
        
    Attributes
    ----------
    shape : tuple
        Shape of the data
    virtual : bool
        If true, the Scalar object is virtual, non-virtual otherwise
    ext_src : LSL data source input object, MAT data source, or None
        External data source represented by the scalar; this data will
        be polled/updated when trials are executed. If the data does
        not represent an external data source, set ext_src to None
    ext_out : output Source
        Data source the tensor pushes data to (only applies to Tensors created
        from a handle)
    data : value of type int, float, complex, str, or bool
        Data value represented by the Scalar object
    volatile : bool
        True if source is volatile (needs to be updated/polled between
        trials), false otherwise
    """

    def __init__(self, sess, shape, data, is_virtual, ext_src, ext_out=None):
        """Init."""
        super().__init__(MPEnums.TENSOR, sess)
        self._shape = tuple(shape)
        self.virtual = is_virtual
        self.ext_src = ext_src
        self.ext_out = ext_out

        if not (data is None):
            self._data = data
        else:
            self._data = np.zeros(shape)

        if ext_src is None:
            self.volatile = False
        else:
            self.volatile = True

        if ext_out is None:
            self.volatile_out = False
        else:
            self.volatile_out = True

    # API Getters
    @property
    def data(self):
        """
        Getter for Tensor data

        Returns
        -------
        Data stored in Tensor : ndarray
        """
        return self._data

    @property
    def shape(self):
        """
        Getter for Tensor shape

        Returns
        -------
        Shape of the Tensor : tuple
        """
        return self._shape

    # API setters
    @data.setter
    def data(self, data):
        """
        Set data of a Tensor. If the current shape of the Tensor
        is different from the shape of the data being inputted,
        you must first change the shape of the Tensor before adding
        the data, or an error will be thrown

        Parameters
        ----------
        data : nd_array
            Data to have the Tensor data changed to

        Raises
        ------
        ValueError
            If the shape of the Tensor is different from the shape of the data
            being input
        """
        # check that the value is a numpy array
        if not isinstance(data, np.ndarray):
            # if the value is a number, convert it to a numpy array
            if isinstance(data, (int, float, complex)):
                data = np.array(data)
            else:
                raise TypeError("Data assigned to Tensors must be a numpy array")

        # special case where every dimension is a singleton
        if (np.prod(np.asarray(data.shape)) == 1 and
                np.prod(np.asarray(self.shape)) == 1):
            while len(self.shape) > len(data.shape):
                data = np.expand_dims(data, axis=0)

            while len(self.shape) < len(data.shape):
                data = np.squeeze(data, axis=0)

        if self.virtual and self.shape != data.shape:
            self._shape = data.shape

        if self.shape == data.shape:
            self._data = data
        else:
            raise ValueError("Mismatched shape")

    @shape.setter
    def shape(self, shape):
        """
        Method to set the shape of a Tensor. Only applies to
        non-virtual tensors and sets all values in the
        modified tensor to 0.

        Parameters
        ----------
        shape : shape_like
            Shape to change the Tensor to

        Raises
        ------
        ValueError
            If the Tensor is non-virtual, the shape cannot be changed

        .. note:: This method is only applicable to virtual Tensors, and will
                  raise a TypeError if called on a non-virtual Tensor.
        """

        if self.virtual:
            self._shape = shape
            # when changing the shape write a zero tensor to data
            self.data = np.zeros(shape)
        else:
            raise TypeError("Cannot change shape of non-virtual tensor")

    def make_copy(self):
        """
        Create and return a deep copy of the tensor

        Returns
        -------
        Tensor object
            Deep copy of the Tensor object
        """
        cpy = Tensor(self.session,
                     self.shape,
                     self.data,
                     self.virtual,
                     self.ext_src)

        # add the copy to the session
        sess = self.session
        sess.add_to_session(cpy)

        return cpy

    def copy_to(self, dest_tensor):
        """
        Copy the attributes of the tensor to another tensor object

        .. note: This method will not copy the virtual and ext_src attributes
                 because these should only be set during creation and modifying could
                 cause unintended consequences.

        Parameters
        ----------
        dest_tensor : Tensor object
            Tensor object where the attributes with the referenced Tensor will
            be copied to
        """
        if dest_tensor.virtual and dest_tensor.shape != self.shape:
            dest_tensor.shape = self.shape
        dest_tensor.data = self.data

    def assign_random_data(self, whole_numbers=False, vmin=0, vmax=1,
                           covariance=False):
        """
        Assign random data to the tensor. This is useful for testing and
        verification purposes.

        Parameters
        ----------
        whole_numbers: bool
            Assigns data that is only whole numbers if True
        vmin: int
            Lower limit for values in the random data
        vmax: int
            Upper limits for values in the random data
        covarinace: bool
            If True, assigns random covariance matrix
        """
        if whole_numbers:
            self.data = np.random.randint(vmin, vmax+1, size=self.shape)
        else:
            num_range = vmax - vmin
            self.data = num_range * np.random.rand(*self.shape) + vmin

        if covariance:
            rank = len(self.shape)
            if rank != 2 and rank != 3:
                raise ValueError("Cannot assign random covariance matrix to " +
                                 "tensor with rank other than 2 or 3")

            if self.shape[-2] != self.shape[-1]:
                raise ValueError("Cannot assign random covariance matrix to " +
                                 "tensor with non-square last two dimensions")

            if rank == 2:
                self.data = np.cov(self.data)
            else:
                for i in range(self.shape[0]):
                    self.data[i,:,:] = np.cov(self.data[i,:,:])

            # add regularization
            self.data += 0.001 * np.eye(self.shape[-1])

    def poll_volatile_data(self, label=None):
        """
        Pull data from external sources or MindPype input data sources.

        Parameters
        ----------
        Label : int, default = None
            Class label corresponding to class data to poll.
        """
        # check if the data is actually volatile, if not just return
        if not self.volatile:
            return

        data = self.ext_src.poll_data(label=label)
        # if we only pulled one trial, remove the first dimension
        if len(data.shape) > 2:
            data = np.squeeze(data, axis=0)
        self.data = data

    def push_volatile_outputs(self, label=None):
        """
        Push data to external sources.
        """
        # check if the output is actually volatile first
        if self.volatile_out:
            # push the data
            self.ext_out.push_data(self.data)

    # Factory Methods
    @classmethod
    def create(cls, sess, shape):
        """
        Factory Method to create a generic, non-virtual, Tensor object. The
        shape must be known to create this object

        Parameters
        ----------
        sess : Session object
            Session where Tensor will exist
        shape : shape_like
            Shape of the Tensor

        """
        t = cls(sess, shape, None, False, None)

        # add the tensor to the session
        sess.add_to_session(t)
        return t

    @classmethod
    def create_virtual(cls, sess, shape=()):
        """
        Factory method to create a virtual Tensor

        Parameters
        ----------

        sess : Session object
            Session where Tensor will exist

        shape : shape_like, default = ()
            Shape of the Tensor, can be changed for virtual tensors
        """
        t = cls(sess, shape, None, True, None)

        # add the tensor to the session
        sess.add_to_session(t)
        return t

    @classmethod
    def create_from_data(cls, sess, data):
        """
        Factory method to create a Tensor from data

        Parameters
        ----------

        sess : Session object
            Session where Tensor will exist

        data : ndarray
            Data to be stored within the array

        """
        if type(data) is list:
            data = np.asarray(data)

        t = cls(sess, data.shape, data, False, None)

        # add the tensor to the session
        sess.add_to_session(t)
        return t

    @classmethod
    def create_from_handle(cls, sess, shape, src):
        """
        Factory method to create a Tensor from a handle/external source

        Parameters
        ----------

        sess : Session object
            Session where Tensor will exist

        shape : shape_like
            Shape of the Tensor

        ext_src : input Source
            Data source the tensor pulls data from (only applies to Tensors
            created from a handle)

        """
        t = cls(sess, shape, None, False, src)

        # add the tensor to the session
        sess.add_to_session(t)
        return t

    @classmethod
    def _create_for_volatile_output(cls, sess, shape, out):
        """
        Create data source for volatile output

        Parameters
        ----------
        sess : Session object
            Session where Tensor will exist

        shape : shape_like
            Shape of the Tensor

        out : output Source
            Data source the tensor pushes data to (only applies to Tensors
            created for volatile output)
        """
        # These tensors should be non-volatile but since init and trial data
        # can be different sizes, they need to be virtual until that is
        # addressed

        t = cls(sess, shape, None, True, None, out)
        sess.add_to_session(t)
        return t

    # utility static methods
    @staticmethod
    def _validate_data(shape, data):
        """
        Method that returns True if  the data within the tensor is the right
        shape and is a numpy ndarray. False otherwise.

        Parameters
        ----------
        data: Tensor
            Data to be validated

        Returns
        -------
        is_valid: bool
            Returns true if the data in the tensor is the correct shape and is a numpy ndarray
        """
        if data is None:
            return False

        if (not (type(data) is np.ndarray)) or (tuple(shape) != data.shape):
            return False

        return True


class Array(MPBase):
    """
    Array containing instances of other MindPype classes. Each array can only
    hold one type of MindPype class.
    
    .. note:: A single array object should only contain one MindPype/data
            object type.

    Parameters
    ----------
    sess : Session object
        Session where the Array object will exist
    capacity : int
        Maximum number of elements to be stored within the array (for
        allocation purposes)
    element_template : any
        The template MindPype element to populate the array (see examples)


    Attributes
    ----------
    virtual : bool
        If true, the Scalar object is virtual, non-virtual otherwise
    volatile : bool
        True if source is volatile (needs to be updated/polled between
        trials), false otherwise
    capacity: int
        Max number of elements that can be stored in the array
    _elements: array
        Elements of the array
    
    Examples
    --------
    >>> # Creating An Array of tensors
    >>> template = Tensor.create(example_session, input_data.shape)
    >>> example = Array.create(example_session, example_capacity, template)

    Return
    ------
    array: Array Object

    """

    def __init__(self, sess, capacity, element_template):
        """ Init """
        super().__init__(MPEnums.ARRAY, sess)

        self.virtual = False  # no virtual arrays for now
        self.volatile = False  # no volatile arrays for now...
        self.volatile_out = False  # no volatile arrays for now...

        self._capacity = capacity

        self._elements = [None] * capacity

        for i in range(capacity):
            self._elements[i] = element_template.make_copy()

    # Returns an element at a particular index
    def get_element(self, index):
        """
        Returns the element at a specific index within an array object.

        Parameters
        ----------
        index : int
            Index is the position within the array with the element will be
            returned. Index should be 0 <= Index < Capacity

        Return
        ------
        any : Data object at index index

        Examples
        --------
        >>> example_element = example_array.get_element(0)


        """
        if index >= self.capacity or index < 0:
            return

        return self._elements[index]

    # Changes the element at a particular index to a specified value
    def set_element(self, index, element):

        """
        Changes the element at a particular index to a specified value

        Parameters
        ----------
        index : int
            Index in the array where the element will
            change. 0 <= Index < capacity

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

        if index >= self._capacity or index < 0:
            raise ValueError("Index out of bounds")

        element.copy_to(self._elements[index])

    # User Facing Getters
    @property
    def capacity(self):
        return self._capacity

    @property
    def num_elements(self):
        # this property is included to allow for seamless abstraction with
        # circle buffer property
        return self._capacity

    @capacity.setter
    def capacity(self, capacity):
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
        >>> new_array = old_array.make_copy()
        """
        cpy = Array(self.session,
                    self.capacity,
                    self.get_element(0))

        for e in range(self.capacity):
            cpy.set_element(e, self.get_element(e))

        # add the copy to the session
        self.session.add_to_session(cpy)

        return cpy

    def copy_to(self, dest_array):
        """
        Copy all the attributes of the array to another array. Note
        these will reference the same objects within the element list

        Parameters
        ----------
        dest_array : Array object
            Array object where the attributes with the referenced array
            will be copied to

        Examples
        --------
        >>> old_array.copy_to(copy_of_old_array)

        """
        dest_array.capacity = self.capacity
        for i in range(self.capacity):
            dest_array.set_element(i, self.get_element(i))

    def assign_random_data(self, whole_numbers=False, vmin=0, vmax=1,
                           covariance=False):
        """
        Assign random data to the array. This is useful for testing and
        verification purposes.

        Parameters
        ----------
        whole_numbers: bool
            Assigns data that is only whole numbers if True
        vmin: int
            Lower limit for values in the random data
        vmax: int
            Upper limits for values in the random data
        covarinace: bool
            If True, assigns random covariance matrix
        """
        for i in range(self.capacity):
            self.get_element(i).assign_random_data(whole_numbers, vmin, vmax,
                                                   covariance)

    def to_tensor(self):
        """
        Stack the elements of the array into a Tensor object.

        Returns
        -------
        Tensor
        """
        element = self.get_element(0)

        if not (element.mp_type == MPEnums.TENSOR or
                (element.mp_type == MPEnums.SCALAR and
                 element.data_type in Scalar._valid_numeric_types())):
            return None

        # extract elements and stack into numpy array
        elements = [self.get_element(i).data for i in range(self.capacity)]
        stacked_elements = np.stack(elements)

        # create tensor
        return Tensor.create_from_data(self.session, stacked_elements)

    # API constructor
    @classmethod
    def create(cls, sess, capacity, element_template):

        """
        Factory method to create array object

        Parameters
        ----------
        sess : Session object
            Session where the Array object will exist
        capacity : int
            Maximum number of elements to be stored within the array
            (for allocation purposes)
        element_template : any
            The template MindPype element to populate the array (see examples)

        """

        a = cls(sess, capacity, element_template)

        # add the array to the session
        sess.add_to_session(a)
        return a


class CircleBuffer(Array):
    """
    A circular buffer/Array for MindPype/data objects.

    Parameters
    ----------
    sess : Session object
        Session where the Array object will exist
    capacity : int
        Maximum number of elements to be stored within the array
        (for allocation purposes)
    element_template : any
        The template MindPype element to populate the array
        (see Array examples)
        
    Attributes
    ----------
    mp_type : MP Enum
        Data source the tensor pushes data to (only applies to Tensors created
        from a handle)
    head : data object
        First element of the circle buffer
    tail : data object
        Last element of the circle buffer

    """

    def __init__(self, sess, capacity, element_template):
        super().__init__(sess, capacity, element_template)
        self.mp_type = MPEnums.CIRCLE_BUFFER  # overwrite

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
        >>> example_num_elements = example_buffer.num_elements()
        """
        if self.is_empty():
            return 0
        else:
            return ((self._tail - self._head) % self._capacity) + 1

    def is_empty(self):
        """
        Checks if circle buffer is empty

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

        if self._head is None and self._tail is None:
            return True
        else:
            return False

    def is_full(self):
        """
        Checks if circle buffer is full

        Return
        ------
        bool : True if circle buffer is empty, false otherwise


        Examples
        --------

        >>> is_empty = example_buffer.is_empty()
        >>> print(is_empty)

            True
        """

        if self._head == ((self._tail + 1) % self._capacity):
            return True
        else:
            return False

    def get_queued_element(self, index):
        """
        Returns the element at a specific index within an Circle Buffer object.

        Parameters
        ----------
        index : int
            Index is the position within the array with the element will be
            returned. Index should be 0 <= Index < Capacity

        Return
        ------
        any : Data object at index index

        Examples
        --------
        >>> example_element = example_circle_buffer.get_element(0)
        """
        if index > self.num_elements:
            return None

        abs_index = (index + self._head) % self._capacity

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

        return super(CircleBuffer, self).get_element(self._head)

    def enqueue(self, obj):
        """
        Enqueue an element into circle buffer

        Parameters
        ----------
        obj: data object
            Object to be added to circle buffer
        """
        if self.is_empty():
            self._head = 0
            self._tail = -1

        elif self.is_full():
            self._head = (self._head + 1) % self.capacity

        self._tail = (self._tail + 1) % self.capacity
        return super().set_element(self._tail, obj)

    def enqueue_chunk(self, cb):
        """
        Enqueue a number of elements from another circle buffer into this
        circle buffer
        
        Parameters
        ----------
        cb: Circle Buffer
            Circle buffer to enqueue into the other Circle buffer
        """

        while not cb.is_empty():
            element = cb.dequeue()
            self.enqueue(element)

    def dequeue(self):
        """ 
        Dequeue element from circle buffer 
        
        Returns
        -------
        ret: data object
            MindPype data object at the head of the circle buffer that is removed
        """
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
        Create and return a deep copy of the Circle Buffer
        The copied Circle Buffer will maintain references to the same objects.
        If a copy of these is also desired, they will need to be copied
        separately.

        Returns
        -------
        cpy: Circle Buffer
            Copy of the circle buffer
        """
        cpy = CircleBuffer(self.session,
                           self.capacity,
                           self.get_element(0))

        for e in range(self.capacity):
            cpy.set_element(e, self.get_element(e))

        # TODO this should be handled using API methods instead
            # copy the head and tail as well
        cpy._tail = self._tail
        cpy._head = self._head

        # add the copy to the session
        self.session.add_to_session(cpy)

        return cpy

    def copy_to(self, dest_array):
        """
        Copy all the attributes of the circle buffer to another circle buffer

        Parameters
        ----------
        dest_array: Circle buffer
            Circle buffer to copy attributes to
        """
        dest_array.capacity = self.capacity
        for i in range(self.capacity):
            dest_array.set_element(i, self.get_element(i))

        if dest_array.mp_type == MPEnums.CIRCLE_BUFFER:
            # copy the head and tail as well
            dest_array._tail = self._tail
            dest_array._head = self._head

    def flush(self):
        """
        Empty the buffer of all elements
        """
        while not self.is_empty():
            self.dequeue()

    def to_tensor(self):
        """
        Copy the elements of the buffer to a Tensor and return the tensor
        """
        if self.is_empty():
            return Tensor.create(self.session, (0,))

        t = super().to_tensor()

        # remove data from tensor that is outside the
        # range of the cb's head and tail
        if self._head < self._tail:
            valid_data = t.data[self._head:self._tail+1]
        else:
            valid_data = np.concatenate((t.data[self._head:],
                                         t.data[:self._tail+1]),
                                        axis=0)

        return Tensor.create_from_data(self.session, valid_data)

    def assign_random_data(self, whole_numbers=False, vmin=0, vmax=1,
                           covariance=False):
        """
        Assign random data to the buffer. This is useful for testing
        and verification purposes.
        """
        super().assign_random_data(whole_numbers, vmin, vmax, covariance)
        self._head = 0
        self._tail = self.capacity - 1

    @classmethod
    def create(cls, sess, capacity, element_template):
        """ 
        Create circle buffer 

        Parameters
        ----------
        sess: Session Object
            Session where graph will exist
        capacity: Int
            Capacity of buffer
        element_template : any
            The template MindPype element to populate the array
            (see Array examples)

        Returns
        -------
        cb: Circle Buffer
        """
        cb = cls(sess, capacity, element_template)

        # add to the session
        sess.add_to_session(cb)

        return cb
