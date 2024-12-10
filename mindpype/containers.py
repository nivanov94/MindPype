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
    virtual : bool, default = False
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
    virtual : bool
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
        Copy the data value to another scalar object.
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
        self, sess, data_type, value=None, virtual=False, ext_src=None, ext_out=None
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

        self.virtual = virtual

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
        Assign random data to the scalar. 
        
        This method assigns a random value to the scalar's `data` attribute.
        This method is typically used for graph verification.

        Parameters
        ----------
        whole_numbers : bool, default=False
            If True, assigns data that is only whole numbers.
        vmin : int, default=0
            Lower limit for values in the random data.
        vmax : int, default=1
            Upper limit for values in the random data.
        covariance : bool, default=False
            Ignored for scalars. This parameter is included for consistency
            with the `assign_random_data` method in the `Tensor` class.
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
        update the scalar's `data` attribute. The label parameter may be used 
        to specify a class label corresponding to the data to be polled.

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

    ## Factory Methods
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
    Represeents a multi-dimensional array of data that can be ingested and
    produced by nodes within MindPype graphs.

    A `Tensor` object contains a multi-dimensional array of data with a
    specified shape.

    Parameters
    ----------
    sess : Session object
        The session object in which the tensor will exist and operate.
    shape : tuple
        Shape of the tensor data array.
    data : ndarray, default = None
        Data value represented by the tensor object. If not specified, a
        default value of zeros will be assigned based on the shape.
    virtual : bool, default = False
        If True, the tensor is a virtual object, otherwise it is non-virtual.
    ext_src : Source, default = None
        A volatile external data source from which the tensor will pull data
        during graph execution. If the tensor does not represent an external
        data source, set ext_src to None.
    ext_out : Source, default = None
        A volatile external data source to which the tensor will push data
        during graph execution. If the tensor does not push data to an
        external source, set ext_out to None.
        
    Attributes
    ----------
    shape : tuple
        Shape of the tensor data array.
    data : ndarray
        Data value represented by the tensor object.
    virtual : bool
        Indicates whether the tensor is a virtual object.
    ext_src : Source
        External data source from which the tensor will pull data during
        graph execution.
    ext_out : output Source
        External data source to which the tensor will push data during
        graph execution.
    volatile : bool
        Indicates whether the tensor is a volatile object that needs to pull
        from external sources during graph execution.
    volatile_out : bool
        Indicates whether the tensor is a volatile object that needs to push
        to external sources during graph execution

    Methods
    -------
    make_copy()
        Create and return a deep copy of the tensor.
    copy_to(dest_tensor)
        Copy the data value to another tensor object.
    assign_random_data(whole_numbers=False, vmin=0, vmax=1, covariance=False)
        Assign random data to the tensor.
    poll_volatile_data(label=None)
        Pull data from external sources and assign it to the tensor's data
        attribute.
    push_volatile_outputs(label=None)
        Push the tensor's data to external sources.
    create(sess, shape)
        Factory method to create a non-virtual Tensor object.
    create_virtual(sess, shape)
        Factory method to create a virtual Tensor object.
    create_from_data(sess, data)
        Factory method to create a Tensor with a specified data array.
    create_from_source(sess, shape, src, direction="input")
        Factory method to create a non-virtual, volatile Tensor object with an
        external data source or destination.
    
    Notes
    -----
    - Virtual tensors should only be used to define connections between nodes
      in a graph. The data within virtual tensors may be modified by `Graph`
      or `Node` methods and they should not be used to store data that is used
      outside of graph execution.
    - Volatile tensors cannot be virtual.
    """

    def __init__(
            self, sess, shape, data=None, virtual=False, ext_src=None, ext_out=None
        ):
        """Init."""
        super().__init__(MPEnums.TENSOR, sess)
        self._shape = tuple(shape)
        self.virtual = virtual
        self.ext_src = ext_src
        self.ext_out = ext_out

        if data is not None:
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

    @property
    def data(self):
        """
        Retrieve the tensor data value.

        This property provides access to the tensor data value stored within
        the `Tensor` object. The data is represented as a multi-dimensional
        array with a specified shape.

        Returns
        -------
        ndarray
            The tensor data value encapsulated by the `Tensor` object
        """
        return self._data

    @property
    def shape(self):
        """
        Retrieve the shape of the tensor data array.

        This property provides access to the shape of the tensor data array
        stored within the `Tensor` object. The shape is represented as a tuple
        of integers indicating the size of each dimension of the array.

        Returns
        -------
        tuple
            The shape of the tensor data array.
        """
        return self._shape

    
    @data.setter
    def data(self, data):
        """
        Set the data of the Tensor.

        This method updates the data stored in the Tensor. If the shape of the 
        provided data does not match the current shape of the Tensor, the shape
        of the Tensor must first be updated using the appropriate method before 
        assigning the new data. An error will be raised if the shapes are 
        incompatible.

        Parameters
        ----------
        data : numpy.ndarray or scalar
            Data to set as the tensor's data attribute. Scalars (e.g., int, 
            float, complex) will be automatically converted into a numpy array.

        Raises
        ------
        TypeError
            If the input data is not a numpy array or cannot be converted 
            to one.
        ValueError
            If the shape of the input data does not match the current shape of
            the Tensor.
        """
        # Ensure the input is a numpy array or convert a scalar to a 
        # numpy array
        if not isinstance(data, np.ndarray):
            if isinstance(data, (int, float, complex)):
                data = np.array(data)  # Convert scalar to numpy array
            else:
                raise TypeError(
                    "Data assigned to Tensor must be a numpy array or a scalar."
                )

        # Handle special case: scalar data with all singleton dimensions
        if (np.prod(data.shape) == 1 and np.prod(self.shape) == 1):
            while len(data.shape) < len(self.shape):
                data = np.expand_dims(data, axis=0)
            while len(data.shape) > len(self.shape):
                data = np.squeeze(data, axis=0)

        # Adjust virtual Tensor's shape if necessary
        if self.virtual and self.shape != data.shape:
            self._shape = data.shape

        # Set the data if shapes match; otherwise, raise an error
        if self.shape == data.shape:
            self._data = data
        else:
            raise ValueError(
                f"Shape mismatch: Tensor shape {self.shape} does not match"
                f"data shape {data.shape}."
            )

    @shape.setter
    def shape(self, shape):
        """
        Set the shape of the Tensor.

        This method changes the shape of the Tensor. If the Tensor is
        non-virtual, the shape cannot be changed and an error will be raised.
        The data attribute of the Tensor will be updated to a new array of 
        zeros.

        Parameters
        ----------
        shape : tuple
            New shape for the Tensor data array.

        Raises
        ------
        TypeError
            If the Tensor is non-virtual and the shape is attempted to be
            changed.
        """
        if self.virtual:
            self._shape = shape
            # when changing the shape write a zero tensor to data
            self.data = np.zeros(shape)
        else:
            raise TypeError("Attempted to change shape of non-virtual Tensor.")

    def make_copy(self):
        """
        Create and return a deep copy of the tensor.

        This method creates a new `Tensor` object with the same attributes as
        the original object, including the shape, data, virtual status, and
        external data sources. The copied tensor is added to the same session
        as the original tensor.

        Returns
        -------
        Tensor
            A deep copy of the tensor.
        """
        cpy = Tensor(
            self.session,
            self.shape,
            self.data,
            self.virtual,
            self.ext_src,
            self.ext_out
        )

        # add the copy to the session
        self.session.add_to_session(cpy)

        return cpy

    def copy_to(self, dest_tensor):
        """
        Copy the data value to another tensor object.

        This method copies the data value from the referenced tensor to the
        destination tensor object. Other attributes including the virtual
        status and external data sources are not copied as they should
        not be modified after the tensor is created.

        Parameters
        ----------
        dest_tensor : Tensor
            Tensor object to which the data value will be copied.

        Raises
        ------
        ValueError
            If the destination tensor's shape does not match the source
            tensor's shape.
        """
        # adjust the shape of the destination tensor if it is a virtual tensor
        if dest_tensor.virtual and dest_tensor.shape != self.shape:
            dest_tensor.shape = self.shape

        dest_tensor.data = self.data

    def assign_random_data(
            self, whole_numbers=False, vmin=0, vmax=1, covariance=False
    ):
        """
        Assign random data to the tensor.

        This method assigns random data to the tensor object. The data type of
        the tensor determines the type of random data that is assigned. The
        data value is generated based on the specified parameters. This method
        is typically used for graph verification.

        Parameters
        ----------
        whole_numbers : bool, default=False
            If True, assigns data that is only whole numbers.
        vmin : int, default=0
            Lower limit for values in the random data.
        vmax : int, default=1
            Upper limit for values in the random data.
        covariance : bool, default=False
            If True, assigns a random covariance matrix to the tensor.

        Raises
        ------
        ValueError
            If the tensor shape is not 2D or 3D when assigning a covariance
            matrix.
        ValueError
            If the tensor shape is not square when assigning a covariance
            matrix.
        """
        if whole_numbers:
            self.data = np.random.randint(vmin, vmax+1, size=self.shape)
        else:
            num_range = vmax - vmin
            self.data = num_range * np.random.rand(*self.shape) + vmin

        if covariance:
            rank = len(self.shape)
            if rank != 2 and rank != 3:
                raise ValueError(
                    "Cannot assign random covariance matrix to "
                    "tensor with rank other than 2 or 3"
                )

            if self.shape[-2] != self.shape[-1]:
                raise ValueError(
                    "Cannot assign random covariance matrix to "
                    "tensor with non-square last two dimensions"
                )

            if rank == 2:
                self.data = np.cov(self.data)
            else:
                for i in range(self.shape[0]):
                    self.data[i,:,:] = np.cov(self.data[i,:,:])

            # add regularization
            r = 0.001 # could be a method parameter in the future
            self.data = (1-r)*self.data + r*np.eye(self.shape[-1])

    def poll_volatile_data(self, label=None):
        """
        Pull data from external sources to update the tensor's data value.

        This method uses the tensor's external data source to pull data and
        update the tensor's `data` attribute. The label parameter may be used 
        to specify a class label corresponding to the data to be polled.

        Parameters
        ----------
        Label : int, default = None
            Class label corresponding to class data to poll, if applicable.
            This is typically used when polling epoched data from a file
            source.
        """
        # check if the data is actually volatile, if not just return
        if self.volatile:
            self.data = self.ext_src.poll_data(label=label)

    def push_volatile_outputs(self, label=None):
        """
        Push the tensor's data to an external source.

        This method uses the tensor's external output source to push data
        during graph execution. The label parameter may be used to specify a
        class label corresponding to the data to be pushed.

        Parameters
        ----------
        Label : int, default = None
            Class label corresponding to class data to push, if applicable.
        """
        # check if the output is actually volatile first
        if self.volatile_out:
            # push the data
            self.ext_out.push_data(self.data)

    ## Factory Methods
    @classmethod
    def create(cls, sess, shape):
        """
        Factory method to create a non-virtual Tensor object and register it
        within a session.

        This method creates a new `Tensor` object with a default data value
        of zeros based on the specified shape. The tensor is added to the
        specified session.

        Parameters
        ----------
        sess : Session 
            Session where the tensor will exist.
        shape : tuple
            Shape of the tensor data array.
        
        Return
        ------
        Tensor
            A reference to the tensor that was created and added to the 
            session.

        """
        t = cls(sess, shape)
        sess.add_to_session(t)
        return t

    @classmethod
    def create_virtual(cls, sess, shape=()):
        """
        Factory method to create a virtual Tensor object and register it within
        a session.

        This method creates a new virtual `Tensor` object with the specified
        shape. The tensor is added to the specified session.

        Parameters
        ----------
        sess : Session
            Session where the tensor will exist.
        shape : tuple, default=()
            Shape of the tensor data array. If not specified, the shape will
            be an empty tuple. Virutal tensors can be dynamically resized
            after creation.
        """
        t = cls(sess, shape, virtual=True)
        sess.add_to_session(t)
        return t

    @classmethod
    def create_from_data(cls, sess, data):
        """
        Factory method to create a Tensor with a specified data array and 
        register it within a session.

        This method creates a new `Tensor` object with the specified data
        array. The tensor is added to the specified session.

        Parameters
        ----------
        sess : Session
            Session where the tensor will exist.
        data : ndarray
            Data value to be stored within the tensor. The shape of the data
            array will be used as the shape of the tensor.
        
        Return
        ------
        Tensor
            A reference to the tensor that was created and added to the session.
        """
        if isinstance(data, (list, tuple)):
            data = np.asarray(data)

        t = cls(sess, data.shape, data)
        sess.add_to_session(t)
        return t

    @classmethod
    def create_from_source(cls, sess, shape, src, direction="input"):
        """
        Factory method to create a non-virtual, volatile Tensor object with an
        external data source or destination.

        This method creates a new `Tensor` object with an external data source
        and adds it to the specified session.

        Parameters
        ----------
        sess : Session
            Session where the tensor will exist.
        shape : tuple
            Shape of the tensor data array.
        src : Source
            External data source from which to pull or push data.
        direction : str, default="input"
            Direction of the external data source. Options are "input" or
            "output". If "input", the tensor will pull data from the source.
            If "output", the tensor will push data to the source.

        Return
        ------
        Tensor
            A reference to the tensor that was created and added to the session.

        Raises
        ------
        ValueError
            If the specified direction is not "input" or "output".
        """
        if direction == "input":
            t = cls(sess, shape, ext_src=src)
        elif direction == "output":
            t = cls(sess, shape, ext_out=src)
        else:
            raise ValueError("Invalid direction")

        # add the tensor to the session
        sess.add_to_session(t)
        return t


class Array(MPBase):
    """
    Represents an array of other MindPype Container objects that can be 
    ingested and produced by nodes within MindPype graphs.

    An `Array` object contains instances of other MindPype Containers (i.e.,
    scalars and tensors). Array elements must be homogenous, meaning that all
    elements must be of the same type (i.e., all scalars or all tensors).
    
    Parameters
    ----------
    sess : Session object
        The session object in which the array will exist and operate.
    capacity : int
        Number of elements within the array.
    element_template : Scalar or Tensor
        A meta tensor or scalar object that will be used to create the array.
        The array will be initialized with copies of this object.

    Attributes
    ----------
    virtual : bool
        Always False. Virtual arrays are not supported. The virtual attribute
        is included for consistency with other MindPype Container objects.
    volatile : bool
        Always False. Volatile arrays are not supported. The volatile attribute
        is included for consistency with other MindPype Container objects.
    capacity: int
        Number of elements within the array.
    elements: list of Scalar or Tensor objects
        Elements of the array

    Methods
    -------
    get_element(index)
        Returns the element at a specific index within the array.
    set_element(index, element)
        Changes the element at a specific index to a specified value.
    make_copy()
        Create and return a deep copy of the array.
    copy_to(dest_array)
        Copy the data value to another array object.
    assign_random_data(whole_numbers=False, vmin=0, vmax=1, covariance=False)
        Assign random data to the elements of the array.
    to_tensor()
        Stack the elements of the array into a new Tensor object.
    create(sess, capacity, element_template)
        Factory method to create an Array object.
    
    See Also
    --------
    Scalar : Represents a single data value within MindPype.
    Tensor : Represents a multi-dimensional array of data within MindPype.
    """

    def __init__(self, sess, capacity, element_template):
        """ Init """
        super().__init__(MPEnums.ARRAY, sess)

        self.capacity = capacity
        self.elements = [None] * capacity

        # create the elements
        for i in range(capacity):
            self.elements[i] = element_template.make_copy()

        ## Set these attributes for API consistency with other containers
        self.virtual = False  # no virtual arrays for now
        self.volatile = False  # no volatile arrays for now...
        self.volatile_out = False  # no volatile arrays for now...

    def get_element(self, index):
        """
        Returns the element from a specified index from the array.

        Parameters
        ----------
        index : int
            Index is the position within the array with the element will be
            returned.

        Return
        ------
        Scalar or Tensor
            The element at the specified index within the array.

        Raises
        ------
        ValueError
            If the specified index is out of bounds for the array.
        """
        # if negative index, convert to positive
        if index < 0:
            index = self.capacity + index

        if index >= self.capacity:
            raise ValueError(
                f"Index {index} out of bounds for Array with "
                f"capacity {self.capacity}."                
            )

        return self.elements[index]

    def set_element(self, index, element):
        """
        Sets the element at a specified index within the array.

        The method will copy the attributes of the parameter element to the
        element at the specified index within the array. The element must be
        of the same type as the other elements within the array.

        Parameters
        ----------
        index : int
            Index within the array to set.

        element : Scalar or Tensor
            The element to set at the specified index. The element must be
            of the same type as the other elements within the array.

        Raises
        ------
        ValueError
            If the specified index is out of bounds for the array.
        """
        # if negative index, convert to positive
        if index < 0:
            index = self.capacity + index

        if index >= self.capacity:
            raise ValueError(
                f"Index {index} out of bounds for Array with "
                f"capacity {self.capacity}."
            )
        element.copy_to(self._elements[index])

    @property
    def num_elements(self):
        # this property is included to allow for seamless abstraction with
        # circle buffer property
        return self.capacity

    def make_copy(self):
        """
        Create and return a deep copy of the array.

        This method creates a new `Array` object with the same attributes as
        the original object, including the capacity and element objects. The
        copied array is added to the same session as the original array.

        Returns
        -------
        Array
            A deep copy of the array.
        """
        cpy = Array(
            self.session,
            self.capacity,
            self.get_element(0)
        )

        # copy the contents of the array to the new array
        for e in range(self.capacity):
            cpy.set_element(e, self.get_element(e))

        # add the copy to the session
        self.session.add_to_session(cpy)

        return cpy

    def copy_to(self, dest_array):
        """
        Copy the contents of the array to another array object. 

        This method copies the contents of the array's elements to the 
        destination array object. The destination array must have the same
        capacity and data element type as the source array.

        Parameters
        ----------
        dest_array : Array 
            Array object to which the data value will be copied.

        Raises
        ------
        ValueError
            If the destination array's capacity does not match the source
            array's capacity.
        TypeError
            If the destination array's element type does not match the source
            array's element type.
        """
        # Validate the destination array's capacity
        if dest_array.capacity != self.capacity:
            raise ValueError(
                f"Cannot copy data from array with capacity {self.capacity} "
                f"to array with capacity {dest_array.capacity}."
            )
    
        # Validate the destination array's element type
        if dest_array.get_element(0).mp_type != self.get_element(0).mp_type:
            raise TypeError(
                "Cannot copy data from array with element type "
                f"{str(dest_array.get_element(0).mp_type)} to array with "
                f"element type {str(dest_array.get_element(0).mp_type)}."
            )

        for i in range(self.capacity):
            dest_array.set_element(i, self.get_element(i))

    def assign_random_data(
            self, whole_numbers=False, vmin=0, vmax=1, covariance=False
    ):
        """
        Assign random data to the elements of the array.

        This method assigns random data to the elements of the array. For each
        element within the array, the element's `assign_random_data` method is
        called to assign random data based on the specified parameters. This
        method is typically used for graph verification.

        Parameters
        ----------
        whole_numbers : bool, default=False
            If True, assigns data that is only whole numbers.
        vmin : int, default=0
            Lower limit for values in the random data.
        vmax : int, default=1
            Upper limit for values in the random data.
        covariance : bool, default=False
            If True, assigns a random covariance matrix to tensor elements
            within the array.

        See Also
        --------
        Tensor.assign_random_data : Assign random data to a Tensor object.
        Scalar.assign_random_data : Assign random data to a Scalar object.
        """
        for i in range(self.capacity):
            self.get_element(i).assign_random_data(
                whole_numbers, vmin, vmax, covariance
            )

    def to_tensor(self):
        """
        Stack the elements of the array into a new Tensor object.

        This method stacks the elements of the array into a new `Tensor` object
        with the same shape as the array. The elements are stacked along the 
        first dimension.

        Returns
        -------
        Tensor
            A new Tensor object with the elements of the array stacked along
            the first dimension.

        Raises
        ------
        TypeError
            If the array contains non-numeric Scalar elements.
        """
        element = self.get_element(0)

        if element.mp_type == MPEnums.SCALAR:
            # ensure all elements are numeric scalars
            for ei in range(self.capacity):
                if not self.get_element(ei).is_numeric:
                    raise TypeError(
                        "Cannot convert Array to Tensor: Array contains "
                        "non-numeric Scalar elements."
                    )

        # extract elements and stack into numpy array
        elements = [self.get_element(i).data for i in range(self.capacity)]
        stacked_elements = np.stack(elements)

        # create tensor
        return Tensor.create_from_data(self.session, stacked_elements)

    @classmethod
    def create(cls, sess, capacity, element_template):
        """
        Factory method to create an Array object and register it within a
        session.

        This method creates a new `Array` object with a specified capacity.
        All elements within the array are initialized with copies of the
        specified element template. The array is added to the specified 
        session.

        Parameters
        ----------
        sess : Session object
            Session where the Array object will exist.
        capacity : int
            Number of elements within the array.
        element_template : Scalar or Tensor
            A meta tensor or scalar object that will be used to create the array.
            The array will be initialized with copies of this object.

        Return
        ------
        Array
            A reference to the array that was created and added to the session.

        Examples
        --------
        >>> a = Array.create(sess, 10, Scalar.create(sess, int))  # create an array of 10 integer scalars
        """
        a = cls(sess, capacity, element_template)
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
