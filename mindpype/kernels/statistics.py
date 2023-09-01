from ..core import MPEnums
from ..kernel import Kernel
from ..graph import Node, Parameter
from ..containers import Scalar

import numpy as np
from scipy.stats import norm, chi2


class CDFKernel(Kernel):
    """
    Calculates the CDF for a distribution given a RV as input. Currently supports normal and chi2 distributions
    
    graph : Graph 
        Graph that the kernel should be added to

    inA : Tensor 
        First input trial data

    outA : Tensor 
        Output trial data

    dist : str, {'norm', 'chi2'}
        Distribution type
    
    df : shape_like
        The shape parameter(s) for the distribution. See scipy.stats.chi2 docstring for more detailed information
    
    loc : array_like, default = 0
        Location Parameter 
    
    scale : array_like, default = 1
        Scale Parameter
    """
    
    def __init__(self,graph,inA,outA,dist,df,loc,scale):
        """
        Kernel takes tensor input of RVs
        """
        super().__init__('CDF',MPEnums.INIT_FROM_NONE,graph)
        self.inputs = [inA]
        self.outputs = [outA]
        self._dist = dist
        self._loc = loc
        self._scale = scale
        self._df = df        
    
    def _initialize(self, init_inputs, init_outputs, labels):

        init_in = init_inputs[0]
        init_out = init_outputs[0]

        if init_out is not None and (init_in is not None and init_in.shape != ()):
            # update output size, as needed
            if init_out.virtual:
                init_out.shape = init_in.shape

            self._process_data(init_inputs, init_outputs)

    
    def _verify(self):
        """
        Verify the inputs and outputs are appropriately sized and typed
        """
        
        d_in = self.inputs[0]
        d_out = self.outputs[0]

        # first ensure the input and output are tensors
        if (d_in.mp_type != MPEnums.TENSOR or 
            d_out.mp_type != MPEnums.TENSOR):
                raise TypeError("CDF Kernel: Input and output must be tensors")
        
        input_shape = d_in.shape        
        
        # if the output is a virtual tensor and dimensionless, 
        # add the dimensions now
        if (d_out.virtual and len(d_out.shape) == 0):
            d_out.shape = input_shape
        
        
        # check that the dimensions of the output match the dimensions of
        # input
        if d_in.shape != d_out.shape:
            raise ValueError("CDF Kernel: Input and output must have the same shape")
        
        # check that the distribution is supported
        if not self._dist in ('norm','chi2'):
            raise ValueError("CDF Kernel: Distribution must be 'norm' or 'chi2'")
        
        if self._dist == 'chi2' and self._df == None:
            raise ValueError("CDF Kernel: Chi2 distribution requires a df parameter")
        
    def _process_data(self, inputs, outputs):
        if self._dist == 'norm':
            outputs[0].data = norm.cdf(inputs[0].data,
                                       loc=self._loc,
                                       scale=self._scale)
        elif self._dist == 'chi2':
            outputs[0].data = chi2.cdf(inputs[0].data,
                                       self._df,
                                       loc=self._loc,
                                       scale=self._scale)

    @classmethod
    def add_cdf_node(cls,graph,inA,outA,dist='norm',df=None,loc=0,scale=1):
        """
        Factory method to create a CDF node
        """
        
        # create the kernel object
        k = cls(graph,inA,outA,dist,df,loc,scale)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,MPEnums.INPUT),
                  Parameter(outA,MPEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node


class CovarianceKernel(Kernel):
    """
    Kernel to compute the covariance of tensors. If the input tensor is 
    unidimensional, will compute the variance. For higher rank tensors,
    highest order dimension will be treated as variables and the second
    highest order dimension will be treated as observations. 
    
    Parameters
    ----------
    graph : Graph 
        Graph that the kernel should be added to

    inA : Tensor or Scalar 
        First input trial data

    outA : Tensor or Scalar 
        Output trial data

    regularization : float, 0 < r < 1
        Regularization parameter


    Tensor size examples:
        Input:  A (kxmxn)
        Output: B (kxnxn)
        
        Input:  A (m)
        Output: B (1)
        
        Input:  A (mxn)
        Output: B (nxn)
        
        Input:  A (hxkxmxn)
        Output: B (hxkxnxn)
    """
    
    def __init__(self,graph,inputA,outputA,regularization):
        super().__init__('Covariance',MPEnums.INIT_FROM_NONE,graph)
        self.inputs = [inputA]
        self.outputs = [outputA]
        self._r = regularization

    def _initialize(self, init_inputs, init_outputs, labels):
        """
        Initialize internal state and initialization output of the kernel
        """

        init_in = init_inputs[0]
        init_out = init_outputs[0]

        if init_out is not None and (init_in is not None and init_in.shape != ()):
            # update output size, as needed
            if init_out.virtual:
                shape = list(init_in.shape)
                shape[-1] = shape[-2]
                init_out.shape = tuple(shape)

            self._process_data(init_inputs, init_outputs)


    def _verify(self):
        """
        Verify the inputs and outputs are appropriately sized
        """

        d_in = self.inputs[0]
        d_out = self.outputs[0]
        
        # first ensure the input and output are tensors
        if (d_in.mp_type != MPEnums.TENSOR or 
            d_out.mp_type != MPEnums.TENSOR):
            raise TypeError("Covariance Kernel: Input and output must be tensors")
        
        if self._r > 1 or self._r < 0:
            raise ValueError("Covariance Kernel: Regularization parameter must be between 0 and 1")
        
        # check the shape
        input_shape = d_in.shape
        input_rank = len(input_shape)
        
        # determine what the output shape should be
        if input_rank < 1 or input_rank > 3:
            raise ValueError("Covariance Kernel: Input must be rank 1, 2, or 3")
        elif input_rank == 1:
            output_shape = (1,)
        else:
            output_shape = list(input_shape)
            output_shape[-1] = output_shape[-2]
            output_shape = tuple(output_shape)
        
        # if the output is virtual and has no defined shape, set the shape now
        if d_out.virtual and len(d_out.shape) == 0:
            d_out.shape = output_shape
        
        # ensure the output tensor's shape equals the expected output shape
        if d_out.shape != output_shape:
            raise ValueError("Covariance Kernel: Output shape does not match expected shape")
        
    def _process_data(self, inputs, outputs):
        """
        Process input data according to outlined kernel function
        """
        shape = inputs[0].shape
        rank = len(shape)
        
        input_data = inputs[0].data
        
        if rank <= 2:
            covmat = np.cov(input_data)
            outputs[0].data = (1/(1+self._r) * 
                                    (covmat + self._r*np.eye(covmat.shape[0])))
        else:
            # reshape the input data so it's rank 3
            input_data = np.reshape(input_data,(-1,) + shape[-2:])
            output_data = np.zeros((input_data.shape[0],input_data.shape[1],
                                    input_data[1]))
            
            # calculate the covariance for each 'trial'
            for i in range(output_data.shape[0]):
                covmat = np.cov(input_data)
                output_data[i,:,:] = (1/(1+self._r) * 
                                        (covmat + self._r*np.eye(covmat.shape[0])))
            
            # reshape the output
            outputs[0].data = np.reshape(output_data,outputs[0].shape)
            
    
    @classmethod
    def add_covariance_node(cls,graph,inputA,outputA,regularization=0):
        """
        Factory method to create a covariance kernel and add it to a graph
        as a generic node object.
        
        Parameters
        ----------
        graph : Graph 
            Graph that the kernel should be added to

        inA : Tensor or Scalar 
            First input trial data

        outA : Tensor or Scalar 
            Output trial data

        regularization : float, 0 < r < 1
            Regularization parameter


        Tensor size examples:
            Input:  A (kxmxn)
            Output: B (kxnxn)
            
            Input:  A (m)
            Output: B (1)
            
            Input:  A (mxn)
            Output: B (nxn)
            
            Input:  A (hxkxmxn)
            Output: B (hxkxnxn)
        """
        
        # create the kernel object
        k = cls(graph,inputA,outputA,regularization)
        
        # create parameter objects for the input and output
        params = (Parameter(inputA,MPEnums.INPUT),
                  Parameter(outputA,MPEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node


class Descriptive:
    def _initialize(self, init_inputs, init_outputs, labels):
        """
        This kernel has no internal state that must be initialized
        """
        init_in = init_inputs[0]
        init_out = init_outputs[0]

        if init_out is not None and (init_in is not None and init_in.shape != ()):
            # update the output shape, as needed
            axis_adjusted = False
            if (len(self.inputs[0].shape) != len(init_in.shape) and
                self._axis >= 0):
                self._axis += 1 # adjust axis assuming stacked data
                axis_adjusted = True
                
            if init_out.virtual:
                phony_out = np.mean(init_in.data,
                                    axis=self._axis,
                                    keepdims=self._keepdims)
                init_out.shape = phony_out.shape
            
            self._process_data(init_inputs,init_outputs)
            
            if axis_adjusted:
                self._axis -= 1 # re-adjust axis


class MaxKernel(Descriptive, Kernel):
    """
    Kernel to extract maximum value along a Tensor axis

    Parameters
    ----------
    graph : Graph 
        Graph that the kernel should be added to

    inA : Tensor 
        Input data (max value will be extracted from here)

    outA : Tensor or Scalar 
        Output trial data

    axis : None or int or tuple of ints
        Axis or axes along which to operate. By default, flattened input in used.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as dimensions with size one.
    """
    
    def __init__(self,graph,inA,outA,axis=None,keepdims=False):
        super().__init__('Max',MPEnums.INIT_FROM_NONE,graph)
        self.inputs = [inA]
        self.outputs = [outA]

        self._axis = axis
        self._keepdims = keepdims

    def _process_data(self, inputs, outputs):
        if outputs[0].mp_type == MPEnums.SCALAR:
            outputs[0].data = np.amax(inputs[0].data).item()
        else:
            outputs[0].data = np.amax(inputs[0].data,
                                      axis=self._axis,
                                      keepdims=self._keepdims)

    @classmethod
    def add_max_node(cls,graph,inA,outA,axis=None,keepdims=False):
        """
        Factory method to create a maximum value kernel 
        and add it to a graph as a generic node object.

        Parameters
        ----------
        graph : Graph 
            Graph that the node should be added to

        inA : Tensor 
            Input data (max value will be extracted from here)

        outA : Tensor or Scalar 
            Output trial data

        axis : None or int or tuple of ints
            Axis or axes along which to operate. By default, flattened input in used.

        keepdims : bool
            If this is set to True, the axes which are reduced are left in the result as dimensions with size one.
        """
        
        # create the kernel object
        k = cls(graph,inA,outA)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,MPEnums.INPUT),
                  Parameter(outA,MPEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node


class MinKernel(Descriptive, Kernel):
    """
    Kernel to extract minimum value within a Tensor

    Parameters
    ----------
    graph : Graph 
        Graph that the kernel should be added to

    inA : Tensor 
        Input data (min value will be extracted from here)

    outA : Tensor or Scalar 
        Output trial data

    axis : None or int or tuple of ints
        Axis or axes along which to operate. By default, flattened input in used.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as dimensions with size one.
    """
    
    def __init__(self,graph,inA,outA,axis=None,keepdims=False):
        super().__init__('Min',MPEnums.INIT_FROM_NONE,graph)
        self.inputs = [inA]
        self.outputs = [outA]
        self._axis = axis
        self._keepdims = keepdims


    def _process_data(self, inputs, outputs):
        if outputs[0].mp_type == MPEnums.SCALAR:
            outputs[0].data = np.amin(inputs[0].data).item()
        else:
            outputs[0].data = np.amin(inputs[0].data,
                                      axis=self._axis,
                                      keepdims=self._keepdims)

    @classmethod
    def add_min_node(cls,graph,inA,outA,axis=None,keepdims=False):
        """
        Factory method to create a minimum value kernel 
        and add it to a graph as a generic node object.

        Calculates the mean of values in a tensor

        Parameters
        ----------
        graph : Graph 
            Graph that the node should be added to

        inA : Tensor 
            Input data (min value will be extracted from here)

        outA : Tensor 
            Output trial data

        axis : None or int or tuple of ints
            Axis or axes along which to operate. By default, flattened input in used.

        keepdims : bool
            If this is set to True, the axes which are reduced are left in the result as dimensions with size one.
        """
        
        # create the kernel object
        k = cls(graph,inA,outA)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,MPEnums.INPUT),
                  Parameter(outA,MPEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node


class MeanKernel(Descriptive, Kernel):
    """
    Calculates the mean of values in a tensor

    Parameters
    ----------
    graph : Graph 
        Graph that the kernel should be added to

    inA : Tensor 
        Input data

    outA : Tensor 
        Output trial data

    axis : None or int or tuple of ints
        Axis or axes along which to operate. By default, flattened input in used.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as dimensions with size one.
    """
    
    def __init__(self,graph,inA,outA,axis=None,keepdims=False):
        """
        Kernal calculates arithmetic mean of values in tensor or array
        """
        super().__init__('Mean',MPEnums.INIT_FROM_NONE,graph)
        self.inputs = [inA]
        self.outputs = [outA]
        self._axis = axis
        self._keepdims = keepdims

    def _process_data(self, inputs, outputs):
        if outputs[0].mp_type == MPEnums.SCALAR:
            outputs[0].data = np.mean(inputs[0].data).item()
        else:
            outputs[0].data = np.mean(inputs[0].data,
                                      axis=self._axis,
                                      keepdims=self._keepdims)

    @classmethod
    def add_mean_node(cls,graph,inA,outA,axis=None,keepdims=False):
        """
        Factory method to create a mean calculating kernel

        Calculates the mean of values in a tensor

        Parameters
        ----------
        graph : Graph 
            Graph that the kernel should be added to

        inA : Tensor 
            Input data

        outA : Tensor 
            Output trial data

        axis : None or int or tuple of ints
            Axis or axes along which to operate. By default, flattened input in used.

        keepdims : bool
            If this is set to True, the axes which are reduced are left in the result as dimensions with size one.
        """
        
        # create the kernel object
        k = cls(graph,inA,outA,axis,keepdims)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,MPEnums.INPUT),
                  Parameter(outA,MPEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node

class StdKernel(Descriptive, Kernel):
    """
    Calculates the standard deviation of values in a tensor

    graph : Graph 
        Graph that the kernel should be added to

    inA : Tensor 
        First input trial data

    outA : Tensor 
        Output trial data

    axis : None or int or tuple of ints, optional
        Axis or axes along which the standard deviation is computed. The default is to compute the standard deviation of the flattened array.

    ddof : int, optional
        Means Delta Degrees of Freedom. The divisor used in calculations is N - ddof, where N represents the number of elements. By default ddof is zero.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as dimensions with size one.
    """
    
    def __init__(self,graph,inA,outA,axis=None,ddof=0,keepdims=False):
        """
        Kernal calculates arithmetic standard deviation of values in tensor
        """
        super().__init__('Std',MPEnums.INIT_FROM_NONE,graph)
        self.inputs = [inA]
        self.outputs = [outA]
        self._axis = axis
        self._ddof = ddof
        self._keepdims = keepdims
        
    def _verify(self):
        super()._verify()

        d_in = self.inputs[0]

        # verify ddof is valid
        if isinstance(self._axis, int):
            N = d_in.shape[self._axis]
        else:
            if self._axis == None:
                dims = d_in.shape
            else:
                dims = [d_in.shape[a] for a in self._axis]
            N = 1
            for dim in dims:
                N *= dim

        if N <= self._ddof:
            raise ValueError("Std Kernel: ddof must be less than the number of elements in the tensor")

    def _process_data(self, inputs, outputs):
        """
        Process data according to outlined kernel function
        """
        outputs[0].data = np.std(inputs[0].data,
                                 axis=self._axis,
                                 ddof=self._ddof,
                                keepdims=self._keepdims)

    @classmethod
    def add_std_node(cls,graph,inA,outA,axis=None,ddof=0,keepdims=False):
        """
        Factory method to add a standard deviation node to a graph

        Calculates the standard deviation of values in a tensor

        graph : Graph 
            Graph that the kernel should be added to
    
        inA : Tensor 
            First input trial data
    
        outA : Tensor 
            Output trial data
    
        axis : None or int or tuple of ints, optional
            Axis or axes along which the standard deviation is computed. The default is to compute the standard deviation of the flattened array.
    
        ddof : int, optional
            Means Delta Degrees of Freedom. The divisor used in calculations is N - ddof, where N represents the number of elements. By default ddof is zero.
    
        keepdims : bool
            If this is set to True, the axes which are reduced are left in the result as dimensions with size one.
        """
        
        # create the kernel object
        k = cls(graph,inA,outA,axis,ddof)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,MPEnums.INPUT),
                  Parameter(outA,MPEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node

class VarKernel(Descriptive, Kernel):
    """
    Calculates the variance of values in a tensor

    graph : Graph 
        Graph that the kernel should be added to

    inA : Tensor or Scalar 
        Input trial data

    outA : Tensor or Scalar 
        Output trial data

    axis : None or int or tuple of ints, optional
        Axis or axes along which the variance is computed. The default is to compute the variance of the flattened array.

    ddof : int, optional
        "Delta Degrees of Freedom": the divisor used in the calculation is N - ddof, where N represents the number of elements. By default ddof is zero.
    
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this option, the result will broadcast correctly against the input array.
    """
    
    def __init__(self,graph,inA,outA,axis,ddof,keep_dims):
        """
        Kernal calculates arithmetic variance of values in tensor
        """
        super().__init__('Var',MPEnums.INIT_FROM_NONE,graph)
        self.inputs = [inA]
        self.outputs = [outA]
        self._axis = axis
        self._ddof = ddof
        self._keepdims = keep_dims
        
    def verify(self):
        super()._verify()

        d_in = self.inputs[0]

        # verify ddof is valid
        if isinstance(self._axis, int):
            N = d_in.shape[self._axis]
        else:
            if self._axis == None:
                dims = d_in.shape
            else:
                dims = [d_in.shape[a] for a in self._axis]
            N = 1
            for dim in dims:
                N *= dim

        if N <= self._ddof:
            raise ValueError("Var Kernel: ddof must be less than the number of elements in the tensor")

    def _process_data(self, inputs, outputs):
        """
        Process data according to outlined kernel function
        """
        outputs[0].data = np.var(inputs[0].data,
                                 axis=self._axis,
                                 ddof=self._ddof,
                                 keepdims=self._keepdims)

    @classmethod
    def add_var_node(cls,graph,inA,outA,axis=None,ddof=0,keep_dims=False):
        """
        Factory method to create a variance kernel

        graph : Graph 
            Graph that the kernel should be added to

        inA : Tensor or Scalar 
            Input trial data

        outA : Tensor or Scalar 
            Output trial data

        axis : None or int or tuple of ints, optional
            Axis or axes along which the variance is computed. The default is to compute the variance of the flattened array.

        ddof : int, optional
            "Delta Degrees of Freedom": the divisor used in the calculation is N - ddof, where N represents the number of elements. By default ddof is zero.
        
        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this option, the result will broadcast correctly against the input array.
            
        """
        
        # create the kernel object
        k = cls(graph,inA,outA,axis,ddof,keep_dims)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,MPEnums.INPUT),
                  Parameter(outA,MPEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node

class ZScoreKernel(Kernel):
    """
    Calculate a z-score for an tensor or scalar input

    graph : Graph 
        Graph that the kernel should be added to

    inA : Tensor or Scalar 
        Input trial data

    outA : Tensor or Scalar 
        Output trial data

    init_data: Tensor or Array 
        Initialization data (n_trials, n_channels, n_samples)
    """
    
    def __init__(self,graph,inA,outA,init_data):
        super().__init__('Zscore',MPEnums.INIT_FROM_DATA,graph)
        self.inputs = [inA]
        self.outputs = [outA]

        if init_data is not None:
            self.init_inputs = [init_data]

        self._mu = 0
        self._sigma = 0
        self._initialized = False

    def _initialize(self, init_inputs, init_outputs, labels):
        """
        Initialize the mean and std. Call initialization_execution if downstream nodes are missing training data
        """

        init_in = init_inputs[0]
        init_out = init_outputs[0]

        if (init_in.mp_type != MPEnums.ARRAY and
            init_in.mp_type != MPEnums.TENSOR and
            init_in.mp_type != MPEnums.CIRCLE_BUFFER):
            raise TypeError("ZScore Kernel: Initialization data must be an array or tensor")

        if init_in.mp_type == MPEnums.TENSOR:
            if len(init_in.squeeze().shape) != 1:
                raise ValueError("ZScore Kernel: Initialization data must be rank 1")
        else:
            e = init_in.get_element(0)
            if e.mp_type == MPEnums.TENSOR:
                if (np.squeeze(e.shape != ())):
                    raise ValueError("ZScore Kernel: Initialization data must be rank 1")
            elif e.mp_type == MPEnums.SCALAR:
                if not e.data_type in Scalar.valid_numeric_types():
                    raise ValueError("ZScore Kernel: Initialization data must be numeric")
            else:
                raise ValueError("ZScore Kernel: Initialization data Arrays must contain tensors or scalars")

        if init_in.mp_type == MPEnums.TENSOR:
            d = init_in.data.squeeze()
        else:
            e = init_in.get_element(0)
            dl = []
            for i in range(init_in.capacity):
                elem_data = init_in.get_element(i).data
                if e.mp_type == MPEnums.TENSOR:
                    dl.append(elem_data)
                else:
                    # convert scalar values to numpy arrays
                    dl.append(np.asarray([elem_data]))

            # stack values into a single numpy array
            d = np.concatenate(dl,axis=0)


        # calc mean and std
        N = d.shape[0]
        self._mu = np.sum(d) / N
        self._sigma = np.sqrt(np.sum((d - self._mu)**2) / (N-1))

        if init_out is not None and (init_in is not None and init_in.shape != ()):
            # set output size, as needed
            if init_out.virtual:
                init_out.shape = init_in.shape

            self._process_data(init_inputs, init_outputs)

    
    def _process_data(self, inputs, outputs):
        """
        Process data according to outlined kernel function
        """
        
        outputs[0].data = (inputs[0].data - self._mu) / self._sigma

    @classmethod
    def add_zscore_node(cls,graph,inA,outA,init_data):
        """
        Factory method to create a z-score value kernel 
        and add it to a graph as a generic node object.

        graph : Graph 
            Graph that the kernel should be added to

        inA : Tensor or Scalar 
            Input trial data

        outA : Tensor or Scalar 
            Output trial data

        init_data: Tensor or Array 
            Initialization data (n_trials, n_channels, n_samples)
        """
        
        # create the kernel object
        k = cls(graph,inA,outA,init_data)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,MPEnums.INPUT),
                  Parameter(outA,MPEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node
