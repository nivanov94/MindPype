from ..core import BCIP, BcipEnums
from ..kernel import Kernel
from ..graph import Node, Parameter
from ..containers import Scalar

import numpy as np
from scipy.stats import norm, chi2


class CDFKernel(Kernel):
    """
    Calculates the CDF for a distribution given a RV as input. Currently supports normal and chi2 distributions
    
    graph : Graph Object
        - Graph that the kernel should be added to

    inA : Tensor object
        - First input trial data

    outA : Tensor object
        - Output trial data

    dist : str, {'norm', 'chi2'}
        - Distribution type
    
    df : shape_like
        - The shape parameter(s) for the distribution. See scipy.stats.chi2 docstring for more detailed information
    
    loc : array_like, default = 0
        - Location Parameter 
    
    scale : array_like, default = 1
        - Scale Parameter
    """
    
    def __init__(self,graph,inA,outA,dist,df,loc,scale):
        """
        Kernel takes tensor input of RVs
        """
        super().__init__('CDF',BcipEnums.INIT_FROM_NONE,graph)
        self._inA  = inA
        self._outA = outA
        self._dist = dist
        self._loc = loc
        self._scale = scale
        self._df = df        
    
        self._init_inA = None
        self._init_outA = None
        
        self._labels = None

    def initialize(self):
        sts = BcipEnums.SUCCESS

        if self._init_outA != None:
            # update output size, as needed
            if len(self._init_outA.shape) == 0:
                self._init_outA.shape = self._init_inA.shape

            sts = self._process_data(self._init_inA, self._init_outA)
        
        return BcipEnums.SUCCESS
        
    
    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized and typed
        """
        
        # first ensure the input and output are tensors
        if (self._inA._bcip_type != BcipEnums.TENSOR or 
            self._outA._bcip_type != BcipEnums.TENSOR):
                return BcipEnums.INVALID_PARAMETERS
        
        input_shape = self._inA.shape        
        
        # if the output is a virtual tensor and dimensionless, 
        # add the dimensions now
        if (self._outA.virtual and len(self._outA.shape) == 0):
            self._outA.shape = input_shape
        
        
        # check that the dimensions of the output match the dimensions of
        # input
        if self._inA.shape != self._outA.shape:
            return BcipEnums.INVALID_PARAMETERS
        
        # check that the distribution is supported
        if not self._dist in ('norm','chi2'):
            return BcipEnums.INVALID_NODE
        
        if self._dist == 'chi2' and self._df == None:
            return BcipEnums.INVALID_NODE
        
        return BcipEnums.SUCCESS
        
    def _process_data(self, input_data, output_data):
        try:
            if self._dist == 'norm':
                output_data.data = norm.cdf(input_data.data,
                                           loc=self._loc,
                                           scale=self._scale)
            elif self._dist == 'chi2':
                output_data.data = chi2.cdf(input_data.data,
                                           self._df,
                                           loc=self._loc,
                                           scale=self._scale)
        except:
            return BcipEnums.EXE_FAILURE
        
        return BcipEnums.SUCCESS

    def execute(self):
        """
        Execute the kernel and calculate the CDF
        """
        return self._process_data(self._inA, self._outA)
    
    @classmethod
    def add_cdf_node(cls,graph,inA,outA,dist='norm',df=None,loc=0,scale=1):
        """
        Factory method to create a CDF node
        """
        
        # create the kernel object
        k = cls(graph,inA,outA,dist,df,loc,scale)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,BcipEnums.INPUT),
                  Parameter(outA,BcipEnums.OUTPUT))
        
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

    graph : Graph Object
        - Graph that the kernel should be added to

    inA : Tensor or Scalar object
        - First input trial data

    outA : Tensor or Scalar object
        - Output trial data

    regularization : float, 0 < r < 1
        - Regularization parameter


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
        super().__init__('Covariance',BcipEnums.INIT_FROM_NONE,graph)
        self._inputA  = inputA
        self._outputA = outputA
        self._r = regularization

        self._init_inA = None
        self._init_outA = None

        self._labels = None

    def initialize(self):
        """
        Initialize internal state and initialization output of the kernel
        """
        sts = BcipEnums.SUCCESS

        if self._init_outA != None:
            # update output size, as needed
            if len(self._init_outA.shape) == 0:
                shape = list(self._init_inA.shape)
                shape[-1] = shape[-2]
                self._init_outA.shape = tuple(shape)

            sts = self._process_data(self._init_inA, self._init_outA)

        return BcipEnums.SUCCESS

    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized
        """
        
        # first ensure the input and output are tensors
        if (self._inputA._bcip_type != BcipEnums.TENSOR or 
            self._outputA._bcip_type != BcipEnums.TENSOR):
            return BcipEnums.INVALID_PARAMETERS
        
        if self._r > 1 or self._r < 0:
            return BcipEnums.INVALID_PARAMETERS
        
        # check the shape
        input_shape = self._inputA.shape
        input_rank = len(input_shape)
        
        # determine what the output shape should be
        if input_rank < 1 or input_rank > 3:
            return BcipEnums.INVALID_PARAMETERS
        elif input_rank == 1:
            output_shape = (1,)
        else:
            output_shape = list(input_shape)
            output_shape[-1] = output_shape[-2]
            output_shape = tuple(output_shape)
        
        # if the output is virtual and has no defined shape, set the shape now
        if self._outputA.virtual and len(self._outputA.shape) == 0:
            self._outputA.shape = output_shape
        
        # ensure the output tensor's shape equals the expected output shape
        if self._outputA.shape != output_shape:
            return BcipEnums.INVALID_PARAMETERS
        else:
            return BcipEnums.SUCCESS
        
    def _process_data(self, input_data1, output_data1):
        """
        Process input data according to outlined kernel function
        """
        shape = input_data1.shape
        rank = len(shape)
        
        input_data = input_data1.data
        
        
        if rank <= 2:
            covmat = np.cov(input_data)
            output_data1.data = (1/(1+self._r) * 
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
            output_data1.data = np.reshape(output_data,self._outputA.shape)
            
        return BcipEnums.SUCCESS
    
    def execute(self):
        """
        Execute the kernel function
        """
        return self._process_data(self._inputA, self._outputA)
    
    @classmethod
    def add_covariance_node(cls,graph,inputA,outputA,regularization=0):
        """
        Factory method to create a covariance kernel and add it to a graph
        as a generic node object.
        
        Parameters
        ----------
        graph : Graph Object
            - Graph that the kernel should be added to

        inA : Tensor or Scalar object
            - First input trial data

        outA : Tensor or Scalar object
            - Output trial data

        regularization : float, 0 < r < 1
            - Regularization parameter


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
        params = (Parameter(inputA,BcipEnums.INPUT),
                  Parameter(outputA,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node


class Descriptive:
    def initialize(self):
        """
        This kernel has no internal state that must be initialized
        """
        sts = BcipEnums.SUCCESS

        if self._init_outA != None:
            # update the output shape
            if len(self._out_initA.shape) == 0:
                phony_out = np.mean(self._init_inA,
                                    axis=self._axis,
                                    keepdims=self._keepdims)
                self._init_outA.shape = phony_out.shape
            
            sts = self._process_data(self._init_inA,self._init_outA)
                
        return sts
 

    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized
        """
        
        # input must be a tensor
        if self._inA._bcip_type != BcipEnums.TENSOR:
            return BcipEnums.INVALID_PARAMETERS

        # output must be a tensor or scalar
        if (self._outA._bcip_type != BcipEnums.TENSOR and 
            self._outA._bcip_type != BcipEnums.SCALAR):
            return BcipEnums.INVALID_PARAMETERS

        # input tensor must contain some values
        if len(self._inA.shape) == 0:
            return BcipEnums.INVALID_PARAMETERS

        # attempt a phony execution to check if axis and keepdims are valid
        # and get the shape of the output
        try:
            phony_in = np.zeros(self._inA.shape)
            phony_out = np.mean(phony_in, axis=self._axis, keepdims=self._keepdims)
        except:
            return BcipEnums.INVALID_PARAMETERS

        # check shape and format of output
        if self._outA._bcip_type == BcipEnums.TENSOR:
            output_shape = phony_out.shape

            if self._outA.virtual and len(self._outA.shape) == 0:
                self._outA.shape = output_shape

            if self._outA.shape != output_shape:
                return BcipEnums.INVALID_PARAMETERS

        else:
            # if the output is a scalar, the operation must produce a single value
            if (self._outA.data_type != float or
                self._axis != None or
                self._keepdims):
                return BcipEnums.INVALID_PARAMETERS

        return BcipEnums.SUCCESS


    def execute(self):
        """
        Execute the kernel function using numpy function
        """
        return self._process_data(self._inA, self._outA)
    


class MaxKernel(Descriptive, Kernel):
    """
    Kernel to extract maximum value along a Tensor axis

    Parameters
    ----------
    graph : Graph Object
        - Graph that the kernel should be added to

    inA : Tensor object
        - Input data (max value will be extracted from here)

    outA : Tensor or Scalar object
        - Output trial data

    axis : None or int or tuple of ints
        - Axis or axes along which to operate. By default, flattened input in used.

    keepdims : bool
        - If this is set to True, the axes which are reduced are left in the result as dimensions with size one.
    """
    
    def __init__(self,graph,inA,outA,axis=None,keepdims=False):
        super().__init__('Max',BcipEnums.INIT_FROM_NONE,graph)
        self._inA   = inA
        self._outA  = outA

        self._axis = axis
        self._keepdims = keepdims

        self._init_inA = None
        self._init_outA = None
        
        self._labels = None
   
    def _process_data(self, input_data, output_data):
        try:
            if output_data._bcip_type == BcipEnums.SCALAR:
                output_data.data = np.amax(input_data.data).item()
            else:
                result = np.amax(input_data.data,
                                 axis=self._axis,
                                 keepdims=self._keepdims)
                if np.isscalar(result):
                    result = np.asarray([result])
                output_data.data = result
        except:
            return BcipEnums.EXE_FAILURE
        
        return BcipEnums.SUCCESS

    @classmethod
    def add_max_node(cls,graph,inA,outA,axis=None,keepdims=False):
        """
        Factory method to create a maximum value kernel 
        and add it to a graph as a generic node object.

        Parameters
        ----------
        graph : Graph Object
            - Graph that the node should be added to

        inA : Tensor object
            - Input data (max value will be extracted from here)

        outA : Tensor or Scalar object
            - Output trial data

        axis : None or int or tuple of ints
            - Axis or axes along which to operate. By default, flattened input in used.

        keepdims : bool
            - If this is set to True, the axes which are reduced are left in the result as dimensions with size one.
        """
        
        # create the kernel object
        k = cls(graph,inA,outA)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,BcipEnums.INPUT),
                  Parameter(outA,BcipEnums.OUTPUT))
        
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
    graph : Graph Object
        - Graph that the kernel should be added to

    inA : Tensor object
        - Input data (min value will be extracted from here)

    outA : Tensor or Scalar object
        - Output trial data

    axis : None or int or tuple of ints
        - Axis or axes along which to operate. By default, flattened input in used.

    keepdims : bool
        - If this is set to True, the axes which are reduced are left in the result as dimensions with size one.
    """
    
    def __init__(self,graph,inA,outA,axis=None,keepdims=False):
        super().__init__('Min',BcipEnums.INIT_FROM_NONE,graph)
        self._in   = inA
        self._out  = outA
        self._axis = axis
        self._keepdims = keepdims

        self._init_inA = None
        self._init_outA = None

        self._labels = None

    def _process_data(self, input_data, output_data):
        try:
            if output_data._bcip_type == BcipEnums.SCALAR:
                output_data.data = np.amin(input_data.data).item()
            else:
                result = np.amin(input_data.data,
                                 axis=self._axis,
                                 keepdims=self._keepdims)
                if np.isscalar(result):
                    result = np.asarray([result])
                output_data.data = result
        except:
            return BcipEnums.EXE_FAILURE
        
        return BcipEnums.SUCCESS

    @classmethod
    def add_min_node(cls,graph,inA,outA,axis=None,keepdims=False):
        """
        Factory method to create a minimum value kernel 
        and add it to a graph as a generic node object.

        Calculates the mean of values in a tensor

        Parameters
        ----------
        graph : Graph Object
            - Graph that the node should be added to

        inA : Tensor object
            - Input data (min value will be extracted from here)

        outA : Tensor object
            - Output trial data

        axis : None or int or tuple of ints
            - Axis or axes along which to operate. By default, flattened input in used.

        keepdims : bool
            - If this is set to True, the axes which are reduced are left in the result as dimensions with size one.
        """
        
        # create the kernel object
        k = cls(graph,inA,outA)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,BcipEnums.INPUT),
                  Parameter(outA,BcipEnums.OUTPUT))
        
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
    graph : Graph Object
        - Graph that the kernel should be added to

    inA : Tensor object
        - Input data

    outA : Tensor object
        - Output trial data

    axis : None or int or tuple of ints
        - Axis or axes along which to operate. By default, flattened input in used.

    keepdims : bool
        - If this is set to True, the axes which are reduced are left in the result as dimensions with size one.
    """
    
    def __init__(self,graph,inA,outA,axis=None,keepdims=False):
        """
        Kernal calculates arithmetic mean of values in tensor or array
        """
        super().__init__('Mean',BcipEnums.INIT_FROM_NONE,graph)
        self._inA  = inA
        self._outA = outA
        self._axis = axis
        self._keepdims = keepdims

        self._init_inA = None
        self._init_outA = None
        

        self._labels = None

    def _process_data(self, input_data, output_data):
        try:
            if output_data._bcip_type == BcipEnums.SCALAR:
                output_data.data = np.mean(input_data.data).item()
            else:
                result = np.mean(input_data.data,
                                 axis=self._axis,
                                 keepdims=self._keepdims)
                if np.isscalar(result):
                    result = np.asarray([result])
                output_data.data = result
        except:
            return BcipEnums.EXE_FAILURE
        
        return BcipEnums.SUCCESS

    @classmethod
    def add_mean_node(cls,graph,inA,outA,axis=None,keepdims=False):
        """
        Factory method to create a mean calculating kernel

        Calculates the mean of values in a tensor

        Parameters
        ----------
        graph : Graph Object
            - Graph that the kernel should be added to

        inA : Tensor object
            - Input data

        outA : Tensor object
            - Output trial data

        axis : None or int or tuple of ints
            - Axis or axes along which to operate. By default, flattened input in used.

        keepdims : bool
            - If this is set to True, the axes which are reduced are left in the result as dimensions with size one.
        """
        
        # create the kernel object
        k = cls(graph,inA,outA,axis,keepdims)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,BcipEnums.INPUT),
                  Parameter(outA,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node

class StdKernel(Descriptive, Kernel):
    """
    Calculates the standard deviation of values in a tensor

    graph : Graph Object
        - Graph that the kernel should be added to

    inA : Tensor object
        - First input trial data

    outA : Tensor object
        - Output trial data

    axis : None or int or tuple of ints, optional
        - Axis or axes along which the standard deviation is computed. The default is to compute the standard deviation of the flattened array.

    ddof : int, optional
        - Means Delta Degrees of Freedom. The divisor used in calculations is N - ddof, where N represents the number of elements. By default ddof is zero.

    keepdims : bool
        - If this is set to True, the axes which are reduced are left in the result as dimensions with size one.
    """
    
    def __init__(self,graph,inA,outA,axis=None,ddof=0,keepdims=False):
        """
        Kernal calculates arithmetic standard deviation of values in tensor
        """
        super().__init__('Std',BcipEnums.INIT_FROM_NONE,graph)
        self._inA  = inA
        self._outA = outA
        self._axis = axis
        self._ddof = ddof
        self._keepdims = keepdims
        
        self._init_inA = None
        self._init_outA = None

        self._labels = None
 
    def verify(self):
        sts = super().verify()

        # verify ddof is valid
        if isinstance(self._axis, int):
            N = self._inA.shape[self._axis]
        else:
            if self._axis == None:
                dims = self._inA.shape
            else:
                dims = [self._inA.shape[a] for a in self._axis]
            N = 1
            for dim in dims:
                N *= dim

        if N <= self._ddof:
            sts = BcipEnums.INVALID_PARAMETERS

        return sts

    def _process_data(self, input_data, output_data):
        """
        Process data according to outlined kernel function
        """
        try:
            result = np.std(input_data.data,
                            axis=self._axis,
                            ddof=self._ddof,
                            keepdims=self._keepdims)

            if output_data._bcip_type == BcipEnums.SCALAR:
                output_data.data = result.item()
            else:
                if np.isscalar(result):
                    result = np.asarray([result])
                output_data.data = result
                
        except:
            return BcipEnums.EXE_FAILURE
        
        return BcipEnums.SUCCESS

    @classmethod
    def add_std_node(cls,graph,inA,outA,axis=None,ddof=0,keepdims=False):
        """
        Factory method to add a standard deviation node to a graph

        Calculates the standard deviation of values in a tensor

        graph : Graph Object
            - Graph that the kernel should be added to
    
        inA : Tensor object
            - First input trial data
    
        outA : Tensor object
            - Output trial data
    
        axis : None or int or tuple of ints, optional
            - Axis or axes along which the standard deviation is computed. The default is to compute the standard deviation of the flattened array.
    
        ddof : int, optional
            - Means Delta Degrees of Freedom. The divisor used in calculations is N - ddof, where N represents the number of elements. By default ddof is zero.
    
        keepdims : bool
            - If this is set to True, the axes which are reduced are left in the result as dimensions with size one.
        """
        
        # create the kernel object
        k = cls(graph,inA,outA,axis,ddof)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,BcipEnums.INPUT),
                  Parameter(outA,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node

class VarKernel(Descriptive, Kernel):
    """
    Calculates the variance of values in a tensor

    graph : Graph Object
        - Graph that the kernel should be added to

    inA : Tensor or Scalar object
        - Input trial data

    outA : Tensor or Scalar object
        - Output trial data

    axis : None or int or tuple of ints, optional
        - Axis or axes along which the variance is computed. The default is to compute the variance of the flattened array.

    ddof : int, optional
        - "Delta Degrees of Freedom": the divisor used in the calculation is N - ddof, where N represents the number of elements. By default ddof is zero.
    
    keepdims : bool, optional
        - If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this option, the result will broadcast correctly against the input array.
    """
    
    def __init__(self,graph,inA,outA,axis,ddof,keep_dims):
        """
        Kernal calculates arithmetic variance of values in tensor
        """
        super().__init__('Var',BcipEnums.INIT_FROM_NONE,graph)
        self._inA  = inA
        self._outA = outA
        self._axis = axis
        self._ddof = ddof
        self._keep_dims = keep_dims
        
        self._init_inA = None
        self._init_outA = None

        self._labels = None


    def verify(self):
        sts = super().verify()

        # verify ddof is valid
        if isinstance(self._axis, int):
            N = self._inA.shape[self._axis]
        else:
            if self._axis == None:
                dims = self._inA.shape
            else:
                dims = [self._inA.shape[a] for a in self._axis]
            N = 1
            for dim in dims:
                N *= dim

        if N <= self._ddof:
            sts = BcipEnums.INVALID_PARAMETERS

        return sts

    def _process_data(self, input_data, output_data):
        """
        Process data according to outlined kernel function
        """
        try:
            result = np.var(input_data.data,
                            axis=self._axis,
                            ddof=self._ddof,
                            keepdims=self._keepdims)

            if output_data._bcip_type == BcipEnums.SCALAR:
                output_data.data = result.item()
            else:
                if np.isscalar(result):
                    result = np.asarray([result])
                output_data.data = result
                
        except:
            return BcipEnums.EXE_FAILURE
        
        return BcipEnums.SUCCESS

    @classmethod
    def add_var_node(cls,graph,inA,outA,axis=None,ddof=0,keep_dims=False):
        """
        Factory method to create a variance kernel

        graph : Graph Object
            - Graph that the kernel should be added to

        inA : Tensor or Scalar object
            - Input trial data

        outA : Tensor or Scalar object
            - Output trial data

        axis : None or int or tuple of ints, optional
            - Axis or axes along which the variance is computed. The default is to compute the variance of the flattened array.

        ddof : int, optional
            - "Delta Degrees of Freedom": the divisor used in the calculation is N - ddof, where N represents the number of elements. By default ddof is zero.
        
        keepdims : bool, optional
            - If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this option, the result will broadcast correctly against the input array.
            
        """
        
        # create the kernel object
        k = cls(graph,inA,outA,axis,ddof,keep_dims)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,BcipEnums.INPUT),
                  Parameter(outA,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node

class ZScoreKernel(Kernel):
    """
    Calculate a z-score for an tensor or scalar input

    graph : Graph Object
        - Graph that the kernel should be added to

    inA : Tensor or Scalar object
        - Input trial data

    outA : Tensor or Scalar object
        - Output trial data

    init_data: Tensor or Array object
        - Initialization data (n_trials, n_channels, n_samples)
    """
    
    def __init__(self,graph,inA,outA,init_data):
        super().__init__('Zscore',BcipEnums.INIT_FROM_DATA,graph)
        self._inA   = inA
        self._out  = outA
        self.initialization_data = init_data

        self._init_inA = None
        self._init_outA = None

        self._mu = 0
        self._sigma = 0
        self._initialized = False

        self._labels = None

    def initialize(self):
        """
        Initialize the mean and std. Call initialization_execution if downstream nodes are missing training data
        """
        if self.initialization_data == None:
            self.initialization_data = self._init_inA

        if (self.initialization_data._bcip_type != BcipEnums.ARRAY and
            self.initialization_data._bcip_type != BcipEnums.TENSOR and
            self.initialization_data._bcip_type != BcipEnums.CIRCLE_BUFFER):
            return BcipEnums.INITIALIZATION_FAILURE

        if self.initialization_data._bcip_type == BcipEnums.TENSOR:
            if len(self.initialization_data.data.squeeze().shape) != 1:
                return BcipEnums.INITIALIZATION_FAILURE
        else:
            e = self.initialization_data.get_element(0)
            if e._bcip_type == BcipEnums.TENSOR:
                if (e.shape != (1,)) and (e.shape != (1,1)):
                    return BcipEnums.INITIALIZATION_FAILURE
            elif e._bcip_type == BcipEnums.SCALAR:
                if not e.data_type in Scalar.valid_numeric_types():
                    return BcipEnums.INITIALIZATION_FAILURE
            else:
                return BcipEnums.INITIALIZATION_FAILURE

        if self.initialization_data._bcip_type == BcipEnums.TENSOR:
            d = self.initialization_data.data.squeeze()
        else:
            e = self.initialization_data.get_element(0)
            dl = []
            for i in range(self.initialization_data.capacity):
                elem_data = self.initialization_data.get_element(i).data
                if e._bcip_type == BcipEnums.TENSOR:
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

        self._initialized = True

        sts = BcipEnums.SUCCESS
        if self._init_outA != None:
            # set output size, as needed
            if len(self._init_outA.shape) == 0:
                self._init_outA.shape = self._init_inA.shape

            sts = self._process_data(self._init_inA, self._init_outA)
        
        return sts

    
    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized and typed
        """
        # input and output must be scalar or tensor
        for param in (self._inA, self._out):
            if (param._bcip_type != BcipEnums.SCALAR and
                param._bcip_type != BcipEnums.TENSOR):
                return BcipEnums.INVALID_PARAMETERS

        if self._inA._bcip_type == BcipEnums.TENSOR:
            # input tensor must contain some values
            if len(self._inA.shape) == 0:
                return BcipEnums.INVALID_PARAMETERS

            # must contain only one non-singleton dimension
            if len(self._inA.data.squeeze().shape) > 1:
                return BcipEnums.INVALID_PARAMETERS

            # if output is a scalar, tensor must contain a single element
            if (self._out._bcip_type == BcipEnums.SCALAR and
                len(self._inA.data.squeeze().shape) != 0):
                return BcipEnums.INVALID_PARAMETERS

        else:
            # input scalar must contain a number
            if not self._inA.data_type in Scalar.valid_numeric_types():
                return BcipEnums.INVALID_PARAMETERS

            if self._out._bcip_type == BcipEnums.SCALAR and self._out.data_type != float:
                return BcipEnums.INVALID_PARAMETERS

            if self._out._bcip_type == BcipEnums.TENSOR and self._out.shape != (1,1):
                return BcipEnums.INVALID_PARAMETERS
            

        if self._out._bcip_type == BcipEnums.TENSOR:
            if self._out.virtual() and len(self._out.shape) == 0:
                self._out.shape = self._inA.shape

            if self._out.shape != self._inA.shape:
                return BcipEnums.INVALID_PARAMETERS

        return BcipEnums.SUCCESS

    def _process_data(self, input_data, output_data):
        """
        Process data according to outlined kernel function
        """
        if not self._initialized:
            return BcipEnums.EXE_FAILURE

        try:
            out_data = (input_data.data - self._mu) / self._sigma

            if (input_data._bcip_type == BcipEnums.TENSOR and
                output_data._bcip_type == BcipEnums.SCALAR):
                # peel back layers of array until the value is extracted
                while isinstance(out_data,np.ndarry):
                    out_data = out_data[0]
                output_data.data = out_data
            elif (input_data._bcip_type == BcipEnums.SCALAR and
                  output_data._bcip_type == BcipEnums.TENSOR):
                output_data.data = np.asarray([[out_data]])
            else:
                output_data.data = out_data
            
        except:
            return BcipEnums.EXE_FAILURE
        
        return BcipEnums.SUCCESS


    def execute(self):
        """
        Execute the kernel function using numpy function
        """
        return self._process_data(self._inA, self._out)
    
    @classmethod
    def add_zscore_node(cls,graph,inA,outA,init_data):
        """
        Factory method to create a z-score value kernel 
        and add it to a graph as a generic node object.

        graph : Graph Object
            - Graph that the kernel should be added to

        inA : Tensor or Scalar object
            - Input trial data

        outA : Tensor or Scalar object
            - Output trial data

        init_data: Tensor or Array object
            - Initialization data (n_trials, n_channels, n_samples)
        """
        
        # create the kernel object
        k = cls(graph,inA,outA,init_data)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,BcipEnums.INPUT),
                  Parameter(outA,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node


