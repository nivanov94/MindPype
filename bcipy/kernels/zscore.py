from types import NoneType
from classes.kernel import Kernel
from classes.node import Node
from classes.parameter import Parameter
from classes.bcip_enums import BcipEnums
from classes.array import Array
from classes.tensor import Tensor
from classes.bcip import BCIP
from classes.scalar import Scalar

import numpy as np


class ZScoreKernel(Kernel):
    """
    Calculate a z-score for an tensor or scalar input
    """
    
    def __init__(self,graph,inA,outA,init_data):
        super().__init__('Zscore',BcipEnums.INIT_FROM_DATA,graph)
        self._in   = inA
        self._out  = outA
        self.initialization_data = init_data

        self._init_inA = None
        self._init_outA = None

        self._mu = 0
        self._sigma = 0
        self._initialized = False

        self.graph = graph

        self._labels = None

    def initialize(self):
        """
        Initialize the mean and std
        """
        if self.initialization_data == None:
            self.initialization_data = self._init_inA

        if not (isinstance(self.initialization_data,Array) or isinstance(self.initialization_data,Tensor)):
            return BcipEnums.INITIALIZATION_FAILURE

        if isinstance(self.initialization_data,Tensor):
            if len(self.initialization_data.data.squeeze().shape) != 1:
                return BcipEnums.INITIALIZATION_FAILURE
        else:
            e = self.initialization_data.get_element(0)
            if isinstance(e,Tensor):
                if (e.data.shape != (1,)) and (e.data.shape != (1,1)):
                    return BcipEnums.INITIALIZATION_FAILURE
            elif isinstance(e,Scalar):
                if not e.data_type in Scalar.valid_numeric_types():
                    return BcipEnums.INITIALIZATION_FAILURE
            else:
                return BcipEnums.INITIALIZATION_FAILURE

        if isinstance(self.initialization_data,Tensor):
            d = self.initialization_data.data.squeeze()
        else:
            e = self.initialization_data.get_element(0)
            dl = []
            for i in range(self.initialization_data.capacity):
                elem_data = self.initialization_data.get_element(i).data
                if isinstance(e,Tensor):
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


        if self._init_outA.__class__ != NoneType:
            return self.initialization_execution()
        
        return BcipEnums.SUCCESS


    
    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized and typed
        """
        

        # input must be scalar or tensor
        if not (isinstance(self._in,Scalar) or isinstance(self._in,Tensor)):
            return BcipEnums.INVALID_PARAMETERS

        # output must be scalar or tensor
        if not (isinstance(self._out,Scalar) or isinstance(self._in,Tensor)):
            return BcipEnums.INVALID_PARAMETERS


        if isinstance(self._in,Tensor):
            # input tensor must contain some values
            if len(self._in.shape) == 0:
                return BcipEnums.INVALID_PARAMETERS

            # must contain only one non-singleton dimension
            if len(self._in.data.squeeze().shape) > 1:
                return BcipEnums.INVALID_PARAMETERS

            # if output is a scalar, tensor must contain a single element
            if isinstance(self._out,Scalar) and len(self._in.data.squeeze().shape) != 0:
                return BcipEnums.INVALID_PARAMETERS

        else:
            # input scalar must contain a number
            if not self._in.data_type in Scalar.valid_numeric_types():
                return BcipEnums.INVALID_PARAMETERS

            if isinstance(self._out,Scalar) and (self._out.data_type != float):
                return BcipEnums.INVALID_PARAMETERS

            if isinstance(self._out,Tensor) and self._out.shape != (1,1):
                return BcipEnums.INVALID_PARAMETERS
            

        if isinstance(self._out,Tensor):
            if self._out.virtual() and len(self._out.shape) == 0:
                self._out.shape = self._in.shape

            if self._out.shape != self._in.shape:
                return BcipEnums.INVALID_PARAMETERS


        # check if the init data is the correct type
        if not (isinstance(self.initialization_data,Array) or isinstance(self.initialization_data,Tensor)):
            return BcipEnums.INITIALIZATION_FAILURE

        if isinstance(self.initialization_data,Tensor):
            if len(self.initialization_data.data.squeeze().shape) != 1:
                return BcipEnums.INITIALIZATION_FAILURE
        else:
            e = self.initialization_data.get_element(0)
            if isinstance(e,Tensor):
                if (e.data.shape != (1,)) and (e.data.shape != (1,1)):
                    return BcipEnums.INITIALIZATION_FAILURE
            elif isinstance(e,Scalar):
                if not e.data_type in Scalar.valid_numeric_types():
                    return BcipEnums.INITIALIZATION_FAILURE
            else:
                return BcipEnums.INITIALIZATION_FAILURE

        return BcipEnums.SUCCESS

    def initialization_execution(self):
        sts = self.process_data(self._init_inA, self._init_outA)
        
        if sts != BcipEnums.SUCCESS:
            return BcipEnums.INITIALIZATION_FAILURE
        
        return sts

    def process_data(self, input_data, output_data):
        if not self._initialized:
            return BcipEnums.EXE_FAILURE

        try:
            out_data = (input_data.data - self._mu) / self._sigma

            if isinstance(input_data,Tensor) and isinstance(output_data,Scalar):
                # peel back layers of array until the value is extracted
                while isinstance(out_data,np.ndarry):
                    out_data = out_data[0]
                output_data.data = out_data
            elif isinstance(input_data,Scalar) and isinstance(output_data,Tensor):
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
        return self.process_data(self._in, self._out)
    
    @classmethod
    def add_zscore_node(cls,graph,inA,outA,init_data):
        """
        Factory method to create a z-score value kernel 
        and add it to a graph as a generic node object.
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
