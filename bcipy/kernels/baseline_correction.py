from ..core import BcipEnums
from ..kernel import Kernel
from ..graph import Node, Parameter
from ..containers import Tensor

import numpy as np

class BaselineCorrectionKernel(Kernel):
    """
    Kernel to conduct baseline correction on data

    Parameters
    ----------
    graph : Graph 
        Graph that the kernel should be added to
    
    inA : Tensor
        Input trial data (n_channels, n_samples) or (n_trials, n_channels, n_samples)

    outA : Tensor
        Output trial data (n_channels, n_samples) or (n_trials, n_channels, n_samples)
    
    baseline_period : np.array or Tensor
        Baseline period to use for baseline correction (n_trials, 2) where column 1 is the start index and column 2 is the end index
        If the same baseline period is to be used for all trials, then the baseline period can be a 1D tensor (2, ) where the first element is the start index and the second element is the end index
    
    """

    def __init__(self, graph, inA, outA, baseline_period = None):
        super().__init__('BaselineCorrection', BcipEnums.INIT_FROM_NONE, graph)
        self.inputs = [inA]
        self.outputs = [outA]

        if baseline_period is not None:
            if type(baseline_period) == list:
                baseline_period = np.array(baseline_period)
            elif type(baseline_period) == Tensor:
                baseline_period = np.array(baseline_period.data)            
        self._baseline_period = baseline_period

    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized
        """
        
        d_in = self.inputs[0]
        d_out = self.outputs[0]
        
        # input and output must be a tensor 
        if (d_in.bcip_type != BcipEnums.TENSOR or
            d_out.bcip_type != BcipEnums.TENSOR):
            return BcipEnums.INVALID_PARAMETERS

        # check the baseline period is valid
        if self._baseline_period is not None:
            # if the baseline period is a 1D tensor, then the same baseline period is used for all trials (ie. self._baseline_period.shape = (2, )
            if len(self._baseline_period.shape) == 1:
                if self._baseline_period.shape[0] != 2:
                    return BcipEnums.INVALID_PARAMETERS
                
                # start index must be less than end index and both must be within the range of the data
                if (self._baseline_period[0] > self._baseline_period[1])\
                    or (self._baseline_period[0] < 0)\
                    or (self._baseline_period[1] > d_in.shape[-1]):
                    return BcipEnums.INVALID_PARAMETERS

            if len(self._baseline_period.shape) == 2:
                if self._baseline_period.shape[1] != 2 or self._baseline_period.shape[0] != d_in.shape[0]:
                    return BcipEnums.INVALID_PARAMETERS
                
                # each start index must be less than end index and both must be within the range of the data
                for i in range(self._baseline_period.shape[0]):
                    if (self._baseline_period[i][0] > self._baseline_period[i][1])\
                        or (self._baseline_period[i][0] < 0)\
                        or (self._baseline_period[i][1] > d_in.shape[-1]):
                        return BcipEnums.INVALID_PARAMETERS
                
        
        # check the output dimensions are valid
        if d_out.virtual and len(d_out.shape) == 0:
            d_out.shape = d_in.shape

        return BcipEnums.SUCCESS


    def initialize(self):
        
        sts = BcipEnums.SUCCESS

        init_in = self.init_inputs[0]
        init_out = self.init_outputs[0]
        
        if init_out is not None and (init_in is not None and init_in.shape != ()):

            if init_out.virtual:
                output_shape = list(init_in.shape)
                init_out.shape = tuple(output_shape)
            
            sts = self._process_data(init_in, init_out)

            # pass on the labels
            self.copy_init_labels_to_output()
        
        return sts

    def execute(self):
        """
        Execute the kernel
        """
        d_in = self.inputs[0]
        d_out = self.outputs[0]
        return self._process_data(d_in, d_out)
    

    def _process_data(self, input, output):
        """
        Process the data
        """

        # if the baseline period is not specified, then return the input
        if self._baseline_period is None:
            input.copy_to(output)
            return BcipEnums.SUCCESS

        # if the baseline period is specified, then perform baseline correction
        else:
            # if the baseline period is a 1D tensor, calculate mean of same indices across all trials
            if len(self._baseline_period.shape) == 1:
                baseline_period = self._baseline_period
                baseline_corrected_data = input.data - np.mean(input.data[..., int(baseline_period[0]):int(baseline_period[1])], axis = -1, keepdims = True)

            else:
                baseline_corrected_data = np.zeros(input.shape)
                for i in range(input.shape[0]):
                    baseline_period = self._baseline_period[i]
                    baseline_corrected_data[i] = input.data[i] - np.mean(input.data[i, ..., int(baseline_period[0]):int(np.ceil(baseline_period[1]))], axis = -1, keepdims = True)
            
            # copy the baseline corrected data to the output
            output.data = baseline_corrected_data

            return BcipEnums.SUCCESS

    @classmethod
    def add_baseline_node(cls, graph, inputA, outputA, baseline_period):
        """
        Factory method to add a baseline correction kernel to a graph
        
        Parameters
        ----------
        graph : Graph 
            Graph that the kernel should be added to

        inputA : Tensor or Scalar 
            Input trial data

        outputA : Tensor or Scalar 
            Output trial data
        
        baseline_period : Tensor
            Baseline period to use for baseline correction (n_trials, 2) where column 1 is the start index and column 2 is the end index
            If the same baseline period is to be used for all trials, then the baseline period can be a 1D tensor (2, ) where the first element is the start index and the second element is the end index
        """

        # create the kernel object
        k = cls(graph,inputA,outputA,baseline_period)
        
        # create parameter objects for the input and output
        params = (Parameter(inputA,BcipEnums.INPUT),
                  Parameter(outputA,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node
