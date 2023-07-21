from ..core import BCIP, BcipEnums
from ..kernel import Kernel
from ..graph import Node, Parameter
from ..containers import Scalar

import numpy as np
import warnings

class ConcatenationKernel(Kernel):
    """
    Kernel to concatenate multiple tensors into a single tensor

    Parameters
    ----------

    graph : Graph 
        Graph that the kernel should be added to
    inA : Tensor 
        First input trial data
    inB : Tensor 
        Second input trial data
    outA : Tensor 
        Output trial data
    axis : int or tuple of ints, default = 0
        The axis along which the arrays will be joined. If axis is None, arrays are flattened before use. Default is 0. 
        See numpy.concatenate for more information
    """
    
    def __init__(self,graph,outA,inA,inB,axis=0):
        """
        Constructor for the Concatenation kernel
        """
        super().__init__('Concatenation',BcipEnums.INIT_FROM_NONE,graph)
        self.inputs = [inA,inB]
        self.outputs = [outA]
        self._axis = axis

    def initialize(self):
        """
        This kernel has no internal state that must be initialized
        """
        sts = BcipEnums.SUCCESS

        # get the init inputs and outputs
        init_inA = self.init_inputs[0]
        init_inB = self.init_inputs[1]
        init_out = self.init_outputs[0]

        if init_out is not None and (init_inA is not None and init_inA.shape != ()):
            # determine if the axis needs to be adjusted for init
            # start by getting the inputs to compare the rank of the init and non-init data
            d_inA = self.inputs[0]
            d_inB = self.inputs[1]
            axis_adjusted = False
            if (len(d_inA.shape)+1 == len(init_inA.shape) and
                len(d_inB.shape)+1 == len(init_inB.shape) and
                self._axis >= 0):
                # adjust axis to accomodate stack of input data
                self._axis += 1
                axis_adjusted = True

            # adjust the output shape if it is virtual
            if init_out.virtual:
                output_sz, _, _ = self._resolve_dims(init_inA, init_inB)
                init_out.shape = output_sz

            # process the init data
            sts = self._process_data(init_inA, init_inB, init_out)

            # adjust the axis back if it was adjusted
            if axis_adjusted:
                self._axis -= 1

            # pass on labels
            labels = self.init_input_labels
            if labels.bcip_type != BcipEnums.TENSOR:
                labels = labels.to_tensor()
            labels.copy_to(self.init_output_labels)
        
        return sts
    
    def _resolve_dims(self, inA, inB):
        sz_A = inA.shape
        sz_B = inB.shape
        concat_axis = self._axis
        
        if len(sz_A) == len(sz_B):
            noncat_sz_A = [d for i,d in enumerate(sz_A) if i!=concat_axis]
            noncat_sz_B = [d for i,d in enumerate(sz_B) if i!=concat_axis]
            output_sz = noncat_sz_A[:]
            output_sz.insert(concat_axis,sz_A[concat_axis]+sz_B[concat_axis])
        elif len(sz_A) == len(sz_B)+1:
            # appending B to A
            noncat_sz_A = [d for i,d in enumerate(sz_A) if i!=concat_axis]
            noncat_sz_B = sz_B
            output_sz = noncat_sz_A[:]
            output_sz.insert(concat_axis,sz_A[concat_axis]+1)
        elif len(sz_A) == len(sz_B)-1:
            noncat_sz_B = [d for i,d in enumerate(sz_B) if i!=concat_axis]
            noncat_sz_A = sz_A
            output_sz = noncat_sz_B[:]
            output_sz.insert(concat_axis,sz_B[concat_axis]+1)
        else:
            output_sz = []
            noncat_sz_A = []
            noncat_sz_B = []

        return tuple(output_sz), noncat_sz_A, noncat_sz_B

    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized
        """
        
        # inA, inB, and outA must be a tensor
        d_inA, d_inB = self.inputs
        d_out = self.outputs[0]
        for param in (d_inA, d_inB, d_out):
            if param._bcip_type != BcipEnums.TENSOR:
                return BcipEnums.INVALID_PARAMETERS
            
        # the dimensions along the catcat axis must be equal
        output_sz, noncat_sz_A, noncat_sz_B = self._resolve_dims(d_inA,d_inB)
        if len(output_sz) == 0:
            return BcipEnums.INVALID_PARAMETERS
        
        # check if the remaining dimensions are the same
        if ((len(noncat_sz_A) != len(noncat_sz_B)) or 
             len(noncat_sz_A) != sum([1 for i,j in 
                                     zip(noncat_sz_A,noncat_sz_B) if i==j])):
            return BcipEnums.INVALID_PARAMETERS
        
        # check the output dimensions are valid
        if d_out.virtual and len(d_out.shape) == 0:
            d_out.shape = output_sz
        
        # ensure the output shape equals the expected output shape
        if d_out.shape != output_sz:
            return BcipEnums.INVALID_PARAMETERS

        return BcipEnums.SUCCESS

    def _process_data(self, input_data1, input_data2, output_data):
        """
        Process input data according to outlined kernel function
        """

        concat_axis = self._axis if self._axis != None else 0
        
        inA_data = input_data1.data
        inB_data = input_data2.data
        
        if len(inA_data.shape) == len(inB_data.shape)+1:
            # add a leading dimension for input B
            inB_data = np.expand_dims(inB_data,axis=0)
        elif len(inB_data.shape) == len(inA_data.shape)+1:
            inA_data = np.expand_dims(inA_data,axis=0)
        
        try:
            out_tensor = np.concatenate((inA_data,
                                         inB_data),
                                        axis=concat_axis)
        except:
            # dimensions are invalid
            return BcipEnums.EXE_FAILURE
        
        # set the data in the output tensor
        output_data.data = out_tensor
        
        return BcipEnums.SUCCESS

    def execute(self):
        """
        Execute the kernel function using numpy functions
        """
        return self._process_data(self.inputs[0], self.inputs[1], self.outputs[0])
    
    @classmethod
    def add_concatenation_node(cls,graph,inA,inB,outA,axis=0):
        """
        Factory method to create a concatenation kernel and add it to a graph
        as a generic node object.

        Parameters
        ----------
        graph : Graph 
            Graph that the kernel should be added to
        inA : Tensor 
            First input trial data
        inB : Tensor 
            Second input trial data
        outA : Tensor 
            Output trial data
        axis : int or tuple of ints, default = 0
            The axis along which the arrays will be joined. If axis is None, arrays are flattened before use. Default is 0. 
            See numpy.concatenate for more information
        """
        
        # create the kernel object
        k = cls(graph,outA,inA,inB,axis)
        
        # create parameter objects for the input and output
        params = (Parameter(outA,BcipEnums.OUTPUT),
                  Parameter(inA,BcipEnums.INPUT),
                  Parameter(inB,BcipEnums.INPUT))
        
    
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node


class EnqueueKernel(Kernel):
    """
    Kernel to enqueue a BCIP object into a BCIP circle buffer

    graph : Graph 
        Graph that the kernel should be added to

    inA : BCIP 
        Input data to enqueue into circle buffer

    queue : CircleBuffer 
        Circle buffer to have data enqueud to

    """
    
    def __init__(self,graph,inA,queue,enqueue_flag):
        super().__init__('Enqueue',BcipEnums.INIT_FROM_NONE,graph)
        self.inputs = [inA, enqueue_flag]
        self.outputs = [queue]

        if enqueue_flag is not None:
            self._gated = True
        else:
            self._gated = False

    def initialize(self):
        """
        This kernel has no internal state that must be initialized
        """
        return BcipEnums.SUCCESS
    
    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized
        """
        
        # first ensure the inputs and outputs are the appropriate type
        d_in = self.inputs[0]
        d_out = self.outputs[0]

        if not isinstance(d_in,BCIP):
            return BcipEnums.INVALID_PARAMETERS
        
        if d_out.bcip_type != BcipEnums.CIRCLE_BUFFER:
            return BcipEnums.INVALID_PARAMETERS

        # check that the buffer's capacity is at least 1
        if d_out.capacity <= 1:
            return BcipEnums.INVALID_PARAMETERS
        
        # if gated, check that the flag is a scalar
        if self._gated:
            enqueue_flag = self.inputs[1]
            if (enqueue_flag.bcip_type != BcipEnums.SCALAR or
                enqueue_flag.data_type not in (int, bool)):
                return BcipEnums.INVALID_PARAMETERS
            
        return BcipEnums.SUCCESS
    
    def execute(self):
        """
        Execute the kernel function using numpy function
        """
        # need to make a deep copy of the object to enqueue
        if not self._gated or self.inputs[1].data:
            cpy = self.inputs[0].make_copy()
            self.outputs[0].enqueue(cpy)
            
        return BcipEnums.SUCCESS
    
    @classmethod
    def add_enqueue_node(cls,graph,inA,queue,enqueue_flag=None):
        """
        Factory method to create a enqueue kernel and add it to a graph as a generic node object.

        graph : Graph 
            Graph that the kernel should be added to

        inA : Tensor or Scalar or Array or CircleBuffer 
            Input data to enqueue into circle buffer

        queue : CircleBuffer 
            Circle buffer to have data enqueud to
            
        enqueue_flag : bool
            (optional) Scalar boolean used to determine if the inputis to be added to the queue
        """
        
        # create the kernel object
        k = cls(graph,inA,queue,enqueue_flag)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,BcipEnums.INPUT),
                  Parameter(queue,BcipEnums.INOUT))
        
        if enqueue_flag is not None:
            params += (Parameter(enqueue_flag, BcipEnums.INPUT),)
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node

class ExtractKernel(Kernel):
    """
    Kernel to extract a portion of a tensor or array

    Parameters
    ----------

    graph : Graph 
        Graph that the kernel should be added to
    inA : Tensor or Array 
        Input trial data
    Indicies : list slices, list of ints
        Indicies within inA from which to extract data
    outA : Tensor 
        Extracted trial data
    reduce_dims : bool, default = False
        Remove singleton dimensions if true, don't squeeze otherwise
    """
    
    def __init__(self,graph,inA,indices,outA,reduce_dims):
        super().__init__('Extract',BcipEnums.INIT_FROM_NONE,graph)
        self.inputs = [inA]
        self.outputs = [outA]
        self._indices = indices
        self._reduce_dims = reduce_dims

    def initialize(self):
        """
        This kernel has no internal state that must be initialized
        """
        sts = BcipEnums.SUCCESS

        init_in = self.init_inputs[0]
        init_out = self.init_outputs[0]

        if init_out is not None and (init_in is not None and init_in.shape != ()):
            # determine init output shape
            init_output_shape = (init_in.shape[0],) + self.outputs[0].shape
            if init_out.virtual:
                if len(init_in.shape) == (len(self.inputs[0].shape)+1):
                    init_out.shape = init_output_shape

                if init_out.shape != init_output_shape:
                    sts = BcipEnums.INITIALIZATION_FAILURE

                if sts == BcipEnums.SUCCESS:
                    # TODO a lot of repeated code from execute below, determine how best to refactor
                    d_in = self.inputs[0]
                    if (init_in.bcip_type != BcipEnums.TENSOR and
                        d_in.bcip_type == BcipEnums.TENSOR):
                        init_in = init_in.to_tensor()

                    # insert an additional slice for the batch dimension
                    if (init_in.bcip_type == BcipEnums.TENSOR):
                        self._indices.insert(0, ":")

                    self._process_data(init_in, init_out)

                    # remove the additional slice for the batch dimension
                    if (init_in.bcip_type == BcipEnums.TENSOR):
                        self._indices.pop(0)

                    # pass on the labels
                    self.copy_init_labels_to_output()

        return sts
    
    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized
        """
        
        # input must be a tensor or array
        # additionally, if the input is a tensor, the output should also be a
        # tensor
        d_in = self.inputs[0]
        d_out = self.outputs[0]
        if (d_in.bcip_type == BcipEnums.TENSOR):
            if (d_out.bcip_type != BcipEnums.TENSOR):
                return BcipEnums.INVALID_PARAMETERS
        elif (d_in.bcip_type != BcipEnums.ARRAY and
              d_in.bcip_type != BcipEnums.CIRCLE_BUFFER):
            return BcipEnums.INVALID_PARAMETERS


        # if the input is an array, then the there should only be a single 
        # dimension to extract with a value of zero
        if d_in.bcip_type != BcipEnums.TENSOR:
            for index in self._indices:
                if not isinstance(index, int):
                    return BcipEnums.INVALID_PARAMETERS
            
                # check that the index to extract do not exceed the capacity
                if index >= self._in.capacity or index < -self._in.capacity:
                    return BcipEnums.INVALID_PARAMETERS

            # if the output is another array, validate that the types match
            in_element = d_in.get_element(0)
            if (d_out.bcip_type == BcipEnums.ARRAY or
                d_out.bcip_type == BcipEnums.CIRCLE_BUFFER):
                out_element = d_out.get_element(0)
                if (in_element.bcip_type != out_element.bcip_type):
                    return BcipEnums.INVALID_PARAMETERS

                # also check that the output has sufficient capacity
                if d_out.capacity < len(self._indices):
                    return BcipEnums.INVALID_PARAMETERS

            # if the output is a scalar, check that the input is an array of compatible scalars
            elif (d_out.bcip_type == BcipEnums.SCARLAR):
                if (in_element.bcip_type != BcipEnums.SCALAR or
                    in_element.data_type != d_out.data_type):
                    return BcipEnums.INVALID_PARAMETERS

                # also verify that only one index is being extracted
                if len(self._indices) != 1:
                    return BcipEnums.INVALID_PARAMETERS

            # if the output is a tensor, ensure its dimensions are valid
            elif d_out.bcip_type == BcipEnums.TENSOR:
                # case 1 : array of tensors
                if in_element.bcip_type == BcipEnums.TENSOR:
                    out_shape = (len(self._indices),) + in_element.shape

                # case 2 : array of scalars
                elif in_element.bcip_type == BcipEnums.SCALAR:
                    out_shape = (len(self._indices),1)

                else:
                    return BcipEnums.INVALID_PARAMETERS

                # if output is virtual, set the shape
                if d_out.virtual and len(d_out.shape) == 0:
                    d_out.shape = out_shape

                # check that the shape is valid
                if d_out.shape != out_shape:
                    return BcipEnums.INVALID_PARAMETERS
       
        elif d_in.bcip_type == BcipEnums.TENSOR:
            if d_out.bcip_type != BcipEnums.TENSOR:
                return BcipEnums.INVALID_PARAMETERS

            # check that the number of dimensions indicated does not exceed 
            # the tensor's rank
            if len(self._indices) != len(d_in.shape):
                warnings.warn("Number of dimensions to extract exceeds the tensor's rank")
                return BcipEnums.INVALID_PARAMETERS
            
            output_sz = []
            for axis in range(len(self._indices)):
                if self._indices[axis] != ":":
                    axis_indices = self._indices[axis]
                    if isinstance(axis_indices,int):
                        axis_indices = (axis_indices,)
                    for index in axis_indices:
                        # check that the index is valid for the given axis
                        if index < -d_in.shape[axis] or index >= d_in.shape[axis]:
                            return BcipEnums.INVALID_PARAMETERS
                    
                    if not self._reduce_dims or len(self._indices[axis]) > 1:
                        output_sz.append(len(axis_indices))
                else:
                    output_sz.append(d_in.shape[axis])
            
            # check that the output tensor's dimensions are valid
            output_sz = tuple(output_sz)
            
            if d_out.virtual and len(d_out.shape) == 0:
                d_out.shape = output_sz
            
            if d_out.shape != output_sz:
                return BcipEnums.INVALID_PARAMETERS
        
        return BcipEnums.SUCCESS

    def _process_data(self, inA, outA):
        if inA.bcip_type != BcipEnums.TENSOR:
            # extract the elements and set in the output array
            if (outA.bcip_type == BcipEnums.ARRAY or
                outA.bcip_type == BcipEnums.CIRCLE_BUFFER):
                for dest_index, src_index in enumerate(self._indices):
                    elem = inA.get_element(src_index) # extract from input
                    outA.set_element(dest_index,elem) # set to output

            elif outA.bcip_type == BcipEnums.SCALAR:
                outA.data = inA.get_element(self._indices[0])

            else:
                # tensor output
                out_array = outA.data
                for dest_index, src_index in enumerate(self._indices):
                    elem_data = inA.get_element(src_index).data
                    out_array[dest_index] = elem_data
                outA.data = out_array
        else:
            # tensor input 
            ix_grid = []
            for axis in range(len(self._indices)):
                axis_indices = self._indices[axis]
                if axis_indices == ":":
                    ix_grid.append([_ for _ in range(inA.shape[axis])])
                else:
                    if isinstance(axis_indices,int):
                        ix_grid.append([axis_indices])
                    else:
                        ix_grid.append(list(axis_indices))

            npixgrid = np.ix_(*ix_grid)
            extr_data = inA.data[npixgrid]
            
            if self._reduce_dims:
                extr_data = np.squeeze(extr_data)
                
            outA.data = extr_data
        
        return BcipEnums.SUCCESS
       

    def execute(self):
        """
        Execute the kernel function
        """
        return self._process_data(self.inputs[0], self.outputs[0])
            
    @classmethod
    def add_extract_node(cls,graph,inA,indices,outA,reduce_dims=False):
        """
        Factory method to create an extract kernel 
        and add it to a graph as a generic node object.

         graph : Graph 
            Graph that the kernel should be added to

        inA : Tensor or Array 
            Input trial data

        Indicies : list slices, list of ints
            Indicies within inA from which to extract data

        outA : Tensor, Scalar, or Array 
            Extracted trial data

        reduce_dims : bool, default = False
            Remove singleton dimensions if true, don't squeeze otherwise
        """
        
        # create the kernel object
        k = cls(graph,inA,indices,outA,reduce_dims)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,BcipEnums.INPUT),
                  Parameter(outA,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node

class StackKernel(Kernel):
    """
    Kernel to stack multiple tensors into a single tensor

    Parameters
    ----------

    graph : Graph
        The graph where the RunningAverageKernel object should be added
    inA : Array 
        Container where specified data will be added to
    outA : Tensor
        Tensor of stacked tensors
    axis : int or None, default = None
        The axis in the result array along which the input arrays are stacked.
    """
    
    def __init__(self,graph,inA,outA,axis=None):
        super().__init__('stack',BcipEnums.INIT_FROM_NONE,graph)
        self.inputs = [inA]
        self.outputs = [outA]
        self._axis = axis

    def initialize(self):
        """
        This kernel has no internal state that must be initialized
        """
        return BcipEnums.SUCCESS
    
    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized
        """
        inA = self.inputs[0]
        outA = self.outputs[0]
                
        # inA must be an array and outA must be a tensor
        if (not ((inA.bcip_type == BcipEnums.ARRAY or 
                  inA.bcip_type == BcipEnums.CIRCLE_BUFFER) and 
            outA.bcip_type == BcipEnums.TENSOR)):
            return BcipEnums.INVALID_PARAMETERS
        
        # if an axis was provided, it must be a scalar
        if self._axis != None and self._axis.bcip_type != BcipEnums.SCALAR:
            return BcipEnums.INVALID_PARAMETERS
        
        stack_axis = self._axis.data if self._axis != None else 0
        
        # ensure that all the tensors in inA are the same size
        tensor_shapes = [inA.get_element(i).shape for i in range(inA.capacity)]
        
        if len(set(tensor_shapes)) != 1:
            # tensors in array are different sizes OR array is empty
            return BcipEnums.INVALID_PARAMETERS
        
        # determine the output dimensions
        output_shape = (tensor_shapes[0][:stack_axis] + (inA.capacity,) 
                         + tensor_shapes[0][stack_axis:])
        
        # check the output dimensions are valid
        if outA.virtual and len(outA.shape) == 0:
            outA.shape = output_shape
        
        # ensure the output shape equals the expected output shape
        if outA.shape != output_shape:
            return BcipEnums.INVALID_PARAMETERS

        return BcipEnums.SUCCESS


    def execute(self):
        """
        Execute the kernel function using numpy functions
        """

        inA = self.inputs[0]
        outA = self.outputs[0]

        stack_axis = self._axis.data if self._axis != None else 0
        
        try:
            input_tensors = [inA.get_element(i) for i in range(inA.capacity)]
            
            input_data = [t.data for t in input_tensors]
            output_data = np.stack(input_data,axis=stack_axis)
        
        except ValueError:
            return BcipEnums.EXE_FAILURE
        
        # set the data of the output tensor
        outA.data = output_data
        
        return BcipEnums.SUCCESS
    
    
    @classmethod
    def add_stack_node(cls,graph,inA,outA,axis=None):
        """
        Factory method to create a stack kernel and add it to a graph
        as a generic node object.
        """
        
        # create the kernel object
        k = cls(graph,inA,outA,axis)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,BcipEnums.INPUT),
                  Parameter(outA,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node

class TensorStackKernel(Kernel):
    """
    Kernel to stack 2 tensors into a single tensor

    Parameters
    ----------

    graph : Graph 
        Graph that the kernel should be added to
    inA : Tensor 
        First input trial data
    inB : Tensor
        Second input trial data
    outA : Tensor 
        Output trial data
    axis : int, default=None
        Axis over which to stack the tensors. If none, the tensors are flattened before they are stacked
    """
    
    def __init__(self,graph,inA,inB,outA,axis=None):
        super().__init__('TensorStack',BcipEnums.INIT_FROM_NONE,graph)
        self.inputs = [inA,inB]
        self.outputs = [outA]
        self._axis = axis

    def initialize(self):
        """
        This kernel has no internal state that must be initialized. 
        """
        sts = BcipEnums.SUCCESS

        init_inA, init_inB = self.init_inputs
        init_out = self.init_outputs[0]

        if init_out is not None and (init_inA is not None and init_inA.shape != ()):
            # adjust the init output shape
            if init_out.virtual:
                init_out.shape = int_inA.shape[:self._axis+1] + (2,) + init_inA.shape[self._axis+1:]

            self._axis += 1 # adjust for batch processing in init TODO will this always be the case?
            sts = self._process_data(init_inA, init_inB, init_out)
            self._axis -= 1 # adjust back for trial processing

        return sts
        
    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized
        """

        inA, inB = self.inputs
        outA = self.outputs[0]
                
        # all params must be tensors
        for param in (inA, inB, outA):
            if param.bcip_type != BcipEnums.TENSOR:
                return BcipEnums.INVALID_PARAMETERS
        
        stack_axis = self._axis
        
        # ensure that all the tensors in inA are the same size
        tensor_shapes = [inA.shape, inB.shape]
        
        if len(set(tensor_shapes)) != 1:
            # tensors in array are different sizes OR array is empty
            return BcipEnums.INVALID_PARAMETERS
        
        # determine the output dimensions
        output_shape = inA.shape[:stack_axis] + (2,) + inA.shape[stack_axis:]
        
        # check the output dimensions are valid
        if outA.virtual and len(outA.shape) == 0:
            outA.shape = output_shape
        
        # ensure the output shape equals the expected output shape
        if outA.shape != output_shape:
            return BcipEnums.INVALID_PARAMETERS

        return BcipEnums.SUCCESS

    def _process_data(self, input_data1, input_data2, output_data):
        """
        Process data according to outlined kernel function.
        """
        stack_axis = self._axis
        
        try:
            input_tensors = [input_data1.data, input_data2.data]
            
            output_data.data = np.stack(input_tensors,axis=stack_axis)
        
        except:
            return BcipEnums.EXE_FAILURE
    
        
        return BcipEnums.SUCCESS

    def execute(self):
        """
        Execute a single trial
        """
        return self._process_data(self.inputs[0], self.inputs[1], self.outputs[0])
    
    
    @classmethod
    def add_tensor_stack_node(cls,graph,inA,inB,outA,axis=0):
        """
        Factory method to create a tensor stack kernel and add it to a graph
        as a generic node object.

        Parameters
        ----------

        graph : Graph 
            Graph that the kernel should be added to
        inA : Tensor or Scalar 
            First input trial data
        inB : Tensor or Scalar 
            Second input trial data
        outA : Tensor or Scalar 
            Output trial data
        axis : int, default=None
            Axis over which to stack the tensors. If none, the tensors are flattened before they are stacked
        
        """
        
        # create the kernel object
        k = cls(graph,inA,inB,outA,axis)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,BcipEnums.INPUT),
                  Parameter(inB,BcipEnums.INPUT),
                  Parameter(outA,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node


