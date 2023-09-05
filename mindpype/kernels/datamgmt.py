from ..core import MPBase, MPEnums
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
        super().__init__('Concatenation',MPEnums.INIT_FROM_NONE,graph)
        self.inputs = [inA,inB]
        self.outputs = [outA]
        self._axis = axis

    def _initialize(self, init_inputs, init_outputs, labels):
        """
        This kernel has no internal state that must be initialized
        """

        # get the init inputs and outputs
        init_inA = init_inputs[0]
        init_inB = init_inputs[1]
        init_out = init_outputs[0]

        for init_in in (init_inA, init_inB):
            if init_in.mp_type != MPEnums.TENSOR:
                init_in = init_in.to_tensor()

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
            self._process_data(init_inA, init_inB, init_out)

            # adjust the axis back if it was adjusted
            if axis_adjusted:
                self._axis -= 1

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

    def _verify(self):
        """
        Verify the inputs and outputs are appropriately sized
        """
        
        # inA, inB, and outA must be a tensor
        d_inA, d_inB = self.inputs
        d_out = self.outputs[0]
        for param in (d_inA, d_inB, d_out):
            if param.mp_type != MPEnums.TENSOR:
                raise TypeError("ConcatenationKernel requires Tensor inputs and outputs")
            
        # the dimensions along the catcat axis must be equal
        output_sz, noncat_sz_A, noncat_sz_B = self._resolve_dims(d_inA,d_inB)
        if len(output_sz) == 0:
            raise TypeError("ConcatenationKernel could not resolve output dimensions")
        
        # check if the remaining dimensions are the same
        if ((len(noncat_sz_A) != len(noncat_sz_B)) or 
             len(noncat_sz_A) != sum([1 for i,j in 
                                     zip(noncat_sz_A,noncat_sz_B) if i==j])):
            raise ValueError("ConcatenationKernel requires non-concatenation dimensions to be equal")
        
        if d_out.virtual and len(d_out.shape) == 0:
            d_out.shape = output_sz
        
        # ensure the output shape equals the expected output shape
        if d_out.shape != output_sz:
            raise ValueError("ConcatenationKernel output shape does not match expected output shape")


    def _process_data(self, inputs, outputs):
        """
        Process input data according to outlined kernel function
        """
        inA_data = inputs[0].data
        inB_data = inputs[1].data

        concat_axis = self._axis if self._axis != None else 0
        
        if len(inA_data.shape) == len(inB_data.shape)+1:
            # add a leading dimension for input B
            inB_data = np.expand_dims(inB_data,axis=0)
        elif len(inB_data.shape) == len(inA_data.shape)+1:
            inA_data = np.expand_dims(inA_data,axis=0)
        
        outputs[0].data = np.concatenate((inA_data,
                                          inB_data),
                                         axis=concat_axis)
        

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

        Returns
        -------
        node : Node
            The node object that was added to the graph containing the concatenation kernel
        """
        
        # create the kernel object
        k = cls(graph,outA,inA,inB,axis)
        
        # create parameter objects for the input and output
        params = (Parameter(outA,MPEnums.OUTPUT),
                  Parameter(inA,MPEnums.INPUT),
                  Parameter(inB,MPEnums.INPUT))
        
    
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node


class EnqueueKernel(Kernel):
    """
    Kernel to enqueue a MindPype object into a MindPype circle buffer

    Parameters
    ----------
    graph : Graph 
        Graph that the kernel should be added to

    inA : MPBase 
        Input data to enqueue into circle buffer

    queue : CircleBuffer 
        Circle buffer to have data enqueud to

    """
    
    def __init__(self,graph,inA,queue,enqueue_flag):
        """
        Constructor for the Enqueue kernel
        """
        super().__init__('Enqueue',MPEnums.INIT_FROM_NONE,graph)
        self.inputs = [inA, queue, enqueue_flag]

        if enqueue_flag is not None:
            self._gated = True
        else:
            self._gated = False

    def _verify(self):
        """
        Verify the inputs and outputs are appropriately sized
        """
        
        # first ensure the inputs and outputs are the appropriate type
        d_in = self.inputs[0]
        d_io = self.inputs[1]

        if not isinstance(d_in,MPBase):
            raise TypeError("EnqueueKernel requires MPBase input")
        
        if d_io.mp_type != MPEnums.CIRCLE_BUFFER:
            raise TypeError("EnqueueKernel requires CircleBuffer output")

        # check that the buffer's capacity is at least 1
        if d_io.capacity <= 1:
            raise ValueError("EnqueueKernel requires CircleBuffer capacity to be at least 1")
        
        # if gated, check that the flag is a scalar
        if self._gated:
            enqueue_flag = self.inputs[2]
            if (enqueue_flag.mp_type != MPEnums.SCALAR or
                enqueue_flag.data_type not in (int, bool)):
                raise TypeError("EnqueueKernel requires enqueue flag to be a scalar boolean")
            
    def _process_data(self, inputs, outputs):
        """
        Execute the kernel function using numpy function
        """
        # need to make a deep copy of the object to enqueue
        if not self._gated or inputs[2].data:
            cpy = inputs[0].make_copy()
            inputs[1].enqueue(cpy)
            
    @classmethod
    def add_enqueue_node(cls,graph,inA,queue,enqueue_flag=None):
        """
        Factory method to create a enqueue kernel and add it to a graph as a generic node object.

        Parameters
        ----------
        graph : Graph 
            Graph that the kernel should be added to

        inA : Tensor or Scalar or Array or CircleBuffer 
            Input data to enqueue into circle buffer

        queue : CircleBuffer 
            Circle buffer to have data enqueud to
            
        enqueue_flag : bool
            (optional) Scalar boolean used to determine if the inputis to be added to the queue

        Returns
        -------
        node : Node
            The node object that was added to the graph containing the enqueue kernel
        """
        
        # create the kernel object
        k = cls(graph,inA,queue,enqueue_flag)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,MPEnums.INPUT),
                  Parameter(queue,MPEnums.INOUT))
        
        if enqueue_flag is not None:
            params += (Parameter(enqueue_flag, MPEnums.INPUT),)
        
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
        super().__init__('Extract',MPEnums.INIT_FROM_NONE,graph)
        self.inputs = [inA]
        self.outputs = [outA]
        self._indices = indices
        self._reduce_dims = reduce_dims

    def _initialize(self, init_inputs, init_outputs, labels):
        """
        This kernel has no internal state that must be initialized
        """
        init_in = init_inputs[0]
        init_out = init_outputs[0]

        if init_in.mp_type != MPEnums.TENSOR:
            init_in = init_in.to_tensor()

        if init_out is not None and (init_in is not None and init_in.shape != ()):
            # determine init output shape
        
            add_batch_dim = False
            if len(init_in.shape) == (len(self.inputs[0].shape)+1):
                init_output_shape = (init_in.shape[0],) + self.outputs[0].shape
                add_batch_dim = True
            
            elif len(init_in.shape) == len(self.inputs[0].shape):
                init_output_shape =  (init_in.shape[0],) + self.outputs[0].shape[1:]

            if init_out.virtual:
                init_out.shape = init_output_shape

                if init_out.shape != init_output_shape:
                    raise ValueError("ExtractKernel initialization output shape does not match expected output shape")

                # TODO a lot of repeated code from execute below, determine how best to refactor
                d_in = self.inputs[0]
                if (init_in.mp_type != MPEnums.TENSOR and
                    d_in.mp_type == MPEnums.TENSOR):
                    init_in = init_in.to_tensor()

                    # insert an additional slice for the batch dimension
                    if add_batch_dim and init_in.mp_type == MPEnums.TENSOR:
                        self._indices.insert(0, ":")

                self._process_data(init_in, init_out)

                # remove the additional slice for the batch dimension
                if add_batch_dim and init_in.mp_type == MPEnums.TENSOR:
                    self._indices.pop(0)
    
    def _verify(self):
        """
        Verify the inputs and outputs are appropriately sized
        """
        
        # input must be a tensor or array
        # additionally, if the input is a tensor, the output should also be a
        # tensor
        d_in = self.inputs[0]
        d_out = self.outputs[0]
        if (d_in.mp_type == MPEnums.TENSOR):
            if (d_out.mp_type != MPEnums.TENSOR):
                raise TypeError("ExtractKernel requires Tensor output if input is a Tensor")
        elif (d_in.mp_type != MPEnums.ARRAY and
              d_in.mp_type != MPEnums.CIRCLE_BUFFER):
            raise TypeError("ExtractKernel requires Tensor or Array input")

        # if the input is an array, then the there should only be a single 
        # dimension to extract
        if d_in.mp_type != MPEnums.TENSOR:
            for index in self._indices:
                if not isinstance(index, int):
                    raise TypeError("ExtractKernel requires integer extraction indicies if input is an Array")
            
                # check that the index to extract do not exceed the capacity
                if index >= self._in.capacity or index < -self._in.capacity:
                    raise ValueError("ExtractKernel requires extraction indicies to be within the capacity of the input Array")

            # if the output is another array, validate that the types match
            in_element = d_in.get_element(0)
            if (d_out.mp_type == MPEnums.ARRAY or
                d_out.mp_type == MPEnums.CIRCLE_BUFFER):
                out_element = d_out.get_element(0)
                if (in_element.mp_type != out_element.mp_type):
                    raise TypeError("ExtractKernel requires Array output to have the same type as the input Array")

                # also check that the output has sufficient capacity
                if d_out.capacity < len(self._indices):
                    raise ValueError("ExtractKernel requires Array output to have sufficient capacity to store the extracted elements")

            # if the output is a scalar, check that the input is an array of compatible scalars
            elif (d_out.mp_type == MPEnums.SCALAR):
                if (in_element.mp_type != MPEnums.SCALAR or
                    in_element.data_type != d_out.data_type):
                    raise TypeError("ExtractKernel requires Scalar output to have the same type as the input Array")

                # also verify that only one index is being extracted
                if len(self._indices) != 1:
                    raise ValueError("ExtractKernel requires only one index extracted when output is a Scalar")

            # if the output is a tensor, ensure its dimensions are valid
            elif d_out.mp_type == MPEnums.TENSOR:
                # case 1 : array of tensors
                if in_element.mp_type == MPEnums.TENSOR:
                    out_shape = (len(self._indices),) + in_element.shape

                # case 2 : array of scalars
                elif in_element.mp_type == MPEnums.SCALAR:
                    out_shape = (len(self._indices),1)

                else:
                    raise TypeError("ExtractKernel requires Array input to contain only Scalars or Tensors")

                # if output is virtual, set the shape
                if d_out.virtual and len(d_out.shape) == 0:
                    d_out.shape = out_shape

                # check that the shape is valid
                if d_out.shape != out_shape:
                    raise ValueError("ExtractKernel Tensor output shape does not match expected output shape")
       
        elif d_in.mp_type == MPEnums.TENSOR:
            if d_out.mp_type != MPEnums.TENSOR:
                raise TypeError("ExtractKernel requires Tensor output if input is a Tensor")

            # check that the number of dimensions indicated does not exceed 
            # the tensor's rank
            if len(self._indices) != len(d_in.shape):
                warnings.warn("Number of dimensions to extract exceeds the tensor's rank")
                raise ValueError("ExtractKernel requires number of dimensions to extract to be less than or equal to the tensor's rank")
            
            output_sz = []
            for axis in range(len(self._indices)):
                if self._indices[axis] != ":":
                    axis_indices = self._indices[axis]
                    if isinstance(axis_indices,int):
                        axis_indices = (axis_indices,)
                    for index in axis_indices:
                        # check that the index is valid for the given axis
                        if index < -d_in.shape[axis] or index >= d_in.shape[axis]:
                            raise ValueError("ExtractKernel extraction index in dimension {} exceeds the input Tensor's shape".format(axis))
                    
                    if not self._reduce_dims or len(self._indices[axis]) > 1:
                        output_sz.append(len(axis_indices))
                else:
                    output_sz.append(d_in.shape[axis])
            
            # check that the output tensor's dimensions are valid
            output_sz = tuple(output_sz)
            
            if d_out.virtual and len(d_out.shape) == 0:
                d_out.shape = output_sz
            
            if d_out.shape != output_sz:
                raise ValueError("ExtractKernel Tensor output shape does not match expected output shape")
        

    def _process_data(self, inputs, outputs):
        inA = inputs[0]
        outA = outputs[0]

        if inA.mp_type != MPEnums.TENSOR:
            # extract the elements and set in the output array
            if (outA.mp_type == MPEnums.ARRAY or
                outA.mp_type == MPEnums.CIRCLE_BUFFER):
                for dest_index, src_index in enumerate(self._indices):
                    elem = inA.get_element(src_index) # extract from input
                    outA.set_element(dest_index,elem) # set to output

            elif outA.mp_type == MPEnums.SCALAR:
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
        

    @classmethod
    def add_extract_node(cls,graph,inA,indices,outA,reduce_dims=False):
        """
        Factory method to create an extract kernel 
        and add it to a graph as a generic node object.

        Parameters
        ----------
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

        Returns
        -------
        node : Node
            The node object that was added to the graph containing the extract kernel
        """
        
        # create the kernel object
        k = cls(graph,inA,indices,outA,reduce_dims)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,MPEnums.INPUT),
                  Parameter(outA,MPEnums.OUTPUT))
        
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
        """
        Constructor for the Stack kernel
        """
        super().__init__('stack',MPEnums.INIT_FROM_NONE,graph)
        self.inputs = [inA]
        self.outputs = [outA]
        self._axis = axis

    def _verify(self):
        """
        Verify the inputs and outputs are appropriately sized
        """
        inA = self.inputs[0]
        outA = self.outputs[0]
                
        # inA must be an array and outA must be a tensor
        if (not ((inA.mp_type == MPEnums.ARRAY or 
                  inA.mp_type == MPEnums.CIRCLE_BUFFER) and 
            outA.mp_type == MPEnums.TENSOR)):
            raise TypeError("StackKernel requires Array input and Tensor output")
        
        # if an axis was provided, it must be a scalar
        if self._axis != None and self._axis.mp_type != MPEnums.SCALAR:
            raise TypeError("StackKernel requires Scalar axis")
        
        stack_axis = self._axis.data if self._axis != None else 0
        
        # ensure that all the tensors in inA are the same size
        tensor_shapes = [inA.get_element(i).shape for i in range(inA.capacity)]
        
        if len(set(tensor_shapes)) != 1:
            # tensors in array are different sizes OR array is empty
            raise ValueError("StackKernel requires all tensors in input array to be the same size")
        
        # determine the output dimensions
        output_shape = (tensor_shapes[0][:stack_axis] + (inA.capacity,) 
                         + tensor_shapes[0][stack_axis:])
        
        # check the output dimensions are valid
        if outA.virtual and len(outA.shape) == 0:
            outA.shape = output_shape
        
        # ensure the output shape equals the expected output shape
        if outA.shape != output_shape:
            raise ValueError("StackKernel output shape does not match expected output shape")
        

    def _process_data(self, inputs, outputs):
        """
        Execute the kernel function using numpy functions
        """

        inA = inputs[0]
        outA = outputs[0]

        stack_axis = self._axis.data if self._axis != None else 0
        
        input_tensors = [inA.get_element(i) for i in range(inA.capacity)]
        
        input_data = [t.data for t in input_tensors]
        outA.data = np.stack(input_data,axis=stack_axis)
        
    @classmethod
    def add_stack_node(cls,graph,inA,outA,axis=None):
        """
        Factory method to create a stack kernel and add it to a graph
        as a generic node object.

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

        Returns
        -------
        node : Node
            The node object that was added to the graph containing the stack kernel
        """
        
        # create the kernel object
        k = cls(graph,inA,outA,axis)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,MPEnums.INPUT),
                  Parameter(outA,MPEnums.OUTPUT))
        
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
        """
        Constructor for the TensorStack kernel
        """
        super().__init__('TensorStack',MPEnums.INIT_FROM_NONE,graph)
        self.inputs = [inA,inB]
        self.outputs = [outA]
        self._axis = axis

    def _initialize(self, init_inputs, init_outputs, labels):
        """
        This kernel has no internal state that must be initialized. 
        """

        init_inA, init_inB = init_inputs
        init_out = init_outputs[0]

        for init_in in (init_inA, init_inB):
            if init_in.mp_type != MPEnums.TENSOR:
                init_in = init_in.to_tensor()

        if init_out is not None and (init_inA is not None and init_inA.shape != ()):
            # adjust the init output shape
            if init_out.virtual:
                init_out.shape = init_inA.shape[:self._axis+1] + (2,) + init_inA.shape[self._axis+1:]

            axis_adjusted = False
            if len(init_inA.shape) == len(self.inputs[0].shape)+1 and self._axis >= 0:
                # adjust axis to accomodate stack of input data
                self._axis += 1
                axis_adjusted = True
            self._process_data([init_inA, init_inB], init_outputs)

            if axis_adjusted:
                self._axis -= 1 # adjust back for trial processing

        
    def _verify(self):
        """
        Verify the inputs and outputs are appropriately sized
        """

        inA, inB = self.inputs
        outA = self.outputs[0]
                
        # all params must be tensors
        for param in (inA, inB, outA):
            if param.mp_type != MPEnums.TENSOR:
                raise TypeError("TensorStackKernel requires Tensor inputs and outputs")
        
        stack_axis = self._axis
        if stack_axis >= len(inA.shape) and stack_axis < -len(inA.shape):
            raise ValueError("TensorStackKernel requires stack axis to be within the rank of the input tensors")
        
        # ensure that all the tensors in inA are the same size
        tensor_shapes = [inA.shape, inB.shape]
        
        if len(set(tensor_shapes)) != 1:
            # tensors in array are different sizes OR array is empty
            raise ValueError("TensorStackKernel requires all tensors in input array to be the same size")   
        
        # determine the output dimensions
        output_shape = inA.shape[:stack_axis] + (2,) + inA.shape[stack_axis:]
        
        # check the output dimensions are valid
        if outA.virtual and len(outA.shape) == 0:
            outA.shape = output_shape
        
        # ensure the output shape equals the expected output shape
        if outA.shape != output_shape:
            raise ValueError("TensorStackKernel output shape does not match expected output shape")


    def _process_data(self, inputs, outputs):
        """
        Process data according to outlined kernel function.
        """
        input_tensors = [inputs[i].data for i in range(len(inputs))]
        outputs[0].data = np.stack(input_tensors,axis=self._axis)
        
    
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
        
        Returns
        -------
        node : Node
            The node object that was added to the graph containing the tensor stack kernel
        """
        
        # create the kernel object
        k = cls(graph,inA,inB,outA,axis)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,MPEnums.INPUT),
                  Parameter(inB,MPEnums.INPUT),
                  Parameter(outA,MPEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node


