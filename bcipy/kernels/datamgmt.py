from ..core import BCIP, BcipEnums
from ..kernel import Kernel
from ..graph import Node, Parameter
from ..containers import Scalar

import numpy as np

class ConcatenationKernel(Kernel):
    """
    Kernel to concatenate multiple tensors into a single tensor

    Parameters
    ----------

    graph : Graph Object
        - Graph that the kernel should be added to

    inA : Tensor object
        - First input trial data

    inB : Tensor object
        - Second input trial data

    outA : Tensor object
        - Output trial data

    axis : int or tuple of ints, default = 0
        - The axis along which the arrays will be joined. If axis is None, arrays are flattened before use. Default is 0. 
        - See numpy.concatenate for more information
    """
    
    def __init__(self,graph,outA,inA,inB,axis=0):
        super().__init__('Concatenation',BcipEnums.INIT_FROM_NONE,graph)
        self._inA  = inA
        self._inB  = inB
        self._outA = outA
        self._axis = axis
        self._init_inA = None
        self._init_inB = None
        self._init_outA = None

        self._labels = None


    def initialize(self):
        """
        This kernel has no internal state that must be initialized
        """
        sts = BcipEnums.SUCCESS

        if self._init_outA != None:
            if len(self._init_outA.shape) == 0:
                output_sz, _, _ = self._resolve_dims(self._init_inA, self._init_inB)
                self._init_outA.shape = output_sz

            sts = self._process_data(self._init_inA, self._init_inB, self._init_outA)
        
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
        for param in (self._inA, self._inB, self._outA):
            if param._bcip_type != BcipEnums.TENSOR:
                return BcipEnums.INVALID_PARAMETERS
        
            
        # the dimensions along the catcat axis must be equal
        output_sz, noncat_sz_A, noncat_sz_B = self._resolve_dims(self._inA,self._inB)
        if len(output_sz) == 0:
            return BcipEnums.INVALID_PARAMETERS
        
        # check if the remaining dimensions are the same
        if ((len(noncat_sz_A) != len(noncat_sz_B)) or 
             len(noncat_sz_A) != sum([1 for i,j in 
                                     zip(noncat_sz_A,noncat_sz_B) if i==j])):
            return BcipEnums.INVALID_PARAMETERS
        
        # check the output dimensions are valid
        if self._outA.virtual and len(self._outA.shape) == 0:
            self._outA.shape = output_sz
        
        # ensure the output shape equals the expected output shape
        if self._outA.shape != output_sz:
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
        except ValueError:
            # dimensions are invalid
            return BcipEnums.EXE_FAILURE
        
        # set the data in the output tensor
        output_data.data = out_tensor
        
        return BcipEnums.SUCCESS

    def execute(self):
        """
        Execute the kernel function using numpy functions
        """
        return self.process_data(self._inA, self._inB, self._outA)
    
    @classmethod
    def add_concatenation_node(cls,graph,inA,inB,outA,axis=0):
        """
        Factory method to create a concatenation kernel and add it to a graph
        as a generic node object.

        Parameters
        ----------
        graph : Graph Object
            - Graph that the kernel should be added to

        inA : Tensor object
            - First input trial data

        inB : Tensor object
            - Second input trial data

        outA : Tensor object
            - Output trial data

        axis : int or tuple of ints, default = 0
            - The axis along which the arrays will be joined. If axis is None, arrays are flattened before use. Default is 0. 
            - See numpy.concatenate for more information
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

    graph : Graph Object
        - Graph that the kernel should be added to

    inA : BCIP Object
        - Input data to enqueue into circle buffer

    queue : Circle Buffer object
        - Circle buffer to have data enqueud to

    """
    
    def __init__(self,graph,inA,queue):
        super().__init__('Enqueue',BcipEnums.INIT_FROM_NONE,graph)
        self._inA  = inA
        self._outA = queue

        self._labels = None
    
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
        if not isinstance(self._inA,BCIP):
            return BcipEnums.INVALID_PARAMETERS
        
        if self._outA._bcip_type != BcipEnums.CIRCLE_BUFFER:
            return BcipEnums.INVALID_PARAMETERS

        # check that the buffer's capacity is at least 1
        if self._outA.capacity <= 1:
            return BcipEnums.INVALID_PARAMETERS
        
        return BcipEnums.SUCCESS
    
    def execute(self):
        """
        Execute the kernel function using numpy function
        """
        # need to make a deep copy of the object to enqueue
        cpy = self._inA.make_copy()
        self._outA.enqueue(cpy)
            
        return BcipEnums.SUCCESS
    
    @classmethod
    def add_enqueue_node(cls,graph,inA,queue):
        """
        Factory method to create a enqueue kernel and add it to a graph as a generic node object.

        graph : Graph Object
        - Graph that the kernel should be added to

        inA : BCIP Object
            - Input data to enqueue into circle buffer

        queue : Circle Buffer object
            - Circle buffer to have data enqueud to
        """
        
        # create the kernel object
        k = cls(graph,inA,queue)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,BcipEnums.INPUT),
                  Parameter(queue,BcipEnums.INOUT))
        
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
    graph : Graph Object
        - Graph that the kernel should be added to

    inA : Tensor or Array object
        - Input trial data

    Indicies : list slices, list of ints
        - Indicies within inA from which to extract data

    outA : Tensor object
        - Extracted trial data

    reduce_dims : bool, default = False
        - Remove singleton dimensions if true, don't squeeze otherwise
    """
    
    def __init__(self,graph,inA,indices,outA,reduce_dims):
        super().__init__('Extract',BcipEnums.INIT_FROM_NONE,graph)
        self._in = inA
        self._out = outA
        self._indices = indices
        self._reduce_dims = reduce_dims

        self._init_inA = None
        self._init_outA = None

        self._labels = None
    
    def initialize(self):
        """
        This kernel has no internal state that must be initialized
        """
        sts = BcipEnums.SUCCESS

        if self._init_outA != None:
            if len(self._init_outA.shape) == 0:
                if len(self._init_inA.shape) == (len(self._in.shape)+1):
                    self._init_outA.shape = (self._init_inA.shape[0],) + self._out.shape
                
                    # TODO a lot of repeated code from execute below, determine how best to refactor
                    ix_grid = [[_ for _ in range(self._init_inA.shape[0])]]
                    for axis in range(len(self._indices)):
                        axis_indices = self._indices[axis]
                        if axis_indices == ":":
                            ix_grid.append([_ for _ in range(self._init_inA.shape[axis])])
                        else:
                            if isinstance(axis_indices,int):
                                ix_grid.append([axis_indices])
                            else:
                                ix_grid.append(list(axis_indices))

                    npixgrid = np.ix_(*ix_grid)
                    extr_data = self._in.data[npixgrid]
                    
                    if self._reduce_dims:
                        extr_data = np.squeeze(extr_data)
                        
                    self._init_outA.data = extr_data

                else:
                    sts = BcipEnums.INITIALIZATION_FAILURE

        return sts
    
    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized
        """
        
        # input must be a tensor or array
        # additionally, if the input is a tensor, the output should also be a
        # tensor
        if (self._in._bcip_type == BcipEnums.TENSOR):
            if (self._out._bcip_type != BcipEnums.TENSOR):
                return BcipEnums.INVALID_PARAMETERS
        elif (self._in._bcip_type != BcipEnums.ARRAY and
              self._in._bcip_type != BcipEnums.CIRCLE_BUFFER):
            return BcipEnums.INVALID_PARAMETERS


        # if the input is an array, then the there should only be a single 
        # dimension to extract with a value of zero
        if self._in._bcip_type != BcipEnums.TENSOR:
            for index in self._indices:
                if not isinstance(index, int):
                    return BcipEnums.INVALID_PARAMETERS
            
                # check that the index to extract do not exceed the capacity
                if index >= self._in.capacity or index < -self._in.capacity:
                    return BcipEnums.INVALID_PARAMETERS

            # if the output is another array, validate that the types match
            in_element = self._in_get_element(0)
            if (self._out._bcip_type == BcipEnums.ARRAY or
                self._out._bcip_type == BcipEnums.CIRCLE_BUFFER):
                out_element = self._out.get_element(0)
                if (in_element._bcip_type != out_element._bcip_type):
                    return BcipEnums.INVALID_PARAMETERS

                # also check that the output has sufficient capacity
                if self._out.capacity < len(self._indices):
                    return BcipEnums.INVALID_PARAMETERS

            # if the output is a scalar, check that the input is an array of compatible scalars
            elif (self._out._bcip_type == BcipEnums.SCARLAR):
                if (in_element._bcip_type != BcipEnums.SCALAR or
                    in_element.data_type != self._out.data_type):
                    return BcipEnums.INVALID_PARAMETERS

                # also verify that only one index is being extracted
                if len(self._indices) != 1:
                    return BcipEnums.INVALID_PARAMETERS

            # if the output is a tensor, ensure its dimensions are valid
            elif self._out._bcip_type == BcipEnums.TENSOR:
                # case 1 : array of tensors
                if in_element._bcip_type == BcipEnums.TENSOR:
                    out_shape = (len(self._indices),) + in_element.shape

                # case 2 : array of scalars
                elif in_element._bcip_type == BcipEnums.SCALAR:
                    out_shape = (len(self._indices),1)

                else:
                    return BcipEnums.INVALID_PARAMETERS

                # if output is virtual, set the shape
                if self._out.virtual and len(self._out.shape) == 0:
                    self._out.shape = out_shape

                # check that the shape is valid
                if self._out.shape != out_shape:
                    return BcipEnums.INVALID_PARAMETERS
       
        elif self._in._bcip_type == BcipEnums.TENSOR:
            if self._out._bcip_type != BcipEnums.TENSOR:
                return BcipEnums.INVALID_PARAMETERS

            # check that the number of dimensions indicated does not exceed 
            # the tensor's rank
            if len(self._indices) != len(self._in.shape):
                return BcipEnums.INVALID_PARAMETERS
            
            output_sz = []
            for axis in range(len(self._indices)):
                if self._indices[axis] != ":":
                    axis_indices = self._indices[axis]
                    if isinstance(axis_indices,int):
                        axis_indices = (axis_indices,)
                    for index in axis_indices:
                        # check that the index is valid for the given axis
                        if index < -self._in.shape[axis] or index >= self._in.shape[axis]:
                            return BcipEnums.INVALID_PARAMETERS
                    
                    if not self._reduce_dims or len(self._indices[axis]) > 1:
                        output_sz.append(len(axis_indices))
                else:
                    output_sz.append(self._in.shape[axis])
            
            # check that the output tensor's dimensions are valid
            output_sz = tuple(output_sz)
            
            if self._out.virtual and len(self._out.shape) == 0:
                self._out.shape = output_sz
            
            if self._out.shape != output_sz:
                return BcipEnums.INVALID_PARAMETERS
        
        return BcipEnums.SUCCESS
        
    def initialization_execution(self):
        """
        Update initialization output if downstream nodes are missing training data
        """
        sts = self.process_data(self._init_inA, self._init_outA)
        
        if sts != BcipEnums.SUCCESS:
            return BcipEnums.INITIALIZATION_FAILURE
        
        return sts

    def execute(self):
        """
        Execute the kernel function
        """
        if self._in._bcip_type != BcipEnums.TENSOR:
            # extract the elements and set in the output array
            if (self._out._bcip_type == BcipEnums.ARRAY or
                self._out._bcip_type == BcipEnums.CIRCLE_BUFFER):
                for dest_index, src_index in enumerate(self._indices):
                    elem = self._in.get_element(src_index) # extract from input
                    self._out.set_element(dest_index,elem) # set to output

            elif self._out._bcip_type == BcipEnums.SCALAR:
                self._out.data = self._in.get_element(self._indices[0])

            else:
                # tensor output
                out_array = self._out.data
                for dest_index, src_index in enumerate(self._indices):
                    elem_data = self._in.get_element(src_index).data
                    out_array[dest_index] = elem_data
                self._out.data = out_array
        else:
            # tensor input 
            ix_grid = []
            for axis in range(len(self._indices)):
                axis_indices = self._indices[axis]
                if axis_indices == ":":
                    ix_grid.append([_ for _ in range(self._in.shape[axis])])
                else:
                    if isinstance(axis_indices,int):
                        ix_grid.append([axis_indices])
                    else:
                        ix_grid.append(list(axis_indices))

            npixgrid = np.ix_(*ix_grid)
            extr_data = self._in.data[npixgrid]
            
            if self._reduce_dims:
                extr_data = np.squeeze(extr_data)
                
            self._out.data = extr_data
        
        return BcipEnums.SUCCESS
    
    @classmethod
    def add_extract_node(cls,graph,inA,indices,outA,reduce_dims=False):
        """
        Factory method to create an extract kernel 
        and add it to a graph as a generic node object.

         graph : Graph Object
            - Graph that the kernel should be added to

        inA : Tensor or Array object
            - Input trial data

        Indicies : list slices, list of ints
            - Indicies within inA from which to extract data

        outA : Tensor, Scalar, or Array object
            - Extracted trial data

        reduce_dims : bool, default = False
            - Remove singleton dimensions if true, don't squeeze otherwise
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

class SetKernel(Kernel):
    """
    Kernel to set a portion of a tensor or array

    Parameters
    ----------
    graph : graph object
        - The graph where the RunningAverageKernel object should be added
    
    inA : Tensor or Array object
        - Container where specified data will be added to

    data : Tensor/Scalar/Array object
        - Data to add to container

    axis : int
        - Axis over which to set the data

    index : array_like
        - Indices of where to change the data within the Container object

    out : Tensor/Array object
        - Output data

    """
    
    # TODO this kernel's behaviour needs to be defined more thoroughly

    def __init__(self,graph,inA,data,axis,index,out):
        super().__init__('Extract',BcipEnums.INIT_FROM_NONE,graph)
        self._inA = inA
        self._out = out
        self._data = data
        self._axis = axis
        self._index  = index

    def initialize(self):
        """
        This kernel has no internal state that must be initialized
        """
        return BcipEnums.SUCCESS
    
    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized
        """
        
        # input must be a tensor, scalar, or array
        # additionally, if the input is a tensor, the output should also be a
        # tensor
        if (self._inA._bcip_type != BcipEnums.SCALAR and
            self._inA._bcip_type != BcipEnums.TENSOR and
            self._inA._bcip_type != BcipEnums.ARRAY and
            self._inA._bcip_type != BcipEnums.CIRCLE_BUFFER):
            return BcipEnums.INVALID_PARAMETERS

        # output must be the same type as input
        if self._inA._bcip_type != self._out._bcip_type:
            return BcipEnums.INVALID_PARAMETERS


        # ARRAY INPUT/OUTPUT
        if (self._inA._bcip_type == BcipEnums.ARRAY or
            self._inA._bcip_type == BcipEnums.CIRCLE_BUFFER):
            # check that the data to set and the source have the same type
            e = self._inA.get_element(0)
            if e._bcip_type != self._data._bcip_type:
                return BcipEnums.INVALID_PARAMETERS

            if e._bcip_type == BcipEnums.SCALAR:
                if e.data_type != self._data.data_type:
                    return BcipEnums.INVALID_PARAMETERS

            # check that the destination index is within the capacity of the array
            if self._out.virtual and self._out.capacity == 0:
                # set the capacity of the array
                self._out.capacity = self._container.capacity

            if (self._inA.capacity != self._out.capacity or
                self._index >= self._out.capacity):
                return BcipEnums.INVALID_PARAMETERS

        # SCALAR INPUT/OUTPUT
        elif self._inA._bcip_type == BcipEnums.SCALAR:
            if (data._bcip_type != BcipEnums.SCALAR or
                data.data_type != self._out.data_type):
                return BcipEnums.INVALID_PARAMETERS

        # TENSOR INPUT/OUTPUT
        else:
        
            # check that the output tensor's dimensions are valid
            output_shape = self._inA.shape
            
            if self._out.virtual and len(self._out.shape) == 0:
                self._out.shape = output_shape
            
            if self._out.shape != output_shape:
                return BcipEnums.INVALID_PARAMETERS
            
            # check that the axis specified does not exceed the tensor's rank
            if self._axis >= len(self._out.shape):
                return BcipEnums.INVALID_PARAMETERS
            
            # check that the index is a valid location in the container tensor
            if self._index >= self._container.shape[self._axis]:
                return BcipEnums.INVALID_PARAMETERS
        
            
            # check if the dimensions of the data to set match the shape 
            # fit the output shape
            ix_grid = []
            for i in len(self._out.shape):
                if i == self._axis:
                    ix_grid.append(list(self._index))
                else:
                    ix_grid.append([_ for _ in range(self._out.shape[i])])
            
            ixgrid = np.ix_(ix_grid)
            set_shape = self._out.data[ixgrid].shape
            if set_shape != self._inA.shape:
                return BcipEnums.INVALID_PARAMETERS
        
        return BcipEnums.SUCCESS
        
    def execute(self):
        """
        Execute the kernel function using numpy function
        """
        
        if self._inA._bcip_type != BcipEnums.TENSOR:
            # copy all the elements of the input container except the the 
            # data to set
            for i in range(self._out.capacity):
                if i == self._index:
                    self._out.set_element(i,self._inA)
                else:
                    self._out.set_element(i,self._container.get_element(i))
        else:
            # tensor case
            ix_grid = []
            for i in len(self._out.shape):
                if i == self._axis:
                    ix_grid.append(list(self._index))
                else:
                    ix_grid.append([_ for _ in range(self._out.shape[i])])
            
            ixgrid = np.ix_(ix_grid)
            out_data = self._container.data
            out_data[ixgrid] = self._inA
            self._out.data = out_data
        
        return BcipEnums.SUCCESS
    
    @classmethod
    def add_set_node(cls,graph,inA,data,axis,index,out):
        """
        Factory method to create a set kernel 
        and add it to a graph as a generic node object.

        Parameters
        ----------
        graph : graph object
            - The graph where the RunningAverageKernel object should be added
        
        inA : Tensor or Array object
            - Container where specified data will be added to

        data : Tensor/Scalar/Array object
            - Data to add to container

        axis : int
            - Axis over which to set the data

        index : array_like
            - Indices of where to change the data within the Container object

        out : Tensor/Array object
            - Output data
        """
        
        # create the kernel object
        k = cls(graph,inA,data,axis,index,out)
        
        # create parameter objects for the input and output
        params = (Parameter(container,BcipEnums.INPUT),
                  Parameter(data,BcipEnums.INPUT),
                  Parameter(index,BcipEnums.INPUT),
                  Parameter(out,BcipEnums.OUTPUT))
        
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
    graph : graph object
        - The graph where the RunningAverageKernel object should be added
    
    inA : Array object
        - Container where specified data will be added to

    outA : Tensor object
        - Tensor of stacked tensors

    axis : int or None, default = None
        - The axis in the result array along which the input arrays are stacked.
    """
    
    def __init__(self,graph,inA,outA,axis=None):
        super().__init__('stack',BcipEnums.INIT_FROM_NONE,graph)
        self._inA  = inA
        self._outA = outA
        self._axis = axis

        self._init_inA = None
        self._init_outA = None
        

        self._labels = None
    
    def initialize(self):
        """
        This kernel has no internal state that must be initialized
        """
        return BcipEnums.SUCCESS
    
    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized
        """
                
        # inA must be an array and outA must be a tensor
        if (not ((self._inA._bcip_type == BcipEnums.ARRAY or 
                  self._inA._bcip_type == BcipEnums.CIRCLE_BUFFER) and 
            self._outA._bcip_type == BcipEnums.TENSOR)):
            return BcipEnums.INVALID_PARAMETERS
        
        # if an axis was provided, it must be a scalar
        if self._axis != None and self._axis._bcip_type != BcipEnums.SCALAR:
            return BcipEnums.INVALID_PARAMETERS
        
        stack_axis = self._axis.data if self._axis != None else 0
        
        # ensure that all the tensors in inA are the same size
        tensor_shapes = [self._inA.get_element(i).shape
                                    for i in range(self._inA.capacity)]
        
        if len(set(tensor_shapes)) != 1:
            # tensors in array are different sizes OR array is empty
            return BcipEnums.INVALID_PARAMETERS
        
        # determine the output dimensions
        output_shape = (tensor_shapes[0][:stack_axis] + (self._inA.capacity,) 
                         + tensor_shapes[0][stack_axis:])
        
        # check the output dimensions are valid
        if self._outA.virtual and len(self._outA.shape) == 0:
            self._outA.shape = output_shape
        
        # ensure the output shape equals the expected output shape
        if self._outA.shape != output_shape:
            return BcipEnums.INVALID_PARAMETERS

        return BcipEnums.SUCCESS


    def execute(self):
        """
        Execute the kernel function using numpy functions
        """
        
        stack_axis = self._axis.data if self._axis != None else 0
        
        try:
            input_tensors = [self._inA.get_element(i) for i 
                                             in range(self._inA.capacity)]
            
            input_data = [t.data for t in input_tensors]
            output_data = np.stack(input_data,axis=stack_axis)
        
        except ValueError:
            return BcipEnums.EXE_FAILURE
        
        # set the data of the output tensor
        self._outA.data = output_data
        
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

    graph : Graph Object
        - Graph that the kernel should be added to

    inA : Tensor object
        - First input trial data

    inB : Tensorobject
        - Second input trial data

    outA : Tensor object
        - Output trial data

    axis : int, default=None
        - Axis over which to stack the tensors. If none, the tensors are flattened before they are stacked
    """
    
    def __init__(self,graph,inA,inB,outA,axis=None):
        super().__init__('TensorStack',BcipEnums.INIT_FROM_NONE,graph)
        self._inA  = inA
        self._inB  = inB
        self._outA = outA
        self._axis = axis

        self._init_inA = None
        self._init_inB = None
        self._init_outA = None

        self._labels = None
    
    def initialize(self):
        """
        This kernel has no internal state that must be initialized. 
        """
        sts = BcipEnums.SUCCESS

        if self._init_outA != None:
            if len(self._init_outA.shape) == 0:
                self._init_outA.shape = self._int_inA.shape[:self._axis+1] + (2,) + self._init_inA.shape[self._axis+1:]

            self._axis += 1 # adjust for batch processing in init TODO will this always be the case?
            sts = self._process_data(self._init_inA, self._init_outA)
            self._axis -= 1 # adjust back for trial processing

        return sts
        
    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized
        """
                
        # all params must be tensors
        for param in (self._inA, self._inB, self._outA):
            if param._bcip_type != BcipEnums.TENSOR:
                return BcipEnums.INVALID_PARAMETERS
        
        stack_axis = self._axis
        
        # ensure that all the tensors in inA are the same size
        tensor_shapes = [self._inA.shape, self._inB.shape]
        
        if len(set(tensor_shapes)) != 1:
            # tensors in array are different sizes OR array is empty
            return BcipEnums.INVALID_PARAMETERS
        
        # determine the output dimensions
        output_shape = self._inA.shape[:stack_axis] + (2,) + self._inA.shape[stack_axis:]
        
        # check the output dimensions are valid
        if self._outA.virtual and len(self._outA.shape) == 0:
            self._outA.shape = output_shape
        
        # ensure the output shape equals the expected output shape
        if self._outA.shape != output_shape:
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
        
        except ValueError:
            return BcipEnums.EXE_FAILURE
    
        
        return BcipEnums.SUCCESS

    def execute(self):
        """
        Execute a single trial
        """
        return self._process_data(self._inA, self._inB, self._outA)
    
    
    @classmethod
    def add_tensor_stack_node(cls,graph,inA,inB,outA,axis=0):
        """
        Factory method to create a tensor stack kernel and add it to a graph
        as a generic node object.

        Parameters
        ----------

        graph : Graph Object
            - Graph that the kernel should be added to

        inA : Tensor or Scalar object
            - First input trial data

        inB : Tensor or Scalar object
            - Second input trial data

        outA : Tensor or Scalar object
            - Output trial data

        axis : int, default=None
            - Axis over which to stack the tensors. If none, the tensors are flattened before they are stacked
        
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


