from ..core import MPEnums
from ..kernel import Kernel
from ..graph import Node, Parameter

import numpy as np


class EpochKernel(Kernel):
    """
    Epochs a continuous signal into a series of smaller segments

    Parameters
    ----------
    graph : Graph
        Graph that the kernel should be added to

    inA : Tensor
        Input data

    outA : Tensor
        Output data

    epoch_length : int
        Length of each epoch in samples

    epoch_stride : int, default=None
        Number of samples between consecutive epochs. If None, defaults to
        epoch_length

    axis : int, default=-1
        Axis along which to epoch the data
    """

    def __init__(self, graph, inA, outA, epoch_length, epoch_stride=None, axis=-1):
        """ Init """
        super().__init__('Epoch', MPEnums.INIT_FROM_NONE, graph)
        self.inputs = [inA]
        self.outputs = [outA]
        self._epoch_length = int(epoch_length)
        if epoch_stride is None:
            self._epoch_stride = self._epoch_length
        else:
            self._epoch_stride = int(epoch_stride)
        self._axis = axis

    def _compute_output_shape(self, input_shape):
        """
        Computes the shape of the output tensor based on the input shape

        Parameters
        ----------

        input_shape: Array
            Shape of input tensor
        """
        output_shape = list(input_shape)
        output_shape[self._axis] = self._epoch_length
        output_shape.insert(self._axis,
                            int(input_shape[self._axis] - self._epoch_length) // self._epoch_stride + 1)
        return tuple(output_shape)

    def _initialize(self, init_inputs, init_outputs, labels):
        """
        This kernel has no internal state that must be initialized.
        Call initialization_execution if downstream nodes are missing training
        data
        """

        init_in = init_inputs[0]
        init_out = init_outputs[0]

        if (init_out is not None and
            init_in is not None and
            init_in.shape != ()):
            if init_in is not None and init_in.mp_type != MPEnums.TENSOR:
                init_in = init_in.to_tensor()

            axis_adjusted = False
            if len(init_in.shape) == len(self.inputs[0].shape)+1 and self._axis >= 0:
                self._axis += 1
                axis_adjusted = True

            # update output size, as needed
            if init_out.virtual:
                init_out.shape = self._compute_output_shape(init_in.shape)

            self._process_data([init_in], init_outputs)

            if axis_adjusted:
                self._axis -= 1

    def _verify(self):
        """
        Verify the inputs and outputs are appropriately sized and typed
        """

        inA = self.inputs[0]
        outA = self.outputs[0]

        if inA.mp_type != MPEnums.TENSOR:
            raise TypeError("Input must be a tensor")

        if outA.mp_type != MPEnums.TENSOR:
            raise TypeError("Output must be a tensor")

        if self._epoch_length <= 0:
            raise ValueError("Epoch length must be greater than 0")

        if self._epoch_stride <= 0:
            raise ValueError("Epoch stride must be greater than 0")

        if self._axis < -len(inA.shape) or self._axis >= len(inA.shape):
            raise ValueError("Axis must be within rank of input tensor")

        if self._axis < 0:
            self._axis += len(inA.shape)

        out_shape = self._compute_output_shape(inA.shape)
        if outA.virtual and len(outA.shape) == 0:
            outA.shape = out_shape

        if outA.shape != out_shape:
            raise ValueError(f"Output shape must be {out_shape}")

    def _process_data(self, inputs, outputs):
        """
        Execute the kernel and epoch the data

        Parameters
        ----------

        inputs: Tensor
            Input data
        
        outputs: Tensor
            Output data
        """

        inA = inputs[0]
        outA = outputs[0]

        # epoch the data
        src_slc = [slice(None)] * len(inA.shape)
        dst_slc = [slice(None)] * len(outA.shape)
        Nepochs = int(inA.shape[self._axis] - self._epoch_length) // self._epoch_stride + 1
        for i_e in range(Nepochs):
            src_slc[self._axis] = slice(i_e*self._epoch_stride,
                                    i_e*self._epoch_stride + self._epoch_length)
            dst_slc[self._axis] = i_e
            outA.data[tuple(dst_slc)] = inA.data[tuple(src_slc)]

    @classmethod
    def add_to_graph(cls, graph, inA, outA, epoch_len, epoch_stride=None,
                     axis=-1, init_inputs=None, labels=None):
        """
        Factory method to create an epoch kernel and add it to a graph as a
        generic node object.

        Parameters
        ----------
        graph : Graph
            Graph that the kernel should be added to

        inA : Tensor
            Input data

        outA : Tensor
            Output data

        epoch_len : int
            Length of each epoch in samples

        epoch_stride : int, default=None
            Number of samples between consecutive epochs. If None, defaults to
            epoch_length

        axis : int, default=-1
            Axis along which to epoch the data
        """

        # create the kernel object
        k = cls(graph, inA, outA, epoch_len, epoch_stride, axis)

        # create parameter objects for the input and output
        params = (Parameter(inA, MPEnums.INPUT),
                  Parameter(outA, MPEnums.OUTPUT))

        # add the kernel to a generic node object
        node = Node(graph, k, params)

        # add the node to the graph
        graph.add_node(node)

        if init_inputs is not None:
            node.add_initialization_data(init_inputs, labels)

        return node
