Advanced Setup
==============

There are a number of more advanced features that may be useful to users

Adding new kernels
------------------
BCIPy was made to be extendable. If there is a particular operation that has not been implemented, it is possible to add it yourself. The following steps will guide you through the process of adding a new kernel.

#. Create a new python file in the ``kernels`` directory. The name of the file should be the name of the kernel you are implementing. For example, if you are implementing a kernel that performs a convolution, the file name should be ``convolution.py``.
#. Within the file, create a class for the kernel. The name of the class should be the same as the file name, followed by ``Kernel``. For example, if the file name is ``convolution.py``, the class name should be ``ConvolutionKernel``.
    * The class should inherit from ``Kernel``.
    * It may be useful to create a docstring for the class. This will be displayed when the user calls ``help`` on the kernel.

    .. code-block:: python

        class ConvolutionKernel(Kernel):
            """Performs a convolution on the input data."""

#. At this point, the kernel is ready to be implemented. The following methods must be implemented:
    * ``__init__``: This method should take in any parameters that are required for the kernel to operate. These parameters should be stored as attributes of the class.
    * ``verify``: This method should check that the input and output tensor parameters passed to the kernel are valid. If they are not, it should return ``BcipEnums.INVALID_PARAMETERS``.
        * With a convolution kernel, the output dimensions can be computed using a manual formula. If the output tensor dimensions do not match the computed dimensions, the input parameters are invalid.
        * Alternatively, it will be shown how the process_data method can be used to verify the input and output tensors.
        * The verify method must also check that the type of the input and output tensors are valid. For example, the input and output tensors for a convolution kernel should be BCIPy ``Tensor`` objects.
        * If the input and output tensors are valid, the method should return ``BcipEnums.SUCCESS``.
    * ``initialize``: This method should perform any initialization that is required before the kernel can be used. This method is particularly relevant when kernels have a component that needs to be trained.
        * For example, a classifier kernel may have a classifier object that need to be initialized.
        * If the initialization is successful, the method should return ``BcipEnums.SUCCESS``.
    * ``execute``: Executes single trial processing
        * For most kernels, this method will simply call ``_process_data``.
    * ``_process_data``: This method should perform the operation on the input data and store the result in the output tensor.
        * For example, a convolution kernel may perform a convolution on the input data and store the result in the output tensor.
        * If the operation is successful, the method should return ``BcipEnums.SUCCESS``.
    * ``add_convolution_node``: a Factory Method (Don't forget the @classmethod decorator) used to create a kernel, adding it to a node, and adding that node to a specified graph.
        * This method should take in the following parameters:
            * ``graph``: The graph to which the node should be added.
            * ``input_tensor_name``: The name of the input tensor to the node.
            * ``second_input_tensor_name``: The name of the second input tensor to the node. This is only required for kernels that take in two input tensors.
            * ``output_tensor_name``: The name of the output tensor to the node.
            * Other parameters that are required for the kernel to operate.

        * The method should create an instance of the kernel, add it to a node, and add that node to the specified graph. The method should return the node that was added to the graph.
        * For example, a convolution kernel may be added to a graph as follows:
  
    .. code-block:: python
        
        @classmethod
        def add_convolution_node(cls, graph, input_tensor_name, second_input_tensor_name, output_tensor_name, stride, padding):
            """Adds a convolution node to the graph."""
            
            kernel = cls(graph, input_tensor_name, second_input_tensor_name, output_tensor_name, stride, padding)
            params = (bcipy.Parameter(input_tensor_name, BcipEnums.INPUT),
                      bcipy.Parameter(second_input_tensor_name, BcipEnums.INPUT),
                      bcipy.Parameter(output_tensor_name, BcipEnums.OUTPUT))

            node = Node(graph, kernel, params)
            graph.add_node(node)
            return node

    * As mentioned, it is possible to use the ``_process_data`` method in ``verify``. For example, by passing two empty numpy arrays to ``_process_data``, the kernel will perform the operation on the empty arrays. 
        * If the output dimensions from ``_process_data`` do not match the expected output dimensions, the input parameters are invalid.
        * NOTE: This may not work for all kernels, and a manual formula may be required to verify the input parameters.

#. At this point, the kernel is ready to be used.


Other functionality
-------------------
