from .core import BCIP, BcipEnums
from abc import ABC, abstractmethod
from .containers import Tensor

class Kernel(BCIP, ABC):
    """
    An abstract base class that defines the minimum set of kernel methods that
    must be defined

    Parameters
    ----------
    name : str
        Name of the kernel
    init_style : BcipEnums Object
        Kernel initialization style, according to BcipEnum class

    Attributes
    ----------
    _name : str
        Name of the kernel
    _init_style : BcipEnums Object
        Kernel initialization style, according to BcipEnum class
    """
    
    def __init__(self,name,init_style,graph):
        """
        Constructor for the Kernel class
        """
        session = graph.session
        super().__init__(BcipEnums.KERNEL,session)
        self._name = name
        self._init_style = init_style

        self._inputs = []
        self._outputs = []

        self._init_inputs = []
        self._init_outputs = []
        self._init_input_labels = None
        self._init_output_labels = None
        
    # API Getters
    @property
    def name(self):
        """
        Returns the name of the kernel

        Returns
        -------
        _name : str
            The name of the kernel

        """

        return self._name
    
    @property
    def init_style(self):
        """
        Returns the initialization style of the kernel

        Returns
        -------
        _init_style : BcipEnums
            The initialization style of the kernel
        """

        return self._init_style
    
    @property
    def init_input_labels(self):
        """
        Returns the labels for the initialization inputs

        Returns
        -------
        _init_input_labels : list
            The labels for the initialization inputs
        """

        return self._init_input_labels
    
    @property
    def init_output_labels(self):
        """
        Returns the labels for the initialization outputs

        Returns
        -------
        _init_output_labels : list
            The labels for the initialization outputs
        """

        return self._init_output_labels
    
    @property
    def inputs(self):
        """
        Returns the inputs of the kernel

        Returns
        -------
        _inputs : list
            The inputs of the kernel
        """

        return self._inputs
    
    @property
    def outputs(self):
        """
        Returns the outputs of the kernel

        Returns
        -------
        _outputs : list
            The outputs of the kernel
        """

        return self._outputs
    
    @property
    def init_inputs(self):
        """
        Returns the inputs of the kernel

        Returns
        -------
        _init_inputs : list
            The inputs of the kernel
        """

        return self._init_inputs
    
    @property
    def init_outputs(self):
        """
        Returns the outputs of the kernel

        Returns
        -------
        _init_outputs : list
            The outputs of the kernel
        """

        return self._init_outputs
    
    @init_input_labels.setter
    def init_input_labels(self,labels):
        """
        Sets the labels for the initialization inputs

        Parameters
        ----------
        labels : list
            The list of labels to be set
        """
        self._init_input_labels = labels

    @init_output_labels.setter
    def init_output_labels(self,labels):
        """
        Sets the labels for the initialization outputs

        Parameters
        ----------
        labels : list
            The list of labels to be set
        """
        self._init_output_labels = labels

    @inputs.setter
    def inputs(self,inputs):
        """
        Sets the inputs of the kernel

        Parameters
        ----------
        inputs : list
            The list of inputs to be set
        """
        self._inputs = inputs

        # update init_inputs so the length matches inputs
        if len(self._init_inputs) != len(self._inputs):
            if len(self._init_inputs) == 0:
                self._init_inputs = [None] * len(self._inputs)
            else:
                self._init_inputs.extend([None] * (len(self._inputs) - len(self._init_inputs)))
    
    @outputs.setter
    def outputs(self,outputs):
        """
        Sets the outputs of the kernel

        Parameters
        ----------
        outputs : list
            The list of outputs to be set
        """
        self._outputs = outputs

        # update init_outputs so the length matches outputs
        if len(self._init_outputs) != len(self._outputs):
            if len(self._init_outputs) == 0:
                self._init_outputs = [None] * len(self._outputs)
            else:
                self._init_outputs.extend([None] * (len(self._outputs) - len(self._init_outputs)))

    @init_inputs.setter
    def init_inputs(self,inputs):
        """
        Sets the initialization inputs of the kernel

        Parameters
        ----------
        inputs : list
            The list of inputs to be set
        """
        self._init_inputs = inputs

    @init_outputs.setter
    def init_outputs(self,outputs):
        """
        Sets the initialization outputs of the kernel

        Parameters
        ----------
        outputs : list
            The list of outputs to be set
        """
        self._init_outputs = outputs

    #@abstractmethod
    #def verify(self):
    #    """
    #    Generic verification abstract method to be defined by individual kernels.
    #    """
    #    pass
    
    #@abstractmethod
    #def execute(self):
    #    """
    #    Generic execution abstract method to be defined by individual kernels.
    #    """
    #    pass
    
    @abstractmethod
    def initialize(self):
        """
        Generic initialization abstract method to be defined by individual kernels.
        """
        pass

    ## INPUT and OUTPUT getter methods
    def get_input(self,index):
        """
        Returns the input at the specified index

        Parameters
        ----------
        index : int
            The index of the input to be returned

        Returns
        -------
        _inputs[index] : Object
            The input at the specified index
        """
        return self._inputs[index]
   
    def get_output(self,index):
        """
        Returns the output at the specified index

        Parameters
        ----------
        index : int
            The index of the output to be returned

        Returns
        -------
        _outputs[index] : Object
            The output at the specified index
        """
        return self._outputs[index]

    def get_init_input(self,index):
        """
        Returns the input at the specified index

        Parameters
        ----------
        index : int
            The index of the input to be returned

        Returns
        -------
        _init_inputs[index] : Object
            The input at the specified index
        """
        return self._init_inputs[index]

    def get_init_output(self,index):
        """
        Returns the output at the specified index

        Parameters
        ----------
        index : int
            The index of the output to be returned

        Returns
        -------
        _init_outputs[index] : Object
            The output at the specified index
        """
        return self._init_outputs[index]
    
    def copy_init_labels_to_output(self):
        """
        Copies the input labels from initialization to the output
        """
        labels = self.init_input_labels
        if labels.bcip_type != BcipEnums.TENSOR:
            labels = labels.to_tensor()

        if self.init_output_labels is None:
            self.init_output_labels = Tensor.create_virtual(self.session)

        labels.copy_to(self.init_output_labels)


    def verification(self):
        """
        Basic verification method that can be applied to all kernels.
        Generates phony inputs and attemps to execute the kernel.
        """

        # generate phony inputs
        phony_inputs = []
        for input in self.inputs:
            phony_input = input.copy()
            phony_input.assign_random_data()
            phony_inputs.append(phony_input)

        # generate phony outputs
        phony_outputs = []
        for output in self.outputs:
            phony_output = output.copy()
            phony_outputs.append(phony_output)

        # if the kernel requires initialization, generate phony init inputs
        if self.init_style == BcipEnums.INIT_REQUIRED:
            phony_init_inputs = []
            for init_input in self.init_inputs:
                phony_init_input = init_input.copy()
                phony_init_input.assign_random_data()
                phony_init_inputs.append(phony_init_input)

            # generate phony init outputs
            phony_init_outputs = []
            for init_output in self.init_outputs:
                phony_init_output = init_output.copy()
                phony_init_outputs.append(phony_init_output)

            # attempt initialization
            try:
                self._initialize(phony_init_inputs, phony_init_outputs)
            except Exception as e:
                raise type(e)(f"{str(e)}\nTest initialization of node {self.name} failed during verification. Please check parameters.").with_traceback(sys.exc_info()[2])

        # attempt kernel execution
        try:
            self._process_data(phony_inputs, phony_outputs)
        except Exception as e:
            raise type(e)(f"{str(e)}\nTest execution of node {self.name} failed during verification. Please check parameters.").with_traceback(sys.exc_info()[2])


    def initialize(self):
        """
        Initialize kernel
        """
        if hasattr(self,'_initialize'):
            self._initialized = False
            self._initialize(self.init_inputs, self.init_outputs)
            self._initialized = True

    def execute(self):
        """
        Execute kernel to process data
        """
        self._process_data(self.inputs, self.outputs)