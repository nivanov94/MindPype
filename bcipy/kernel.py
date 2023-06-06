from .core import BCIP, BcipEnums
from abc import ABC, abstractmethod

class Kernel(BCIP, ABC):
    """
    An abstract base class that defines the minimum set of kernel methods that
    must be defined

    Parameters
    ----------

    name : str
        - Name of the kernel
    init_style : BcipEnums Object
        - Kernel initialization style, according to BcipEnum class

    Attributes
    ----------
    
    _name : str
        - Name of the kernel
    _init_style : BcipEnums Object
        - Kernel initialization style, according to BcipEnum class
    """
    
    def __init__(self,name,init_style,graph):
        session = graph.session
        super().__init__(BcipEnums.KERNEL,session)
        self._name = name
        self._init_style = init_style
        
    # API Getters
    @property
    def name(self):
        """
        Returns the name of the kernel

        Returns
        -------
        _name : str
        """

        return self._name
    
    @property
    def init_style(self):
        """
        Returns the initialization style of the kernel

        Returns
        -------
        _init_style : BcipEnums Object
        """

        return self._init_style
    
    @abstractmethod
    def verify(self):
        """
        Generic verification abstract method to be defined by individual kernels.
        """
        pass
    
    @abstractmethod
    def execute(self):
        """
        Generic execution abstract method to be defined by individual kernels.
        """
        pass
    
    @abstractmethod
    def initialize(self):
        """
        Generic initialization abstract method to be defined by individual kernels.
        """
        pass
