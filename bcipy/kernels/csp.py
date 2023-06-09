from ..core import BcipEnums
from ..kernel import Kernel
from ..graph import Node, Parameter
from .kernel_utils import extract_nested_data
from ..containers import Tensor

import numpy as np
import pyriemann
from scipy.linalg import eigh
from scipy.special import binom
from itertools import combinations as iter_combs


class CommonSpatialPatternKernel(Kernel):
    """
    CSP Filter Kernel that applies a set of common spatial patter filters to tensors of covariance matrices

    Parameters
    ----------
    graph : Graph
        Graph that the kernel should be added to
    inA : Tensor or Scalar
        First input trial data
    outA : Tensor or Scalar 
        Output trial data

    """
    
    def __init__(self,graph,inA,outA,
                 init_style,init_params,
                 num_filts,Ncls,multi_class_mode):
        """
        Constructor for CSP filter kernel
        """
        super().__init__('CSP',init_style,graph)
        self._inA = inA
        self._outA = outA     
        
        self._num_filts = num_filts
        self._init_params = init_params
        self.multi_class_mode = multi_class_mode
        self.num_classes = Ncls

        if 'initialization_data' in init_params:
            self._init_inA = init_params['initialization_data']
        else:
            self._init_inA = None

        if 'labels' in init_params:
            self._init_labels_in = init_params['labels']
        else:
            self._init_labels_in = None

        self._init_outA = None
        self._init_labels_out = None
        
        if init_style == BcipEnums.INIT_FROM_DATA:
            # model will be trained using data in tensor object at later time
            self._initialized = False
            self._W = None
            
        elif init_style == BcipEnums.INIT_FROM_COPY:
            # model is copy of predefined MDM model object
            self._W = init_params['filters']
            self._initialized = True
    

    def initialize(self):
        """
        Set the filter values
        """
        sts = BcipEnums.SUCCESS
        
        if self.init_style == BcipEnums.INIT_FROM_DATA:
            self._initialized = False # clear initialized flag
            
            if ((self._init_inA._bcip_type != BcipEnums.TENSOR and
                 self._init_inA._bcip_type != BcipEnums.ARRAY  and
                 self._init_inA._bcip_type != BcipEnums.CIRCLE_BUFFER) or
                (self._init_labels_in._bcip_type != BcipEnums.TENSOR and
                 self._init_labels_in._bcip_type != BcipEnums.ARRAY  and
                 self._init_labels_in._bcip_type != BcipEnums.CIRCLE_BUFFER)):
                return BcipEnums.INITIALIZATION_FAILURE
        
        
            if self._init_inA._bcip_type == BcipEnums.TENSOR: 
                X = self._init_inA.data
            else:
                try:
                    # extract the data from a potentially nested array of tensors
                    X = extract_nested_data(self._init_inA)
                except:
                    return BcipEnums.INITIALIZATION_FAILURE    
        
            if self._init_labels_in._bcip_type == BcipEnums.TENSOR:    
                y = self._init_labels_in.data
            else:
                try:
                    y = extract_nested_data(self._init_labels_in)
                except:
                    return BcipEnums.INITIALIZATION_FAILURE
            
            sts = self._compute_filters(X,y)
        
        # compute init output
        if sts == BcipEnums.SUCCESS and self._init_outA != None:
            if self._init_inA._bcip_type != BcipEnums.TENSOR:
                init_input_tensor = Tensor.create_from_data(self.session, X.shape, X)
            else:
                init_input_tensor = self._init_inA
                
            # adjust the shape of init output tensor
            if len(init_input_tensor.shape) == 3:
                self._init_outA.shape = (init_input_tensor.shape[0], self._W.shape[1], init_input_tensor.shape[2])
 
            sts = self._process_data(init_input_tensor, self._init_outA)

            # pass on the labels
            if self._init_labels_in._bcip_type != BcipEnums.TENSOR:
                input_labels = self._init_labels_in.to_tensor()
            else:
                input_labels = self._init_labels_in
            input_labels.copy_to(self._init_labels_out)

        if sts == BcipEnums.SUCCESS:
            self._initialized = True
        
        return sts
    

    def _process_data(self, input_data, output_data):
        """
        Process input data according to outlined kernel function
        """

        try:
            output_data.data = np.matmul(self._W.T, input_data.data) 
            return BcipEnums.SUCCESS
        except:
            return BcipEnums.EXE_FAILURE


    def _compute_filters(self,X,y):
        """
        Compute CSP filters
        """

        # ensure the shapes are valid
        if len(X.shape) == 2:
            X = X[np.newaxis, :, :]

        if len(y.shape) == 2:
            y = np.squeeze(y)

        if len(X.shape) != 3 or len(y.shape) != 1:
            return BcipEnums.INITIALIZATION_FAILURE
        
        if X.shape[0] != y.shape[0]:
            return BcipEnums.INITIALIZATION_FAILURE

        unique_labels = np.unique(y)
        Nl = unique_labels.shape[0]
        
        if Nl != self.num_classes:
            return BcipEnums.INITIALIZATION_FAILURE
        
        if Nl == 2:
            self._W = self._compute_binary_filters(X,y)

        else:
            if self.multi_class_mode not in ('OVA', 'PW'):
                return BcipEnums.INITIALIZATION_FAILURE

            _, Nc, Ns = X.shape

            if self.multi_class_mode == 'OVA':
                # one vs. all
                self._W = np.zeros((Nc,Nl*self._num_filts))

                for il, l in enumerate(unique_labels):
                    yl = np.copy(y)
                    yl[y==l] = 1 # target
                    yl[y!=l] = 0 # non-target
                    self._W[:, il*self._num_filts:(il+1)*self._num_filts] = self._compute_binary_filters(X,yl)

            else:
                # pairwise 
                Nf = int(binom(Nl,2)) # number of pairs
                self._W = np.zeros((Nc, Nf*self._num_filts))

                for il, (l1,l2) in enumerate(iter_combs(unique_labels,2)):
                    # get trials from each label
                    Xl1 = X[y==l1,:,:]
                    Xl2 = X[y==l2,:,:]

                    # create feature and label matrices using the current label pair
                    yl = np.concatenate((l1 * np.ones(Xl1.shape[0],),
                                         l2 * np.ones(Xl2.shape[0])),
                                        axis=0)
                    Xl = np.concatenate((Xl1,Xl2),
                                        axis=0)

                    self._W[:, il*self._num_filts:(il+1)*self._num_filts] = self.compute_binary_filters(Xl, yl)
    
        return BcipEnums.SUCCESS

    def _compute_binary_filters(self, X, y):
        """
        Compute binary CSP filters
        """
        _ , Nc, Ns = X.shape

        # start by calculating the mean covariance matrix for each class
        C = pyriemann.utils.covariance.covariances(X)

        # remove any trials that are not positive definite
        pd = np.asarray([np.all(np.linalg.eigvals(Ci)) for Ci in C])
        C = C[pd==1]
        y = y[pd==1]

        C_bar = np.zeros((2, Nc, Nc))
        labels = np.unique(y)
        for i, label in enumerate(labels):
            C_bar[i,:,:] = np.mean(C[y==label,:,:], axis=0)

        C_total = np.sum(C_bar, axis = 0)

        # get the whitening matrix
        d, U = np.linalg.eig(C_total)
        
        # filter any eigenvalues close to zero
        d[np.isclose(d, 0)] = 0
        U = U[:,d!=0]
        d = d[d!=0]
        
        # construct the whitening matrix
        P = np.matmul(np.diag(d ** (-1/2)), U.T) 

        C_tot_white = np.matmul(P,np.matmul(C_total,P.T))

         # apply the whitening transform to both class covariance matrices
        C1_bar_white = np.matmul(P,np.matmul(C_bar[0,:,:],P.T))

        l, V = eigh(C1_bar_white, C_tot_white)

        # sort the eigenvectors in order of eigenvalues
        ix = np.flip(np.argsort(l)) 
        V = V[:,ix]
        
        # extract the specified number of filters
        m = self._num_filts // 2
        W = np.concatenate((V[:,:m], V[:,-m:]), axis=1)

        # rotate the filters back into the channel space
        W = np.matmul(P.T,W)
        
        return W
    
    
    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized and typed
        """
        
        # first ensure the input and output are tensors
        if (self._inA._bcip_type != BcipEnums.TENSOR or 
            self._outA._bcip_type != BcipEnums.TENSOR):
            return BcipEnums.INVALID_PARAMETERS
        
        # input tensor should be two- or three-dimensional
        if len(self._inA.shape) != 2 and len(self._inA.shape) != 3:
            return BcipEnums.INVALID_PARAMETERS
        
        if self.num_classes < 2:
            return BcipEnums.INVALID_PARAMETERS
        
        if (self.num_classes > 2 and 
            self.multi_class_mode not in ('OVA', 'PW')):
            return BcipEnums.INVALID_PARAMETERS

        # if the output is a virtual tensor and dimensionless, 
        # add the dimensions now
        if self.num_classes == 2:
            filt_multiplier = 1
        else:
            if self.multi_class_mode == 'OVA':
                filt_multiplier = self.num_classes
            else:
                filt_multiplier = int(binom(self.num_classes,2))
                
        if len(self._inA.shape) == 2:
            out_sz = (self._num_filts*filt_multiplier,self._inA.shape[1])
        else:
            out_sz =  (self._inA.shape[0], self._num_filts*filt_multiplier, self._inA.shape[2])
        
        if self._outA.virtual and len(self._outA.shape) == 0:
            self._outA.shape = out_sz

        if self._outA.shape != out_sz:
            return BcipEnums.INVALID_PARAMETERS

        return BcipEnums.SUCCESS
        
    def execute(self):
        """
        Execute the kernel by classifying the input trials
        """
        return self._process_data(self._inA, self._outA)
    
    @classmethod
    def add_uninitialized_CSP_node(cls,graph,inA,outA,
                                   initialization_data,labels,
                                   num_filts,Ncls=2,multi_class_mode='OVA'):
        """
        Factory method to create a CSP filter node and add it to a graph
        
        Note that the node will have to be initialized prior 
        to execution of the kernel.

        Parameters
        ----------

        graph : Graph 
            Graph that the kernel should be added to

        inA : Tensor or Scalar 
            First input trial dat

        outA : Tensor or Scalar 
            Output trial data
        
        initialization_data : Tensor
            Initialization data to configure the filters (n_trials, n_channels, n_samples)
    
        labels : Tensor
            Labels corresponding to initialization data class labels (n_trials, )

        num_filts : int
            Number of spatial filters to apply to trial data.        
        
        """
        
        # create the kernel object            
        init_params = {'initialization_data' : initialization_data, 
                       'labels'              : labels}
        
        k = cls(graph,inA,outA,BcipEnums.INIT_FROM_DATA,init_params,
                num_filts,Ncls,multi_class_mode)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,BcipEnums.INPUT),
                  Parameter(outA,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node
    
    
    @classmethod
    def add_initialized_CSP_node(cls,graph,inA,outA,filters):
        """
        Factory method to create a pre-initialized CSP filter node

        Parameters
        ----------
        
        graph : Graph 
            Graph that the kernel should be added to

        inA : Tensor or Scalar 
            First input trial dat

        outA : Tensor or Scalar 
            Output trial data
        
        filters : Tensor 
            Tensor containing precalculated spatial filters to be applied to input trial data      
        """
        
        # create the kernel object
        init_params = {'filters' : filters}
        k = cls(graph,inA,outA,BcipEnums.INIT_FROM_COPY,init_params,filters.shape[1])
        
        # create parameter objects for the input and output
        params = (Parameter(inA,BcipEnums.INPUT),
                  Parameter(outA,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node
