"""
Created on Fri Jan 17 12:14:43 2020

@author: ivanovn
"""

from classes.kernel import Kernel
from classes.node import Node
from classes.parameter import Parameter
from classes.tensor import Tensor
from classes.scalar import Scalar
from classes.array import Array
from classes.bcip_enums import BcipEnums

from math import exp, log, sqrt
import numpy as np
from scipy import signal, linalg, dot

from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.distance import distance_riemann

# TODO - this may belong within a utils library as it will be used by more than
# one kernel
def _filter(filt,X):
    
    axis = next((i for i, ex in enumerate(X.shape) if ex != 1))

    if filt.implementation == 'ba':
        Y = signal.filtfilt(filt.coeffs['b'],
                            filt.coeffs['a'],
                            X,
                            axis=axis)
    else:
        Y = signal.sosfiltfilt(filt.coeffs['sos'],
                               X,
                               axis=axis)
    
    return Y

def _cov(X):
    return np.cov(X)

def _dist_mean(S, T, stats='arithmetic'):
    """
    Calculate the mean distance of covariance matrices within T and the 
    covariance matrix in S using arithmetic or geometric statistics.
    """
    
    
    if stats == 'arithmetic':
        mu = 1/T.shape[0] * sum([distance_riemann(S,T[i,:,:]) 
                                      for i in range(T.shape[0])])
    else:
        mu = exp(1/T.shape[0]*sum([log(distance_riemann(S,T[i,:,:]))
                                        for i in range(T.shape[0])]))

    return mu

def _dist_std(S, T, stats='arithmetic'):
    """
    Calculate the standard deviation of the distance of covariance matrices 
    within T and the covariance matrix in S
    using arithmetic or geometric statistics.
    """

    d = [distance_riemann(S,T[i,:,:]) for i in range(T.shape[0])]
    mu = _dist_mean(S,T,stats)

    if stats == 'arithmetic':
        sigma = sqrt(1/T.shape[0] * sum([(di-mu)**2 for di in d]))
    else:
        sigma = exp(sqrt(1/T.shape[0] * sum([log(di/mu)**2 for di in d])))
        
    return sigma

def _z_score(d,mu,sigma,stats='arithmetic'):
    if stats == 'arithmetic':
        return (d - mu) / sigma
    else:
        return log(d/mu) / log(sigma)
    

class RiemannPotatoKernel(Kernel):
    """
    Riemannian potato artifact detection detector
    """
    
    def __init__(self,graph,inA,out_label,out_score,thresh,update,alpha,
                 initialization_data,filt,
                 stats_type,init_stopping_crit,k):
        """
        Kernel takes Tensor input and produces scalar label representing
        the predicted class
        """
        super().__init__('RiemannPotato',BcipEnums.INIT_FROM_DATA,graph)
        self._inA  = inA
        self._out_label = out_label
        self._out_score = out_score

        self._labels = None
        
        self._thresh = thresh
        self._mean = None
        self._std  = None
        self._ref  = None
        self._q = None
        self.initialization_data = initialization_data
        
        self._filt = filt
        
        self._stats_type = stats_type
        
        if init_stopping_crit == 'iterative':
            self._init_stop_mode = 'iterative'
            self._k = k
        else:
            # do not apply multiple passes. Same as iterative with k=1
            self._init_stop_mode = 'iterative'
            self._k = 1
        
        if update == 'static':
            self._alpha = 0
            self._update = 'static'
        elif update == 'moving_avg':
            self._alpha = alpha
            self._update = 'moving_avg'
        else:
            self._update = 'cumulative'
            
        self._initialized = False
          
    
    def initialize(self):
        """
        Set reference covariance matrix, mean, and standard deviation
        """
        
        if (not isinstance(self.initialization_data,Tensor)) and \
        (not isinstance(self.initialization_data,Array)):
            return BcipEnums.INITIALIZATION_FAILURE
        
        if isinstance(self.initialization_data,Tensor):
            X = self.initialization_data.data
        else:
            X = [self.initialization_data.get_element(i).data 
                        for i in range(self.initialization_data.capacity)]
            X = np.stack(X,axis=0)
        
        if len(X.shape) != 3:
            return BcipEnums.INITIALIZATION_FAILURE
        
        
        XXt = np.zeros((X.shape[0],X.shape[1],X.shape[1]))
        
        for i in range(X.shape[0]):
            # filter the data
            Xfilt = _filter(self._filt,X[i,:,:])
        
            # calculate the covariance matrices
            Xcov = _cov(Xfilt)
            
            # regularize the cov mat
            r = 0.01
            XXt[i,:,:] = 1/(1+r)*(Xcov + r*np.eye(Xcov.shape[0]))

        
        for i in range(self._k):
            X_clean = []
            # compute the mean covariance matrix
            S  = mean_covariance(XXt)
            mu = _dist_mean(S,XXt,self._stats_type)
            sigma = _dist_std(S,XXt,self._stats_type)
                
            self._q = XXt
            
            for j in range(XXt.shape[0]):
                if _z_score(distance_riemann(XXt[j,:,:], S),
                                mu,sigma,self._stats_type) < self._thresh:
                    X_clean.append(XXt[j,:,:])
            
            if len(X_clean) == XXt.shape[0]:
                break
            
            XXt = np.stack(X_clean)
            
        self._ref = S
        self._mean = mu
        self._std = sigma
        
        self._initialized = True
        return BcipEnums.SUCCESS
        
    
    
    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized and typed
        """
        
        # first ensure the input is a tensor
        if not isinstance(self._inA,Tensor):
            return BcipEnums.INVALID_PARAMETERS
        
        input_rank = len(self._inA.shape)
        if input_rank > 3 or input_rank < 2:
            return BcipEnums.INVALID_PARAMETERS
        
        if input_rank == 3:
            output_sz = self._inA.shape[0]
        else:
            output_sz = 1
        
        
        for output in (self._out_label,self._out_score):
            if output == None:
                continue
            
            if not (isinstance(output,Tensor)
                or (isinstance(output,Scalar))):
            
                return BcipEnums.INVALID_PARAMETERS
            
            if isinstance(output,Tensor):
                
                if output.virtual and len(output.shape) == 0:
                    output.shape = (output_sz,1)
            
                if output.shape != (output_sz,1):
                    return BcipEnums.INVALID_PARAMETERS
            
            elif (isinstance(output,Scalar)
                  and (output_sz > 1 or
                  not (output.data_type in Scalar.valid_numeric_types()))):
                return BcipEnums.INVALID_PARAMETERS
            
        
        # do not support filtering directly with zpk filter repesentation
        if self._filt.implementation == 'zpk':
            return BcipEnums.NOT_SUPPORTED
        
        
        return BcipEnums.SUCCESS
        
    def execute(self):
        """
        Execute the kernel by classifying the input trials
        """
        if not self._initialized:
            return BcipEnums.EXE_FAILURE_UNINITIALIZED
        
        if len(self._inA.shape) == 2:
            in_data = np.expand_dims(self._inA.data,axis=0)
        else:
            in_data = self._inA.data
        
        scores = []
        labels = []
        for i in range(in_data.shape[0]):
            X = _filter(self._filt,in_data[i,:,:])
            X = _cov(X)
            
            # regularize
            r = 0.01
            X = 1/(1+r) * (X + r*np.eye(X.shape[0]))
            
            dt = distance_riemann(self._ref,X)
            scores.append(_z_score(dt,self._mean,self._std,self._stats_type))
            
            if scores[-1] > self._thresh:
                labels.append(1) # artifact
            else:
                labels.append(0)
                
                # update the potato
                if self._update != 'static':
                    inv_sqrt_ref = linalg.fractional_matrix_power(self._ref,-1/2)
                    
                    self._q = np.stack((X,self._q))
                    mean_cov = mean_covariance(self._q)
                    
                    A = dot(inv_sqrt_ref,dot(mean_cov,inv_sqrt_ref))
                    
                    if self._update == 'moving_avg':
                        alpha = self._alpha
                    else:
                        alpha = 1/self._q.shape[0]
                        
                    A = linalg.fractional_matrix_power(A,alpha)
                    self._ref = dot(inv_sqrt_ref,dot(A,inv_sqrt_ref))
                    
                    if self._stats_type == 'arithmetic':
                        self._mean = (1-alpha)*self._mean + alpha*dt
                        self._std = sqrt((1-alpha)*(self._std**2) + 
                                         alpha*((dt - self._mean)**2))
                    else:
                        self._mean = exp((1-alpha)*log(self._mean) + 
                                         alpha*log(dt))
                        self._std = exp(sqrt((1-alpha)*(log(self._std)**2) + 
                                             alpha*(log(dt/self._mean)**2)))
        
        
        if isinstance(self._out_label,Tensor):
            data = np.asarray(labels)
            self._out_label.data = data
        elif isinstance(self._out_label,Scalar):
            self._out_label.data = labels[0]
        
        if isinstance(self._out_score,Tensor):
            data = np.expand_dims(np.asarray(scores),axis=1)
            self._out_score.data = data
        elif isinstance(self._out_score,Scalar):
            self._out_score.data = scores[0]
                    
            
        return BcipEnums.SUCCESS
    
    
    @classmethod
    def add_riemann_potato_node(cls,graph,inA,filt,initialization_data,
                                out_labels=None,out_scores=None,
                                thresh=2.5,update='static',alpha=0.1,
                                stats_type='geometric',
                                init_stopping_crit='iterative',k=3):
        """
        Factory method to create a riemann potato artifact detector
        """
        
        # create the kernel object            

        k = cls(graph,inA,out_labels,out_scores,thresh,update,alpha,
                initialization_data,filt,stats_type,
                init_stopping_crit,k)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,BcipEnums.INPUT),)
        
        if out_labels != None:
            params += (Parameter(out_labels,BcipEnums.OUTPUT),)
        
        if out_scores != None:
            params += (Parameter(out_scores,BcipEnums.OUTPUT),)
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node

