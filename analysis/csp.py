# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 10:42:59 2020

@author: Nick
"""

from itertools import combinations as iter_combs
from scipy.special import binom
import numpy as np



class CSP:
    """
    Vanilla CSP - no feature optimization, single band
    """
    
    def __init__(self,eval_type,
                 classes,m=2,multi_class_ext='OVR',win_sz=None,
                 win_type='sliding',step_sz=None,decay=0.8):
        
        self.eval_type = eval_type
        self.classes = classes
        self.win_sz = win_sz
        self.win_type = win_type
        self.step_sz = step_sz
        self.decay = decay
        self.multi_class_ext = multi_class_ext
        self.m = m
    
    
    def extract_feats(self,train_set,test_set=None):
        """
         using provided data

        """
        if self.eval_type == 'static' and test_set == None:
            raise("Static analysis requires a test set param")
        
        if self.eval_type == 'dynamic' and test_set != None:
            raise("Dynamic analysis requires no test set")
        
        
        # create classifier
        if self.eval_type == 'static':
            Xtr, ytr = train_set
            Xte, yte = test_set

            # remove frequency dim
            Xtr = np.squeeze(Xtr,axis=1)
            Xte = np.squeeze(Xte,axis=1)
            
            # Calculate spatial filters and extract features
            W = self._calc_csp_filters(Xtr,ytr)
            
            # filter
            Xtr_filt = self._apply_csp_filt(W,Xtr)
            Xte_filt = self._apply_csp_filt(W,Xtr)
            
            # log-var feats
            Xtr_feats = self._ext_log_var(Xtr_filt)
            Xte_feats = self._ext_log_var(Xte_filt)
            
            return ((Xtr_feats,ytr), (Xte_feats,yte))
                
            
        elif self.eval_type == 'dynamic':
            # split the train set into train and test
            # with a sliding window to simulate co-adaptive online session
            
            X = train_set[0]
            y = train_set[1]
            
            X = np.squeeze(X,axis=1)
            
            data = [None] * self.classes
            labels = np.unique(y)
            Nl = labels.shape[0]
        
            if Nl != self.classes:
                raise("Too many labels")
        
            for i_l in range(Nl):
                label = labels[i_l]
                data[i_l] = X[y==label,:,:]
        
            # calculate the maximum number of trials available for each class
            Nt = min([x.shape[0] for x in data])
        
            if self.win_sz == None:
                win_sz = Nt - self.step_sz
            else:
                win_sz = self.win_sz

            if Nt < (win_sz + self.step_sz):
                # not enough trials
                return (X,y)
        
            clsf_blocks = (Nt - win_sz - self.step_sz) // self.step_sz + 1


            _,Ns,Nc = X.shape
            if self.multi_class_ext == 'PW':
                Nfilts = binom(self.classes)
            else:
                Nfilts = self.classes
                
            X_feats = np.zeros((clsf_blocks,Nt,Nfilts,2*self.m))

        
            for i_s in range(clsf_blocks):
                # update training and test set for this block
                if self.win_type == 'sliding':
                    # extract data
                    train_sz = win_sz
                
                    Xtr = np.zeros((self.classes*win_sz,Ns,Nc))
                    Xte = np.zeros((self.classes*self.step_sz,Ns,Nc))
                
                    ytr = -1 * np.ones((self.classes*win_sz,))
                    yte = -1 * np.ones((self.classes*self.step_sz,))
                
                    train_start = i_s * self.step_sz
                    train_stop = train_start + win_sz
                
                    test_start = train_stop

                
                elif self.win_type == 'expanding':
                
                    train_sz = win_sz + i_s*self.step_sz
                
                    Xtr = np.zeros((self.classes*train_sz,Ns,Nc))
                
                    ytr = -1 * np.ones((self.classes*train_sz,))
                
                    train_start = 0
                    train_stop = win_sz + i_s*self.step_sz
                
                else:
                    raise("Invalid window type")
                
                for i_c in range(self.classes):
                    
                    l = labels[i_c]
                    
                    # training data
                    Ctr = data[i_c][train_start:train_stop,:,:]
                    Xtr[i_c*train_sz:(i_c+1)*train_sz,:,:] = Ctr
                    
                    ytr[i_c*train_sz:(i_c+1)*train_sz] = l * np.ones((train_sz,))
                    
                
                # Calculate spatial filters and extract features
                W_step = self._calc_csp_filters(Xtr,ytr)
            
                # filter
                X_filt_step = self._apply_csp_filt(W_step,X)
            
                # log-var feats
                X_feats[i_s,:,:,:] = self._ext_log_var(X_filt_step)
                        
            return (X_feats,ytr)

        else:
            raise("invalid evaluation type")
    
    
    
    def _calc_csp_filters(self,X,y):
        
        if self.classes == 2:
            return self._calc_binary_csp_filters(X,y)
        
        if self.multi_class_ext == 'OVR':
            _, Ns, Nc = X.shape
            labels = np.unique(y)
            Nl = labels.shape[0]
            W = np.zeros((Nl,Nc,2*self.m))
            
            for i_l in range(Nl):
                l = labels[i_l]
                yl = np.copy(y)
                yl[y==l] = 1
                yl[y!=l] = 0
                
                W[i_l,:,:] = self._calc_binary_csp_filters(X,yl)
            
            return W
                
        elif self.multi_class_ext == 'PW':
            _, Ns, Nc = X.shape
            labels = np.unique(y)
            Nl = labels.shape[0]
            
            Nf = binom(Nl,2)
            
            W = np.zeros((Nf,Nc,2*self.m))
            
            i = 0
            for (l1,l2) in iter_combs(labels,2):
                Xl1 = X[y==l1,:,:]
                Xl2 = X[y==l2,:,:]
                yl = np.concatenate((l1 * np.ones(Xl1.shape[0],),
                                     l2 * np.ones(Xl2.shape[0])),
                                    axis=0)
                Xl = np.concatenate((Xl1,Xl2),
                                    axis=0)                
                
                W[i,:,:] = self._calc_binary_csp_filters(Xl,yl)
                i += 1
            
            return W
        
        else:
            raise("Invalid multi-class extention")
        
    
    def _calc_binary_csp_filters(self,X,y):
        _, Ns, Nc = X.shape
        
        labels = np.unique(y)
        Nl = labels.shape[0]
        
        if Nl != 2:
            raise("invalid number of labels")
        
        # calc mean cov mats for each class
        C = np.zeros((2,Nc,Nc))
        for i_l in range(2):
            l = labels[i_l]
            X_l = X[y==l]
            Nt = X_l.shape[0]
            X_l = np.transpose(X_l,(2,1,0))
            X_l = np.reshape(X_l,(Nc,Ns*Nt))
            C[i_l,:,:] = np.cov(X_l)
                
        d, V = np.linalg.eig(np.mean(C,axis=0))
    
        ix = np.flip(np.argsort(d))
        d = d[ix]
        V = V[:,ix]
            
        M = np.matmul(V,np.diag(d ** (-1/2))) # whitening matrix

        dC = C[0,:,:] - C[1,:,:]
        S = np.matmul(M.T,np.matmul(dC,M))
        d, W = np.linalg.eig(S)
        W = np.matmul(M,W)
            
        ix = np.flip(np.argsort(d))
        d = d[ix]
        W = W[:,ix]
    
        W = np.concatenate((W[:,:self.m],W[:,-self.m:]),axis=1)
    
        return W
    
    def _apply_csp_filts(self,W,X):
        Nt,Ns,Ncx = X.shape
        
        if len(W.shape) == 2:
            Ncw, Nf = W.shape
            W = np.expand_dims(W,axis=0)
            X_filt = np.zeros((Nt,1,Ns,Ncx))
        else:
            Ncl, Ncw, Nf = W.shape
            X_filt = np.zeros((Nt,Ncl,Ns,Ncx))
            
        if Ncw != Ncx:
            raise("Channel mismatch")
        
        for i_t in range(Nt):
            X_filt[i_t,:,:,:] = np.matmul(X[i_t,:,:],W)
            
        return X_filt
    
    def _ext_log_var(self,X):
        X_var = np.var(X,axis=2)
        X_log_var = np.log(X_var)
        
        return X_log_var