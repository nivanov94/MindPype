# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 10:46:02 2020

@author: Nick
"""

from pyriemann.classification import MDM as pyriemMDM
from pyriemann.classification import FgMDM as pyriemFgMDM
from pyriemann.classification import TSclassifier as pyriemTSC

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as skLDA

import numpy as np

def _generate_confusion_mat(y,y_bar):
    labels = np.unique(y)
    Nl = labels.shape[0]
    
    confuse_mat = np.zeros((Nl,Nl))
    for i in range(Nl):
        true_label = labels[i]
        for j in range(Nl):
            pred_label = labels[j]
            confuse_mat[i,j] = np.sum( y_bar[y == true_label] == pred_label )
    
    return confuse_mat


class ClassifierEval():
    """
    Base class for classifer assessment
    """
    
    def __init__(self,eval_type,classes,win_sz=None,
                 win_type='sliding',step_sz=None,decay=0.8):
        """
        Create classifier eval object
        """
        self.eval_type = eval_type
        self.classes = classes
        self.win_sz = win_sz
        self.win_type = win_type
        self.step_sz = step_sz
        self.decay = decay
        
        
        
    def static_eval(self,clsf,Xtr,ytr,Xte,yte):
        
        ytr_bar = clsf.fit_predict(Xtr,ytr)
        train_res = _generate_confusion_mat(ytr, ytr_bar)
        
        yte_bar = clsf.predict(Xte)
        test_res = _generate_confusion_mat(yte,yte_bar)
        
        results = {'Train'  : train_res,
                   'Test'   : test_res}
        
        return results
    
    def dynamic_eval(self,clsf,X,y):
        
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
            return {'Train' : np.zeros((2,1)),
                    'Test'  : np.zeros((2,1))}
        
        clsf_blocks = (Nt - win_sz - self.step_sz) // self.step_sz + 1

        results = {'Train' : np.zeros((clsf_blocks,Nl,Nl)),
                   'Test'  : np.zeros((clsf_blocks,Nl,Nl))}

        Nc = X.shape[-1]
        
        for i_s in range(clsf_blocks):
            # update training and test set for this block
            if self.win_type == 'sliding':
                # extract data
                train_sz = win_sz
                
                Xtr = np.zeros((self.classes*win_sz,Nc,Nc))
                Xte = np.zeros((self.classes*self.step_sz,Nc,Nc))
                
                ytr = -1 * np.ones((self.classes*win_sz,))
                yte = -1 * np.ones((self.classes*self.step_sz,))
                
                train_start = i_s * self.step_sz
                train_stop = train_start + win_sz
                
                test_start = train_stop
                test_stop  = test_start + self.step_sz
                
            elif self.win_type == 'expanding':
                Nc = X.shape[-1]
                
                train_sz = win_sz + i_s*self.step_sz
                
                Xtr = np.zeros((self.classes*train_sz,Nc,Nc))
                Xte = np.zeros((self.classes*self.step_sz,Nc,Nc))
                
                ytr = -1 * np.ones((self.classes*train_sz,))
                yte = -1 * np.ones((self.classes*self.step_sz,))
                
                train_start = 0
                train_stop = win_sz + i_s*self.step_sz
                
                test_start = train_stop
                test_stop  = test_start + self.step_sz
                
            else:
                raise("Invalid window type")
                
            for i_c in range(self.classes):
                    
                l = labels[i_c]
                    
                # training data
                Ctr = np.squeeze(data[i_c][train_start:train_stop,:,:])
                Xtr[i_c*train_sz:(i_c+1)*train_sz,:,:] = Ctr
                    
                ytr[i_c*train_sz:(i_c+1)*train_sz] = l * np.ones((train_sz,))
                    
                # testing data
                Cte = np.squeeze(data[i_c][test_start:test_stop,:,:])
                Xte[i_c*self.step_sz:(i_c+1)*self.step_sz,:,:] = Cte
                    
                yte[i_c*self.step_sz:(i_c+1)*self.step_sz] = l * np.ones((self.step_sz,))
                
            # train
            ytr_bar = clsf.fit_predict(Xtr, ytr)
            results['Train'][i_s,:,:] = _generate_confusion_mat(ytr, ytr_bar)
                
            # test
            yte_bar = clsf.predict(Xte)
            results['Test'][i_s,:,:] = _generate_confusion_mat(yte,yte_bar)
                
        return results
    
class MDM(ClassifierEval):
    """
    Class for evaluating MDM classifiers
    """
    
    def __init__(self,eval_type,classes,win_sz=None,
                 win_type='sliding',step_sz=None,decay=0.8):
        super().__init__(eval_type,classes,win_sz,
                         win_type,step_sz,decay)
    
    def evaluate(self,train_set,test_set=None):
        """
        Evaluate classifier using provided data

        """
        if self.eval_type == 'static' and test_set == None:
            raise("Static analysis requires a test set param")
        
        if self.eval_type == 'dynamic' and test_set != None:
            raise("Dynamic analysis requires no test set")
        
        
        clsf = pyriemMDM()

        if self.eval_type == 'static':
            Xtr, ytr = train_set
            Xte, yte = test_set

            # remove frequency dim
            Xtr = np.squeeze(Xtr,axis=1)
            Xte = np.squeeze(Xte,axis=1)

            results = self.static_eval(clsf,Xtr,ytr,Xte,yte)
            
        elif self.eval_type == 'dynamic':
            # split the train set into train and test
            # with a sliding window to simulate co-adaptive online session
            
            X = train_set[0]
            y = train_set[1]
            
            # remove frequency dim
            X = np.squeeze(X,axis=1)
                        
            results = self.dynamic_eval(clsf,X,y)

        else:
            raise("invalid evaluation type")
            
        return results


class TangentSpace(ClassifierEval):
    """
    Riemann tangent space classifier
    """
    
    def __init__(self,eval_type,classes,clf=LogisticRegression(),
                 win_sz=None,win_type='sliding',
                 step_sz=None,decay=0.8):
        super().__init__(eval_type,classes,win_sz,
                         win_type,step_sz,decay)
        
        self.clf = clf
    
    def evaluate(self,train_set,test_set=None):
        """
        Evaluate classifier using provided data

        """
        if self.eval_type == 'static' and test_set == None:
            raise("Static analysis requires a test set param")
        
        if self.eval_type == 'dynamic' and test_set != None:
            raise("Dynamic analysis requires no test set")
        
        
        clsf = pyriemTSC(clf=self.clf)

        if self.eval_type == 'static':
            Xtr, ytr = train_set
            Xte, yte = test_set

            # remove frequency dim
            Xtr = np.squeeze(Xtr,axis=1)
            Xte = np.squeeze(Xte,axis=1)

            results = self.static_eval(clsf,Xtr,ytr,Xte,yte)
            
        elif self.eval_type == 'dynamic':
            # split the train set into train and test
            # with a sliding window to simulate co-adaptive online session
            
            X = train_set[0]
            y = train_set[1]
            
            # remove frequency dim
            X = np.squeeze(X,axis=1)
                        
            results = self.dynamic_eval(clsf,X,y)

        else:
            raise("invalid evaluation type")
            
        return results

class FgMDM(ClassifierEval):
    """
    Riemann tangent space classifier
    """
    
    def __init__(self,eval_type,
                 classes,win_sz=None,
                 win_type='sliding',step_sz=None,decay=0.8):
        super().__init__(eval_type,classes,win_sz,
                         win_type,step_sz,decay)
        
    
    def evaluate(self,train_set,test_set=None):
        """
        Evaluate classifier using provided data

        """
        if self.eval_type == 'static' and test_set == None:
            raise("Static analysis requires a test set param")
        
        if self.eval_type == 'dynamic' and test_set != None:
            raise("Dynamic analysis requires no test set")
        
        
        clsf = pyriemFgMDM()

        if self.eval_type == 'static':
            Xtr, ytr = train_set
            Xte, yte = test_set

            # remove frequency dim
            Xtr = np.squeeze(Xtr,axis=1)
            Xte = np.squeeze(Xte,axis=1)

            results = self.static_eval(clsf,Xtr,ytr,Xte,yte)
            
        elif self.eval_type == 'dynamic':
            # split the train set into train and test
            # with a sliding window to simulate co-adaptive online session
            
            X = train_set[0]
            y = train_set[1]
            
            # remove frequency dim
            X = np.squeeze(X,axis=1)
                        
            results = self.dynamic_eval(clsf,X,y)

        else:
            raise("invalid evaluation type")
            
        return results


class rLDA(ClassifierEval):
    
    def __init__(self,eval_type,classes,win_sz=None,
                 win_type='sliding',step_sz=None,decay=0.8):
        super().__init__(eval_type,classes,win_sz=None,
                 win_type='sliding',step_sz=None,decay=0.8)
        
        
    def evaluate(self,train_set,test_set=None):
        
        if self.eval_type == 'static' and test_set == None:
            raise("Static analysis requires a test set param")
        
        if self.eval_type == 'dynamic' and test_set != None:
            raise("Dynamic analysis requires no test set")
        
        
        clsf = skLDA(solver='lsqr',shrinkage='auto')
        
        if self.eval_type == 'static':
            Xtr, ytr = train_set
            Xte, yte = test_set


            results = self.static_eval(clsf,Xtr,ytr,Xte,yte)
            
        elif self.eval_type == 'dynamic':
            # split the train set into train and test
            # with a sliding window to simulate co-adaptive online session
            
            X = train_set[0]
            y = train_set[1]
                        
            results = self.dynamic_eval(clsf,X,y)

        else:
            raise("invalid evaluation type")
            
        return results