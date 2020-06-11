# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 10:48:25 2020

@author: Nick
"""

from .clsf_eval import ClassifierEval, _generate_confusion_mat
from scipy.special import binom
from scipy.stats import mode
from itertools import combinations as iter_combs
import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as skLDA

class EnsembleClassifierEval(ClassifierEval):
    
    def __init__(self,eval_type,ext,classes,win_sz=None,
                 win_type='sliding',step_sz=None,decay=0.8):
        super().__init__(eval_type,classes,win_type,step_sz,decay)
        self.ext = ext
    
    def static_eval(self,clsf,Xtr,ytr,Xte,yte):
        
        labels = np.unique(ytr)
        Nl = labels.shape[0]
        
        if self.ext == 'PW':
            Nclf = binom(Nl)
            ytr_bar = -1 * np.ones((Nclf,Xtr.shape[0]))
            yte_bar = -1 * np.ones((Nclf,Xte.shape[0]))
            
            # train/test all the binary classifiers
            i_c = 0
            for (l1,l2) in iter_combs(labels,2):
                # train
                Xl1 = Xtr[ytr == l1,i_c,:]
                Xl2 = Xtr[ytr == l2,i_c,:]
                
                X = np.concatenate((Xl1,Xl2),
                                   axis=0)
                y = np.concatenate((l1 * np.ones((Xl1.shape[0],)),
                                    l2 * np.ones((Xl2.shape[0],))),
                                   axis=0)
                
                clsf.fit(X,y)
                
                # eval train set
                ytr_bar[i_c,:] = clsf.predict(Xtr[:,i_c,:])
                
                # eval test set
                yte_bar[i_c,:] = clsf.predict(Xte[:,i_c,:])
                
                i_c += 1
            
            # determine predicted class from  mode of all classifiers
            ytr_bar, _ = mode(ytr_bar,axis=0)
            yte_bar, _ = mode(yte_bar,axis=0)
            
            train_res = _generate_confusion_mat(ytr, ytr_bar)
            test_res = _generate_confusion_mat(yte, yte_bar)

        elif self.ext == 'OVR':
            labels = np.unique(ytr)
            Nl = labels.shape[0]
            
            ytr_bar = -1 * np.ones((Nl,Xtr.shape[0]))
            yte_bar = -1 * np.ones((Nl,Xte.shape[0]))
            
            # train/test all the binary classifiers
            for i_c in range(Nl):
                l1 = labels[i_c]
                # train
                ytr_l = np.zeros((ytr.shape[0],))
                ytr_l[y == l1] = 1
                ytr_l[y != l1] = 0
                
                clsf.fit(Xtr,ytr_l)
                
                # eval train set
                ytr_bar[i_c,:] = clsf.predict_proba(Xtr[:,i_c,:])[:,1]
                
                # eval test set
                yte_bar[i_c,:] = clsf.predict_proba(Xte[:,i_c,:])[:,1]
                
                i_c += 1
            
            # determine predicted class from  mode of all classifiers
            ytr_bar, _ = np.argmax(ytr_bar,axis=0)
            yte_bar, _ = np.argmax(yte_bar,axis=0)
            
            train_res = _generate_confusion_mat(ytr, ytr_bar)
            test_res = _generate_confusion_mat(yte, yte_bar)
            
        else:
            raise("Invalid multi-class extension")
        
        
        results = {'Train'  : train_res,
                   'Test'   : test_res}
        
        return results
    
    def dynamic_eval(self,clsf,X,y):
        
        if len(X.shape) == 3:
            X = np.expand_dims(X,axis=0)
        
        data = [None] * self.classes
        labels = np.unique(y)
        Nl = labels.shape[0]
        
        if Nl != self.classes:
            raise("Too many labels")
        
        for i_l in range(Nl):
            label = labels[i_l]
            data[i_l] = X[:,y==label,:,:]
        
        # calculate the maximum number of trials available for each class
        Nt = min([x.shape[1] for x in data])
        
        if self.win_sz == None:
            win_sz = Nt - self.step_sz
        else:
            win_sz = self.win_sz

        Nblocks,_,Nclf,Nf = X.shape

        results = {'Train' : np.zeros((Nblocks,Nl,Nl)),
                   'Test'  : np.zeros((Nblocks,Nl,Nl))}

        
        for i_s in range(Nblocks):
            # update training and test set for this block
            if self.win_type == 'sliding':
                # extract data
                train_sz = win_sz
                
                Xtr = np.zeros((self.classes*win_sz,Nclf,Nf))
                Xte = np.zeros((self.classes*self.step_sz,Nclf,Nf))
                
                ytr = -1 * np.ones((self.classes*win_sz,))
                yte = -1 * np.ones((self.classes*self.step_sz,))
                
                train_start = i_s * self.step_sz
                train_stop = train_start + win_sz
                
                test_start = train_stop
                test_stop  = test_start + self.step_sz
                
            elif self.win_type == 'expanding':                
                train_sz = win_sz + i_s*self.step_sz
                
                Xtr = np.zeros((self.classes*train_sz,Nclf,Nf))
                Xte = np.zeros((self.classes*self.step_sz,Nclf,Nf))
                
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
                Ctr = data[i_c][i_s,train_start:train_stop,:,:]
                Xtr[i_c*train_sz:(i_c+1)*train_sz,:,:] = Ctr
                    
                ytr[i_c*train_sz:(i_c+1)*train_sz] = l * np.ones((train_sz,))
                    
                # testing data
                Cte = data[i_c][i_s,test_start:test_stop,:,:]
                Xte[i_c*self.step_sz:(i_c+1)*self.step_sz,:,:] = Cte
                    
                yte[i_c*self.step_sz:(i_c+1)*self.step_sz] = l * np.ones((self.step_sz,))
                

            step_results = self.static_eval(clsf,Xtr,ytr,Xte,yte)
            
            results['Train'][i_s,:,:] = step_results['Train']
            results['Test'][i_s,:,:] = step_results['Test']
                
        return results
    
class rLDA(EnsembleClassifierEval):
    
    def __init__(self,eval_type,ext,classes,win_sz=None,
                 win_type='sliding',step_sz=None,decay=0.8):
        super().__init__(eval_type,ext,classes,win_sz=None,
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