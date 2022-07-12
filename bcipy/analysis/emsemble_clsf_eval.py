# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 10:48:25 2020

@author: Nick
"""

from clsf_eval import ClassifierEval
from scipy.special import binom
from scipy.stats import mode
from itertools import combinations as iter_combs
import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as skLDA
from sklearn.metrics import confusion_matrix

class EnsembleClassifierEval(ClassifierEval):
    
    def __init__(self,eval_type,ext,classes,win_sz=None,
                 win_type='sliding',step_sz=None,decay=0.8):
        super().__init__(eval_type,classes,win_type,step_sz,decay)
        self.ext = ext
    
    def static_eval(self,clsf,Xtr,ytr,Xte,yte):
        
        labels = np.unique(ytr)
        Nl = labels.shape[0]
        
        if self.ext == 'PW':
            Nclf = int(binom(Nl,2))
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
            
            ytr_bar = np.squeeze(ytr_bar)
            yte_bar = np.squeeze(yte_bar)
            
            train_res = confusion_matrix(ytr, ytr_bar)
            test_res = confusion_matrix(yte, yte_bar)

        elif self.ext == 'OVR':

            ytr_bar = -1 * np.ones((Nl,Xtr.shape[0]))
            yte_bar = -1 * np.ones((Nl,Xte.shape[0]))
            
            # train/test all the binary classifiers
            for i_c in range(Nl):
                l1 = labels[i_c]
                # train
                ytr_l = np.zeros((ytr.shape[0],))
                ytr_l[ytr == l1] = 1
                ytr_l[ytr != l1] = 0
                
                clsf.fit(Xtr[:,i_c,:],ytr_l)
                
                # eval train set
                ytr_bar[i_c,:] = clsf.predict_proba(Xtr[:,i_c,:])[:,1]
                
                # eval test set
                yte_bar[i_c,:] = clsf.predict_proba(Xte[:,i_c,:])[:,1]
                
                i_c += 1
            
            # determine predicted class from  mode of all classifiers
            ytr_bar = [labels[l] for l in np.argmax(ytr_bar,axis=0)]
            yte_bar = [labels[l] for l in np.argmax(yte_bar,axis=0)]
            
            train_res = confusion_matrix(ytr, ytr_bar)
            test_res = confusion_matrix(yte, yte_bar)
            
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
        
        
    def evaluate_train_test(self,train_set,test_set):
        
        clsf = skLDA(solver='lsqr',shrinkage='auto')
        
        
        Xtr, ytr = train_set
        Xte, yte = test_set

        best_fb, cv_res = self.select_best_feature_hyperparams(clsf,Xtr,ytr)

        # extract best frequency band
        Xtr = Xtr[:,best_fb,:,:]
        Xte = Xte[:,best_fb,:,:]

        results = self.static_eval(clsf,Xtr,ytr,Xte,yte)
        
        return results, cv_res    
    
    def select_best_feature_hyperparams(self,clsf,X,y):
        """
        Use K-fold CV to select best frequency band for user

        """
        X = np.transpose(X,axes=(1,0,2,3))
        Nfb,Nt,Nf,Nch = X.shape
        
        cv_res = np.zeros((Nfb,2))
        
        labels = np.unique(y)
        Nc = labels.shape[0]
        
        # np arrays for CV
        # assuming 5-fold CV with 45 training samples per class
        Xtr = np.zeros((Nc*36,Nf,Nch))
        Xte = np.zeros((Nc*9,Nf,Nch))
        ytr = np.zeros((Nc*36,))
        yte = np.zeros((Nc*9,))
        
        
        class_indices = [None] * Nc
        for i_l in range(Nc):
            class_indices[i_l] = np.squeeze(np.argwhere(y==labels[i_l]))
            
            ytr[i_l*36:(i_l+1)*36] = labels[i_l]
            yte[i_l*9:(i_l+1)*9] = labels[i_l]
        
        cv_scr = np.zeros((5,))
        
        for i_f in range(Nfb):
            Xf = X[i_f,:,:,:]
            
            for i_cv in range(5):
                for i_l in range(Nc):
                    l = labels[i_l]
                    l_cv_te_slice_indices = class_indices[i_l][i_cv*9:(i_cv+1)*9]
                    l_cv_tr_slice_indices = np.setdiff1d(class_indices[i_l],
                                                         l_cv_te_slice_indices)
                    
                    Xtr[i_l*36:(i_l+1)*36,:,:] = Xf[l_cv_tr_slice_indices,:,:]
                    Xte[i_l*9:(i_l+1)*9,:,:] = Xf[l_cv_te_slice_indices,:,:]
                
                
                fold_res = self.static_eval(clsf,Xtr,ytr,Xte,yte)
                fold_acc = np.sum(np.diag(fold_res['Test'])) / np.sum(fold_res['Test'])
                cv_scr[i_cv] = fold_acc
            
            cv_res[i_f,0] = np.mean(cv_scr)
            cv_res[i_f,1] = 2 * np.std(cv_scr) # 95% conf. interval           
            
        
        best_fb = np.argmax(cv_res[:,0])
        
        return best_fb, cv_res
