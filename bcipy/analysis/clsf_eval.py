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

from sklearn.metrics import confusion_matrix, log_loss
from sklearn.model_selection import cross_val_score, StratifiedKFold


class ClassifierEval():
    """
    Base class for classifer assessment
    """
    
    def __init__(self,classes,win_sz=None,
                 win_type='sliding',step_sz=None,decay=0.8,seed_sz=45):
        """
        Create classifier eval object
        """
        self.classes = classes
        self.win_sz = win_sz
        self.win_type = win_type
        self.step_sz = step_sz
        self.decay = decay
        self.seed_sz = seed_sz
        
    def extract_dynamic_init_training_set(self,X,y):
        data = [None] * self.classes
        labels = np.unique(y)
        Nl = labels.shape[0]
        
        if Nl != self.classes:
            raise("Too many labels")
        
        for i_l in range(Nl):
            label = labels[i_l]
            data[i_l] = X[y==label,:,:]
        
        _, Nfb, Ns, Nc = X.shape
        Xtr = np.zeros((self.win_sz*self.classes,Nfb,Ns,Nc))
        ytr = np.zeros((self.win_sz*self.classes,))
        
        for i_c in range(self.classes):
            l = labels[i_c]
                    
            # training data
            Ctr = np.squeeze(data[i_c][:self.win_sz,:,:,:])
            Xtr[i_c*self.win_sz:(i_c+1)*self.win_sz,:,:,:] = Ctr
                    
            ytr[i_c*self.win_sz:(i_c+1)*self.win_sz] = l * np.ones((self.win_sz,))
                    
        return Xtr, ytr
        
    def static_eval(self,clsf,Xtr,ytr,Xte,yte):
                
        clsf.fit(Xtr,ytr)
        ytr_bar = clsf.predict(Xtr)
        train_res = confusion_matrix(ytr, ytr_bar)
        
        yte_bar = clsf.predict(Xte)
        test_res = confusion_matrix(yte,yte_bar)
        
        train_prob = clsf.predict_proba(Xtr)
        test_prob = clsf.predict_proba(Xte)
        
        train_cross_ent = log_loss(ytr,train_prob,labels=np.unique(ytr))
        test_cross_ent = log_loss(yte,test_prob,labels=np.unique(yte))
        
        results = {'Train-confusion-mat'  : train_res,
                   'Test-confusion-mat'   : test_res,
                   'Train-cross-entropy'  : train_cross_ent,
                   'Test-cross-entropy'   : test_cross_ent}
        
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
            clsf.fit(Xtr,ytr)
            ytr_bar = clsf.predict(Xtr)
            results['Train'][i_s,:,:] = confusion_matrix(ytr, ytr_bar)
                
            # test
            yte_bar = clsf.predict(Xte)
            results['Test'][i_s,:,:] = confusion_matrix(yte,yte_bar)
                
        return results
    
class MDM(ClassifierEval):
    """
    Class for evaluating MDM classifiers
    """
    
    def __init__(self,classes,win_sz=None,
                 win_type='sliding',step_sz=None,decay=0.8,seed_sz=45):
        super().__init__(classes,win_sz,
                         win_type,step_sz,decay,seed_sz)
    
    def evaluate_train_test(self,train_set,test_set):
        """
        Evaluate classifier using provided data

        """
 
        clsf = pyriemMDM()

        Xtr, ytr = train_set
        Xte, yte = test_set

        best_fb, cv_res = self.select_best_feature_hyperparams(clsf,Xtr,ytr)

        # extract best frequency band
        Xtr = Xtr[:,best_fb,:,:]
        Xte = Xte[:,best_fb,:,:]

        results = self.static_eval(clsf,Xtr,ytr,Xte,yte)
        
        return results, cv_res
    
    def evaluate_dynamic_online(self,dataset):
        
        # split the train set into train and test
        # with a sliding window to simulate co-adaptive online session
        
        clsf = pyriemMDM()
        
        X = dataset[0]
        y = dataset[1]
        
        # get initial training data seed
        Xtr_init, ytr_init = self.extract_dynamic_init_training_set(X,y)

        best_fb, cv_res = self.select_best_feature_hyperparams(clsf,
                                                                Xtr_init,
                                                                ytr_init)
        # extract best frequency band
        X = X[:,best_fb,:,:]
        
        results = self.dynamic_eval(clsf,X,y)

        return results, cv_res
    
    def select_best_feature_hyperparams(self,clsf,X,y):
        """
        Use K-fold CV to select best frequency band for user

        """
        X = np.transpose(X,axes=(1,0,2,3))
        Nfb = X.shape[0]
        
        cv_res = np.zeros((Nfb,2))
        
        for i_f in range(Nfb):
            Xf = X[i_f,:,:,:]
            cv_scr = cross_val_score(clsf,Xf,y=y,
                                     scoring='accuracy',
                                     cv=StratifiedKFold(n_splits=5))
            
            cv_res[i_f,0] = np.mean(cv_scr)
            cv_res[i_f,1] = 2 * np.std(cv_scr) # 95% conf. interval           
            
        
        best_fb = np.argmax(cv_res[:,0])
        
        return best_fb, cv_res


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
    
    def __init__(self,
                 classes,win_sz=None,
                 win_type='sliding',step_sz=None,decay=0.8):
        super().__init__(classes,win_sz,
                         win_type,step_sz,decay)
        
    
    def evaluate_train_test(self,train_set,test_set):
        """
        Evaluate classifier using provided data

        """

        clsf = pyriemFgMDM()

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
        Nfb = X.shape[0]
        
        cv_res = np.zeros((Nfb,2))
        
        for i_f in range(Nfb):
            Xf = X[i_f,:,:,:]
            cv_scr = cross_val_score(clsf,Xf,y=y,
                                     scoring='accuracy',
                                     cv=StratifiedKFold(n_splits=5))
            
            cv_res[i_f,0] = np.mean(cv_scr)
            cv_res[i_f,1] = 2 * np.std(cv_scr) # 95% conf. interval           
            
        
        best_fb = np.argmax(cv_res[:,0])
        
        return best_fb, cv_res


class rLDA(ClassifierEval):
    """
       regularized LDA classifier
    """
    
    def __init__(self,
                 classes,win_sz=None,
                 win_type='sliding',step_sz=None,decay=0.8):
        super().__init__(classes,win_sz,
                         win_type,step_sz,decay)
        
    
    def evaluate_train_test(self,train_set,test_set):
        """
        Evaluate classifier using provided data

        """

        clsf = skLDA(solver='lsqr',shrinkage='auto')

        Xtr, ytr = train_set
        Xte, yte = test_set

        if len(Xtr.shape) > 3:
            Xtr = np.squeeze(Xtr)
        
        if len(Xte.shape) > 3:
            Xte = np.squeeze(Xte)

        best_fb, cv_res = self.select_best_feature_hyperparams(clsf,Xtr,ytr)

        # extract best frequency band
        Xtr = Xtr[:,best_fb,:]
        Xte = Xte[:,best_fb,:]

        results = self.static_eval(clsf,Xtr,ytr,Xte,yte)
        
        return results, cv_res
    
    
    def select_best_feature_hyperparams(self,clsf,X,y):
        """
        Use K-fold CV to select best frequency band for user

        """
        X = np.transpose(X,axes=(1,0,2))
        Nfb,Nt,Nfeats = X.shape
        
        cv_res = np.zeros((Nfb,2))
        
        for i_f in range(Nfb):
            Xf = X[i_f,:,:]
            cv_scr = cross_val_score(clsf,Xf,y=y,
                                     scoring='accuracy',
                                     cv=StratifiedKFold(n_splits=5))
            
            cv_res[i_f,0] = np.mean(cv_scr)
            cv_res[i_f,1] = 2 * np.std(cv_scr) # 95% conf. interval           
            
        
        best_fb = np.argmax(cv_res[:,0])
        
        return best_fb, cv_res
