import mindpype as mp
import numpy as np
import pickle
import pytest
from sklearn.svm import SVC

class CoreUnitTest():
    """Unit test for Session class in core.py"""
    def __init__(self):
        self.__session = mp.Session.create()
    
    def TestFindObjFunc(self):
        """Ensure find_obj executes without error and returns correct object using session id"""
        raw_data = np.zeros((3,3,3))

        t = mp.Tensor.create_from_data(self.__session, raw_data)
        s = mp.Scalar.create_from_value(self.__session, 'test')         
        misc = mp.Classifier.create_SVM(self.__session)

        ## how to do id of session?

        t_obj = self.__session.find_obj(t.session_id)
        assert t_obj == t

        s_obj = self.__session.find_obj(s.session_id)  
        assert s_obj == s

        misc_obj = self.__session.find_obj(misc.session_id)   ## this is returning None
        print(misc.mp_type)
        assert misc_obj == misc    

        ## can we do ext src?
    
    def TestAddToSession(self):
        """Ensure add_to_session correctly MindPype adds object to session"""
        ## obj not inheriting MPBase???

        test_string = 'test'

        try:
            self.__session.add_to_session(test_string)
        except ValueError:
            pass
    
def test_execute():
    t = CoreUnitTest()
    
    t.TestFindObjFunc()

    t.TestAddToSession()
    
test_execute()