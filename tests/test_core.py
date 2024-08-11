import mindpype as mp
import numpy as np
import pickle
import pytest

class CoreUnitTest():
    def __init__(self):
        self.__session = mp.Session.create()
    
    def TestFindObjFunc(self, raw_data):
        obj1 = mp.Tensor.create_from_data(self.__session, raw_data)
        obj2 = mp.Scalar.create_from_value(self.__session, 'test')
        obj = self.__session.find_obj(obj1.id)
        return obj
    
    def TestSaveSessionFunc(self):   
        obj1 = mp.Tensor.create_from_data(self.__session, np.zeros((3,3,3)))
        obj2 = mp.Scalar.create_from_value(self.__session, 'test')
        output = self.__session.save_session(file='test.pickle')
        return output
    
    def TestAddToSessionError(self):
        test_string = 'test'
        self.__session.add_to_session(test_string)
    
def test_execute():
    Test = CoreUnitTest()
    res = Test.TestSaveSessionFunc()
    with open("test.pickle", "rb") as f:
        x = pickle.load(f)
    
    assert isinstance(res['pipeline'], mp.Session)
    
    raw_data = np.zeros((3,3,3))
    res = Test.TestFindObjFunc(raw_data)
    assert (res.data == raw_data).all()
    
    with pytest.raises(ValueError) as e_info:
        res = Test.TestAddToSessionError()
    
test_execute()