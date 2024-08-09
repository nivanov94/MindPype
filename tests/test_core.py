import mindpype as mp
import numpy as np
import pickle

class SaveSessionUnitTest():
    def __init__(self):
        self.__session = mp.Session.create()
    
    def TestSaveSessionExecution(self):   
        obj1 = mp.Tensor.create_from_data(self.__session, np.zeros((3,3,3)))
        obj2 = mp.Scalar.create_from_value(self.__session, 'test')
        output = self.__session.save_session(file='test.pickle')
        return output
    
def test_execute():
    Test = SaveSessionUnitTest()
    res = Test.TestSaveSessionExecution()
    with open("test.pickle", "rb") as f:
        x = pickle.load(f)
    
    assert isinstance(res['pipeline'], mp.Session)
    
test_execute()