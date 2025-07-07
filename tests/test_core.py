import mindpype as mp
import numpy as np
import pytest
from sklearn.svm import SVC

class CoreUnitTest():
    """Unit test for Session class in core.py"""
    
    def TestFindObjFunc(self):
        """Ensure find_obj executes without error and returns correct object using session id"""
        session = mp.Session.create()
        graph = mp.Graph.create(session)

        raw_data = np.zeros((3,3,3))

        t = mp.Tensor.create_from_data(session, raw_data)
        s = mp.Scalar.create_from_value(session, 'test')         
        misc = mp.Classifier.create_SVM(session)
        src = mp.source.InputLSLStream.create_marker_uncoupled_data_stream(session, active=False)

        session_obj = session.find_obj(session.session_id)
        assert session_obj == session

        t_obj = session.find_obj(t.session_id)
        assert t_obj == t

        s_obj = session.find_obj(s.session_id)  
        assert s_obj == s

        misc_obj = session.find_obj(misc.session_id) 
        assert misc_obj == misc    

        src_obj = session.find_obj(src.session_id)
        assert src_obj == src

        graph_obj = session.find_obj(graph.session_id)
        assert graph_obj == None
    
    def TestAddToSession(self):
        """Ensure add_to_session correctly MindPype adds object to session"""
        session = mp.Session.create()
        graph = mp.Graph.create(session)

        test_string = 'test'
        # Only objects inheriting MPBase can be added to a session
        try:
            session.add_to_session(test_string)
        except ValueError:
            pass

        src = mp.source.InputLSLStream.create_marker_uncoupled_data_stream(session, active=False)
        session.add_to_session(src)

        inA = np.zeros((1,1))
        inB = np.ones((1,1))
        out = np.zeros((1,1))
        kernel = mp.kernels.AdditionKernel(graph, inA, inB, out)
        # Kernal cannot be directly added to session
        try:
            session.add_to_session(kernel)
        except ValueError:
            pass
    
def test_execute():
    t = CoreUnitTest()
    
    t.TestFindObjFunc()

    t.TestAddToSession()
    
test_execute()