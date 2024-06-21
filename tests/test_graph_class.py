import mindpype as mp
import sys, os
class GraphUnitTest:

    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)
        
    
    def TestGraphCreation(self):
        return self.__graph.mp_type
    
    def TestGraphVerification(self):
        self.__graph.verify()

    def TestGraphInitialization(self):
        self.__graph.initialize()

    def TestGraphExecution(self):
        sys.stdout = open(os.devnull, 'w')
        self.__graph.execute()
        sys.stdout = sys.__stdout__

    def TestGraphScheduler(self):
        mp.Graph.add_node(self.__graph, mp.graph.Node(self.__graph, None, {}))
    
def test_graph():
    
    GraphUnitTest_Object = GraphUnitTest()
    assert GraphUnitTest_Object.TestGraphCreation() == mp.MPEnums.GRAPH

    try:
        GraphUnitTest_Object.TestGraphScheduler()
        print("Graph Scheduler Test: PASSED")
    except:
        print("Graph Scheduler Test: FAILED")
    
    try:
        GraphUnitTest_Object.TestGraphVerification()
        print("Graph Verification Test: PASSED")
    except:
        print("Graph Verification Test: FAILED")

    try:
        GraphUnitTest_Object.TestGraphInitialization()
        print("Graph Initialization Test: PASSED")
    except:
        print("Graph Initialization Test: FAILED")
    
    try:
        GraphUnitTest_Object.TestGraphExecution()
        print("Graph Execution Test: PASSED")
    except:
        print("Graph Execution Test: FAILED")
    
    

    