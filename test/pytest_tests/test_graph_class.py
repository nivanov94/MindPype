from bcipy import bcipy
import sys, os
class GraphUnitTest:

    def __init__(self):
        self.__session = bcipy.Session.create()
        self.__graph = bcipy.Graph.create(self.__session)
        
    
    def TestGraphCreation(self):
        return self.__graph._bcip_type
    
    def TestGraphVerification(self):
        return self.__graph.verify()

    def TestGraphInitialization(self):
        return self.__graph.initialize()

    def TestGraphExecution(self):
        sys.stdout = open(os.devnull, 'w')
        sts = self.__graph.execute()
        sys.stdout = sys.__stdout__
        return sts

    def TestGraphScheduler(self):
        sts = bcipy.Graph.add_node(self.__graph, bcipy.graph.Node(self.__graph, None, {}))
        return sts
    
def test_graph():
    
    GraphUnitTest_Object = GraphUnitTest()
    try:
        assert GraphUnitTest_Object.TestGraphCreation() == bcipy.BcipEnums.GRAPH
        print("Graph Creation Test: PASSED")
    except AssertionError:
        print("Graph Creation Test: FAILED")
    
    try:
        assert GraphUnitTest_Object.TestGraphVerification() == bcipy.BcipEnums.SUCCESS
        print("Graph Verification Test: PASSED")
    except AssertionError:
        print("Graph Verification Test: FAILED")

    try:
        assert GraphUnitTest_Object.TestGraphInitialization() == bcipy.BcipEnums.SUCCESS
        print("Graph Initialization Test: PASSED")
    except AssertionError:
        print("Graph Initialization Test: FAILED")
    
    try:
        assert GraphUnitTest_Object.TestGraphExecution() == bcipy.BcipEnums.SUCCESS
        print("Graph Execution Test: PASSED")
    except AssertionError:
        print("Graph Execution Test: FAILED")
    
    try:
        assert GraphUnitTest_Object.TestGraphScheduler() == bcipy.BcipEnums.SUCCESS
        print("Graph Scheduler Test: PASSED")
    except AssertionError as e:
        print("Graph Scheduler Test: FAILED")

    