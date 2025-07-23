import mindpype as mp
import numpy as np

class DataManagment():
    def TestEnqueueKernel(self):
        session = mp.Session.create()
        graph = mp.Graph.create(session)

        queue = mp.CircleBuffer.create(session, 2, mp.Scalar.create(session, int))
        scalar = mp.Scalar.create_from_value(session, 5)

        node = mp.kernels.datamgmt.EnqueueKernel.add_to_graph(graph, scalar, queue)

        graph.verify()
        graph.initialize()
        graph.execute()

def test_execute():
    test = DataManagment()

    test.TestEnqueueKernel()