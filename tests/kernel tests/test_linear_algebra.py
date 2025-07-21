import mindpype as mp
import numpy as np

class LinearAlgebraUnitTest:
    def TestKernelCreation(self):
        session = mp.Session.create()
        graph1 = mp.Graph.create(session)
        graph2 = mp.Graph.create(session)

        inA = mp.Tensor.create_from_data(session, [[1,2], [3,4]])
        inB = mp.Tensor.create_from_data(session, [[2,2], [2,2]])
        in_bad = mp.Scalar.create(session, int)
        out = mp.Tensor.create(session, (2,2))
        out1 = mp.Tensor.create(session, (2,2))

        node = mp.kernels.MatrixMultKernel.add_to_graph(graph1, inA, inB, out)
        
        node_bad = mp.kernels.MatrixMultKernel.add_to_graph(graph1, inA, in_bad, out1)

        # input/output must be tensors
        try:
            graph1.verify()
            graph1.initialize()
            graph1.execute()
        except TypeError:
            pass
        

        in_data = [mp.Tensor.create_from_data(session, [1,2,3]), mp.Tensor.create_from_data(session, [2,2])]
        labels = mp.Tensor.create_from_data(session, [1,2])
        # labels.data = np.array([1,2,3])

        node = mp.kernels.MatrixMultKernel.add_to_graph(graph2, inA, inB, out, init_inputs=in_data, init_labels=labels)

        graph2.verify()
        graph2.initialize()
        graph2.execute()


    def TestComputeOutputSize(self):
        session = mp.Session.create()
        graph1 = mp.Graph.create(session)
        graph2 = mp.Graph.create(session)
        graph3 = mp.Graph.create(session)
        graph4 = mp.Graph.create(session)
        graph5 = mp.Graph.create(session)

        inA_inner_2 = mp.Tensor.create_from_data(session, [[1,2], [1,2]])
        inA_inner_2_b = mp.Tensor.create_from_data(session, [[1], [1], [1]])
        inB_inner_2 = mp.Tensor.create_from_data(session, [[1,2,3], [1,2,3]])

        inA_inner_3 = mp.Tensor.create_from_data(session, [[[1,2], [1,2]], [[1,2], [1,2]]])
        inB_inner_3 = mp.Tensor.create_from_data(session, [[[1,2], [1,2]], [[1,2], [1,2]]])
        inB_inner_3_b = mp.Tensor.create_from_data(session, [[[1,2], [1,2], [1,2]]])

        out1 = mp.Tensor.create(session, (2,3))
        out2 = mp.Tensor.create(session, (2,2,3))
        out3 = mp.Tensor.create(session, (2,2,2))
        out4 = mp.Tensor.create(session, (2,2,2))
        out5 = mp.Tensor.create(session, (2,2))
        out6 = mp.Tensor.create(session, (2,2))
        out7 = mp.Tensor.create(session, (2,2))
        out8 = mp.Tensor.create(session, (2,2))
        out9 = mp.Tensor.create(session, (2,2))

        n1 = mp.kernels.MatrixMultKernel.add_to_graph(graph1, inA_inner_2, inB_inner_2, out1)
        # Inner dimensions of input tensors must match
        n2 = mp.kernels.MatrixMultKernel.add_to_graph(graph1, inA_inner_2_b, inB_inner_2, out5)

        n3 = mp.kernels.MatrixMultKernel.add_to_graph(graph2, inA_inner_3, inB_inner_2, out2)
        # Inner dimensions of input tensors must match
        n4 = mp.kernels.MatrixMultKernel.add_to_graph(graph2, inA_inner_3, inA_inner_2_b, out6)

        n5 = mp.kernels.MatrixMultKernel.add_to_graph(graph3, inA_inner_2, inB_inner_3, out3)
        # Inner dimensions of input tensors must match
        n6 = mp.kernels.MatrixMultKernel.add_to_graph(graph3, inA_inner_2_b, inA_inner_3, out7)

        n7 = mp.kernels.MatrixMultKernel.add_to_graph(graph4, inA_inner_3, inB_inner_3, out4)
        # Inner dimensions of input tensors must match
        n8 = mp.kernels.MatrixMultKernel.add_to_graph(graph4, inA_inner_3, inB_inner_3_b, out8)

        n9 = mp.kernels.MatrixMultKernel.add_to_graph(graph5, mp.Tensor.create_from_data(session, [[[[1,2]]]]), mp.Tensor.create_from_data(session, [[[1,2,3]]]), out9)

        # Inner dimensions of input tensors must match
        try:
            graph1.verify()
            graph1.initialize()
            graph1.execute()
        except ValueError:
            pass

        try:
            graph2.verify()
            graph2.initialize()
            graph2.execute()
        except ValueError:
            pass

        try:
            graph3.verify()
            graph3.initialize()
            graph3.execute()
        except ValueError:
            pass

        try:
            graph4.verify()
            graph4.initialize()
            graph4.execute()
        except ValueError:
            pass


def test_execute():
    test = LinearAlgebraUnitTest()
    
    test.TestKernelCreation()
    test.TestComputeOutputSize()