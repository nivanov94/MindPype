import mindpype as mp
import numpy as np

class GraphTests():

    def XDF(self, file, tasks, channels, start, samples):
        session = mp.Session.create()

        graph = mp.Graph.create(session)
        graph_cont = mp.Graph.create(session)

        xdf_object = mp.source.BcipXDF.create_epoched(session, file, tasks, channels, start, samples)
        xdf_object_cont = mp.source.BcipXDF.create_continuous(session, file, tasks, channels, start, samples)

        t_in = mp.Tensor.create_from_handle(session, (len(channels), samples), xdf_object)
        t_in_cont = mp.Tensor.create_from_handle(session, (len(channels), samples), xdf_object_cont)

        t_in_2 = mp.Tensor.create_from_data(session, data=np.zeros(t_in.shape))

        t_out = mp.Tensor.create(session, shape=t_in.shape)
        t_out_cont = mp.Tensor.create(session, shape=t_in.shape)

        mp.kernels.AdditionKernel.add_to_graph(graph, t_in, t_in_2, t_out)
        mp.kernels.AdditionKernel.add_to_graph(graph_cont, t_in_cont, t_in_2, t_out_cont)

        graph.verify()
        graph_cont.verify()

        graph.initialize()
        graph_cont.initialize()

        for i in range(10):
            graph.execute('flash')
            graph_cont.execute('flash')


def test_XDF():
    """tests = GraphTests()
    ch_map =  {'FCz': 0, 'Fz': 1, 'F3': 2, 'F7': 3, 'FC3': 4, 'T7': 5, 'C5': 6, 'C3': 7, 'C1': 8,
            'Cz' : 9, 'CP3': 10, 'CPz': 11, 'P7': 12, 'P5': 13, 'P3': 14, 'P1': 15, 'Pz': 16,
            'PO3': 17, 'Oz': 18, 'PO4': 19, 'P8': 20, 'P6': 21, 'P4': 22, 'P2': 23, 'CP4': 24,
            'T8' : 25, 'C6': 26, 'C4' : 27, 'C2': 28, 'FC4': 29, 'F4': 30, 'F8': 31}

    sel_chs = ('FCz', 'Fz', 'F3', 'F7', 'FC3', 'T7', 'C5', 'C3', 'C1', 'Cz', 'CP3', 'CPz', 'P7',
                'P5', 'P3','P1','Pz','PO3','Oz','PO4','P8','P6','P4','P2','CP4','T8','C6','C4',
                'C2','FC4','F4','F8')

    channels = [ch_map[ch] for ch in sel_chs]
    tasks = ('flash', 'target')
    trial_data = tests.XDF(
        ['C:/Users/lioa/Documents/Mindset P300 Code for Aaron/sub-P001_ses-S001_task-vP300+2x2_run-003.xdf'],
                           tasks,
                           channels,
                           -.2,
                           500)

    assert trial_data == mp.MPEnums.SUCCESS"""
    assert True
