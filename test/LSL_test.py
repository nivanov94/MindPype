import bcipy
import numpy as np
import pylsl

def LSL_test(channels, start, samples):
    session = bcipy.Session.create()

    graph = bcipy.Graph.create(session)


    lsl_object = bcipy.source.V2LSLStream.create_marker_coupled_data_stream(
        session, "type='EEG'",
        channels, -0.2, marker_fmt="^SPACE pressed$"
        )
    
    t_in = bcipy.Tensor.create_from_handle(session, (len(channels), samples), lsl_object)
    
    t_in_2 = bcipy.Tensor.create_from_data(session, shape=t_in.shape, data=np.zeros(t_in.shape))

    t_out = bcipy.Tensor.create(session, shape=t_in.shape)
    
    Add = bcipy.kernels.AdditionKernel.add_addition_node(graph, t_in, t_in_2, t_out)


    sts1 = graph.verify()
    
    if sts1 != bcipy.BcipEnums.SUCCESS:
        print(sts1)
        print("Test Failed D=")
        return sts1

    sts2 = graph.initialize()

    if sts2 != bcipy.BcipEnums.SUCCESS:
        print(sts2)
        print("Test Failed D=")
        return sts2
    
    i = 0

    marker_stream = pylsl.StreamInlet(pylsl.resolve_bypred("type='Markers'")[0])


    sts = bcipy.BcipEnums.SUCCESS
    while i < 10 and sts == bcipy.BcipEnums.SUCCESS:
        marker = marker_stream.pull_sample(timeout=1)
        print(marker, flush=True)
        if marker[0] == None:
            print("Waiting for marker")
        elif marker[0][0] == 'SPACE pressed':
            sts = graph.execute(500)
            print(t_in.data[23,50])
            print(t_out.data[23,50])   
            i+=1 



ch_map =  {'FCz': 0, 'Fz': 1, 'F3': 2, 'F7': 3, 'FC3': 4, 'T7': 5, 'C5': 6, 'C3': 7, 'C1': 8, 
           'Cz' : 9, 'CP3': 10, 'CPz': 11, 'P7': 12, 'P5': 13, 'P3': 14, 'P1': 15, 'Pz': 16, 
           'PO3': 17, 'Oz': 18, 'PO4': 19, 'P8': 20, 'P6': 21, 'P4': 22, 'P2': 23, 'CP4': 24, 
           'T8' : 25, 'C6': 26, 'C4' : 27, 'C2': 28, 'FC4': 29, 'F4': 30, 'F8': 31}
    
sel_chs = ('FCz',
               'Fz',
               'F3',
               'F7',
               'FC3',
               'T7',
               'C5',
               'C3',
               'C1',
               'Cz',
               'CP3',
               'CPz',
               'P7',
               'P5',
               'P3',
               'P1',
               'Pz',
               'PO3',
               'Oz',
               'PO4',
               'P8',
               'P6',
               'P4',
               'P2',
               'CP4',
               'T8',
               'C6',
               'C4',
               'C2',
               'FC4',
               'F4',
               'F8'
              )

if __name__ == '__main__':
    
    channels = [ch_map[ch] for ch in sel_chs]
    trial_data = LSL_test(channels, -.2, 500)
