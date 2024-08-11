import numpy as np

def compute_ss(A):
    steady_state = []
    for i in range(A.shape[0]):
        w, v = np.linalg.eig(A[i].T)
        w = np.abs(w)
        v = np.abs(v)
        steady_state.append(v[:,np.argmax(w)]/np.sum(v[:,np.argmax(w)]))

    steady_state = np.stack(steady_state, axis=0)
    return steady_state


A = np.array([[[0.587, 0.147, 0.266],
               [0.112, 0.752, 0.136],
               [0.140, 0.127, 0.733]],
              [[0.688, 0.103, 0.209],
               [0.114, 0.768, 0.118],
               [0.146, 0.122, 0.732]],
              [[0.668, 0.146, 0.186],
               [0.105, 0.749, 0.146],
               [0.163, 0.154, 0.683]]])
#ss = compute_ss(A)
#print(ss)

ss = np.asarray([[22, 18, 16],
                 [20, 21, 18],
                 [25, 17, 13]])
ss = ss/(ss.sum(axis=1)[:, np.newaxis])
print(ss)

def find_map_path(path, ss, A):
    map_path = None
    max_prob = 0

    for c in range(ss.shape[0]):
        prob = ss[c,path[0]]

        for i in range(1,len(path)):
            prob *= A[c,path[i-1],path[i]]

        print(f"\tProb for path {c}: {prob:.6f}")
        if prob > max_prob:
            max_prob = prob
            map_path = c

    return map_path  

paths = [
    # Path 0
    [
        [0,2,0,2,0],
        [0,2,0],
        [2,0,2,1,2,1],
        [0,2,0,2,0],
        [1,2,1],
        [0,2,0,1],
        [1,0,1],
        [0,1,2,0,2,0,1],
        [2,0,2,1,2,0],
        [0,2,0,2,1]
    ],
    # Path 1
    [
        [2,0,2],
        [0,2,0],
        [2,0,2,0,2],
        [2,1,2,1],
        [0,2,1,2],
        [1],
        [2,1,2,1,0],
        [1,2,1,2,0,2],
        [2,0,2,1,0,2],
        [1]
    ],
    # Path 2
    [
        [0,2],
        [0,2],
        [0,2,1,2],
        [1,0,2],
        [0,1,2,1,2],
        [0,2],
        [1,0,2,0,2],
        [0,1,2,1,0],
        [0,2,0,2,0],
        [0,2,1,2]
    
    ]
]
for i_c, c_paths in enumerate(paths):
    for i_p, path in enumerate(c_paths):
        print(f"Path {i_p} in class {i_c}: {path}")
        map_path = find_map_path(np.array(path), ss, A)
        print(f"MAP path for path {i_p} in class {i_c}: {map_path}\n")
#map_path = find_map_path(path, ss, A)
#print(f"MAP path: {map_path}")