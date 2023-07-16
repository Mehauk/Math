import numpy as np

def Convolution(l1: list, l2: list, N: int = 0):
    list_final = []
    N = max(len(l1), len(l2), N)

    l1.extend([0]*(N-len(l1)))
    l2.extend([0]*(N-len(l2)))

    l2.reverse()

    for _ in range(N):
        l2t = l2[:-1]
        l2 = [l2[-1]]
        l2.extend(l2t)
        list_final.append(np.dot(l1, l2))
    
    return list_final
