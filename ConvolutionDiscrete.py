import numpy as np


def Convolution(l1: list, l2: list, axis1: int, axis2: int):
    list_final = []
    index_list = []
    protrude = axis2
    list_usable = [0]*len(l1)
    list_usable[0] = l2[0]
    start_index = -protrude + -axis1
    i = 1
    
    while list_usable.count(0) != len(list_usable):
        list_final.append(np.dot(l1, list_usable))
        
        index_list.append(start_index)
        start_index += 1
        list_usable.pop()
        try:
            list_usable.insert(0, l2[i])
        except IndexError as e:
            list_usable.insert(0, 0)
        print(list_usable)
        i += 1
        
    
    # list_final = list(reversed(list_final))
    # index_list = list(reversed(index_list))
    print("CONVOLUTED: " + str(list_final))
    print("Start: " + str(index_list.index(0)))
    return list_final