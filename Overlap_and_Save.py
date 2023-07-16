import ConvolutionCiclicalDiscrete as cc
import numpy as np
import copy

def OverlapSave(h:list, x:list, N:int=0):
    y = []
    M = len(h)
    N = max(N, M)
    T = (len(x))//(N+1-M) + 1

    h.extend([0]*(N-len(h)))

    block = [0]*N
    print("\n\n---------------------------------\nBLOCKS, Circular Convolution, DFT\n")
    for i in range(T):
        x_temp = x[(N+1-M)*i:(N+1-M)*(i+1)]
        block = block[-(M-1):]
        block.extend(x_temp)
        block.extend([0]*(N-len(block)))
        y_writer = cc.Convolution(copy.deepcopy(block), copy.deepcopy(h), N)
        DX = np.fft.fft(block, N)
        DH = np.fft.fft(h, N)
        print(f"B{i}:", block)
        print(f"y{i}:", y_writer)
        # print("DFT{x" + str(i) + "} = ", DX)
        # print("DFT{h" + str(i) + "} = ", DH)
        # print("y"+ str(i) + " = " +"IDFT{X*H" + str(i) + "} = ", np.fft.ifft([x*y for x, y in zip(DH, DX)]), '\n')
        y.extend(y_writer[M-1:])
    
    print("\n-------\nResult:")
    print("y =", y, "\n")
    return y
