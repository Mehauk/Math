import ConvolutionCiclicalDiscrete as C
import ConvolutionDiscrete as L
import Overlap_and_Save as Block
import numpy as np


if __name__ == "__main__":
    # Initialize Vars

    A = [1, 4, 10, 18, 27, 36, 45, 54, 63, 71, 77, 79, 77, 71, 63, 54, 45, 36 , 27, 18, 10, 4, 1, 0]
    B = [1, 3, 3, 5, 7, 6, 7, 8, 9, 1, 9, 7, 7, 3, 2, 4, 3, 1, 1]
    h0 = [1, 2, 3, 3, 4]

    x = [0, 1+3j, 2, 1-3j]
    h = [2, -1-1j, 4, -1+1j]
    x1 = [1, 2, 3, 4]
    x2 = [1, 3, -1, -2]
    X = [1, 2, -2, 3]

    #C.Convolution, L.Convolution, Block.OverlapSave, np.fft.fft, np.fft.ifft
    # print(B[3], B[5], B[8], A[2], A[3], A[8], A[10])
    # print(np.fft.ifft([a*b for a,b in zip(x,h)], 4))
    # print(C.Convolution(x1, x2, 4))
    # print(np.fft.fft(X, 10))

    Block.OverlapSave(h0, B, 10)
    # print(np.fft.ifft([DX*DH for DX,DH in zip(np.fft.fft(B, 10), np.fft.ifft(h0, 10))], 10))

    # xr = [0, 0, 0, 0, 1, 3, 3, 5, 7, 6]
    # print(C.Convolution(xr, h0, 10))