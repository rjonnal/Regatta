"""
Created 14 November 2025.
@author: RSJ

In RegATTA we need to perform DFTs on arrays of varying dimensionality.
This script tests the efficiency of FFTN on for 2D DFT compared to FFT2.
Using FFTN for all DFTs, regardless of dimensionality, makes the code simpler,
i.e., less dimensionality checking, etc.
"""

import numpy as np
import cProfile

N = 4096
arr = np.random.randn(N,N)
iterations = 10


def fft2_test(arr):
    for k in range(iterations):
        np.fft.fft2(arr)

def fftn_test(arr):
    for k in range(iterations):
        np.fft.fftn(arr)

cProfile.run('fft2_test(arr)')
cProfile.run('fftn_test(arr)')
