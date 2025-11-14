"""
Created on 14 November 2025
@author:RSJ

In this test we generate some random data and check the `register` method
of `reference_image.ReferenceImage`.
"""
import regatta
import numpy as np
from matplotlib import pyplot as plt
import sys,os,glob

ss,sd,sf = 120,15,25

sample = np.random.rand(ss,sd,ss)# + 1j*np.random.rand(100,10,20)

rs,rd,rs = 100,10,20
ref_data = sample[:rs,:rd,:rs]


# create a ReferenceImage object based on the sample data
refim = regatta.reference_image.ReferenceImage(ref_data)


# Case 1: register a 3D target to the reference image
shifts = [2,3,4]
target = sample[shifts[0]:rs+shifts[0],shifts[1]:rd+shifts[1],shifts[2]:rs+shifts[2]]

res = refim.register(target)
print(res)

# Case 2: register a 2D target to the reference image
target = sample[50,shifts[1]:rd+shifts[1],shifts[2]:rs+shifts[2]]

res = refim.register(target)
print(res)
