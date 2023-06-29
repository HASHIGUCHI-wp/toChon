import taichi as ti
import numpy as np
import meshio
import datetime
import os
from pyevtk.hl import *
import sympy as sy



# num_surface = 73
num_surface = 133
# P_surface_str = [67, 59, 53, 54, 43, 33, 36, 39, 25, 26, 55, 57, 45, 44, 32, 42, 23, 56, 27, 58, 46, 47, 34, 41, 35, 62, 65, 49, 28, 48, 31, 38, 40, 64, 60, 29, 51, 50, 22, 37, 24, 63, 61, 30, 52, 73]
P_surface_str = [125, 115, 46, 105, 128, 84, 57, 75, 73, 123, 47, 110, 85, 69, 63, 83, 122, 114, 48, 87, 43, 66, 82, 90, 121, 109, 49, 89, 59, 81, 67, 120, 92, 50, 91, 112, 62, 68, 80, 124, 51, 94, 107, 93, 65, 44, 79, 119, 104, 52, 96, 95, 64, 74, 78, 118, 53, 108, 98, 97, 61, 72, 77, 117, 106, 54, 99, 60, 76, 71, 100, 126, 55, 101, 102, 111, 58, 70, 45, 116, 113, 103, 56, 133, 86]
NP_surface_str = "{"
FirstNP = True

for _s in range(num_surface):
    s = _s + 1
    if not(s in P_surface_str):
        NP_surface_str += '' if FirstNP else ','
        NP_surface_str += str(s)
        FirstNP = False
            
NP_surface_str += "}"
    
print(NP_surface_str)
