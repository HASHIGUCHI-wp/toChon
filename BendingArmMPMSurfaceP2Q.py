import meshio
import numpy as np
import sympy as sy
import taichi as ti
import datetime
import os
import pandas as pd
from pyevtk.hl import *
import sys
from Solid_P2Q import Solid_P2Q
from Solid_P1T import Solid_P1T
from Fluid_MPM import Fluid_MPM
from Solid_MPM import Solid_MPM
from addWaterSurfaceP2Q import addWaterSurfaceP2Q

ti.init()

USER = "Hashiguchi"
USING_MACHINE = "GILES"
SCHEME = "MPMSurface"
ADD_INFO_LIST = False
EXPORT = True
EXPORT_NUMPY = True
FOLDER_NAME = "BendingArmP1T"
DEBUG = False
PI = np.pi
INVERSE_NORM = False
WEAK_S = True

FIX = 1
SEARCH = 3

DONE = 1
BIG = 1.0e20
EXIST = 1
DIVERGENCE = 1

ATTENUATION = True
SLIP = True

length_s, width_s, height_s = 62.0, 15.0, 19.0
length_f_add, width_f_add, height_f_add = 3.0, 3.0, 3.0

ELE_s, SUR_s = "hexahedron27", "quad9"
ELE_f, SUR_f = "hexahedron", "quad"
# ELE_s, SUR_s = "tetra", "triangle"

if ELE_s == "hexahedron27" :
    # mesh_name_s = "MoyashiTransfinite2"
    # FOLDER_NAME = "BendingArmP2Q"
    FOLDER_NAME = "BendingArmSurfaceP2Q"
    mesh_name_s = "BendingArmTransfiniteShort"
    
elif ELE_s == "tetra" :
    mesh_name_s = "BendingArmTetraSize0.35-0.4"
    mesh_name_s = "BendingArmTetraSize0.35-0.4_ver2"
    FOLDER_NAME = "BendingArmP1T"
    

mesh_name_f_init = "initWaterQuad3"
mesh_name_f_add = "addWaterQuad3"

DATE = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')


dx_mesh = 1.0
dx_mesh_f = 0.25


rho_s = 1068.72
# rho_s = 1.0
young_s, nu_s = 82737.21, 0.36

rho_f = 0.9975e3
mu_f = 1.002e-3
gamma_f = 7.0     ## water
kappa_f = 2.0e6
dt = 0.000215
dt = 0.00215

num_add = 100
vel_add = 5.0
add_span_time = length_f_add / np.abs(vel_add)
add_span_time_step = int(add_span_time // dt) + 1
max_number = add_span_time_step * num_add
max_number *= 2
output_span = max_number // 100
# output_span = 1

output_span_numpy = output_span * 5

dx = dx_mesh_f * 2
area_start = ti.Vector([-10.0, -10.0, -80.0])
area_end = ti.Vector([80.0, 25.0, 30.0])

diri_area_start = [0.0, 0.0, 0.0]
diri_area_end = [0.0, width_s, height_s]





if USING_MACHINE == "PC" or USING_MACHINE == "CERVO":
    dir_mesh = "./mesh_file"
    dir_export = "./consequence" + "/" + FOLDER_NAME + "/" + DATE 
    dir_vtu = dir_export + "/" + "vtu"
    dir_numpy = dir_export + "/" + "numpy"
    if EXPORT :
        os.makedirs(dir_export, exist_ok=True)
        os.makedirs(dir_vtu, exist_ok=True)
        if EXPORT_NUMPY :
            os.makedirs(dir_numpy, exist_ok=True)
    ti.init(arch=ti.cpu, default_fp=ti.f64)
            
elif USING_MACHINE == "CERVO" :
    dir_mesh = "./mesh_file/Moyashi"
    dir_export = "./consequence" + "/" + FOLDER_NAME + "/" + DATE 
    dir_vtu = dir_export + "/" + "vtu"
    dir_numpy = dir_export + "/" + "numpy"
    if EXPORT :
        os.makedirs(dir_export, exist_ok=True)
        os.makedirs(dir_vtu, exist_ok=True)
        if EXPORT_NUMPY :
            os.makedirs(dir_numpy, exist_ok=True)
    ti.init(arch=ti.gpu, default_fp=ti.f64)
        
elif USING_MACHINE == "GILES":
    dir_mesh = '/home/hashiguchi/mpm_simulation/geometry/BendingArm'
    dir_export = '/home/hashiguchi/mpm_simulation/result/' + FOLDER_NAME + "/" + DATE
    dir_vtu = dir_export + "/" + "vtu"
    dir_numpy = dir_export + "/" + "numpy"
    if EXPORT :
        os.makedirs(dir_export, exist_ok=True)
        os.makedirs(dir_vtu,  exist_ok=True)
        os.makedirs(dir_numpy, exist_ok=True)
    ti.init(arch=ti.gpu, default_fp=ti.f64, device_memory_fraction=0.9)
        

msh_s = meshio.read(dir_mesh + "/" + mesh_name_s + ".msh")
msh_f_init = meshio.read(dir_mesh + "/" + mesh_name_f_init + ".msh")
msh_f_add = meshio.read(dir_mesh + "/" + mesh_name_f_add + ".msh")




dim = 3
nip = 3
sip = 6
err = 1.0e-5
num_type = 2
la_s, mu_s = young_s * nu_s / ((1 + nu_s) * (1 - 2*nu_s)) , young_s / (2 * (1 + nu_s))
eta1_s, eta2_s = 0.01 * la_s, 0.01 * mu_s
# eta1_s, eta2_s = 0.0 * la_s, 0.0 * mu_s
sound_s = ti.sqrt((la_s + 2 * mu_s) / rho_s)
grav = 0.0
gi = ti.Vector([-grav, 0.0, 0.0])


lambda_f = 2/3 * mu_f
sound_f = np.sqrt(kappa_f / rho_f)

sound_max = sound_f if sound_f > sound_s else sound_s
dt_max = 0.1 * dx_mesh / sound_s


vel_add_vec = ti.Vector([vel_add, 0.0, 0.0])

inv_dx = 1 / dx

box_size = area_end - area_start
nx, ny, nz = int(box_size.x * inv_dx + 1), int(box_size.y * inv_dx + 1), int(box_size.z * inv_dx + 1)
prepare_point = -1000


print("dt_max", dt_max)
print("dt", dt)


ChairBeindingObj = addWaterSurfaceP2Q(
msh_s = msh_s,
msh_f_init = msh_f_init,
msh_f_add = msh_f_add,
mesh_name_s = mesh_name_s,
mesh_name_f_init = mesh_name_f_init,
mesh_name_f_add = mesh_name_f_add,
dim = dim,
nip = nip,
sip = sip,
young_s = young_s,
ELE_s = ELE_s,
SUR_s = SUR_s,
ELE_f = ELE_f,
nu_s = nu_s,
la_s = la_s,
mu_s = mu_s,
rho_s = rho_s,
eta1_s= eta1_s,
eta2_s= eta2_s,
rho_f = rho_f,
mu_f = mu_f,
gamma_f = gamma_f,
kappa_f = kappa_f,
lambda_f = lambda_f,
length_f_add = height_f_add,
dt = dt,
dx = dx,
nx = nx,
ny = ny,
nz = nz,
gi = gi,
grav = grav,
prepare_point = prepare_point,
area_start = area_start,
area_end = area_end,
num_add = num_add,
dx_mesh = dx_mesh,
dx_mesh_f = dx_mesh_f,
diri_area_start = diri_area_start,
diri_area_end = diri_area_end,
vel_add_vec = vel_add_vec,
vel_add = vel_add,
max_number = max_number,
output_span = output_span,
output_span_numpy = output_span_numpy,
add_span_time_step = add_span_time_step,
ATTENUATION_s = ATTENUATION,
EXPORT = EXPORT,
EXPORT_NUMPY = EXPORT_NUMPY,
SEARCH = SEARCH,
SCHEME = SCHEME,
SLIP = SLIP,
DATE = DATE,
WEAK_S= WEAK_S,
dir_numpy = dir_numpy,
dir_export = dir_export,
dir_vtu = dir_vtu
)

ChairBeindingObj.init()
ChairBeindingObj.main()
