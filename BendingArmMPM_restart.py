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
from Fluid_MPM import Fluid_MPM
from Solid_MPM import Solid_MPM
from MoyashiMPM import addWater

ti.init()

CHANGE_PARAMETER = True
USER = "Hashiguchi"
USING_MACHINE = "GILES"
ADD_INFO_LIST = False
EXPORT = True
EXPORT_NUMPY = True
FOLDER_NAME = "BendingArmP1T"
DATE = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

# DATE_PRE = "2023_09_19_09_54_03/2023_09_19_11_22_23/2023_09_19_13_20_44"    # Change
# DATE_PRE = "2023_09_20_18_08_59/2023_09_20_21_40_34"
DATE_PRE = "2023_09_20_23_16_38/2023_09_21_00_33_42/2023_09_21_15_05_38/2023_09_21_17_02_06/2023_09_21_18_24_26/2023_09_21_20_11_27/2023_09_21_22_16_18/2023_09_22_00_08_13/2023_09_22_03_52_50/2023_09_22_06_04_52/2023_09_22_13_41_29"
OUTPUT_TIMES = 1100 # Change

DEBUG = False
PI = np.pi
INVERSE_NORM = False

FIX = 1
PRESS_LABEL = 2
SEARCH = 3

DONE = 1
BIG = 1.0e20
EXIST = 1
DIVERGENCE = 1

LOC_PRE = 1

if USING_MACHINE == "PC" :
    ti.init(arch=ti.cpu, default_fp=ti.f64)
    dir_mesh = "./mesh_file/"
    # if EXPORT :
    #     os.makedirs(dir_export, exist_ok=True)
    #     os.makedirs(dir_vtu, exist_ok=True)
    #     if EXPORT_NUMPY :
    #         os.makedirs(dir_numpy,  exist_ok=True)
            
elif USING_MACHINE == "CERVO" :
    ti.init(arch=ti.gpu, default_fp=ti.f64)
    dir_mesh = "./mesh_file/Moyashi/"
    # if EXPORT :
    #     print("dir_export", dir_export)
    #     os.makedirs(dir_export, exist_ok=True)
    #     os.makedirs(dir_vtu, exist_ok=True)
    #     if EXPORT_NUMPY :
    #         os.makedirs(dir_numpy, exist_ok=True)
        
elif USING_MACHINE == "GILES":
    ti.init(arch=ti.gpu, default_fp=ti.f64, device_memory_fraction=0.9)
    dir_pre = "/home/hashiguchi/mpm_simulation/result/" + FOLDER_NAME + "/" + DATE_PRE
    dir_numpy_pre = dir_pre + "/" + "numpy"
    dir_export = dir_pre + "/" + DATE
    dir_vtu = dir_export + "/" + "vtu"
    dir_numpy = dir_export + "/" + "numpy"
    dir_pre_info = dir_pre
    dir_mesh = '/home/hashiguchi/mpm_simulation/geometry/BendingArm'
    pre_info = pd.read_csv(dir_pre_info + "/" + "Information", index_col=0, header=None)
    if EXPORT :
        os.makedirs(dir_export, exist_ok=True)
        os.makedirs(dir_vtu,  exist_ok=True)
        os.makedirs(dir_numpy,  exist_ok=True)

print(pre_info)

# dim = int(pre_info.loc["dim"][LOC_PRE])
# nip = int(pre_info.loc["dim"][LOC_PRE])
dim = 3
nip = 3


mesh_name_s = pre_info.loc["mesh_name_s"][LOC_PRE]
mesh_name_f_init = pre_info.loc["mesh_name_f_init"][LOC_PRE]
mesh_name_f_add = pre_info.loc["mesh_name_f_add"][LOC_PRE]

ELE_s = pre_info.loc["element_s"][LOC_PRE]
ELE_f = pre_info.loc["element_f"][LOC_PRE]
SUR_s = pre_info.loc["surface_s"][LOC_PRE]

SCHEME = pre_info.loc["Scheme"][LOC_PRE]
ATTENUATION = bool(pre_info.loc["Attenuation"][LOC_PRE])
SLIP = bool(pre_info.loc["Slip"][LOC_PRE])

rho_s = float(pre_info.loc["rho_s"][LOC_PRE])
young_s = float(pre_info.loc["young_s"][LOC_PRE])
nu_s = float(pre_info.loc["nu_s"][LOC_PRE])
la_s = float(pre_info.loc["la_s"][LOC_PRE])
mu_s = float(pre_info.loc["mu_s"][LOC_PRE])

rho_f = float(pre_info.loc["rho_f"][LOC_PRE])
mu_f = float(pre_info.loc["mu_f"][LOC_PRE])
gamma_f = float(pre_info.loc["gamma_f"][LOC_PRE])
kappa_f = float(pre_info.loc["kappa_f"][LOC_PRE])
lambda_f = float(pre_info.loc["lambda_f"][LOC_PRE])
length_f_add = float(pre_info.loc["length_f_add"][LOC_PRE])

grav = float(pre_info.loc["grav"][LOC_PRE])
vel_add = float(pre_info.loc["vel_add"][LOC_PRE])

dt = float(pre_info.loc["dt"][LOC_PRE])
dx = float(pre_info.loc["dx"][LOC_PRE])
dx_mesh = float(pre_info.loc["dx_mesh"][LOC_PRE])
dx_mesh_f = float(pre_info.loc["dx_mesh_f"][LOC_PRE])

num_add = int(pre_info.loc["num_add"][LOC_PRE])
add_span_time_step = int(pre_info.loc["add_span_time_step"][LOC_PRE])
max_number_pre = int(pre_info.loc["max_number"][LOC_PRE])
output_span = int(pre_info.loc["output_span"][LOC_PRE])
output_span_pre = int(pre_info.loc["output_span"][LOC_PRE])
output_span_numpy = int(pre_info.loc["output_span_numpy"][LOC_PRE])
add_span_time_step = int(pre_info.loc["add_span_time_step"][LOC_PRE])


area_start = ti.Vector([
    float(pre_info.loc["area_start_x"][LOC_PRE]),
    float(pre_info.loc["area_start_y"][LOC_PRE]),
    float(pre_info.loc["area_start_z"][LOC_PRE])
])
area_end = ti.Vector([
    float(pre_info.loc["area_end_x"][LOC_PRE]),
    float(pre_info.loc["area_end_y"][LOC_PRE]),
    float(pre_info.loc["area_end_z"][LOC_PRE])
])
diri_area_start = [
    float(pre_info.loc["diri_area_start_x"][LOC_PRE]),
    float(pre_info.loc["diri_area_start_y"][LOC_PRE]),
    float(pre_info.loc["diri_area_start_z"][LOC_PRE]),
]
diri_area_end = [
    float(pre_info.loc["diri_area_end_x"][LOC_PRE]),
    float(pre_info.loc["diri_area_end_y"][LOC_PRE]),
    float(pre_info.loc["diri_area_end_z"][LOC_PRE]),
]




if CHANGE_PARAMETER :
    # pass
    # area_start = ti.Vector([-100, -20.0, -90.0])
    area_end = ti.Vector([65.0, 20.0, 50.0])
    # dx *= 1.1
    # dt *= 0.9
    # num_add *= 2
    # max_number *= 2
    # output_span = 1
    max_number = max_number_pre
    num_add = 900
    dx *= 1.0
    
    
    # num_add *= 2
    # max_number = num_add * add_span_time_step
    
        

        

msh_s = meshio.read(dir_mesh + "/" + mesh_name_s + ".msh")
msh_f_init = meshio.read(dir_mesh + "/" + mesh_name_f_init + ".msh")
msh_f_add = meshio.read(dir_mesh + "/" + mesh_name_f_add + ".msh")

err = 1.0e-5
sound_s = ti.sqrt((la_s + 2 * mu_s) / rho_s)
gi = ti.Vector([0.0, 0.0, -grav])
sound_f = np.sqrt(kappa_f / rho_f)
sound_max = sound_f if sound_f > sound_s else sound_s
dt_max = 0.1 * dx_mesh / sound_s

vel_add_vec = ti.Vector([- vel_add, 0.0, 0.0])
inv_dx = 1 / dx
box_size = area_end - area_start
nx, ny, nz = int(box_size.x * inv_dx + 1), int(box_size.y * inv_dx + 1), int(box_size.z * inv_dx + 1)
prepare_point = -1000

print("dt_max", dt_max)
print("dt", dt)

ti.data_oriented
class restartAddWater(addWater) :
    def __init__(self) :
        addWater.__init__(self,
            msh_s = msh_s,
            msh_f_init = msh_f_init,
            msh_f_add = msh_f_add,
            mesh_name_s = mesh_name_s,
            mesh_name_f_init = mesh_name_f_init,
            mesh_name_f_add = mesh_name_f_add,
            dim = dim,
            nip = nip,
            young_s = young_s,
            ELE_s = ELE_s,
            SUR_s = SUR_s,
            ELE_f = ELE_f,
            nu_s = nu_s,
            la_s = la_s,
            mu_s = mu_s,
            rho_s = rho_s,
            rho_f = rho_f,
            mu_f = mu_f,
            gamma_f = gamma_f,
            kappa_f = kappa_f,
            lambda_f = lambda_f,
            length_f_add = length_f_add,
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
            dir_numpy = dir_numpy,
            dir_export = dir_export,
            dir_vtu = dir_vtu
         )
    
    def init(self):
        self.set_init_restart()
        self.set_f_add()
        self.set_f_init()
        self.set_s_init()
        self.get_S_fix()
        self.get_F_add()
        self.get_es_press()
        self.set_esN_pN_press()
        
        if self.ELE_s == "hexahedran27" :
            self.leg_weights_roots(nip)
            self.set_tN_pN_s()
            self.cal_Ja_Ref_s()
            
        self.cal_m_p_s()
        
        self.set_info()
        self.update_info()
        self.export_info()
        self.export_program()
        self.export_calculation_domain()
        
    
    
    def set_init_restart(self):
        self.add_times[None] = int(np.load(dir_numpy_pre + "/" + "add_times_{:05d}.npy".format(OUTPUT_TIMES)))
        self.output_times[None] = OUTPUT_TIMES
        self.time_steps[None] = 0
        self.num_p_f_active[None] = self.num_p_f_init + self.add_times[None] * self.num_p_f_add
        self.protruding[None] = float(np.load(dir_numpy_pre + "/" + "protruding_{:05d}.npy".format(OUTPUT_TIMES)))
        
        print("protruding", self.protruding[None])
        
    def set_f_init(self) :
        num_active_f = np.load(dir_numpy_pre + "/" + "pos_p_f_{:05d}.npy".format(OUTPUT_TIMES)).shape[0]

        pos_p_f_np = np.zeros((self.num_p_f_all, dim), dtype=np.float32)
        vel_p_f_np = np.zeros((self.num_p_f_all, dim), dtype=np.float32)
        C_p_f_np = np.zeros((self.num_p_f_all, dim, dim), dtype=np.float32)
        sigma_p_f_np = np.zeros((self.num_p_f_all, dim, dim), dtype=np.float32)
        P_p_f_np = np.zeros((self.num_p_f_all), dtype=np.float32)
        m_p_f_np = np.zeros((self.num_p_f_all), dtype=np.float32)
        rho_p_f_np = np.zeros((self.num_p_f_all), dtype=np.float32)

        pos_p_f_np[:num_active_f] = np.load(dir_numpy_pre + "/" + "pos_p_f_{:05d}.npy".format(OUTPUT_TIMES))
        vel_p_f_np[:num_active_f] = np.load(dir_numpy_pre + "/" + "vel_p_f_{:05d}.npy".format(OUTPUT_TIMES))
        C_p_f_np[:num_active_f] = np.load(dir_numpy_pre + "/" + "C_p_f_{:05d}.npy".format(OUTPUT_TIMES))
        sigma_p_f_np[:num_active_f] = np.load(dir_numpy_pre + "/" + "sigma_p_f_{:05d}.npy".format(OUTPUT_TIMES))
        P_p_f_np[:num_active_f] = np.load(dir_numpy_pre + "/" + "P_p_f_{:05d}.npy".format(OUTPUT_TIMES))
        m_p_f_np[:num_active_f] = np.load(dir_numpy_pre + "/" + "m_p_f_{:05d}.npy".format(OUTPUT_TIMES))
        rho_p_f_np[:num_active_f] = np.load(dir_numpy_pre + "/" + "rho_p_f_{:05d}.npy".format(OUTPUT_TIMES))


        self.pos_p_f.from_numpy(pos_p_f_np)
        self.vel_p_f.from_numpy(vel_p_f_np)
        self.C_p_f.from_numpy(C_p_f_np)
        self.sigma_p_f.from_numpy(sigma_p_f_np)
        self.P_p_f.from_numpy(P_p_f_np)
        self.m_p_f.from_numpy(m_p_f_np)
        self.rho_p_f.from_numpy(rho_p_f_np)
        
    def set_s_init(self) :
        self.pos_p_s.from_numpy(np.load(dir_numpy_pre + "/" + "pos_p_s_{:05d}.npy".format(OUTPUT_TIMES)))
        self.pos_p_s_rest.from_numpy(msh_s.points)
        self.vel_p_s.from_numpy(np.load(dir_numpy_pre + "/" + "vel_p_s_{:05d}.npy".format(OUTPUT_TIMES)))
        self.C_p_s.from_numpy(np.load(dir_numpy_pre + "/" + "C_p_s_{:05d}.npy".format(OUTPUT_TIMES)))
        
        self.tN_pN_arr_s.from_numpy(msh_s.cells_dict[self.ELE_s])
        self.esN_pN_arr_s.from_numpy(msh_s.cells_dict[self.SUR_s])
        
    
    def update_info(self) :
        data_restart = {
            "DATE_PRE" : DATE_PRE,
            "output_times_restart" : OUTPUT_TIMES
        }
        self.data.update(data_restart) 
        
        
restartAddWaterObj = restartAddWater()

if __name__ == '__main__':
    restartAddWaterObj.init()
    restartAddWaterObj.main()
