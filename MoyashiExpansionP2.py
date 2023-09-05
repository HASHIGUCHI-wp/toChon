import meshio
import numpy as np
import sympy as sy
import taichi as ti
import datetime
import os
import pandas as pd
import math
import sys
from Solid_P2Q import Solid_P2Q

ti.init(arch=ti.cpu, default_fp=ti.f64)

USER = "Hashiguchi"
USING_MACHINE = "PC"
SCHEME = "FEM"
ADD_INFO_LIST = False
EXPORT = True
EXPORT_NUMPY = False
FOLDER_NAME = "MoyashiExpansionP2"
PRESS_TIME_CHANGE = "CONST"
DEBUG = False
ATTENUATION = True
DONE = 1
PI = np.pi
INVERSE_NORM = True
NAN = math.nan
BIG = 1.0e20

EXIST = 1
DIVERGENCE = 1
FIX = 1
PRESS_LABEL = 2
alpha_press = 0.3

if EXPORT:
    DATE = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    dir_export = "./consequence" + "/" + FOLDER_NAME + "/" + DATE + "/"
    dir_vtu = dir_export + "/" + "vtu"
    os.makedirs(dir_export, exist_ok=True)
    os.makedirs(dir_vtu, exist_ok=True)

    if EXPORT_NUMPY :
        dir_numpy = dir_export + "/" "numpy"
        os.makedirs(dir_numpy, exist_ok=True)


mesh_dir = "./mesh_file/"
mesh_name_s = "MoyashiTransfinite2"
msh_s = meshio.read(mesh_dir + mesh_name_s + ".msh")

if USING_MACHINE == "PC" :
    ti.init(arch=ti.cpu, default_fp=ti.f64)
    mesh_dir = "./mesh_file/"
    if EXPORT :
        info_list_dir = "./Consequence/" + FOLDER_NAME
        export_dir = "./Consequence/" + FOLDER_NAME + "/" + DATE + "/"
        os.makedirs(export_dir, exist_ok=True)
        os.makedirs(export_dir + "vtu" + "/",  exist_ok=True)
        
elif USING_MACHINE == "ATLAS":
    ti.init(arch=ti.gpu, default_fp=ti.f64, device_memory_fraction=0.9)
    mesh_dir = '/home/hashiguchi/mpm_simulation/geometry/BendingArmAir/'
    if EXPORT :
        info_list_dir = '/home/hashiguchi/mpm_simulation/result/' + FOLDER_NAME
        export_dir = '/home/hashiguchi/mpm_simulation/result/' + FOLDER_NAME + "/" + DATE + "/"
        os.makedirs(export_dir, exist_ok=True)
        os.makedirs(export_dir + "vtu" + "/",  exist_ok=True)


dim = 3
nip = 3
err = 1.0e-5
rho_s = 4e1
dx_mesh = 0.25
Z_FIX = 63.0
young_s, nu_s = 4e5, 0.3
la_s, mu_s = young_s * nu_s / ((1 + nu_s) * (1 - 2*nu_s)) , young_s / (2 * (1 + nu_s))
sound_s = ti.sqrt((la_s + 2 * mu_s) / rho_s)
grav = 9.81
gi = ti.Vector([0.0, - grav, 0.0])
Press = alpha_press * young_s

max_number = 20000
output_span = 100
dt_max = 0.1 * dx_mesh / sound_s
dt = 0.0215

print("dt_max", dt_max)
print("dt", dt)


@ti.data_oriented
class Expansion(Solid_P2Q):
    def __init__(self) -> None:
        super().__init__(
            msh_s = msh_s,
            rho_s = rho_s,
            young_s = young_s,
            nu_s = nu_s,
            la_s = la_s,
            mu_s = mu_s,
            dt = dt,
            nip = nip,
            dim = dim,
            gi = ti.Vector([0.0, 0.0, 0.0]),
            press_const = Press,
            ATTENUATION_s = ATTENUATION
        )
        self.output_times = ti.field(dtype=ti.i32, shape=())
        self.divergence = ti.field(dtype=ti.i32, shape=())

    @ti.kernel
    def set_sN_fix(self):
        for p in range(self.num_p_s):
            pos_p_z = self.pos_p_s_rest[p].z
            if ti.abs(pos_p_z - Z_FIX) < err : 
                self.sN_fix[p] = FIX

    
    def export_info(self):
        if EXPORT:
            data_base = {
                "date" : DATE ,
                "Scheme" : SCHEME,
                "mesh_name_s" : mesh_name_s,
                "max_number" : max_number,
                "output_span" : output_span,
                "dt" : dt
            }
            
            if PRESS_TIME_CHANGE == "CONST" : 
                data_press = {
                    "Press_Time_Change" : PRESS_TIME_CHANGE,
                    "Press" : Press,
                    "alpha_press" : alpha_press
                }
            # elif PRESS_TIME_CHANGE == "LINEAR" or PRESS_TIME_CHANGE == "TORIGO":
            #     data_press = {
            #         "Press_Time_Change" : PRESS_TIME_CHANGE,
            #         "Press_Max" : Press_max,   
            #         "Press_Min" : Press_min, 
            #         "alpha_press" : alpha_press,
            #         "period_step" : period_step
            #     }
            self.set_data_solid()
            data_base.update(data_press)
            data_base.update(self.data_solid)
            
            s = pd.Series(data_base)
            s.to_csv(export_dir + "Information", header=False)

    def export_program(self):
        if EXPORT:
            with open(__file__, mode="r", encoding="utf-8") as fr:
                prog = fr.read()
            with open(export_dir + "/program.txt", mode="w") as fw:
                fw.write(prog)
                fw.flush()

    

    def export_numpy(self) :
        if EXPORT_NUMPY :
            pos_p_np = self.pos_p_s.to_numpy()
            np.save(dir_numpy + "/" + "pos_p{:05d}".format(self.output_times[None]) , pos_p_np)
        

    def whether_continue(self) :
        self.whether_continue_s()
        


    def main(self):
        self.set_s_init(msh_s)
        self.get_es_press(msh_s)
        self.set_esN_pN_press()
        self.set_sN_fix()
        self.leg_weights_roots(nip)
        self.set_tN_pN()
        self.cal_Ja_Ref()
        self.cal_m_p()
        self.export_info()
        self.export_program()


        for time_step in range(max_number):

            with ti.Tape(self.StrainEnergy):
                self.cal_StrainEnergy()

            self.f_p_ext_s.fill(0)
            self.cal_Press_all()
            self.plus_f_p_ext_from_press()
            self.cal_alpha_Dum()
            self.plus_vel_pos_p()
            self.whether_continue()


            if time_step % output_span == 0 :
                print(time_step)
                print(self.pos_p_s)
                if EXPORT : 
                    self.export_Solid(msh_s, dir_vtu + "/" + "SOLID{:05d}.vtu".format(self.output_times[None]))
                    self.export_numpy()
                    self.output_times[None] += 1




ExpansionObj = Expansion()

if __name__ == '__main__':
    ExpansionObj.main()
