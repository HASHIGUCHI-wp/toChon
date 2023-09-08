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

ti.init()

RESTART = False
CHANGE_PARAMETER = True
USER = "Hashiguchi"
USING_MACHINE = "CERVO"
SCHEME = "MPM"
ADD_INFO_LIST = False
EXPORT = True
EXPORT_NUMPY = True
FOLDER_NAME = "MoyashiAddWaterP2"
DEBUG = False
PI = np.pi
INVERSE_NORM = False
# ELEMENT_SOLID = "P2Q"
# ELEMENT_FLUID = "P1Q"

FIX = 1
PRESS_LABEL = 2
SEARCH = 3

DONE = 1
BIG = 1.0e20
EXIST = 1
DIVERGENCE = 1

LOC_PRE = 1

length_s, width_s, height_s = 18.0, 12.0, 63.0
length_f_add, width_f_add, height_f_add = 4.5, 6.0, 6.0

DATE = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

if RESTART :
    DATE_PRE = "2023_09_05_12_55_06"
    OUTPUT_TIMES = 107
    dir_pre = "./consequence" + "/" + FOLDER_NAME + "/" + DATE_PRE
    dir_numpy_pre = dir_pre + "/" + "numpy"
    dir_export = dir_pre + "/" + DATE
    dir_vtu = dir_export + "/" + "vtu"
    dir_numpy = dir_export + "/" + "numpy"
    dir_pre_info = "./consequence" + "/" + FOLDER_NAME + "/" + DATE_PRE + "/"
    pre_info = pd.read_csv(dir_pre_info + "/" + "Information", index_col=0, header=None)

    mesh_name_s = pre_info.loc["mesh_name_s"][LOC_PRE]
    mesh_name_f_init = pre_info.loc["mesh_name_f_init"][LOC_PRE]
    mesh_name_f_add = pre_info.loc["mesh_name_f_add"][LOC_PRE]

    ATTENUATION = bool(pre_info.loc["Attenuation"][LOC_PRE])
    SLIP = bool(pre_info.loc["Slip"][LOC_PRE])

    rho_s = float(pre_info.loc["rho_s"][LOC_PRE])
    young_s = float(pre_info.loc["young_s"][LOC_PRE])
    nu_s = float(pre_info.loc["nu_s"][LOC_PRE])
    rho_f = float(pre_info.loc["rho_f"][LOC_PRE])
    mu_f = float(pre_info.loc["mu_f"][LOC_PRE])
    # gamma_f = float(pre_info.loc["gamma_f"][LOC_PRE])
    gamma_f = 7.0
    kappa_f = float(pre_info.loc["kappa_f"][LOC_PRE])

    dt = float(pre_info.loc["dt"][LOC_PRE])
    dx = float(pre_info.loc["dx"][LOC_PRE])
    dx_mesh = 0.75
    dx_mesh_f = 0.75 / 2
    num_add = int(pre_info.loc["num_add"][LOC_PRE])
    vel_add = float(pre_info.loc["vel_add"][LOC_PRE])
    add_span_time_step = int(pre_info.loc["add_span_time_step"][LOC_PRE])
    max_number = int(pre_info.loc["max_number"][LOC_PRE])
    output_span = int(pre_info.loc["output_span"][LOC_PRE])

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
        area_start = ti.Vector([-10., -10., -10.])
        area_end = ti.Vector([45.0, 25.0, 73.0])



else :
    dir_export = "./consequence" + "/" + FOLDER_NAME + "/" + DATE 
    dir_vtu = dir_export + "/" + "vtu"
    dir_numpy = dir_export + "/" + "numpy"

    mesh_name_s = "MoyashiTransfinite2"
    mesh_name_f_init = "MoyashiWaterInit2"
    mesh_name_f_add = "MoyashiWaterAdd2"

    ATTENUATION = False
    SLIP = True


    dx_mesh = 0.75
    dx_mesh_f = 0.75 / 2
    rho_s = 4e1
    young_s, nu_s = 4e5, 0.3
    rho_f = 0.9975e3
    mu_f = 1.002e-3
    gamma_f = 7.0     ## water
    kappa_f = 2.0e6
    dt = 0.000215

    num_add = 20
    vel_add = 10.0
    add_span_time = height_f_add / np.abs(vel_add)
    add_span_time_step = int(add_span_time // dt) + 1
    max_number = add_span_time_step * num_add
    output_span = max_number // 1000

    dx = dx_mesh
    area_start = ti.Vector([-10.0, -7.0, -40.0])
    area_end = ti.Vector([30.0, 19.0, 73.0])
    diri_area_start = [0.0, 0.0, 63.0]
    diri_area_end = [18.0, 12.0, 63.0]








if USING_MACHINE == "PC" :
    ti.init(arch=ti.cpu, default_fp=ti.f64)
    mesh_dir = "./mesh_file/"
    if EXPORT :
        os.makedirs(dir_export, exist_ok=True)
        os.makedirs(dir_vtu, exist_ok=True)
        if EXPORT_NUMPY :
            os.makedirs(dir_numpy,  exist_ok=True)
            
elif USING_MACHINE == "CERVO" :
    ti.init(arch=ti.gpu, default_fp=ti.f64)
    mesh_dir = "./mesh_file/Moyashi/"
    if EXPORT :
        os.makedirs(dir_export, exist_ok=True)
        os.makedirs(dir_vtu, exist_ok=True)
        if EXPORT_NUMPY :
            os.makedirs(dir_numpy, exist_ok=True)
        
elif USING_MACHINE == "ATLAS":
    ti.init(arch=ti.gpu, default_fp=ti.f64, device_memory_fraction=0.9)
    mesh_dir = '/home/hashiguchi/mpm_simulation/geometry/BendingArmAir/'
    if EXPORT :
        info_list_dir = '/home/hashiguchi/mpm_simulation/result/' + FOLDER_NAME
        export_dir = '/home/hashiguchi/mpm_simulation/result/' + FOLDER_NAME + "/" + DATE + "/"
        os.makedirs(export_dir, exist_ok=True)
        os.makedirs(export_dir + "vtu" + "/",  exist_ok=True)
        

msh_s = meshio.read(mesh_dir + mesh_name_s + ".msh")
msh_f_init = meshio.read(mesh_dir + mesh_name_f_init + ".msh")
msh_f_add = meshio.read(mesh_dir + mesh_name_f_add + ".msh")

ELE_s, SUR_s = "hexahedron27", "quad9"
ELE_f, SUR_f = "hexahedron", "quad"


dim = 3
nip = 3
err = 1.0e-5
num_type = 2
la_s, mu_s = young_s * nu_s / ((1 + nu_s) * (1 - 2*nu_s)) , young_s / (2 * (1 + nu_s))
sound_s = ti.sqrt((la_s + 2 * mu_s) / rho_s)
grav = 0.0
gi = ti.Vector([0.0, 0.0, 0.0])


lambda_f = 2/3 * mu_f
sound_f = np.sqrt(kappa_f / rho_f)

sound_max = sound_f if sound_f > sound_s else sound_s
dt_max = 0.1 * dx_mesh / sound_s


vel_add_vec = ti.Vector([0.0, 0.0, - vel_add])

inv_dx = 1 / dx

box_size = area_end - area_start
nx, ny, nz = int(box_size.x * inv_dx + 1), int(box_size.y * inv_dx + 1), int(box_size.z * inv_dx + 1)
prepare_point = -1000


print("dt_max", dt_max)
print("dt", dt)


@ti.data_oriented
class addWater(Solid_MPM, Fluid_MPM, Solid_P2Q):
    def __init__(self) -> None:
        self.time_steps = ti.field(dtype=ti.i32, shape=())
        self.output_times = ti.field(dtype=ti.i32, shape=())
        self.add_times = ti.field(dtype=ti.i32, shape=())
        
        self.ELE_f, self.SUR_f = ELE_f, SUR_f
        self.num_p_s = msh_s.points.shape[0]
        self.num_p_f_init = msh_f_init.cells_dict[self.ELE_f].shape[0]
        self.num_p_f_add = msh_f_add.cells_dict[self.ELE_f].shape[0]
        self.num_p_f_all = self.num_p_f_init + num_add * self.num_p_f_add
        self.num_p_f_active = ti.field(dtype=ti.i32, shape=())
        self.num_p = self.num_p_s + self.num_p_f_all
        self.num_node_ele_f = msh_f_init.cells_dict[self.ELE_f].shape[1]
        
        
        Solid_P2Q.__init__(self, 
            msh_s = msh_s,
            rho_s = rho_s,
            young_s = young_s,
            nu_s = nu_s,
            la_s = la_s,
            mu_s = mu_s,
            dt = dt,
            nip = nip,
            gi = gi
        )
        Fluid_MPM.__init__(self,
            dim = dim,
            rho_f = rho_f,
            mu_f = mu_f,
            gamma_f = gamma_f,
            kappa_f = kappa_f,
            lambda_f = lambda_f,
            dt = dt,
            dx = dx,
            nx = nx, 
            ny = ny, 
            nz = nz,
            ELE_f = ELE_f,
            area_start = area_start,
            area_end = area_end,
            gi = gi
        )
        Solid_MPM.__init__(self,
            dt = dt,
            num_p_s = self.num_p_s,
            dx = dx,
            nx = nx,
            ny = ny,
            nz = nz,
            area_start = area_start,
            area_end = area_end
        )
        


        self.exist_Ix = ti.field(dtype=ti.i32, shape=nx)
        self.exist_Iy = ti.field(dtype=ti.i32, shape=ny)
        self.exist_Iz = ti.field(dtype=ti.i32, shape=nz)
        self.domain_edge = ti.field(dtype=ti.i32, shape=(dim, 2))

        self.pos_p_f_add = ti.Vector.field(dim, dtype=float, shape=self.num_p_f_add)
        self.rho_p_f_add = ti.field(dtype=float, shape=self.num_p_f_add)
        self.m_p_f_add = ti.field(dtype=float, shape=self.num_p_f_add)

        self.exist_edge = ti.field(dtype=ti.i32, shape=())
        self.divergence = ti.field(dtype=ti.i32, shape=())

        Solid_P2Q.set_taichi_field(self)
        Solid_MPM.set_taichi_field(self)
        Fluid_MPM.set_taichi_field(self, self.num_p_f_all)
        self.set_init_restart()
        self.set_f_add()
        self.set_f_init()
        self.set_s_init()
        self.get_S_fix()
        self.get_F_add()
        self.get_es_press(msh_s)
        self.set_esN_pN_press()
        self.leg_weights_roots(nip)
        self.set_tN_pN_s()
        self.cal_Ja_Ref_s()
        self.cal_m_p_s()
        self.export_info()
        self.export_program()
        self.export_calculation_domain()


    def set_init_restart(self) :
        if RESTART :
            self.add_times[None] = int(np.load(dir_numpy_pre + "/" + "add_times_{:05d}.npy".format(OUTPUT_TIMES)))
            self.output_times[None] = OUTPUT_TIMES
            self.time_steps[None] = self.output_times[None] * output_span
        self.num_p_f_active[None] = self.num_p_f_init + self.add_times[None] * self.num_p_f_add


    def set_f_add(self):
        pos_f_add_np = np.zeros((self.num_p_f_add, dim), dtype=np.float64)
        for _f in range(self.num_node_ele_f) :
            f_arr = msh_f_add.cells_dict[self.ELE_f][:, _f]
            pos_f_add_np += msh_f_add.points[f_arr, :]
        pos_f_add_np /= self.num_node_ele_f
        self.pos_p_f_add.from_numpy(pos_f_add_np)
        self.m_p_f_add.fill(rho_f * (dx_mesh_f)**dim)
        self.rho_p_f_add.fill(rho_f)

        if DEBUG:
            print(self.pos_p_f_add)

    def set_f_init(self):
        if RESTART : 
            num_active_f = np.load(dir_numpy_pre + "/" + "pos_p_f_{:05d}.npy".format(OUTPUT_TIMES)).shape[0]

            pos_p_f_np = np.zeros((self.num_p_f, dim), dtype=np.float32)
            vel_p_f_np = np.zeros((self.num_p_f, dim), dtype=np.float32)
            C_p_f_np = np.zeros((self.num_p_f, dim, dim), dtype=np.float32)
            sigma_p_f_np = np.zeros((self.num_p_f, dim, dim), dtype=np.float32)
            P_p_f_np = np.zeros((self.num_p_f), dtype=np.float32)
            m_p_f_np = np.zeros((self.num_p_f), dtype=np.float32)
            rho_p_f_np = np.zeros((self.num_p_f), dtype=np.float32)

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

        else : 
            pos_p_f_np = np.zeros((self.num_p_f, dim), dtype=np.float64)
            for _f in range(self.num_node_ele_f) :
                f_arr = msh_f_init.cells_dict[self.ELE_f][:, _f]
                pos_p_f_np[:self.num_p_f_init] += msh_f_init.points[f_arr, :]
            pos_p_f_np /= self.num_node_ele_f
            pos_p_f_np[self.num_p_f_init:, :] = prepare_point
            self.pos_p_f.from_numpy(pos_p_f_np)

            rho_p_f_np = np.zeros((self.num_p_f), dtype=np.float64)
            rho_p_f_np[:self.num_p_f_init] = rho_f 
            self.rho_p_f.from_numpy(rho_p_f_np)

            m_p_f_np = np.zeros((self.num_p_f), dtype=np.float64)
            m_p_f_np[:self.num_p_f_init] = rho_f * (dx_mesh_f)**dim
            self.m_p_f.from_numpy(m_p_f_np)

    def set_s_init(self):
        if RESTART :
            self.pos_p_s.from_numpy(np.load(dir_numpy_pre + "/" + "pos_p_s_{:05d}.npy".format(OUTPUT_TIMES)))
            self.vel_p_s.from_numpy(np.load(dir_numpy_pre + "/" + "vel_p_s_{:05d}.npy".format(OUTPUT_TIMES)))
            self.C_p_s.from_numpy(np.load(dir_numpy_pre + "/" + "C_p_s_{:05d}.npy".format(OUTPUT_TIMES)))
        else :
            self.pos_p_s.from_numpy(msh_s.points)

        self.pos_p_s_rest.from_numpy(msh_s.points)
        self.tN_pN_arr_s.from_numpy(msh_s.cells_dict[self.ELE_s])
        self.esN_pN_arr_s.from_numpy(msh_s.cells_dict[self.SUR_s])
        

    @ti.kernel
    def add_f(self):
        for _f in range(self.num_p_f_add):
            f = _f + self.num_p_f_init + self.num_p_f_add * self.add_times[None]
            self.pos_p_f[f] = self.pos_p_f_add[_f]
            self.m_p_f[f] = self.m_p_f_add[_f]
            self.rho_p_f[f] = rho_f



    def get_S_fix(self):
        S_fix_start_x = int((diri_area_start[0] - area_start[0]) * inv_dx - 0.5)
        S_fix_start_y = int((diri_area_start[1] - area_start[1]) * inv_dx - 0.5)
        S_fix_start_z = int((diri_area_start[2] - area_start[2]) * inv_dx - 0.5)
        
        S_fix_end_x = int((diri_area_end[0] - area_start[0]) * inv_dx - 0.5) + 2
        S_fix_end_y = int((diri_area_end[1] - area_start[1]) * inv_dx - 0.5) + 2
        S_fix_end_z = int((diri_area_end[2] - area_start[2]) * inv_dx - 0.5) + 2
        
        nx_fix = S_fix_end_x - S_fix_start_x + 1
        ny_fix = S_fix_end_y - S_fix_start_y + 1
        nz_fix = S_fix_end_z - S_fix_start_z + 1
        
        S_fix_np = np.zeros(0, dtype=np.int32)
        for _Ix in range(nx_fix):
            for _Iy in range(ny_fix):
                for _Iz in range(nz_fix):
                    Ix, Iy, Iz = _Ix + S_fix_start_x, _Iy + S_fix_start_y, _Iz + S_fix_start_z
                    IxIyIz = Iz * (nx * ny) + Iy * (nx) + Ix
                    S_fix_np = np.append(S_fix_np, IxIyIz)
        
        S_fix_np = np.unique(S_fix_np)
        self.num_S_fix = S_fix_np.shape[0]
        self.S_fix = ti.field(dtype=ti.i32, shape=self.num_S_fix)
        self.S_fix.from_numpy(S_fix_np)
        print("num_S_fix", self.num_S_fix)

    
    def get_F_add(self):
        F_add_np = np.zeros(0, dtype=np.int32)
        for _p in range(msh_f_add.points.shape[0]):
            pos_p_this = msh_f_add.points[_p, :]
            base_x = int((pos_p_this[0] - area_start[0]) * inv_dx - 0.5)
            base_y = int((pos_p_this[1] - area_start[1]) * inv_dx - 0.5)
            base_z = int((pos_p_this[2] - area_start[2]) * inv_dx - 0.5)
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        I_cand = [base_x + i, base_y + j, base_z + k]
                        IxIyIz = I_cand[2] * (nx * ny) + I_cand[1] * (nx) + I_cand[0]
                        F_add_np = np.append(F_add_np, IxIyIz)
        F_add_np = np.unique(F_add_np)
        self.num_F_add = F_add_np.shape[0]
        self.F_add = ti.field(dtype=ti.i32, shape=self.num_F_add)
        self.F_add.from_numpy(F_add_np)        
        print("num_F_add", self.num_F_add)


    
    def export_info(self):
        if EXPORT:
            data = {
                "date" : DATE ,
                "dir_export" : dir_export,
                "Scheme" : SCHEME,
                "Slip" : SLIP,
                "dim" : dim,
                "mesh_name_s" : mesh_name_s,
                "mesh_name_f_init" : mesh_name_f_init,
                "mesh_name_f_add" : mesh_name_f_add,
                "Attenuation" : ATTENUATION,
                "max_number" : max_number,
                "output_span" : output_span,
                "add_span_time_step" : add_span_time_step,
                "num_add" : num_add,
                "vel_add" : vel_add,
                "element_s" : self.ELE_s,
                "element_f" : self.ELE_f,
                "surface_s" : self.SUR_s,
                "dt" : dt,
                "dx" : dx,
                "dx_mesh" : dx_mesh,
                "dx_mesh_f" : dx_mesh_f,
                "area_start_x" : area_start.x,
                "area_start_y" : area_start.y,
                "area_start_z" : area_start.z,
                "area_end_x" : area_end.x,
                "area_end_y" : area_end.y,
                "area_end_z" : area_end.z,
                "diri_area_start_x" : diri_area_start[0],
                "diri_area_start_y" : diri_area_start[1],
                "diri_area_start_z" : diri_area_start[2],
                "diri_area_end_x" : diri_area_end[0],
                "diri_area_end_y" : diri_area_end[1],
                "diri_area_end_z" : diri_area_end[2],
                "young_s" : young_s,
                "nu_s" : nu_s,
                "rho_s" : rho_s,
                "la_s" : la_s,
                "mu_s" : mu_s,
                "mu_f" : mu_f,
                "lambda_f" : lambda_f,
                "kappa_f" : kappa_f,
                "rho_f" : rho_f,
                "gamma_f" : gamma_f,
                "nip" : nip,
                "grav" : grav
            }
            
            if RESTART :
                data_restart = {
                    "Restart"  : RESTART,
                    "output_times_restart" : OUTPUT_TIMES
                }
                data.update(data_restart)    

            s = pd.Series(data)
            s.to_csv(dir_export + "/" + "Information", header=False)

    def export_program(self):
        if EXPORT :
            with open(__file__, mode="r", encoding="utf-8") as fr:
                prog = fr.read()
            with open(dir_export + "/" + "program.txt", mode="w") as fw:
                fw.write(prog)
                fw.flush()

    def export_calculation_domain(self) :
        if EXPORT:
            pos = np.array([
                [area_start.x, area_start.y, area_start.z], 
                [area_end.x, area_start.y, area_start.z],
                [area_end.x, area_end.y, area_start.z],
                [area_start.x, area_end.y, area_start.z],
                [area_start.x, area_start.y, area_end.z], 
                [area_end.x, area_start.y, area_end.z],
                [area_end.x, area_end.y, area_end.z],
                [area_start.x, area_end.y, area_end.z]
            ])
            pointsToVTK(
                dir_export + "/" + "vtu" + "/" + "Domain",
                pos[:, 0].copy(),
                pos[:, 1].copy(),
                pos[:, 2].copy()
            )


    

    def export_numpy(self):
        np.save(dir_numpy + "/" + "pos_p_s_{:05d}".format(self.output_times[None]), self.pos_p_s.to_numpy())
        np.save(dir_numpy + "/" + "vel_p_s_{:05d}".format(self.output_times[None]), self.vel_p_s.to_numpy())
        np.save(dir_numpy + "/" + "C_p_s_{:05d}".format(self.output_times[None]), self.C_p_s.to_numpy())

        num_f_end = self.num_p_f_init + self.add_times[None] * self.num_p_f_add
        np.save(dir_numpy + "/" + "pos_p_f_{:05d}".format(self.output_times[None]), self.pos_p_f.to_numpy()[:num_f_end])
        np.save(dir_numpy + "/" + "vel_p_f_{:05d}".format(self.output_times[None]), self.vel_p_f.to_numpy()[:num_f_end])
        np.save(dir_numpy + "/" + "C_p_f_{:05d}".format(self.output_times[None]), self.C_p_f.to_numpy()[:num_f_end])
        np.save(dir_numpy + "/" + "sigma_p_f_{:05d}".format(self.output_times[None]), self.sigma_p_f.to_numpy()[:num_f_end])
        np.save(dir_numpy + "/" + "P_p_f_{:05d}".format(self.output_times[None]), self.P_p_f.to_numpy()[:num_f_end])
        np.save(dir_numpy + "/" + "rho_p_f_{:05d}".format(self.output_times[None]), self.rho_p_f.to_numpy()[:num_f_end])
        np.save(dir_numpy + "/" + "m_p_f_{:05d}".format(self.output_times[None]), self.m_p_f.to_numpy()[:num_f_end])

        np.save(dir_numpy + "/" + "add_times_{:05d}".format(self.output_times[None]), self.add_times[None])




    @ti.kernel
    def diri_norm_S(self):
        for _IxIyIz in range(self.num_S_fix):
            IxIyIz = self.S_fix[_IxIyIz]
            iz, ixiy = IxIyIz // (nx*ny), IxIyIz % (nx*ny)
            iy, ix = ixiy // nx, ixiy % nx
            self.norm_S[ix, iy, iz] = ti.Vector([0.0, 0.0, 0.0])
            
        for _IxIyIz in range(self.num_F_add):
            IxIyIz = self.F_add[_IxIyIz]
            iz, ixiy = IxIyIz // (nx*ny), IxIyIz % (nx*ny)
            iy, ix = ixiy // nx, ixiy % nx
            self.norm_S[ix, iy, iz] = ti.Vector([0.0, 0.0, 0.0]) 

    @ti.kernel
    def p2g(self):
        for p in range(self.num_p_s + self.num_p_f_active[None]) :
            if p < self.num_p_s :
                Solid_MPM._p2g(self, p)
            else :
                Fluid_MPM._p2g(self, p - self.num_p_s)
    


    @ti.kernel
    def set_domain_edge(self):
        left, right = 0, nx - 1
        front, back = 0, ny - 1
        bottom, upper = 0, nz - 1
        while self.exist_Ix[left + SEARCH] == 0 : left += SEARCH
        while self.exist_Iy[front + SEARCH] == 0 : front += SEARCH
        while self.exist_Iz[bottom + SEARCH] == 0 : bottom += SEARCH
        while self.exist_Ix[right - SEARCH] == 0 : right -= SEARCH
        while self.exist_Iy[back - SEARCH] == 0 : back -= SEARCH
        while self.exist_Iz[upper - SEARCH] == 0 : upper -= SEARCH
        self.domain_edge[0, 0], self.domain_edge[0, 1] = left, right
        self.domain_edge[1, 0], self.domain_edge[1, 1] = front, back
        self.domain_edge[2, 0], self.domain_edge[2, 1] = bottom, upper
        
        if left <= 1 or front <= 1 or bottom <= 1 or right >= nx - 2 or back >= ny - 2 or upper >= nz - 2 :
            self.exist_edge[None] = EXIST 

    @ti.kernel
    def plus_p_I_by_contact(self):
        for ix, iy, iz in ti.ndrange(
            (self.domain_edge[0, 0],self.domain_edge[0, 1]),
            (self.domain_edge[1, 0],self.domain_edge[1, 1]),
            (self.domain_edge[2, 0],self.domain_edge[2, 1])
            ):

            if self.m_S[ix, iy, iz] > 0.0 or self.m_F[ix, iy, iz]  > 0.0:
                p_I_F_tilde = self.p_F[ix, iy, iz] + dt * self.f_F[ix, iy, iz]
                p_I_S_tilde = self.p_S[ix, iy, iz]
                if self.norm_S[ix, iy, iz].norm_sqr() > 0.0 : self.norm_S[ix, iy, iz] = self.norm_S[ix, iy, iz].normalized()
                norm_S_this = self.norm_S[ix, iy, iz]
                CDT1 = self.m_F[ix, iy, iz] > 0.0 and self.m_S[ix, iy, iz] > 0.0
                CDT2 = (self.m_F[ix, iy, iz] * p_I_S_tilde - self.m_S[ix, iy, iz] * p_I_F_tilde).dot(norm_S_this) > 0.0
                if CDT1 and CDT2:
                    f_I_nor = 1 / (dt * (self.m_F[ix, iy, iz] + self.m_S[ix, iy, iz])) * (self.m_S[ix, iy, iz] * self.p_F[ix, iy, iz] - self.m_F[ix, iy, iz] * self.p_S[ix, iy, iz]).dot(norm_S_this)
                    f_I_nor += 1 / (self.m_S[ix, iy, iz] + self.m_F[ix, iy, iz]) * self.m_S[ix, iy, iz] * self.f_F[ix, iy, iz].dot(norm_S_this)

                    f_S_cnt = f_I_nor * norm_S_this
                    self.p_S[ix, iy, iz] = p_I_S_tilde + dt * f_S_cnt
                    self.p_F[ix, iy, iz] = p_I_F_tilde - dt * f_S_cnt
                else :
                    self.p_S[ix, iy, iz] = p_I_S_tilde
                    self.p_F[ix, iy, iz] = p_I_F_tilde


    @ti.kernel
    def diri_p_I(self):
        for _IxIyIz in range(self.num_S_fix):
            IxIyIz = self.S_fix[_IxIyIz]
            iz, ixiy = IxIyIz // (nx*ny), IxIyIz % (nx*ny)
            iy, ix = ixiy // nx, ixiy % nx
            self.p_S[ix, iy, iz] = ti.Vector([0.0, 0.0, 0.0])
            
        for _IxIyIz in range(self.num_F_add):
            IxIyIz = self.F_add[_IxIyIz]
            iz, ixiy = IxIyIz // (nx*ny), IxIyIz % (nx*ny)
            iy, ix = ixiy // nx, ixiy % nx
            self.p_F[ix, iy, iz] = vel_add_vec * self.m_F[ix, iy, iz]


    @ti.kernel
    def g2p(self):
        for p in range(self.num_p_s + self.num_p_f_active[None]) :
            if p < self.num_p_s :
                Solid_MPM._g2p(self, p)
            else :
                Fluid_MPM._g2p(self, p - self.num_p_s)
    

    @ti.kernel
    def plus_pos_p(self) :
        for p in range(self.num_p_s + self.num_p_f_active[None]) :
            if p < self.num_p_s : 
                self.pos_p_s[p] += self.d_pos_p_s[p]
                if not(self.pos_p_s[p].x < BIG) :
                    self.divergence[None] = DIVERGENCE
            else : 
                self._plus_pos_p_f(p - self.num_p_s)
                
    @ti.kernel
    def update_p_F(self):
        for f in range(self.num_p_f_active[None]) :
            self._update_p_F(f)
            
    @ti.kernel
    def cal_L_p_f(self) :
        for f in range(self.num_p_f_active[None]) :
            self._cal_L_p_f(f)
            
    @ti.kernel
    def cal_rho_sigma_p_f(self):
        for f in range(self.num_p_f_active[None]) :
            self._cal_rho_sigma_p_f(f)
            
            

    def clear(self) :
        self.clear_grid()
        self.exist_Ix.fill(0)
        self.exist_Iy.fill(0)
        self.exist_Iz.fill(0)
        self.domain_edge.fill(0)

    @ti.kernel
    def clear_grid(self) :
        for ix, iy, iz in ti.ndrange(
            (self.domain_edge[0, 0],self.domain_edge[0, 1]),
            (self.domain_edge[1, 0],self.domain_edge[1, 1]),
            (self.domain_edge[2, 0],self.domain_edge[2, 1])
            ):
            self.m_S[ix, iy, iz] = 0.0
            self.m_F[ix, iy, iz] = 0.0
            self.p_S[ix, iy, iz] = [0.0, 0.0, 0.0]
            self.p_F[ix, iy, iz] = [0.0, 0.0, 0.0]
            self.f_F[ix, iy, iz] = [0.0, 0.0, 0.0]
            self.norm_S[ix, iy, iz] = [0.0, 0.0, 0.0]


    def whether_continue(self):
        if self.exist_edge[None] == EXIST :
            sys.exit("Error : Particles exist near the edge of the computational domain. Please extend the computational domain and restart the simulation.")

        if self.divergence[None] == DIVERGENCE:
            sys.exit("Error : The values diverged.")
            
                        
    

    def main(self):
        print("roop start")
        while self.time_steps[None] < max_number :
            if self.time_steps[None] % 100 == 0:
                print(self.time_steps[None])
            
            if self.time_steps[None] % add_span_time_step == 0:
                if self.add_times[None] <= num_add:
                    self.add_f()
                    self.add_times[None] += 1
                    self.num_p_f_active[None] += self.num_p_f_add
                    
                    
            if self.time_steps[None] % output_span == 0:
                print(self.time_steps[None])
                
                if EXPORT:
                    self.export_Solid(msh_s, dir_vtu + "/" + "SOLID{:05d}.vtu".format(self.output_times[None]))
                    self.export_Fluid(dir_vtu + "/" + "FLUID{:05d}".format(self.output_times[None]))
                    if EXPORT_NUMPY :
                        self.export_numpy()
                    self.output_times[None] += 1
                
            with ti.Tape(self.StrainEnergy):
                self.cal_StrainEnergy()
                
            self.cal_norm_S()
            self.p2g()
            self.set_domain_edge()
            self.diri_norm_S()
            self.plus_p_I_by_contact()
            self.diri_p_I()
            
            self.g2p()
            
            self.p_F.fill(0)
            self.update_p_F()
            self.diri_p_I()
            self.cal_L_p_f()
            self.cal_rho_sigma_p_f()
            self.plus_pos_p()
            self.clear()

            self.whether_continue()

            self.time_steps[None] += 1



ChairBeindingObj = addWater()

if __name__ == '__main__':
    ChairBeindingObj.main()
