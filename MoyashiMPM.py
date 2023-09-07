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

ti.init(arch=ti.cpu, default_fp=ti.f64)

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

SOLID = 0
FLUID = 1

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
    dir_export = "./consequence" + "/" + FOLDER_NAME + "/" + DATE + "/"
    dir_vtu = dir_export + "/" + "vtu"
    dir_numpy = dir_export + "/" + "numpy"

    mesh_name_s = "MoyashiTransfinite2"
    mesh_name_f_init = "MoyashiWaterInit2"
    mesh_name_f_add = "MoyashiWaterAdd2"

    ATTENUATION = False
    SLIP = True


    dx_mesh = 0.75
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
    area_start = ti.Vector([-5.0, -5.0, -5.0])
    area_end = ti.Vector([30.0, 17.0, 73.0])
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
class addWater(Solid_P2Q):
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
            gi = ti.Vector([0.0, 0.0, 0.0])
        )
        self.time_steps = ti.field(dtype=ti.i32, shape=())
        self.output_times = ti.field(dtype=ti.i32, shape=())
        self.add_times = ti.field(dtype=ti.i32, shape=())
        self.ELE_f, self.SUR_f = "hexahedron", "quad"

        self.num_p_f_init = msh_f_init.cells_dict[self.ELE_f].shape[0]
        self.num_p_f_add = msh_f_add.cells_dict[self.ELE_f].shape[0]
        self.num_p_f = self.num_p_f_init + num_add * self.num_p_f_add
        self.num_p = self.num_p_s + self.num_p_f
        self.num_node_ele_f = msh_f_init.cells_dict[self.ELE_f].shape[1]
        self.num_p_active = ti.field(dtype=ti.i32, shape=())

        self.d_pos_p_s = ti.Vector.field(dim, dtype=float, shape=self.num_p_s)
        self.C_p_s = ti.Matrix.field(dim, dim, dtype=float, shape=self.num_p_s)

        self.m_p_f = ti.field(dtype=float, shape=self.num_p_f)
        self.rho_p_f = ti.field(dtype=float, shape=self.num_p_f)
        self.P_p_f = ti.field(dtype=float, shape=self.num_p_f)
        self.pos_p_f = ti.Vector.field(dim, dtype=float, shape=self.num_p_f)
        self.d_pos_p_f = ti.Vector.field(dim, dtype=float, shape=self.num_p_f)
        self.vel_p_f = ti.Vector.field(dim, dtype=float, shape=self.num_p_f)
        self.C_p_f = ti.Matrix.field(dim, dim, dtype=float, shape=self.num_p_f)
        self.sigma_p_f = ti.Matrix.field(dim, dim, dtype=float, shape=self.num_p_f)
        self.L_p_f = ti.Matrix.field(dim, dim, dtype=float, shape=self.num_p_f)

        self.m_I = ti.field(dtype=float, shape=(nx, ny, nz, num_type))
        self.p_I = ti.Vector.field(dim, dtype=float, shape=(nx, ny, nz, num_type))
        self.f_F = ti.Vector.field(dim, dtype=float, shape=(nx, ny, nz))
        self.norm_S = ti.Vector.field(dim, dtype=float, shape=(nx, ny, nz))
        self.exist_Ix = ti.field(dtype=ti.i32, shape=nx)
        self.exist_Iy = ti.field(dtype=ti.i32, shape=ny)
        self.exist_Iz = ti.field(dtype=ti.i32, shape=nz)
        self.domain_edge = ti.field(dtype=ti.i32, shape=(dim, 2))

        self.pos_p_f_add = ti.Vector.field(dim, dtype=float, shape=self.num_p_f_add)
        self.rho_p_f_add = ti.field(dtype=float, shape=self.num_p_f_add)
        self.m_p_f_add = ti.field(dtype=float, shape=self.num_p_f_add)

        self.exist_edge = ti.field(dtype=ti.i32, shape=())
        self.divergence = ti.field(dtype=ti.i32, shape=())

        self.set_init_restart()
        self.set_f_add()
        self.set_f_init()
        self.set_s_init()
        self.get_diri_I()
        self.get_add_I()
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
        self.num_p_active[None] = self.num_p_s + self.num_p_f_init + self.add_times[None] * self.num_p_f_add


    def set_f_add(self):
        pos_f_add_np = np.zeros((self.num_p_f_add, dim), dtype=np.float64)
        for _f in range(self.num_node_ele_f) :
            f_arr = msh_f_add.cells_dict[self.ELE_f][:, _f]
            pos_f_add_np += msh_f_add.points[f_arr, :]
        pos_f_add_np /= self.num_node_ele_f
        self.pos_p_f_add.from_numpy(pos_f_add_np)
        self.m_p_f_add.fill(rho_f * (dx_mesh / 2)**dim)
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
            m_p_f_np[:self.num_p_f_init] = rho_f * (dx_mesh / 2)**dim
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



    def get_diri_I(self):
        diri_ix_s = int((diri_area_start[0] - area_start[0]) * inv_dx - 0.5)
        diri_iy_s = int((diri_area_start[1] - area_start[1]) * inv_dx - 0.5)
        diri_iz_s = int((diri_area_start[2] - area_start[2]) * inv_dx - 0.5)
        
        diri_ix_e = int((diri_area_end[0] - area_start[0]) * inv_dx - 0.5) + 2
        diri_iy_e = int((diri_area_end[1] - area_start[1]) * inv_dx - 0.5) + 2
        diri_iz_e = int((diri_area_end[2] - area_start[2]) * inv_dx - 0.5) + 2
        
        nx_diri = diri_ix_e - diri_ix_s + 1
        ny_diri = diri_iy_e - diri_iy_s + 1
        nz_diri = diri_iz_e - diri_iz_s + 1
        
        diri_IxIyIz_np = np.zeros(0, dtype=np.int32)
        for _Ix in range(nx_diri):
            for _Iy in range(ny_diri):
                for _Iz in range(nz_diri):
                    Ix, Iy, Iz = _Ix + diri_ix_s, _Iy + diri_iy_s, _Iz + diri_iz_s
                    IxIyIz = Iz * (nx * ny) + Iy * (nx) + Ix
                    diri_IxIyIz_np = np.append(diri_IxIyIz_np, IxIyIz)
        
        diri_IxIyIz_np = np.unique(diri_IxIyIz_np)
        self.num_diri_I = diri_IxIyIz_np.shape[0]
        self.diri_I = ti.field(dtype=ti.i32, shape=self.num_diri_I)
        self.diri_I.from_numpy(diri_IxIyIz_np)
        print("num_diri_I", self.num_diri_I)

    
    def get_add_I(self):
        add_IxIyIz_np = np.zeros(0, dtype=np.int32)
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
                        add_IxIyIz_np = np.append(add_IxIyIz_np, IxIyIz)
        add_IxIyIz_np = np.unique(add_IxIyIz_np)
        self.num_add_I = add_IxIyIz_np.shape[0]
        self.add_I = ti.field(dtype=ti.i32, shape=self.num_add_I)
        self.add_I.from_numpy(add_IxIyIz_np)        
        print("num_add_I", self.num_add_I)


    
    def export_info(self):
        if EXPORT:
            data = {
                "date" : DATE ,
                "dir_export" : dir_export,
                "Scheme" : SCHEME,
                "Slip" : SLIP,
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
                "dt" : dt,
                "dx" : dx,
                "dx_mesh" : dx_mesh,
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
        with open(__file__, mode="r", encoding="utf-8") as fr:
            prog = fr.read()
        with open(dir_export + "/" + "program.txt", mode="w") as fw:
            fw.write(prog)
            fw.flush()

    def export_calculation_domain(self) :
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


    def export_Fluid(self):
        num_f_end = self.num_p_f_init + self.add_times[None] * self.num_p_f_add
        pos_p_np = self.pos_p_f.to_numpy()[:num_f_end, :]
        P_p_np = self.P_p_f.to_numpy()[:num_f_end]
        point_data = {"pressure": P_p_np.copy()}
        pointsToVTK(
            dir_export + "/" + "vtu" + "/" + "FLUID{:05d}".format(self.output_times[None]),
            pos_p_np[:, 0].copy(),
            pos_p_np[:, 1].copy(),
            pos_p_np[:, 2].copy(),
            data=point_data
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
    def cal_norm_S(self):
        for g in range(self.num_gauss_press):
            _es, mn = g // (nip**2), g % (nip**2)
            m, n = mn // nip, mn % nip
            k1, k2 = ti.Vector([0.0, 0.0, 0.0]), ti.Vector([0.0, 0.0, 0.0])
            pos_a = ti.Vector([0.0, 0.0, 0.0])
            for _a1 in ti.static(range(3)):
                for _a2 in ti.static(range(3)):
                    a = self.esN_pN_press[_es, _a1, _a2]
                    pos_a += self.v_Gauss[_a1, m] * self.v_Gauss[_a2, n] * self.pos_p_s[a]
                    k1 += self.dv_Gauss[_a1, m] * self.v_Gauss[_a2, n] * self.pos_p_s[a]
                    k2 += self.v_Gauss[_a1, m] * self.dv_Gauss[_a2, n] * self.pos_p_s[a]
            k3 = k1.cross(k2)
            norm = k3.normalized()
            norm *= -1.0 if INVERSE_NORM else 1.0
            base = ti.cast((pos_a - area_start) * inv_dx - 0.5, ti.i32)
            fx = (pos_a - area_start) * inv_dx - ti.cast(base, float)
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    for k in ti.static(range(3)):
                        ix, iy, iz = base.x + i, base.y + j, base.z + k
                        self.norm_S[ix, iy, iz] += w[i].x * w[j].y * w[k].z * norm



    @ti.kernel
    def diri_norm_S(self):
        for _IxIyIz in range(self.num_diri_I):
            IxIyIz = self.diri_I[_IxIyIz]
            iz, ixiy = IxIyIz // (nx*ny), IxIyIz % (nx*ny)
            iy, ix = ixiy // nx, ixiy % nx
            self.norm_S[ix, iy, iz] = ti.Vector([0.0, 0.0, 0.0])
            
        for _IxIyIz in range(self.num_add_I):
            IxIyIz = self.add_I[_IxIyIz]
            iz, ixiy = IxIyIz // (nx*ny), IxIyIz % (nx*ny)
            iy, ix = ixiy // nx, ixiy % nx
            self.norm_S[ix, iy, iz] = ti.Vector([0.0, 0.0, 0.0]) 

    @ti.kernel
    def p2g(self):
        for p in range(self.num_p_active[None]) :
            if p < self.num_p_s :
                self.p2g_SOLID(p)
            else :
                self.p2g_FLUID(p - self.num_p_s)

    @ti.func
    def p2g_SOLID(self, s : ti.int32) :
        base = ti.cast((self.pos_p_s[s] - area_start) * inv_dx - 0.5, ti.i32)
        fx = (self.pos_p_s[s] - area_start) * inv_dx - ti.cast(base, float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    ix, iy, iz = base.x + i, base.y + j, base.z + k
                    I = ti.Vector([i, j, k])
                    dist = (float(I) - fx) * dx
                    NpI = w[i].x * w[j].y * w[k].z
                    f_p_int = - self.pos_p_s.grad[s]
                    self.m_I[ix, iy, iz, SOLID] += NpI * self.m_p_s[s]
                    self.p_I[ix, iy, iz, SOLID] += NpI * (self.m_p_s[s] * (self.vel_p_s[s] + self.C_p_s[s] @ dist) + dt * f_p_int)
                    self.exist_Ix[ix], self.exist_Iy[iy], self.exist_Iz[iz] = EXIST, EXIST, EXIST

    @ti.func
    def p2g_FLUID(self, f : ti.int32) :
        base = ti.cast((self.pos_p_f[f] - area_start) * inv_dx - 0.5, ti.i32)
        fx = (self.pos_p_f[f] - area_start) * inv_dx - ti.cast(base, float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        dw = [(fx - 1.5)  * inv_dx, -2 * (fx - 1) * inv_dx, (fx - 0.5) * inv_dx]
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    ix, iy, iz = base.x + i, base.y + j, base.z + k
                    I = ti.Vector([i, j, k])
                    dist = (float(I) - fx) * dx
                    NpI = w[i].x * w[j].y * w[k].z
                    dNpIdx = ti.Vector([dw[i].x * w[j].y * w[k].z, w[i].x * dw[j].y * w[k].z, w[i].x * w[j].y * dw[k].z])
                    self.m_I[ix, iy, iz, FLUID] += NpI * self.m_p_f[f]
                    self.p_I[ix, iy, iz, FLUID] += NpI * (self.m_p_f[f] * (self.vel_p_f[f] + self.C_p_f[f] @ dist))
                    self.f_F[ix, iy, iz] += NpI * self.m_p_f[f] * gi
                    self.f_F[ix, iy, iz][0] += - self.m_p_f[f] / self.rho_p_f[f] * (self.sigma_p_f[f][0, 0] * dNpIdx[0] + self.sigma_p_f[f][0, 1] * dNpIdx[1] + self.sigma_p_f[f][0, 2] * dNpIdx[2])
                    self.f_F[ix, iy, iz][1] += - self.m_p_f[f] / self.rho_p_f[f] * (self.sigma_p_f[f][1, 0] * dNpIdx[0] + self.sigma_p_f[f][1, 1] * dNpIdx[1] + self.sigma_p_f[f][1, 2] * dNpIdx[2])
                    self.f_F[ix, iy, iz][2] += - self.m_p_f[f] / self.rho_p_f[f] * (self.sigma_p_f[f][2, 0] * dNpIdx[0] + self.sigma_p_f[f][2, 1] * dNpIdx[1] + self.sigma_p_f[f][2, 2] * dNpIdx[2])
                    self.exist_Ix[ix], self.exist_Iy[iy], self.exist_Iz[iz] = EXIST, EXIST, EXIST


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

        # for IxIyIz in range(nx*ny*nz):
        #     iz, ixiy = IxIyIz // (nx*ny), IxIyIz % (nx*ny)
        #     iy, ix = ixiy // nx, ixiy % nx

            if self.m_I[ix, iy, iz, SOLID] > 0.0 or self.m_I[ix, iy, iz, FLUID]  > 0.0:
                p_I_F_tilde = self.p_I[ix, iy, iz, FLUID] + dt * self.f_F[ix, iy, iz]
                p_I_S_tilde = self.p_I[ix, iy, iz, SOLID]
                if self.norm_S[ix, iy, iz].norm_sqr() > 0.0 : self.norm_S[ix, iy, iz] = self.norm_S[ix, iy, iz].normalized()
                norm_S_this = self.norm_S[ix, iy, iz]
                CDT1 = self.m_I[ix, iy, iz, FLUID] > 0.0 and self.m_I[ix, iy, iz, SOLID] > 0.0
                CDT2 = (self.m_I[ix, iy, iz, FLUID] * p_I_S_tilde - self.m_I[ix, iy, iz, SOLID] * p_I_F_tilde).dot(norm_S_this) > 0.0
                if CDT1 and CDT2:
                # if CDT1:
                    f_I_nor = 1 / (dt * (self.m_I[ix, iy, iz, FLUID] + self.m_I[ix, iy, iz, SOLID])) * (self.m_I[ix, iy, iz, SOLID] * self.p_I[ix, iy, iz, FLUID] - self.m_I[ix, iy, iz, FLUID] * self.p_I[ix, iy, iz, SOLID]).dot(norm_S_this)
                    f_I_nor += 1 / (self.m_I[ix, iy, iz, SOLID] + self.m_I[ix, iy, iz, FLUID]) * self.m_I[ix, iy, iz, SOLID] * self.f_F[ix, iy, iz].dot(norm_S_this)

                    f_S_cnt = f_I_nor * norm_S_this
                    self.p_I[ix, iy, iz, SOLID] = p_I_S_tilde + dt * f_S_cnt
                    self.p_I[ix, iy, iz, FLUID] = p_I_F_tilde - dt * f_S_cnt
                else :
                    self.p_I[ix, iy, iz, SOLID] = p_I_S_tilde
                    self.p_I[ix, iy, iz, FLUID] = p_I_F_tilde


    @ti.kernel
    def diri_p_I(self):
        for _IxIyIz in range(self.num_diri_I):
            IxIyIz = self.diri_I[_IxIyIz]
            iz, ixiy = IxIyIz // (nx*ny), IxIyIz % (nx*ny)
            iy, ix = ixiy // nx, ixiy % nx
            self.p_I[ix, iy, iz, SOLID] = ti.Vector([0.0, 0.0, 0.0])
            
        for _IxIyIz in range(self.num_add_I):
            IxIyIz = self.add_I[_IxIyIz]
            iz, ixiy = IxIyIz // (nx*ny), IxIyIz % (nx*ny)
            iy, ix = ixiy // nx, ixiy % nx
            self.p_I[ix, iy, iz, FLUID] = vel_add_vec * self.m_I[ix, iy, iz, FLUID]

    @ti.kernel
    def cal_alpha_Dum(self) :
        if ATTENUATION :
            uKu, uMu = 0.0, 0.0
            for p in range(self.num_p_s):
                if self.pN_fix[p] == FIX : continue
                u_this = self.pos_p_s[p] - self.pos_p_s_rest[p]
                uKu += u_this.dot(self.pos_p_s.grad[p])
                uMu += dim * self.m_p_s[p] * u_this.norm_sqr()
            self.alpha_Dum[None] = 2 * ti.sqrt(uKu / uMu) if uMu > 0.0 else 0.0

    @ti.kernel
    def g2p(self):
        for p in range(self.num_p_active[None]) :
            if p < self.num_p_s :
                self.g2p_SOLID(p)
            else :
                self.g2p_FLUID(p - self.num_p_s)

    @ti.func
    def g2p_SOLID(self, s : ti.int32):
        base = ti.cast((self.pos_p_s[s] - area_start) * inv_dx - 0.5, ti.i32)
        fx = (self.pos_p_s[s] - area_start) * inv_dx - ti.cast(base, float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_C_p, new_vel_p, new_d_pos_p = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), ti.Vector([0.0, 0.0, 0.0]), ti.Vector([0.0, 0.0, 0.0])
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    ix, iy, iz = base.x + i, base.y + j, base.z + k
                    dist = (float(ti.Vector([i, j, k])) - fx) * dx
                    NpI = w[i].x * w[j].y * w[k].z
                    vel_this = self.p_I[ix, iy, iz, SOLID] / self.m_I[ix, iy, iz, SOLID]
                    vel_this = ti.Vector([0.0, 0.0, 0.0]) if self.m_I[ix, iy, iz, SOLID] == 0.0 else vel_this
                    new_C_p += 4 * inv_dx**2 * NpI * vel_this.outer_product(dist)
                    new_vel_p += NpI * vel_this
                    new_d_pos_p += NpI * vel_this * dt
        self.C_p_s[s] = new_C_p
        self.vel_p_s[s] = new_vel_p
        self.d_pos_p_s[s] = new_d_pos_p

    @ti.func
    def g2p_FLUID(self, f : ti.i32) :
        base = ti.cast((self.pos_p_f[f] - area_start) * inv_dx - 0.5, ti.i32)
        fx = (self.pos_p_f[f] - area_start) * inv_dx - ti.cast(base, float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_C_p, new_vel_p, new_d_pos_p = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), ti.Vector([0.0, 0.0, 0.0]), ti.Vector([0.0, 0.0, 0.0])
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    ix, iy, iz = base.x + i, base.y + j, base.z + k
                    dist = (float(ti.Vector([i, j, k])) - fx) * dx
                    NpI = w[i].x * w[j].y * w[k].z
                    vel_this = self.p_I[ix, iy, iz, FLUID] / self.m_I[ix, iy, iz, FLUID]
                    vel_this = ti.Vector([0.0, 0.0, 0.0]) if self.m_I[ix, iy, iz, FLUID] == 0.0 else vel_this
                    new_C_p += 4 * inv_dx**2 * NpI * vel_this.outer_product(dist)
                    new_vel_p += NpI * vel_this
                    new_d_pos_p += NpI * vel_this * dt
        self.C_p_f[f] = new_C_p
        self.vel_p_f[f] = new_vel_p
        self.d_pos_p_f[f] = new_d_pos_p



    @ti.kernel
    def update_p_I(self):
        for f in range(self.num_p_f_init + self.add_times[None] * self.num_p_f_add):
            base = ti.cast((self.pos_p_f[f] - area_start) * inv_dx - 0.5, ti.i32)
            fx = (self.pos_p_f[f] - area_start) * inv_dx - ti.cast(base, float)
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    for k in ti.static(range(3)):
                        ix, iy, iz = base.x + i, base.y + j, base.z + k
                        dist = (float(ti.Vector([i, j, k])) - fx) * dx
                        NpI = w[i].x * w[j].y * w[k].z
                        self.p_I[ix, iy, iz, FLUID] += NpI * (self.m_p_f[f] * (self.vel_p_f[f] + self.C_p_f[f] @ dist))

    @ti.kernel
    def cal_L_p(self):
        for f in range(self.num_p_f_init + self.add_times[None] * self.num_p_f_add):
            base = ti.cast((self.pos_p_f[f] - area_start) * inv_dx - 0.5, ti.i32)
            fx = (self.pos_p_f[f] - area_start) * inv_dx - ti.cast(base, float)
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2] 
            dw = [(fx - 1.5)  * inv_dx, -2 * (fx - 1) * inv_dx, (fx - 0.5) * inv_dx]
            new_L_p_f = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    for k in ti.static(range(3)):
                        ix, iy, iz = base.x + i, base.y + j, base.z + k
                        vel_I_this = self.p_I[ix, iy, iz, FLUID] / self.m_I[ix, iy, iz, FLUID]
                        vel_I_this = [0.0, 0.0, 0.0] if self.m_I[ix, iy, iz, FLUID] == 0 else vel_I_this
                        dv = ti.Matrix([
                            [dw[i].x * w[j].y * w[k].z * vel_I_this.x, w[i].x * dw[j].y * w[k].z * vel_I_this.x, w[i].x * w[j].y * dw[k].z * vel_I_this.x],
                            [dw[i].x * w[j].y * w[k].z * vel_I_this.y, w[i].x * dw[j].y * w[k].z * vel_I_this.y, w[i].x * w[j].y * dw[k].z * vel_I_this.y],
                            [dw[i].x * w[j].y * w[k].z * vel_I_this.z, w[i].x * dw[j].y * w[k].z * vel_I_this.z, w[i].x * w[j].y * dw[k].z * vel_I_this.z]
                        ])
                        new_L_p_f += dv
            self.L_p_f[f] = new_L_p_f

    @ti.kernel
    def cal_rho_sigma_p(self):
        Iden = ti.Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        for f in range(self.num_p_f_init + self.add_times[None] * self.num_p_f_add):
            tr_Dep = self.L_p_f[f].trace() * dt
            self.rho_p_f[f] /= 1 + tr_Dep
            P_this = kappa_f * ((self.rho_p_f[f] / rho_f)**gamma_f - 1)
            epsilon_dot = 0.5 * (self.L_p_f[f] + self.L_p_f[f].transpose())
            self.sigma_p_f[f] = 2 * mu_f * epsilon_dot + (lambda_f - P_this) * Iden
            self.P_p_f[f] = P_this

    @ti.kernel
    def plus_pos_p(self) :
        for p in range(self.num_p_active[None]) :
            if p < self.num_p_s : 
                self.pos_p_s[p] += self.d_pos_p_s[p]
                if self.pos_p_s[p].x < BIG : self.divergence[None] == DIVERGENCE
            else : 
                self.pos_p_f[p - self.num_p_s] += self.d_pos_p_f[p - self.num_p_s]
                if self.pos_p_f[p - self.num_p_s].x < BIG : self.divergence[None] == DIVERGENCE

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
            self.m_I[ix, iy, iz, SOLID] = 0.0
            self.m_I[ix, iy, iz, FLUID] = 0.0
            self.p_I[ix, iy, iz, SOLID] = [0.0, 0.0, 0.0]
            self.p_I[ix, iy, iz, FLUID] = [0.0, 0.0, 0.0]
            self.f_F[ix, iy, iz] = [0.0, 0.0, 0.0]
            self.norm_S[ix, iy, iz] = [0.0, 0.0, 0.0]


    def whether_continue(self):
        if self.exist_edge[None] == EXIST :
            sys.exit("Error : Particles exist near the edge of the computational domain. Please extend the computational domain and restart the simulation.")

        if self.divergence[None] == DIVERGENCE:
            sys.exit("Error : The values diverged.")
                        
#     @ti.kernel
#     def plus_vel_pos_p_s(self):
#         if ATTENUATION :
#             beta = 0.5 * dt * self.alpha_Dum[None]
#             for p in range(self.num_p_s_s):
#                 if self.pN_fix[p] == FIX : continue
#                 f_p_init = - self.pos_p_s.grad[p]
#                 self.vel_p[p] = (1 - beta) / (1 + beta) * self.vel_p[p] + dt * (self.f_p_ext[p] + f_p_init) / (self.m_p_s[p] * (1 + beta))
#                 self.pos_p_s[p] = self.pos_p_s[p] + dt * self.vel_p[p]
                
#         else :
#             for p in range(self.num_p_s_s):
#                 if self.pN_fix[p] == FIX : continue
#                 f_p_init = - self.pos_p_s.grad[p]
#                 self.vel_p[p] += dt * (self.f_p_ext[p] + f_p_init) / self.m_p_s[p]
#                 self.pos_p_s[p] += dt * self.vel_p[p]


    def main(self):
        print("roop start")
        while self.time_steps[None] < max_number :
            if self.time_steps[None] % 100 == 0:
                print(self.time_steps[None])
            
            if self.time_steps[None] % add_span_time_step == 0:
                if self.add_times[None] <= num_add:
                    self.add_f()
                    self.add_times[None] += 1
                    self.num_p_active[None] += self.num_p_f_add
                    
                    
            if self.time_steps[None] % output_span == 0:
                print(self.time_steps[None])
                if EXPORT:
                    self.export_Solid(msh_s, dir_vtu + "/" + "SOLID{:05d}.vtu".format(self.output_times[None]))
                    self.export_Fluid()
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
            
            self.p_I.fill(0)
            self.update_p_I()
            self.diri_p_I()
            self.cal_L_p()
            self.cal_rho_sigma_p()
            self.plus_pos_p()
            self.clear()

            self.whether_continue()

            self.time_steps[None] += 1




ChairBeindingObj = addWater()

if __name__ == '__main__':
    ChairBeindingObj.main()
