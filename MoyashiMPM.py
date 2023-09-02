import meshio
import numpy as np
import sympy as sy
import taichi as ti
import datetime
import os
import pandas as pd
from pyevtk.hl import *


ti.init(arch=ti.cpu, default_fp=ti.f64)

USER = "Hashiguchi"
USING_MACHINE = "PC"
SCHEME = "MPM"
ADD_INFO_LIST = False
EXPORT = True
FOLDER_NAME = "MoyashiAddWaterP2"
PRESS_TIME_CHANGE = "CONST"
DEBUG = True
ATTENUATION = False
DONE = 1
PI = np.pi
INVERSE_NORM = False
ELEMENT_SOLID = "P2Q"
ELEMENT_FLUID = "P1Q"
SLIP = True

SOLID = 0
FLUID = 1

FIX = 1
PRESS_LABEL = 2

if EXPORT:
    DATE = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    dir_export = "./consequence" + "/" + FOLDER_NAME + "/" + DATE + "/"
    os.makedirs(dir_export, exist_ok=True)
    os.makedirs(dir_export + "/" + "vtu", exist_ok=True)


mesh_dir = "./mesh_file/"
mesh_name_s = "MoyashiTransfinite2"
mesh_name_f_init = "MoyashiWaterInit2"
mesh_name_f_add = "MoyashiWaterAdd2"

msh_s = meshio.read(mesh_dir + mesh_name_s + ".msh")
msh_f_init = meshio.read(mesh_dir + mesh_name_f_init + ".msh")
msh_f_add = meshio.read(mesh_dir + mesh_name_f_add + ".msh")


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
dx_mesh = 0.75
Z_FIX = 63.0
num_type = 2

young_s, nu_s = 4e5, 0.3
la_s, mu_s = young_s * nu_s / ((1 + nu_s) * (1 - 2*nu_s)) , young_s / (2 * (1 + nu_s))
sound_s = ti.sqrt((la_s + 2 * mu_s) / rho_s)
grav = 9.81
gi = ti.Vector([0.0, 0.0, 0.0])
length_s, width_s, height_s = 18.0, 12.0, 63.0

length_f_add, width_f_add, height_f_add = 4.5, 6.0, 6.0
rho_f = 0.9975e3
mu_f = 1.002e-3
lambda_f = 2/3 * mu_f
gamma_f = 7.0     ## water
kappa_f = 2.0e6
sound_f = np.sqrt(kappa_f / rho_f)

sound_max = sound_f if sound_f > sound_s else sound_s
dt_max = 0.1 * dx_mesh / sound_s
dt = 0.000215

num_add = 20
vel_add = 10.0
vel_add_vec = ti.Vector([0.0, 0.0, -vel_add])
add_span_time = height_f_add / np.abs(vel_add)
add_span_time_step = int(add_span_time // dt) + 1

dx = dx_mesh
inv_dx = 1 / dx
area_start = ti.Vector([-5.0, -5.0, -5.0])
area_end = ti.Vector([30.0, 17.0, 68.0])
box_size = area_end - area_start
nx, ny, nz = int(box_size.x * inv_dx + 1), int(box_size.y * inv_dx + 1), int(box_size.z * inv_dx + 1)
diri_area_start = [0.0, 0.0, 63.0]
diri_area_end = [18.0, 12.0, 63.0]
prepare_point = -1000


max_number = add_span_time_step * num_add
output_span = max_number // 1000

print("dt_max", dt_max)
print("dt", dt)

if DEBUG :
    print(msh_f_init)
    # print(msh_s)
    # print("num_p_s", msh_s.points.shape[0])
    # print("num_es", msh_s.cells_dict["quad9"].shape[0])
    # print("num_t", msh_s.cells_dict["hexahedron27"].shape[0])
    # print("interval")
    # print(msh_s.cell_data['gmsh:physical'][0].shape)
    # print(np.array(msh_s.cell_data['gmsh:physical'])[2].shape)

    # for es in range(msh_s.cells_dict["quad9"].shape[0]) :
    #     print(msh_s.cell_data['gmsh:physical'][0][es])


@ti.data_oriented
class addWater():
    def __init__(self) -> None:
        self.output_times = ti.field(dtype=ti.i32, shape=())
        self.add_times = ti.field(dtype=ti.i32, shape=())
        self.ELE_s, self.SUR_s = "hexahedron27", "quad9"
        self.ELE_f, self.SUR_f = "hexahedron", "quad"

        self.num_p_s = msh_s.points.shape[0]
        self.num_p_f_init = msh_f_init.cells_dict[self.ELE_f].shape[0]
        self.num_p_f_add = msh_f_add.cells_dict[self.ELE_f].shape[0]
        self.num_p_f = self.num_p_f_init + num_add * self.num_p_f_add
        self.num_p = self.num_p_s + self.num_p_f
        self.num_t_s, self.num_node_ele_s = msh_s.cells_dict[self.ELE_s].shape
        self.num_es_s, self.num_node_sur_s = msh_s.cells_dict[self.SUR_s].shape
        self.num_node_ele_f = msh_f_init.cells_dict[self.ELE_f].shape[1]
        self.num_gauss_s = self.num_t_s * nip**dim
        self.num_p_active = ti.field(dtype=ti.i32, shape=())
        self.num_p_active[None] = self.num_p_s + self.num_p_f_init + self.add_times[None] * self.num_p_f_add

        self.m_p_s = ti.field(dtype=float, shape=self.num_p_s)
        self.pos_p_s = ti.Vector.field(dim, dtype=float, shape=self.num_p_s, needs_grad=True)
        self.pos_p_s_rest = ti.Vector.field(dim, dtype=float, shape=self.num_p_s)
        self.vel_p_s = ti.Vector.field(dim, dtype=float, shape=self.num_p_s)
        self.d_pos_p_s = ti.Vector.field(dim, dtype=float, shape=self.num_p_s)
        self.C_p_s = ti.Matrix.field(dim, dim, dtype=float, shape=self.num_p_s)
        self.Ja_Ref_s = ti.Matrix.field(dim, dim, dtype=float, shape=(self.num_t_s * nip**dim))
        self.tN_pN_s_arr_s = ti.field(dtype=ti.i32, shape=(self.num_t_s, self.num_node_ele_s))
        self.tN_pN_s = ti.field(dtype=ti.i32, shape=(self.num_t_s, 3, 3, 3))
        self.esN_pN_arr_s = ti.field(dtype=ti.i32, shape=(self.num_es_s, self.num_node_sur_s))
        self.StrainEnergy = ti.field(dtype=float, shape=(), needs_grad=True)
        self.alpha_Dum = ti.field(dtype=float, shape=())
        self.pN_fix = ti.field(dtype=ti.i32, shape=self.num_p_s)

        self.pos_p_s.from_numpy(msh_s.points)
        self.pos_p_s_rest.from_numpy(msh_s.points)
        self.tN_pN_s_arr_s.from_numpy(msh_s.cells_dict[self.ELE_s])
        self.esN_pN_arr_s.from_numpy(msh_s.cells_dict[self.SUR_s])

        self.m_p_f = ti.field(dtype=float, shape=self.num_p_f)
        self.rho_p_f = ti.field(dtype=float, shape=self.num_p_f)
        self.P_p_f = ti.field(dtype=float, shape=self.num_p_f)
        self.pos_p_f = ti.Vector.field(dim, dtype=float, shape=self.num_p_f)
        self.d_pos_p_f = ti.Vector.field(dim, dtype=float, shape=self.num_p_f)
        self.vel_p_f = ti.Vector.field(dim, dtype=float, shape=self.num_p_f)
        self.C_p_f = ti.Matrix.field(dim, dim, dtype=float, shape=self.num_p_f)
        self.sigma_p_f = ti.Matrix.field(dim, dim, dtype=float, shape=self.num_p_f)
        self.epsilon_p_f = ti.Matrix.field(dim, dim, dtype=float, shape=self.num_p_f)
        self.L_p_f = ti.Matrix.field(dim, dim, dtype=float, shape=self.num_p_f)

        self.m_I = ti.field(dtype=float, shape=(nx, ny, nz, num_type))
        self.p_I = ti.Vector.field(dim, dtype=float, shape=(nx, ny, nz, num_type))
        self.f_F = ti.Vector.field(dim, dtype=float, shape=(nx, ny, nz))
        self.norm_S = ti.Vector.field(dim, dtype=float, shape=(nx, ny, nz))

        self.pos_p_f_add = ti.Vector.field(dim, dtype=float, shape=self.num_p_f_add)
        self.rho_p_f_add = ti.field(dtype=float, shape=self.num_p_f_add)
        self.m_p_f_add = ti.field(dtype=float, shape=self.num_p_f_add)


        self.set_f_add()
        self.set_f_init()
        self.get_diri_I()
        self.get_add_I()
        self.get_es_press()
        self.set_esN_pN_press()
        self.leg_weights_roots(nip)
        self.set_tN_pN_s()
        self.cal_Ja_Ref_s()
        self.cal_m_p_s()
        self.export_info()
        self.export_program()
        self.export_calculation_domain()

        # self.pos_gauss_press = ti.Vector.field(dim, dtype=float, shape=self.num_gauss_s_press)
        # self.norm_gauss_press = ti.Vector.field(dim, dtype=float, shape=self.num_gauss_s_press)

        # self.cal_pos_gauss_press()
        # self.export_pos_gauss_press()

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
        pos_p_f_np = np.zeros((self.num_p_f, dim), dtype=np.float64)
        for _f in range(self.num_node_ele_f) :
            f_arr = msh_f_init.cells_dict[self.ELE_f][:, _f]
            pos_p_f_np[:self.num_p_f_init] += msh_f_init.points[f_arr, :]
        pos_p_f_np /= self.num_node_ele_f
        pos_p_f_np[self.num_p_f_init:, :] = prepare_point
        self.pos_p_f.from_numpy(pos_p_f_np)

        print(self.pos_p_f)

        rho_p_f_np = np.zeros((self.num_p_f), dtype=np.float64)
        rho_p_f_np[:self.num_p_f_init] = rho_f 
        self.rho_p_f.from_numpy(rho_p_f_np)

        m_p_f_np = np.zeros((self.num_p_f), dtype=np.float64)
        m_p_f_np[:self.num_p_f_init] = rho_f * (dx_mesh / 2)**dim
        self.m_p_f.from_numpy(m_p_f_np)

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


    
    def get_es_press(self) :
        es_press_np = np.arange(0, self.num_es_s)[msh_s.cell_data['gmsh:physical'][0] == PRESS_LABEL]
        self.num_es_s_press = es_press_np.shape[0]
        self.num_gauss_s_press = self.num_es_s_press * nip**2
        self.es_press = ti.field(dtype=ti.i32, shape=self.num_es_s_press)
        self.es_press.from_numpy(es_press_np)
        self.esN_pN_press = ti.field(dtype=ti.i32, shape=(self.num_es_s_press, 3, 3))

    @ti.kernel
    def set_esN_pN_press(self):
        for _es in range(self.num_es_s_press):
            es = self.es_press[_es]
            self.esN_pN_press[_es, 0, 0] = self.esN_pN_arr_s[es, 0]
            self.esN_pN_press[_es, 2, 0] = self.esN_pN_arr_s[es, 1]
            self.esN_pN_press[_es, 2, 2] = self.esN_pN_arr_s[es, 2]
            self.esN_pN_press[_es, 0, 2] = self.esN_pN_arr_s[es, 3]
            self.esN_pN_press[_es, 1, 0] = self.esN_pN_arr_s[es, 4]
            self.esN_pN_press[_es, 2, 1] = self.esN_pN_arr_s[es, 5]
            self.esN_pN_press[_es, 1, 2] = self.esN_pN_arr_s[es, 6]
            self.esN_pN_press[_es, 0, 1] = self.esN_pN_arr_s[es, 7]
            self.esN_pN_press[_es, 1, 1] = self.esN_pN_arr_s[es, 8]




    def leg_weights_roots(self, n):
        self.gauss_x = ti.field(dtype=float, shape=nip)
        self.gauss_w = ti.field(dtype=float, shape=nip)

        self.v_Gauss = ti.field(dtype=float, shape=(3, nip))
        self.dv_Gauss = ti.field(dtype=float, shape=(3, nip))

        x = sy.Symbol('x')
        roots = sy.Poly(sy.legendre(n, x)).all_roots()  # n次ルジャンドル多項式の根を求める
        x_i = np.array([rt.evalf(20) for rt in roots], dtype=np.float64) 
        w_i = np.array([(2*(1-rt**2)/(n*sy.legendre(n-1, rt))**2).evalf(20) for rt in roots], dtype=np.float64)
        self.gauss_x.from_numpy(x_i)
        self.gauss_w.from_numpy(w_i)
        
        for m in range(n):
            self.v_Gauss[0, m] = 1 / 2 * self.gauss_x[m] * (self.gauss_x[m] - 1)
            self.v_Gauss[1, m] = - (self.gauss_x[m] + 1) * (self.gauss_x[m] - 1)
            self.v_Gauss[2, m] = 1 / 2 * (self.gauss_x[m] + 1) * self.gauss_x[m]

            self.dv_Gauss[0, m] = self.gauss_x[m] - 1 / 2
            self.dv_Gauss[1, m] = - 2 * self.gauss_x[m]
            self.dv_Gauss[2, m] = self.gauss_x[m] + 1 / 2
        
    @ti.kernel
    def set_tN_pN_s(self) :
        for t in range(self.num_t_s):
            # tN_pN_s_arr_s = msh_s.cells_dict['hexahedron27'][t]
            self.tN_pN_s[t, 0, 2, 2] = self.tN_pN_s_arr_s[t,0]
            self.tN_pN_s[t, 0, 0, 2] = self.tN_pN_s_arr_s[t,1]
            self.tN_pN_s[t, 0, 0, 0] = self.tN_pN_s_arr_s[t,2]
            self.tN_pN_s[t, 0, 2, 0] = self.tN_pN_s_arr_s[t,3]

            self.tN_pN_s[t, 2, 2, 2] = self.tN_pN_s_arr_s[t,4]
            self.tN_pN_s[t, 2, 0, 2] = self.tN_pN_s_arr_s[t,5]
            self.tN_pN_s[t, 2, 0, 0] = self.tN_pN_s_arr_s[t,6]
            self.tN_pN_s[t, 2, 2, 0] = self.tN_pN_s_arr_s[t,7]

            self.tN_pN_s[t, 0, 1, 2] = self.tN_pN_s_arr_s[t,8]
            self.tN_pN_s[t, 0, 0, 1] = self.tN_pN_s_arr_s[t,9]
            self.tN_pN_s[t, 0, 1, 0] = self.tN_pN_s_arr_s[t,10]
            self.tN_pN_s[t, 0, 2, 1] = self.tN_pN_s_arr_s[t,11]

            self.tN_pN_s[t, 2, 1, 2] = self.tN_pN_s_arr_s[t,12]
            self.tN_pN_s[t, 2, 0, 1] = self.tN_pN_s_arr_s[t,13]
            self.tN_pN_s[t, 2, 1, 0] = self.tN_pN_s_arr_s[t,14]
            self.tN_pN_s[t, 2, 2, 1] = self.tN_pN_s_arr_s[t,15]

            self.tN_pN_s[t, 1, 2, 2] = self.tN_pN_s_arr_s[t,16]
            self.tN_pN_s[t, 1, 0, 2] = self.tN_pN_s_arr_s[t,17]
            self.tN_pN_s[t, 1, 0, 0] = self.tN_pN_s_arr_s[t,18]
            self.tN_pN_s[t, 1, 2, 0] = self.tN_pN_s_arr_s[t,19]

            self.tN_pN_s[t, 1, 2, 1] = self.tN_pN_s_arr_s[t,20]
            self.tN_pN_s[t, 1, 0, 1] = self.tN_pN_s_arr_s[t,21]
            self.tN_pN_s[t, 1, 1, 2] = self.tN_pN_s_arr_s[t,22]
            self.tN_pN_s[t, 1, 1, 0] = self.tN_pN_s_arr_s[t,23]

            self.tN_pN_s[t, 0, 1, 1] = self.tN_pN_s_arr_s[t,24]
            self.tN_pN_s[t, 2, 1, 1] = self.tN_pN_s_arr_s[t,25]
            self.tN_pN_s[t, 1, 1, 1] = self.tN_pN_s_arr_s[t,26]


    @ti.kernel
    def cal_Ja_Ref_s(self):
        for g in range(self.num_gauss_s):
            t, mnl = g // (nip**3), g % (nip**3)
            m, nl = mnl // (nip**2), mnl % (nip**2)
            n, l  = nl // nip, nl % nip
            for _a1 in ti.static(range(3)):
                for _a2 in ti.static(range(3)):
                    for _a3 in ti.static(range(3)):
                        a = self.tN_pN_s[t, _a1, _a2, _a3]
                        for pd in ti.static(range(dim)):
                            self.Ja_Ref_s[g][pd, 0] += self.dv_Gauss[_a1, m] * self.v_Gauss[_a2, n] * self.v_Gauss[_a3, l] * self.pos_p_s_rest[a][pd]
                            self.Ja_Ref_s[g][pd, 1] += self.v_Gauss[_a1, m] * self.dv_Gauss[_a2, n] * self.v_Gauss[_a3, l] * self.pos_p_s_rest[a][pd]
                            self.Ja_Ref_s[g][pd, 2] += self.v_Gauss[_a1, m] * self.v_Gauss[_a2, n] * self.dv_Gauss[_a3, l] * self.pos_p_s_rest[a][pd]

    @ti.kernel
    def cal_m_p_s(self):
        for g in range(self.num_gauss_s):
            t, mnl = g // (nip**3), g % (nip**3)
            m, nl = mnl // (nip**2), mnl % (nip**2)
            n, l  = nl // nip, nl % nip
            ja_ref_s = self.Ja_Ref_s[g]
            det_ja_ref_s = ja_ref_s.determinant()
            det_ja_ref_s = ti.abs(det_ja_ref_s)
            for _a1 in ti.static(range(3)):
                for _a2 in ti.static(range(3)):
                    for _a3 in ti.static(range(3)):
                        a = self.tN_pN_s[t, _a1, _a2, _a3]
                        self.m_p_s[a] += rho_s * self.v_Gauss[_a1, m] * self.v_Gauss[_a2, n] * self.v_Gauss[_a3, l] * self.gauss_w[m] * self.gauss_w[n] * self.gauss_w[l] * det_ja_ref_s

    
    def export_info(self):
        if EXPORT:
            data = {
                "date" : DATE ,
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
                "young_s" : young_s,
                "nu_s" : nu_s,
                "rho_s" : rho_s,
                "mu_f" : mu_f,
                "lambda_f" : lambda_f,
                "kappa_f" : kappa_f,
                "rho_f" : rho_f
            }
                
            s = pd.Series(data)
            s.to_csv(export_dir + "Information", header=False)

    def export_program(self):
        with open(__file__, mode="r", encoding="utf-8") as fr:
            prog = fr.read()
        with open(export_dir + "/program.txt", mode="w") as fw:
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


    def export_Solid(self):
        cells = [
            (self.ELE_s, msh_s.cells_dict[self.ELE_s])
        ]
        mesh_ = meshio.Mesh(
            msh_s.points,
            cells,
            point_data = {
                "displacememt" : self.pos_p_s.to_numpy() - msh_s.points
            },
            cell_data = {
                # "sigma_max" : [sigma_max.to_numpy()],
                # "sigma_mu" : [sigma_mu.to_numpy()],
                # "U_ele" : [U_ele.to_numpy()]
            }
        )
        mesh_.write(dir_export + "/" + "vtu" + "/" + "SOLID{:05d}.vtu".format(self.output_times[None]))

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


    @ti.kernel
    def cal_StrainEnergy(self):
        for g in range(self.num_gauss_s):
            t, mnl = g // (nip**3), g % (nip**3)
            m, nl = mnl // (nip**2), mnl % (nip**2)
            n, l  = nl // nip, nl % nip
            ja_ref_s = self.Ja_Ref_s[g]
            det_ja_ref_s = ja_ref_s.determinant()
            det_ja_ref_s = ti.abs(det_ja_ref_s)
            inv_Ja_ref_s = ja_ref_s.inverse()
            FiJ = ti.Matrix([[0.0,0.0,0.0], [0.0,0.0,0.0], [0.0,0.0,0.0]])
            for _a1 in ti.static(range(3)):
                for _a2 in ti.static(range(3)):
                    for _a3 in ti.static(range(3)):
                        a = self.tN_pN_s[t, _a1, _a2, _a3]
                        dNadt = ti.Vector([
                            self.dv_Gauss[_a1, m] * self.v_Gauss[_a2, n] * self.v_Gauss[_a3, l], 
                            self.v_Gauss[_a1, m] * self.dv_Gauss[_a2, n] * self.v_Gauss[_a3, l],
                            self.v_Gauss[_a1, m] * self.v_Gauss[_a2, n] * self.dv_Gauss[_a3, l]
                        ])
                        dNadx = inv_Ja_ref_s @ dNadt
                        FiJ += dNadx.outer_product(self.pos_p_s[a])
            I1 = (FiJ @ FiJ.transpose()).trace()
            J = FiJ.determinant()
            element_energy = 0.5 * mu_s * (I1 - dim) - mu_s * ti.log(J) + 0.5 * la_s * ti.log(J)**2
            self.StrainEnergy[None] += element_energy * self.gauss_w[m] * self.gauss_w[n] * self.gauss_w[l] * det_ja_ref_s


    @ti.kernel 
    def cal_norm_S(self):
        for g in range(self.num_gauss_s_press):
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
    def cal_pos_gauss_press(self): 
        for g in range(self.num_gauss_s_press):
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
            self.pos_gauss_press[g] = pos_a
            self.norm_gauss_press[g] = norm

    def export_pos_gauss_press(self) :
        pos_p_np = self.pos_gauss_press.to_numpy()
        pointsToVTK(
            dir_export + "/" + "vtu" + "/" + "pos_gauss_press".format(self.output_times[None]),
            pos_p_np[:, 0].copy(),
            pos_p_np[:, 1].copy(),
            pos_p_np[:, 2].copy(),
            data = {
                "norm" : self.norm_gauss_press.to_numpy().copy()
            }
        )

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

    @ti.kernel
    def plus_p_I_by_contact(self):
        for IxIyIz in range(nx*ny*nz):
            iz, ixiy = IxIyIz // (nx*ny), IxIyIz % (nx*ny)
            iy, ix = ixiy // nx, ixiy % nx
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
            else : 
                self.pos_p_f[p - self.num_p_s] += self.d_pos_p_f[p - self.num_p_s]
                        
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
        for time_step in range(max_number):
            if time_step % 100 == 0:
                print(time_step)
            
            if time_step % add_span_time_step == 0:
                if self.add_times[None] <= num_add:
                    self.add_f()
                    self.add_times[None] += 1
                    self.num_p_active[None] += self.num_p_f_add
                    
                    
            if time_step % output_span == 0:
                print(time_step)
                if EXPORT:
                    self.export_Solid()
                    self.export_Fluid()
                    self.output_times[None] += 1
                
            with ti.Tape(self.StrainEnergy):
                self.cal_StrainEnergy()
                
            self.m_I.fill(0)
            self.p_I.fill(0)
            self.f_F.fill(0)
            self.norm_S.fill(0)
                
            self.cal_norm_S()
            self.p2g()
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




ChairBeindingObj = addWater()

if __name__ == '__main__':
    ChairBeindingObj.main()
