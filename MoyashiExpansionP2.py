import meshio
import numpy as np
import sympy as sy
import taichi as ti
import datetime
import os
import pandas as pd
import math
import sys

ti.init(arch=ti.cpu, default_fp=ti.f64)

USER = "Hashiguchi"
USING_MACHINE = "PC"
SCHEME = "FEM"
ADD_INFO_LIST = False
EXPORT = False
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
    os.makedirs(dir_export, exist_ok=True)
    os.makedirs(dir_export + "/" + "vtu", exist_ok=True)

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
la, mu = young_s * nu_s / ((1 + nu_s) * (1 - 2*nu_s)) , young_s / (2 * (1 + nu_s))
sound_s = ti.sqrt((la + 2 * mu) / rho_s)
grav = 9.81
gi = ti.Vector([0.0, - grav, 0.0])
Press = alpha_press * young_s

max_number = 20000
output_span = 100
dt_max = 0.1 * dx_mesh / sound_s
dt = 0.000215
# dt = 0.0215

print("dt_max", dt_max)
print("dt", dt)

if DEBUG :
    print(msh_s)
    print("num_p", msh_s.points.shape[0])
    print("num_es", msh_s.cells_dict["quad9"].shape[0])
    print("num_t", msh_s.cells_dict["hexahedron27"].shape[0])
    print("interval")
    # print(msh_s.cell_data['gmsh:physical'][0].shape)
    # print(np.array(msh_s.cell_data['gmsh:physical'])[2].shape)

    # for es in range(msh_s.cells_dict["quad9"].shape[0]) :
    #     print(msh_s.cell_data['gmsh:physical'][0][es])


@ti.data_oriented
class Expansion():
    def __init__(self) -> None:
        self.output_times = ti.field(dtype=ti.i32, shape=())
        self.ELE, self.SUR = "hexahedron27", "quad9"
        self.num_p = msh_s.points.shape[0]
        self.num_t, self.num_node_ele = msh_s.cells_dict[self.ELE].shape
        self.num_es, self.num_node_sur = msh_s.cells_dict[self.SUR].shape
        self.num_gauss = self.num_t * nip**dim
        self.m_p = ti.field(dtype=float, shape=self.num_p)
        self.pos_p = ti.Vector.field(dim, dtype=float, shape=self.num_p, needs_grad=True)
        self.pos_p_rest = ti.Vector.field(dim, dtype=float, shape=self.num_p)
        self.vel_p = ti.Vector.field(dim, dtype=float, shape=self.num_p)
        self.f_p_ext = ti.Vector.field(dim, dtype=float, shape=self.num_p)
        self.Ja_Ref = ti.Matrix.field(dim, dim, dtype=float, shape=(self.num_t * nip**dim))
        self.tN_pN_arr = ti.field(dtype=ti.i32, shape=(self.num_t, self.num_node_ele))
        self.tN_pN = ti.field(dtype=ti.i32, shape=(self.num_t, 3, 3, 3))
        self.esN_pN_arr = ti.field(dtype=ti.i32, shape=(self.num_es, self.num_node_sur))
        self.StrainEnergy = ti.field(dtype=float, shape=(), needs_grad=True)
        self.alpha_Dum = ti.field(dtype=float, shape=())

        self.pN_fix = ti.field(dtype=ti.i32, shape=self.num_p)

        self.pos_p.from_numpy(msh_s.points)
        self.pos_p_rest.from_numpy(msh_s.points)
        self.tN_pN_arr.from_numpy(msh_s.cells_dict[self.ELE])
        self.esN_pN_arr.from_numpy(msh_s.cells_dict[self.SUR])

        self.divergence = ti.field(dtype=ti.i32, shape=())

        
        self.get_es_press()
        self.set_esN_pN_press()
        self.set_pN_fix()
        self.leg_weights_roots(nip)
        self.set_tN_pN()
        self.cal_Ja_Ref()
        self.cal_m_p()
        self.export_info()
        self.export_program()
        


        if DEBUG:
            self.SUM_AREA = ti.field(dtype=float, shape=())
            # self.cal_SUM_AREA()
            self.cal_f_p_ext()


            # print("SUM_AREA", self.SUM_AREA[None])
            # print("SUM_AREA_ANA", 2 * (6 * 12 + 6 * 12 + 12 * 12) + 2 * (6 * 45 + 4.5 * 45))

            # Sum = 0.0
            # for p in range(self.num_p) :
            #     Sum += self.m_p[p]

            # Sum_f = 0.0
            # for p in range(self.num_p):
            #     if ti.abs(self.pos_p_rest[p].z - 3.0) < err :
            #         Sum_f += self.f_p_ext[p].z

            # print(Sum_f)
            # print(- (6 * 12 + 6 * 45) * PRESS)
            # print(- (6 * 12) * PRESS)
            # print(- 6 * 12 * PRESS)


    
    def get_es_press(self) :
        es_press_np = np.arange(0, self.num_es)[msh_s.cell_data['gmsh:physical'][0] == PRESS_LABEL]
        self.num_es_press = es_press_np.shape[0]
        self.num_gauss_press = self.num_es_press * nip**2
        self.es_press = ti.field(dtype=ti.i32, shape=self.num_es_press)
        self.es_press.from_numpy(es_press_np)
        self.esN_pN_press = ti.field(dtype=ti.i32, shape=(self.num_es_press, 3, 3))

    @ti.kernel
    def set_esN_pN_press(self):
        for _es in range(self.num_es_press):
            es = self.es_press[_es]
            self.esN_pN_press[_es, 0, 0] = self.esN_pN_arr[es, 0]
            self.esN_pN_press[_es, 2, 0] = self.esN_pN_arr[es, 1]
            self.esN_pN_press[_es, 2, 2] = self.esN_pN_arr[es, 2]
            self.esN_pN_press[_es, 0, 2] = self.esN_pN_arr[es, 3]
            self.esN_pN_press[_es, 1, 0] = self.esN_pN_arr[es, 4]
            self.esN_pN_press[_es, 2, 1] = self.esN_pN_arr[es, 5]
            self.esN_pN_press[_es, 1, 2] = self.esN_pN_arr[es, 6]
            self.esN_pN_press[_es, 0, 1] = self.esN_pN_arr[es, 7]
            self.esN_pN_press[_es, 1, 1] = self.esN_pN_arr[es, 8]

        # es0 = self.es_press[0]
        # p0 = self.esN_pN_arr[es0, 0]
        # p1 = self.esN_pN_arr[es0, 1]
        # p2 = self.esN_pN_arr[es0, 2]
        # p3 = self.esN_pN_arr[es0, 3]
        # p4 = self.esN_pN_arr[es0, 4]
        # p5 = self.esN_pN_arr[es0, 5]
        # p6 = self.esN_pN_arr[es0, 6]
        # p7 = self.esN_pN_arr[es0, 7]
        # p8 = self.esN_pN_arr[es0, 8]

        # for _p in range(9):
        #     p = self.esN_pN_arr[es0, _p]
        #     print(self.pos_p_rest[p])

        # for _es in range(self.num_es_press):
        #     es = self.es_press[_es]



    @ti.kernel
    def set_pN_fix(self):
        for p in range(self.num_p):
            pos_p_z = self.pos_p_rest[p].z
            if ti.abs(pos_p_z - Z_FIX) < err : 
                self.pN_fix[p] = FIX

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
    def set_tN_pN(self) :
        for t in range(self.num_t):
            # tN_pN_arr = msh_s.cells_dict['hexahedron27'][t]
            self.tN_pN[t, 0, 2, 2] = self.tN_pN_arr[t,0]
            self.tN_pN[t, 0, 0, 2] = self.tN_pN_arr[t,1]
            self.tN_pN[t, 0, 0, 0] = self.tN_pN_arr[t,2]
            self.tN_pN[t, 0, 2, 0] = self.tN_pN_arr[t,3]

            self.tN_pN[t, 2, 2, 2] = self.tN_pN_arr[t,4]
            self.tN_pN[t, 2, 0, 2] = self.tN_pN_arr[t,5]
            self.tN_pN[t, 2, 0, 0] = self.tN_pN_arr[t,6]
            self.tN_pN[t, 2, 2, 0] = self.tN_pN_arr[t,7]

            self.tN_pN[t, 0, 1, 2] = self.tN_pN_arr[t,8]
            self.tN_pN[t, 0, 0, 1] = self.tN_pN_arr[t,9]
            self.tN_pN[t, 0, 1, 0] = self.tN_pN_arr[t,10]
            self.tN_pN[t, 0, 2, 1] = self.tN_pN_arr[t,11]

            self.tN_pN[t, 2, 1, 2] = self.tN_pN_arr[t,12]
            self.tN_pN[t, 2, 0, 1] = self.tN_pN_arr[t,13]
            self.tN_pN[t, 2, 1, 0] = self.tN_pN_arr[t,14]
            self.tN_pN[t, 2, 2, 1] = self.tN_pN_arr[t,15]

            self.tN_pN[t, 1, 2, 2] = self.tN_pN_arr[t,16]
            self.tN_pN[t, 1, 0, 2] = self.tN_pN_arr[t,17]
            self.tN_pN[t, 1, 0, 0] = self.tN_pN_arr[t,18]
            self.tN_pN[t, 1, 2, 0] = self.tN_pN_arr[t,19]

            self.tN_pN[t, 1, 2, 1] = self.tN_pN_arr[t,20]
            self.tN_pN[t, 1, 0, 1] = self.tN_pN_arr[t,21]
            self.tN_pN[t, 1, 1, 2] = self.tN_pN_arr[t,22]
            self.tN_pN[t, 1, 1, 0] = self.tN_pN_arr[t,23]

            self.tN_pN[t, 0, 1, 1] = self.tN_pN_arr[t,24]
            self.tN_pN[t, 2, 1, 1] = self.tN_pN_arr[t,25]
            self.tN_pN[t, 1, 1, 1] = self.tN_pN_arr[t,26]

    # @ti.kernel
    # def cal_Ja_Ref_gauss(self):
    #     for g in range(self.num_gauss_press):



    @ti.kernel
    def cal_Ja_Ref(self):
        for g in range(self.num_gauss):
            t, mnl = g // (nip**3), g % (nip**3)
            m, nl = mnl // (nip**2), mnl % (nip**2)
            n, l  = nl // nip, nl % nip
            for _a1 in ti.static(range(3)):
                for _a2 in ti.static(range(3)):
                    for _a3 in ti.static(range(3)):
                        a = self.tN_pN[t, _a1, _a2, _a3]
                        for pd in ti.static(range(dim)):
                            self.Ja_Ref[g][pd, 0] += self.dv_Gauss[_a1, m] * self.v_Gauss[_a2, n] * self.v_Gauss[_a3, l] * self.pos_p_rest[a][pd]
                            self.Ja_Ref[g][pd, 1] += self.v_Gauss[_a1, m] * self.dv_Gauss[_a2, n] * self.v_Gauss[_a3, l] * self.pos_p_rest[a][pd]
                            self.Ja_Ref[g][pd, 2] += self.v_Gauss[_a1, m] * self.v_Gauss[_a2, n] * self.dv_Gauss[_a3, l] * self.pos_p_rest[a][pd]

    @ti.kernel
    def cal_m_p(self):
        for g in range(self.num_gauss):
            t, mnl = g // (nip**3), g % (nip**3)
            m, nl = mnl // (nip**2), mnl % (nip**2)
            n, l  = nl // nip, nl % nip
            ja_ref = self.Ja_Ref[g]
            det_ja_ref = ja_ref.determinant()
            det_ja_ref = ti.abs(det_ja_ref)
            for _a1 in ti.static(range(3)):
                for _a2 in ti.static(range(3)):
                    for _a3 in ti.static(range(3)):
                        a = self.tN_pN[t, _a1, _a2, _a3]
                        self.m_p[a] += rho_s * self.v_Gauss[_a1, m] * self.v_Gauss[_a2, n] * self.v_Gauss[_a3, l] * self.gauss_w[m] * self.gauss_w[n] * self.gauss_w[l] * det_ja_ref

    
    def export_info(self):
        if EXPORT:
            data = {
                "date" : DATE ,
                "Scheme" : SCHEME,
                "mesh_name_s" : mesh_name_s,
                "Attenuation" : ATTENUATION,
                "max_number" : max_number,
                "output_span" : output_span,
                "element" : self.ELE,
                "surface"  : self.SUR,
                "dt" : dt,
                "young_s" : young_s,
                "nu_s" : nu_s,
                "rho_s" : rho_s
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
                
            data.update(data_press)
            
            s = pd.Series(data)
            s.to_csv(export_dir + "Information", header=False)

    def export_program(self):
        if EXPORT:
            with open(__file__, mode="r", encoding="utf-8") as fr:
                prog = fr.read()
            with open(export_dir + "/program.txt", mode="w") as fw:
                fw.write(prog)
                fw.flush()

    def export_Solid(self):
        cells = [
            (self.ELE, msh_s.cells_dict[self.ELE])
        ]
        mesh_ = meshio.Mesh(
            msh_s.points,
            cells,
            point_data = {
                "displacememt" : self.pos_p.to_numpy() - msh_s.points
            },
            cell_data = {
                # "sigma_max" : [sigma_max.to_numpy()],
                # "sigma_mu" : [sigma_mu.to_numpy()],
                # "U_ele" : [U_ele.to_numpy()]
            }
        )
        mesh_.write(dir_export + "/" + "vtu" + "/" + "SOLID{:05d}.vtu".format(self.output_times[None]))

    def export_numpy(self) :
        if EXPORT_NUMPY :
            pos_p_np = self.pos_p.to_numpy()
            np.save(dir_numpy + "/" + "pos_p{:05d}".format(self.output_times[None]) , pos_p_np)
            if np.any(pos_p_np == NAN) :
                print("ERROR")



    @ti.kernel
    def cal_StrainEnergy(self):
        for g in range(self.num_gauss):
            t, mnl = g // (nip**3), g % (nip**3)
            m, nl = mnl // (nip**2), mnl % (nip**2)
            n, l  = nl // nip, nl % nip
            ja_ref = self.Ja_Ref[g]
            det_ja_ref = ja_ref.determinant()
            det_ja_ref = ti.abs(det_ja_ref)
            inv_Ja_ref = ja_ref.inverse()
            FiJ = ti.Matrix([[0.0,0.0,0.0], [0.0,0.0,0.0], [0.0,0.0,0.0]])
            for _a1 in ti.static(range(3)):
                for _a2 in ti.static(range(3)):
                    for _a3 in ti.static(range(3)):
                        a = self.tN_pN[t, _a1, _a2, _a3]
                        dNadt = ti.Vector([
                            self.dv_Gauss[_a1, m] * self.v_Gauss[_a2, n] * self.v_Gauss[_a3, l], 
                            self.v_Gauss[_a1, m] * self.dv_Gauss[_a2, n] * self.v_Gauss[_a3, l],
                            self.v_Gauss[_a1, m] * self.v_Gauss[_a2, n] * self.dv_Gauss[_a3, l]
                        ])
                        dNadx = inv_Ja_ref @ dNadt
                        FiJ += dNadx.outer_product(self.pos_p[a])
            I1 = (FiJ @ FiJ.transpose()).trace()
            J = FiJ.determinant()
            element_energy = 0.5 * mu * (I1 - dim) - mu * ti.log(J) + 0.5 * la * ti.log(J)**2
            self.StrainEnergy[None] += element_energy * self.gauss_w[m] * self.gauss_w[n] * self.gauss_w[l] * det_ja_ref

    @ti.kernel
    def cal_SUM_AREA(self):
        for _es in range(self.num_es_press):
            area = 0.0
            for m in ti.static(range(nip)):
                for n in ti.static(range(nip)):
                    k1, k2 = ti.Vector([0.0, 0.0, 0.0]), ti.Vector([0.0, 0.0, 0.0])
                    for _a1 in ti.static(range(3)):
                        for _a2 in ti.static(range(3)):
                            a = self.esN_pN_press[_es, _a1, _a2]
                            # print(self.pos_p[a])
                            k1 += self.dv_Gauss[_a1, m] * self.v_Gauss[_a2, n] * self.pos_p[a]
                            k2 += self.v_Gauss[_a1, m] * self.dv_Gauss[_a2, n] * self.pos_p[a]
                    k3 = k1.cross(k2)
                    J = k3.norm()
                    area += J * self.gauss_w[m] * self.gauss_w[n]
            self.SUM_AREA[None] += area
            print(area)


    @ti.kernel
    def cal_f_p_ext(self):
        for g in range(self.num_gauss_press):
            _es, mn = g // (nip**2), g % (nip**2)
            m, n = mn // nip, mn % nip
            k1, k2 = ti.Vector([0.0, 0.0, 0.0]), ti.Vector([0.0, 0.0, 0.0])
            pos_a = ti.Vector([0.0, 0.0, 0.0])
            for _a1 in ti.static(range(3)):
                for _a2 in ti.static(range(3)):
                    a = self.esN_pN_press[_es, _a1, _a2]
                    # print(self.pos_p[a])
                    pos_a += self.v_Gauss[_a1, m] * self.v_Gauss[_a2, n] * self.pos_p[a]
                    k1 += self.dv_Gauss[_a1, m] * self.v_Gauss[_a2, n] * self.pos_p[a]
                    k2 += self.v_Gauss[_a1, m] * self.dv_Gauss[_a2, n] * self.pos_p[a]
            k3 = k1.cross(k2)
            J = k3.norm()
            norm = k3.normalized()
            norm *= -1.0 if INVERSE_NORM else 1.0
            for _a1 in ti.static(range(3)):
                for _a2 in ti.static(range(3)):
                    a = self.esN_pN_press[_es, _a1, _a2]
                    self.f_p_ext[a] += Press * norm * self.v_Gauss[_a1, m] * self.v_Gauss[_a2, n] * J * self.gauss_w[m] * self.gauss_w[n]
            # self.SUM_AREA[None] += J * self.gauss_w[m] * self.gauss_w[n]


    @ti.kernel
    def cal_alpha_Dum(self) :
        if ATTENUATION :
            uKu, uMu = 0.0, 0.0
            for p in range(self.num_p):
                if self.pN_fix[p] == FIX : continue
                u_this = self.pos_p[p] - self.pos_p_rest[p]
                uKu += u_this.dot(self.pos_p.grad[p])
                uMu += dim * self.m_p[p] * u_this.norm_sqr()
            self.alpha_Dum[None] = 2 * ti.sqrt(uKu / uMu) if uMu > 0.0 else 0.0

                        
    @ti.kernel
    def plus_vel_pos_p(self):
        if ATTENUATION :
            beta = 0.5 * dt * self.alpha_Dum[None]
            for p in range(self.num_p):
                if self.pN_fix[p] == FIX : continue
                f_p_init = - self.pos_p.grad[p]
                self.vel_p[p] = (1 - beta) / (1 + beta) * self.vel_p[p] + dt * (self.f_p_ext[p] + f_p_init) / (self.m_p[p] * (1 + beta))
                self.pos_p[p] = self.pos_p[p] + dt * self.vel_p[p]
                # if self.pos_p[p].x == float('nan') :
                if  not(self.pos_p[p].x < BIG) :
                # if  False :
                    self.divergence[None] = DIVERGENCE
                    
                
        else :
            for p in range(self.num_p):
                if self.pN_fix[p] == FIX : continue
                f_p_init = - self.pos_p.grad[p]
                self.vel_p[p] += dt * (self.f_p_ext[p] + f_p_init) / self.m_p[p]
                self.pos_p[p] += dt * self.vel_p[p]

    def whether_continue(self) :
        # pass
        # pos_p_np = self.pos_p.to_numpy()
        # divergence = not(np.all(pos_p_np < BIG))
        # if divergence :
        #     sys.exit("Error : The values diverged.")

        if self.divergence[None] == DIVERGENCE :
            sys.exit("Error : The values diverged.")


    def main(self):
        for time_step in range(max_number):
            self.f_p_ext.fill(0)

            with ti.Tape(self.StrainEnergy):
                self.cal_StrainEnergy()

            self.cal_f_p_ext()
            self.cal_alpha_Dum()
            self.plus_vel_pos_p()

            if time_step % output_span == 0 :
                print(time_step)
                print(self.pos_p)
                self.whether_continue()
                if EXPORT : 
                    self.export_Solid()
                    self.export_numpy()
                    self.output_times[None] += 1




ChairBeindingObj = Expansion()

if __name__ == '__main__':
    ChairBeindingObj.main()
