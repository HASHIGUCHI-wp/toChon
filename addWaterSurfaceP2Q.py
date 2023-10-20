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



@ti.data_oriented
class addWaterSurfaceP2Q(Solid_MPM, Fluid_MPM, Solid_P2Q):
    def __init__(self,
            msh_s, msh_f_init, msh_f_add,
            dim, nip, sip,
            mesh_name_s, mesh_name_f_init, mesh_name_f_add,
            ELE_s, SUR_s, ELE_f,
            young_s, nu_s, la_s, mu_s, rho_s, eta1_s, eta2_s,
            rho_f, mu_f, gamma_f, kappa_f, lambda_f, length_f_add,
            dt, dx, nx, ny, nz, gi,
            dx_mesh, dx_mesh_f,
            prepare_point, area_start, area_end,
            diri_area_start, diri_area_end,
            vel_add, vel_add_vec, grav,
            num_add, max_number, output_span, output_span_numpy, add_span_time_step,
            ATTENUATION_s, EXPORT, EXPORT_NUMPY, SEARCH, SCHEME, DATE, WEAK_S,
            SLIP,
            dir_vtu, dir_numpy, dir_export,
            EXIST = 1,
            BIG = 1.0e20
        ):
        self.dim = dim
        self.nip, self.sip = nip, sip
        self.mesh_name_s, self.mesh_name_f_init, self.mesh_name_f_add = mesh_name_s, mesh_name_f_init, mesh_name_f_add
        self.ELE_s, self.SUR_s, self.ELE_f = ELE_s, SUR_s, ELE_f
        self.msh_s, self.msh_f_init, self.msh_f_add = msh_s, msh_f_init, msh_f_add
        self.young_s, self.nu_s, self.la_s, self.mu_s, self.rho_s, self.eta1_s, self.eta2_s = young_s, nu_s, la_s, mu_s, rho_s, eta1_s, eta2_s
        self.rho_f, self.mu_f, self.gamma_f, self.kappa_f, self.lambda_f, self.length_f_add = rho_f, mu_f, gamma_f, kappa_f, lambda_f, length_f_add
        self.dt, self.dx, self.nx, self.ny, self.nz, self.gi = dt, dx, nx, ny, nz, gi 
        self.dx_mesh, self.dx_mesh_f = dx_mesh, dx_mesh_f
        self.prepare_point = prepare_point
        self.area_start, self.area_end = area_start, area_end
        self.vel_add, self.vel_add_vec, self.grav = vel_add, vel_add_vec, grav
        self.num_add, self.max_number, self.output_span, self.output_span_numpy, self.add_span_time_step = num_add, max_number, output_span, output_span_numpy, add_span_time_step
        self.diri_area_start, self.diri_area_end = diri_area_start, diri_area_end
        self.inv_dx = 1 / self.dx
        self.ATTENUATION_s, self.SEARCH, self.SCHEME, self.DATE, self.WEAK_S = ATTENUATION_s, SEARCH, SCHEME, DATE, WEAK_S
        self.SLIP = SLIP
        self.EXPORT, self.EXPORT_NUMPY = EXPORT, EXPORT_NUMPY
        self.dir_vtu, self.dir_numpy, self.dir_export = dir_vtu, dir_numpy, dir_export
        self.BIG = BIG
        self.EXIST, self.DIVERGENCE = 1, 1
        self.S0, self.S1, self.S2 = 0, 1, 2
        self.COL_SURFACE, self.COL_VOLUE = 0, 1
        self.CLEAR = 0
        self.X_FIX = 0.0
        
        print("ATTENUATION_s", ATTENUATION_s)
        print("self.ATTENUATION_s", self.ATTENUATION_s)
        
        self.time_steps = ti.field(dtype=ti.i32, shape=())
        self.output_times = ti.field(dtype=ti.i32, shape=())
        self.add_times = ti.field(dtype=ti.i32, shape=())
        self.protruding = ti.field(dtype=float, shape=())
        
        self.num_p_s = msh_s.points.shape[0]
        self.num_p_f_init = msh_f_init.cells_dict[self.ELE_f].shape[0]
        self.num_p_f_add = msh_f_add.cells_dict[self.ELE_f].shape[0]
        self.num_p_f_all = self.num_p_f_init + self.num_add * self.num_p_f_add
        self.num_p_f_active = ti.field(dtype=ti.i32, shape=())
        self.num_p_f_active[None] = self.num_p_f_init
        self.num_p = self.num_p_s + self.num_p_f_all
        self.num_t_f_init, self.num_node_ele_f = msh_f_init.cells_dict[self.ELE_f].shape
        self.num_t_f_add = msh_f_add.cells_dict[self.ELE_f]
        
        if self.ELE_s == "hexahedron27" :
            print("ELE_s", ELE_s)
            Solid_P2Q.__init__(self, 
                msh_s = self.msh_s,
                rho_s = self.rho_s,
                young_s = self.young_s,
                nu_s = self.nu_s,
                la_s = self.la_s,
                mu_s = self.mu_s,
                dt = self.dt,
                nip = self.nip,
                gi = self.gi,
                ATTENUATION_s = self.ATTENUATION_s,
                WEAK = self.WEAK_S
            )
            
        Fluid_MPM.__init__(self,
            dim = self.dim,
            rho_f = self.rho_f,
            mu_f = self.mu_f,
            gamma_f = self.gamma_f,
            kappa_f = self.kappa_f,
            lambda_f = self.lambda_f,
            dt = self.dt,
            dx = self.dx,
            nx = self.nx, 
            ny = self.ny, 
            nz = self.nz,
            area_start = self.area_start,
            area_end = self.area_end,
            gi = self.gi
        )
        Solid_MPM.__init__(self,
            dt = self.dt,
            num_p_s = self.num_p_s,
            dx = self.dx,
            nx = self.nx,
            ny = self.ny,
            nz = self.nz,
            area_start = self.area_start,
            area_end = self.area_end
        )
        


        self.exist_Ix = ti.field(dtype=ti.i32, shape=self.nx)
        self.exist_Iy = ti.field(dtype=ti.i32, shape=self.ny)
        self.exist_Iz = ti.field(dtype=ti.i32, shape=self.nz)
        self.domain_edge = ti.field(dtype=ti.i32, shape=(self.dim, 2))

        self.pos_p_f_add = ti.Vector.field(self.dim, dtype=float, shape=self.num_p_f_add)
        self.rho_p_f_add = ti.field(dtype=float, shape=self.num_p_f_add)
        self.m_p_f_add = ti.field(dtype=float, shape=self.num_p_f_add)

        self.exist_edge = ti.field(dtype=ti.i32, shape=())
        self.divergence = ti.field(dtype=ti.i32, shape=())
        self.COL_EDGE = 0
        self.COL_SURFACE = 1
        self.COL_VOLUE = 2

        Solid_P2Q.set_taichi_field(self)
        self.set_taichi_field_SOLID_MPM_SURFACE()
        Fluid_MPM.set_taichi_field(self, self.num_p_f_all)
        
    def set_taichi_field_SOLID_MPM_SURFACE(self) :
        self.norm_S = ti.Vector.field(self.dim, dtype=float, shape=(self.nx, self.ny, self.nz))
        self.m_S = ti.field(dtype=float, shape=(self.nx, self.ny, self.nz))
        self.p_S = ti.Vector.field(self.dim, dtype=float, shape=(self.nx, self.ny, self.nz))
        self.f_S = ti.Vector.field(self.dim, dtype=float, shape=(self.nx, self.ny, self.nz))
        # self.num_type_S = 3
        # self.num_contact = 2
        # self.m_S = ti.field(dtype=float, shape=(self.num_type_S, self.nx, self.ny, self.nz))
        # self.f_S = ti.field(dtype=float, shape=(self.num_contact, self.nx, self.ny, self.nz))
        # self.p_S = ti.Vector.field(self.dim, dtype=float, shape=(self.num_type_S, self.nx, self.ny, self.nz))
        # self.norm_CTT = ti.Vector.field(self.dim, dtype=float, shape=(self.num_contact, self.nx, self.ny, self.nz))
        # self.CTT_Type_s = ti.field(dtype=ti.i32, shape=self.num_type_S)
        # self.CTT_Type_s.from_numpy(np.array([0, 1, 1]))
        # self.CTT_Sign_Type_s = ti.field(float, shape=self.num_type_S)
        # self.CTT_Sign_Type_s.from_numpy(np.array([1.0, 1.0, -1.0]))
        # self.MAP_norm_Type_s = ti.field(float, shape=self.num_type_S)
        # self.MAP_norm_Type_s.from_numpy(np.array([1.0, 1.0, 0.0]))
        # print(self.CTT_Type_s)
        # sys.exit("set_taichi_field_SOLID_MPM_SURFACE")
        
    
    def init(self) :
        self.set_s_init()
        self.leg_weights_roots(self.nip)
        self.leg_weights_roots_surface(self.sip)
        self.set_sN_fix()
        self.get_es_press()
        self.set_esN_pN_press()
        self.set_tN_pN_s()
        self.set_Ja_Ref_s()
        self.cal_m_p_s()
        print(self.m_p_s.to_numpy().sum())
        print(self.rho_s * self.num_t_s)
        self.set_p_edge()
        self.cal_m_es_press()
        self.set_f_add()
        self.set_f_init()
        self.get_F_add()
            
        self.set_info()
        self.export_info()
        self.export_program()
        self.export_calculation_domain()
        
    
    def set_Ja_Ref_s(self) :
        self.Ja_Ref_s = ti.Matrix([
            [0.5 * self.dx_mesh, 0.0, 0.0],
            [0.0, 0.5 * self.dx_mesh, 0.0],
            [0.0, 0.0, 0.5 * self.dx_mesh]
        ])
    
    def get_es_press(self) :
        es_label_arr = self.msh_s.cell_data['gmsh:physical'][self.COL_SURFACE]
        es_press_np = np.arange(0, self.num_es_s)[es_label_arr == 0]
        self.num_es_press = es_press_np.shape[0]
        self.num_gauss_press = self.num_es_press * self.sip**2
        self.esN_pN_press = ti.field(dtype=ti.i32, shape=(self.num_es_press, 3, 3))
        self.es_press = ti.field(dtype=ti.i32, shape=self.num_es_press)
        self.es_press.from_numpy(es_press_np)
        type_es_press_np = es_label_arr[es_press_np]
        self.type_es_press = ti.field(dtype=ti.i32, shape=self.num_es_press)
        self.type_es_press.from_numpy(type_es_press_np)
        # sys.exit("get_es_press")
        
        
    
    @ti.kernel
    def cal_m_p_s(self) :
        for g in range(self.num_gauss):
            t, mnl = g // (self.nip**3), g % (self.nip**3)
            m, nl = mnl // (self.nip**2), mnl % (self.nip**2)
            n, l  = nl // self.nip, nl % self.nip
            ja_ref = self.Ja_Ref_s
            det_ja_ref = ja_ref.determinant()
            det_ja_ref = ti.abs(det_ja_ref)
            for _a1 in ti.static(range(3)):
                for _a2 in ti.static(range(3)):
                    for _a3 in ti.static(range(3)):
                        a = self.tN_pN_s[t, _a1, _a2, _a3]
                        self.m_p_s[a] += self.rho_s * self.v_Gauss[_a1, m] * self.v_Gauss[_a2, n] * self.v_Gauss[_a3, l] * self.gauss_w[m] * self.gauss_w[n] * self.gauss_w[l] * det_ja_ref
                        
    def set_p_edge(self) :
        LABEL_EDGE = 1
        self.EDGE = 1
        l_edge_np = np.arange(0, self.num_l_s)[self.msh_s.cell_data['gmsh:physical'][self.COL_EDGE] == LABEL_EDGE]
        num_l_boundary = l_edge_np.shape[0]
        p_edge_np = np.unique(self.msh_s.cells_dict["line3"][l_edge_np, :].reshape(num_l_boundary, self.num_node_edg_s))
        pN_boundary_np = np.zeros(self.num_p_s, dtype=np.float32)
        pN_boundary_np[p_edge_np] = self.EDGE
        self.pN_boundary = ti.field(dtype=float, shape=self.num_p_s)
        self.pN_boundary.from_numpy(pN_boundary_np)
        
    def cal_m_es_press(self) :
        self.set_p_press()
        self.m_gauss_press = ti.field(dtype=float, shape=(self.num_gauss_press))
        self._cal_m_es_press()

        print(self.m_gauss_press.to_numpy().sum())
        print(self.m_p_s.to_numpy()[self.p_press.to_numpy()].sum())
        
        
    def set_p_press(self):
        es_press_np = self.es_press.to_numpy()
        p_press_np = np.unique(self.msh_s.cells_dict[self.SUR_s][es_press_np].reshape(self.num_es_press * self.num_node_sur_s))
        self.num_p_press = p_press_np.shape[0]
        self.p_press = ti.field(dtype=ti.i32, shape=self.num_p_press)
        self.p_press.from_numpy(p_press_np)
    
    
    @ti.kernel
    def cal_sum_area_p_press(self) :
        for _es in range(self.num_es_press) :
            es = self.es_press[_es]
            a, b, c = self.esN_pN_arr_s[es, 0], self.esN_pN_arr_s[es, 1], self.esN_pN_arr_s[es, 2]
            vec_ab, vec_ac = self.pos_p_s_rest[b] - self.pos_p_s_rest[a], self.pos_p_s_rest[c] - self.pos_p_s_rest[a] 
            area = 0.5 * ti.abs(vec_ab.cross(vec_ac).norm())
            for _alpha in ti.static(range(self.num_node_sur_s)) :
                p = self.esN_pN_arr_s[es, _alpha]
                _p = self.pN_p_press[p]
                self.sum_area_p_press[_p] += area
                
    @ti.kernel
    def _cal_m_es_press(self) :
        print("dx_mesh", self.dx_mesh)
        for g in range(self.num_gauss_press) :
            _es, mn = g // (self.sip**2), g % (self.sip**2)
            m, n = mn // self.sip, mn % self.sip
            Ja_Res_es = ti.Matrix([[0.5 * self.dx_mesh, 0.0], [0.0, 0.5 * self.dx_mesh]])
            det_Ja_Res_es = Ja_Res_es.determinant()
            det_Ja_Res_l = 0.5 * self.dx_mesh
            for _a in ti.static(range(3)) :
                for _b in ti.static(range(3) ):
                    p = self.esN_pN_press[_es, _a, _b]
                    for l in ti.static(range(self.nip)) :
                        self.m_gauss_press[g] += self.rho_s * self.v_Gauss_sur[_a, m] * self.v_Gauss_sur[_b, n] * self.gauss_w_sur[m] * self.gauss_w_sur[n] * det_Ja_Res_es * self.v_Gauss[0, l] * self.gauss_w[l] * det_Ja_Res_l / (self.pN_boundary[p] + 1.0)
                        # self.m_gauss_press[g] += self.rho_s * self.v_Gauss_sur[_a, m] * self.v_Gauss_sur[_b, n] * self.gauss_w_sur[m] * self.gauss_w_sur[n] * det_Ja_Res_es * self.v_Gauss[0, l] * self.gauss_w[l] * det_Ja_Res_l
        
    def clear_other_m_es_press(self) :
        self.sum_area_p_press = self.CLEAR
        self.p_press = self.CLEAR
        self.pN_p_press = self.CLEAR
        


    def set_f_add(self):
        pos_f_add_np = np.zeros((self.num_p_f_add, self.dim), dtype=np.float64)
        for _f in range(self.num_node_ele_f) :
            f_arr = self.msh_f_add.cells_dict[self.ELE_f][:, _f]
            pos_f_add_np += self.msh_f_add.points[f_arr, :]
        pos_f_add_np /= self.num_node_ele_f
        self.pos_p_f_add.from_numpy(pos_f_add_np)
        self.m_p_f_add.fill(self.rho_f * (self.dx_mesh_f)**self.dim)
        self.rho_p_f_add.fill(self.rho_f)



    def set_f_init(self):   
        pos_p_f_np = np.zeros((self.num_p_f_all, self.dim), dtype=np.float64)
        for _f in range(self.num_node_ele_f) :
            f_arr = self.msh_f_init.cells_dict[self.ELE_f][:, _f]
            pos_p_f_np[:self.num_p_f_init] += self.msh_f_init.points[f_arr, :]
        pos_p_f_np /= self.num_node_ele_f
        pos_p_f_np[self.num_p_f_init:, :] = self.prepare_point
        self.pos_p_f.from_numpy(pos_p_f_np)

        rho_p_f_np = np.zeros((self.num_p_f_all), dtype=np.float64)
        rho_p_f_np[:self.num_p_f_init] = self.rho_f 
        self.rho_p_f.from_numpy(rho_p_f_np)

        m_p_f_np = np.zeros((self.num_p_f_all), dtype=np.float64)
        m_p_f_np[:self.num_p_f_init] = self.rho_f * (self.dx_mesh_f)**self.dim
        self.m_p_f.from_numpy(m_p_f_np)
    
    # def cal_m_p_f(self) :
    #     for t in range(self.num_t_f_init + self.num_t_f_add) :
    #         if t < self.num_t_f_init :
    #             self._cal_m_p_f_init(t)
    #         else :
    #             self.cal_m_p_f_add(t - self.num_t_f_init)
                        

    def set_s_init(self):
        self.pos_p_s.from_numpy(self.msh_s.points)
        self.pos_p_s_rest.from_numpy(self.msh_s.points)
        self.tN_pN_arr_s.from_numpy(self.msh_s.cells_dict[self.ELE_s])
        self.esN_pN_arr_s.from_numpy(self.msh_s.cells_dict[self.SUR_s])
        
    
    def set_sN_fix(self) :
        sN_fix_np = (self.msh_s.points[:, 0] == self.X_FIX).astype(np.int32)
        self.sN_fix.from_numpy(sN_fix_np)
        

    @ti.kernel
    def add_f(self):
        for _f in range(self.num_p_f_add):
            f = _f + self.num_p_f_init + self.num_p_f_add * self.add_times[None]
            self.pos_p_f[f] = self.pos_p_f_add[_f]
            self.m_p_f[f] = self.m_p_f_add[_f]
            self.rho_p_f[f] = self.rho_f


    
    def get_F_add(self):
        F_add_np = np.zeros(0, dtype=np.int32)
        for _p in range(self.msh_f_add.points.shape[0]):
            pos_p_this = self.msh_f_add.points[_p, :]
            base_x = int((pos_p_this[0] - self.area_start[0]) * self.inv_dx - 0.5)
            base_y = int((pos_p_this[1] - self.area_start[1]) * self.inv_dx - 0.5)
            base_z = int((pos_p_this[2] - self.area_start[2]) * self.inv_dx - 0.5)
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        I_cand = [base_x + i, base_y + j, base_z + k]
                        IxIyIz = I_cand[2] * (self.nx * self.ny) + I_cand[1] * (self.nx) + I_cand[0]
                        F_add_np = np.append(F_add_np, IxIyIz)
        F_add_np = np.unique(F_add_np)
        self.num_F_add = F_add_np.shape[0]
        self.F_add = ti.field(dtype=ti.i32, shape=self.num_F_add)
        self.F_add.from_numpy(F_add_np)        
        print("num_F_add", self.num_F_add)
        
    
        

    def set_info(self) :
        print("self.ATTENUATION_s", self.ATTENUATION_s)
        self.data = {
                "date" : self.DATE ,
                "dir_export" : self.dir_export,
                "date" : self.DATE,
                "Scheme" : self.SCHEME,
                "Slip" : self.SLIP,
                "SEARCH" : self.SEARCH,
                "dim" : self.dim,
                "mesh_name_s" : self.mesh_name_s,
                "mesh_name_f_init" : self.mesh_name_f_init,
                "mesh_name_f_add" : self.mesh_name_f_add,
                "Attenuation" : self.ATTENUATION_s,
                "max_number" : self.max_number,
                "output_span" : self.output_span,
                "output_span_numpy" : self.output_span_numpy,
                "add_span_time_step" : self.add_span_time_step,
                "num_add" : self.num_add,
                "vel_add" : self.vel_add,
                "element_s" : self.ELE_s,
                "element_f" : self.ELE_f,
                "surface_s" : self.SUR_s,
                "dt" : self.dt,
                "dx" : self.dx,
                "dx_mesh" : self.dx_mesh,
                "dx_mesh_f" : self.dx_mesh_f,
                "area_start_x" : self.area_start.x,
                "area_start_y" : self.area_start.y,
                "area_start_z" : self.area_start.z,
                "area_end_x" : self.area_end.x,
                "area_end_y" : self.area_end.y,
                "area_end_z" : self.area_end.z,
                "diri_area_start_x" : self.diri_area_start[0],
                "diri_area_start_y" : self.diri_area_start[1],
                "diri_area_start_z" : self.diri_area_start[2],
                "diri_area_end_x" : self.diri_area_end[0],
                "diri_area_end_y" : self.diri_area_end[1],
                "diri_area_end_z" : self.diri_area_end[2],
                "young_s" : self.young_s,
                "nu_s" : self.nu_s,
                "rho_s" : self.rho_s,
                "la_s" : self.la_s,
                "mu_s" : self.mu_s,
                "eta1_s" : self.eta1_s,
                "eta2_s" : self.eta2_s,
                "mu_f" : self.mu_f,
                "lambda_f" : self.lambda_f,
                "kappa_f" : self.kappa_f,
                "rho_f" : self.rho_f,
                "gamma_f" : self.gamma_f,
                "length_f_add" : self.length_f_add,
                "nip" : self.nip,
                "grav" : self.grav
            }
        
        print("gi", self.gi)
    
    def export_info(self):
        if self.EXPORT:
            s = pd.Series(self.data)
            s.to_csv(self.dir_export + "/" + "Information", header=False)

    def export_program(self):
        if self.EXPORT :
            with open(__file__, mode="r", encoding="utf-8") as fr:
                prog = fr.read()
            with open(self.dir_export +  "/" + "program.txt", mode="w") as fw:
                fw.write(prog)
                fw.flush()

    def export_calculation_domain(self) :
        if self.EXPORT:
            pos = np.array([
                [self.area_start.x, self.area_start.y, self.area_start.z], 
                [self.area_end.x, self.area_start.y, self.area_start.z],
                [self.area_end.x, self.area_end.y, self.area_start.z],
                [self.area_start.x, self.area_end.y, self.area_start.z],
                [self.area_start.x, self.area_start.y, self.area_end.z], 
                [self.area_end.x, self.area_start.y, self.area_end.z],
                [self.area_end.x, self.area_end.y, self.area_end.z],
                [self.area_start.x, self.area_end.y, self.area_end.z]
            ])
            pointsToVTK(
                self.dir_export + "/" + "vtu" + "/" + "Domain",
                pos[:, 0].copy(),
                pos[:, 1].copy(),
                pos[:, 2].copy()
            )


    

    def export_numpy(self):
        np.save(self.dir_numpy + "/" + "pos_p_s_{:05d}".format(self.output_times[None]), self.pos_p_s.to_numpy())
        np.save(self.dir_numpy + "/" + "vel_p_s_{:05d}".format(self.output_times[None]), self.vel_p_s.to_numpy())
        np.save(self.dir_numpy + "/" + "C_p_s_{:05d}".format(self.output_times[None]), self.C_p_s.to_numpy())

        num_f_end = self.num_p_f_init + self.add_times[None] * self.num_p_f_add
        np.save(self.dir_numpy + "/" + "pos_p_f_{:05d}".format(self.output_times[None]), self.pos_p_f.to_numpy()[:num_f_end])
        np.save(self.dir_numpy + "/" + "vel_p_f_{:05d}".format(self.output_times[None]), self.vel_p_f.to_numpy()[:num_f_end])
        np.save(self.dir_numpy + "/" + "C_p_f_{:05d}".format(self.output_times[None]), self.C_p_f.to_numpy()[:num_f_end])
        np.save(self.dir_numpy + "/" + "sigma_p_f_{:05d}".format(self.output_times[None]), self.sigma_p_f.to_numpy()[:num_f_end])
        np.save(self.dir_numpy + "/" + "P_p_f_{:05d}".format(self.output_times[None]), self.P_p_f.to_numpy()[:num_f_end])
        np.save(self.dir_numpy + "/" + "rho_p_f_{:05d}".format(self.output_times[None]), self.rho_p_f.to_numpy()[:num_f_end])
        np.save(self.dir_numpy + "/" + "m_p_f_{:05d}".format(self.output_times[None]), self.m_p_f.to_numpy()[:num_f_end])

        np.save(self.dir_numpy + "/" + "add_times_{:05d}".format(self.output_times[None]), self.add_times[None])
        np.save(self.dir_numpy + "/" + "protruding_{:05d}".format(self.output_times[None]), self.protruding[None])

    @ti.kernel
    def cal_f_p_int_s_from_WEAK(self) :
        for g in range(self.num_gauss):
            t, mnl = g // (self.nip**3), g % (self.nip**3)
            m, nl = mnl // (self.nip**2), mnl % (self.nip**2)
            n, l  = nl // self.nip, nl % self.nip
            ja_ref = self.Ja_Ref_s
            det_ja_ref = ja_ref.determinant()
            det_ja_ref = ti.abs(det_ja_ref)
            inv_trs_Ja_ref = ja_ref.inverse().transpose()
            F = ti.Matrix([[0.0,0.0,0.0], [0.0,0.0,0.0], [0.0,0.0,0.0]])
            for _a1 in ti.static(range(3)):
                for _a2 in ti.static(range(3)):
                    for _a3 in ti.static(range(3)):
                        a = self.tN_pN_s[t, _a1, _a2, _a3]
                        NT = ti.Vector([self.dv_Gauss[_a1, m] * self.v_Gauss[_a2, n] * self.v_Gauss[_a3, l], self.v_Gauss[_a1, m] * self.dv_Gauss[_a2, n] * self.v_Gauss[_a3, l], self.v_Gauss[_a1, m] * self.v_Gauss[_a2, n] * self.dv_Gauss[_a3, l]])
                        NX = inv_trs_Ja_ref @ NT
                        F += self.pos_p_s[a].outer_product(NX)
            inv_trs_F = F.inverse().transpose()
            det_F = F.determinant()
            for _a1 in ti.static(range(3)):
                for _a2 in ti.static(range(3)):
                    for _a3 in ti.static(range(3)):
                        p = self.tN_pN_s[t, _a1, _a2, _a3]
                        NT = ti.Vector([self.dv_Gauss[_a1, m] * self.v_Gauss[_a2, n] * self.v_Gauss[_a3, l], self.v_Gauss[_a1, m] * self.dv_Gauss[_a2, n] * self.v_Gauss[_a3, l], self.v_Gauss[_a1, m] * self.v_Gauss[_a2, n] * self.dv_Gauss[_a3, l]])
                        NX = inv_trs_Ja_ref @ NT
                        self.f_p_int_s[p] += (- self.mu_s * F + (self.mu_s - self.la_s * ti.log(det_F)) * inv_trs_F) @ NX * det_ja_ref * self.gauss_w[m] * self.gauss_w[n] * self.gauss_w[l]

    @ti.kernel
    def diri_norm_S(self):
        for _IxIyIz in range(self.num_S_fix):
            IxIyIz = self.S_fix[_IxIyIz]
            iz, ixiy = IxIyIz // (self.nx * self.ny), IxIyIz % (self.nx * self.ny)
            iy, ix = ixiy // self.nx, ixiy % self.nx
            self.norm_CTT[ix, iy, iz] = ti.Vector([0.0, 0.0, 0.0])
            
        for _IxIyIz in range(self.num_F_add):
            IxIyIz = self.F_add[_IxIyIz]
            iz, ixiy = IxIyIz // (self.nx * self.ny), IxIyIz % (self.nx * self.ny)
            iy, ix = ixiy // self.nx, ixiy % self.nx
            self.norm_CTT[ix, iy, iz] = ti.Vector([0.0, 0.0, 0.0]) 

    @ti.kernel
    def p2g(self):
        for g_or_f in range(self.num_gauss_press + self.num_p_f_active[None]) :
            if g_or_f < self.num_gauss_press :
                g = g_or_f
                self._p2g(g)
            else :
                f = g_or_f - self.num_gauss_press
                Fluid_MPM._p2g(self, f)
    
    @ti.func
    def _p2g(self, g : int) :
        _es, mn = g // (self.sip**2), g % (self.sip**2)
        m, n = mn // self.sip, mn % self.sip
        # TYPE_S = self.type_es_press[_es]
        # TYPE_CTT = self.CTT_Type_s[TYPE_S]
        pos_g, vel_g = ti.Vector([0.0, 0.0, 0.0]), ti.Vector([0.0, 0.0, 0.0])
        k1, k2 = ti.Vector([0.0, 0.0, 0.0]), ti.Vector([0.0, 0.0, 0.0])
        for _a1 in ti.static(range(3)) :
            for _a2 in ti.static(range(3)):
                p = self.esN_pN_press[_es, _a1, _a2]
                pos_g += self.v_Gauss_sur[_a1, m] * self.v_Gauss_sur[_a2, n] * self.pos_p_s[p]
                vel_g += self.v_Gauss_sur[_a1, m] * self.v_Gauss_sur[_a2, n] * self.vel_p_s[p]
                k1 += self.dv_Gauss_sur[_a1, m] * self.v_Gauss_sur[_a2, n] * self.pos_p_s[p]
                k2 += self.v_Gauss_sur[_a1, m] * self.dv_Gauss_sur[_a2, n] * self.pos_p_s[p]
        k3 = k1.cross(k2)
        norm_g = k3.normalized()
        base = ti.cast((pos_g - self.area_start) * self.inv_dx - 0.5, ti.i32)
        fx = (pos_g - self.area_start) * self.inv_dx - ti.cast(base, float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    ix, iy, iz = base.x + i, base.y + j, base.z + k
                    NpI = w[i].x * w[j].y * w[k].z
                    self.m_S[ix, iy, iz] += NpI * self.m_gauss_press[g] 
                    self.p_S[ix, iy, iz] += NpI * self.m_gauss_press[g] * vel_g
                    self.norm_S[ix, iy, iz] += NpI * norm_g
                    self.exist_Ix[ix], self.exist_Iy[iy], self.exist_Iz[iz] = self.EXIST, self.EXIST, self.EXIST
                    # self.m_S[TYPE_S, ix, iy, iz] += NpI * self.m_gauss_press[g] 
                    # self.p_S[TYPE_S, ix, iy, iz] += NpI * self.m_gauss_press[g] * vel_g
                    # self.norm_CTT[TYPE_CTT, ix, iy, iz] += NpI * self.MAP_norm_Type_s[TYPE_S] * norm_g
    

        


    @ti.kernel
    def set_domain_edge(self):
        left, right = 0, self.nx - 1
        front, back = 0, self.ny - 1
        bottom, upper = 0, self.nz - 1
        while self.exist_Ix[left + self.SEARCH] == 0 : left += self.SEARCH
        while self.exist_Iy[front + self.SEARCH] == 0 : front += self.SEARCH
        while self.exist_Iz[bottom + self.SEARCH] == 0 : bottom += self.SEARCH
        while self.exist_Ix[right - self.SEARCH] == 0 : right -= self.SEARCH
        while self.exist_Iy[back - self.SEARCH] == 0 : back -= self.SEARCH
        while self.exist_Iz[upper - self.SEARCH] == 0 : upper -= self.SEARCH
        self.domain_edge[0, 0], self.domain_edge[0, 1] = left, right
        self.domain_edge[1, 0], self.domain_edge[1, 1] = front, back
        self.domain_edge[2, 0], self.domain_edge[2, 1] = bottom, upper
        
        if left <= 1 or front <= 1 or bottom <= 1 or right >= self.nx - 2 or back >= self.ny - 2 or upper >= self.nz - 2 :
            self.exist_edge[None] = self.EXIST 

    @ti.kernel
    def on_grid(self):
        for ix, iy, iz in ti.ndrange(
            (self.domain_edge[0, 0],self.domain_edge[0, 1]),
            (self.domain_edge[1, 0],self.domain_edge[1, 1]),
            (self.domain_edge[2, 0],self.domain_edge[2, 1])
            ):
            
            self.CTT_S0_F_and_update_p_F(ix, iy, iz)
            # self.CTT_S1_S2(ix, iy, iz)
            # self._plus_p_F(ix, iy, iz)
            
    @ti.func
    def _plus_p_F(self, ix : int, iy : int, iz : int) :
        if self.m_F[ix, iy, iz] > 0.0 :
            self.p_F[ix, iy, iz] += self.dt * self.f_F[ix, iy, iz]
        
    
    @ti.func
    def CTT_S0_F_and_update_p_F(self, ix : int, iy : int, iz : int) :
        if self.m_S[ix, iy, iz] > 0.0 or self.m_F[ix, iy, iz]  > 0.0:
                p_I_F_tilde = self.p_F[ix, iy, iz] + self.dt * self.f_F[ix, iy, iz]
                p_I_S_tilde = self.p_S[ix, iy, iz]
                if self.norm_S[ix, iy, iz].norm_sqr() > 0.0 : self.norm_S[ix, iy, iz] = self.norm_S[ix, iy, iz].normalized()
                norm_S_this = self.norm_S[ix, iy, iz]
                CDT1 = self.m_F[ix, iy, iz] > 0.0 and self.m_S[ix, iy, iz] > 0.0
                CDT2 = (self.m_F[ix, iy, iz] * p_I_S_tilde - self.m_S[ix, iy, iz] * p_I_F_tilde).dot(norm_S_this) > 0.0
                if CDT1 and CDT2:
                    f_I_nor = 1 / (self.dt * (self.m_F[ix, iy, iz] + self.m_S[ix, iy, iz])) * (self.m_S[ix, iy, iz] * self.p_F[ix, iy, iz] - self.m_F[ix, iy, iz] * self.p_S[ix, iy, iz]).dot(norm_S_this)
                    f_I_nor += 1 / (self.m_S[ix, iy, iz] + self.m_F[ix, iy, iz]) * self.m_S[ix, iy, iz] * self.f_F[ix, iy, iz].dot(norm_S_this)

                    f_S_cnt = f_I_nor * norm_S_this
                    self.p_S[ix, iy, iz] = p_I_S_tilde + self.dt * f_S_cnt
                    self.p_F[ix, iy, iz] = p_I_F_tilde - self.dt * f_S_cnt
                    self.f_S[ix, iy, iz] = f_I_nor * norm_S_this
                else :
                    self.p_S[ix, iy, iz] = p_I_S_tilde
                    self.p_F[ix, iy, iz] = p_I_F_tilde

        # if self.m_S[self.S0, ix, iy, iz] > 0.0 or self.m_F[ix, iy, iz]  > 0.0:
        #     TYPE_CTT = 0
        #     p_F_tilde = self.p_F[ix, iy, iz] + self.dt * self.f_F[ix, iy, iz]
        #     p_F = self.p_F[ix, iy, iz]
        #     p_S = self.p_S[self.S0, ix, iy, iz]
        #     m_S, m_F = self.m_S[self.S0, ix, iy, iz], self.m_F[ix, iy, iz]
        #     f_F = self.f_F[ix, iy, iz]
        #     if self.norm_CTT[TYPE_CTT, ix, iy, iz].norm_sqr() > 0.0 : self.norm_CTT[TYPE_CTT, ix, iy, iz] = self.norm_CTT[TYPE_CTT, ix, iy, iz].normalized()
        #     norm_CTT = self.norm_CTT[TYPE_CTT, ix, iy, iz]
        #     CDT2 = (m_F * p_S - m_S * p_F_tilde).dot(norm_CTT) > 0.0
        #     if CDT2:
        #         f_I_nor = 1 / (self.dt * (m_F + m_S)) * (m_S * p_F - m_F * p_S).dot(norm_CTT)
        #         f_I_nor += 1 / (m_S + m_F) * m_S * f_F.dot(norm_CTT)

        #         self.f_S[TYPE_CTT, ix, iy, iz] = f_I_nor
        #         self.p_F[ix, iy, iz] = p_F_tilde - self.dt * self.f_S[TYPE_CTT, ix, iy, iz] * norm_CTT
        #     else :
        #         self.p_F[ix, iy, iz] = p_F_tilde
                
                
    @ti.func
    def CTT_S1_S2(self, ix : int, iy : int, iz : int) :
        if self.m_S[self.S1, ix, iy, iz] > 0.0 and self.m_S[self.S2, ix, iy, iz] > 0.0 :
            TYPE_CTT = 1
            if self.norm_CTT[TYPE_CTT, ix, iy, iz].norm_sqr() > 0.0 : self.norm_CTT[TYPE_CTT, ix, iy, iz] = self.norm_CTT[TYPE_CTT, ix, iy, iz].normalized()
            norm_CTT = self.norm_CTT[TYPE_CTT, ix, iy, iz]
            m_S1, m_S2 = self.m_S[self.S1, ix, iy, iz], self.m_S[self.S2, ix, iy, iz]
            p_S1, p_S2 = self.p_S[self.S1, ix, iy, iz], self.p_S[self.S2, ix, iy, iz]
            CDT2 = (m_S2 * p_S1 - m_S1 * p_S2).dot(norm_CTT) > 0.0
            if CDT2 :
                f_I_nor = 1 / (self.dt * (m_S1 + m_S2)) * (m_S1 * p_S2 - m_S2 * p_S1).dot(norm_CTT)
                self.f_S[TYPE_CTT, ix, iy, iz] = f_I_nor
                


    @ti.kernel
    def diri_p_F(self):
        for _IxIyIz in range(self.num_F_add):
            IxIyIz = self.F_add[_IxIyIz]
            iz, ixiy = IxIyIz // (self.nx*self.ny), IxIyIz % (self.nx*self.ny)
            iy, ix = ixiy // self.nx, ixiy % self.nx
            self.p_F[ix, iy, iz] = self.vel_add_vec * self.m_F[ix, iy, iz]
            
    

    @ti.kernel
    def g2p(self) :
        for g_or_f in range(self.num_gauss_press + self.num_p_f_active[None]) :
            if g_or_f < self.num_gauss_press :
                g = g_or_f
                self._g2p(g)
            else :
                f = g_or_f - self.num_p_f_active[None]
                Fluid_MPM._g2p(self, f)
            
    @ti.func
    def _g2p(self, g : int) :
        _es, mn = g // (self.sip**2), g % (self.sip**2)
        m, n = mn // self.sip, mn % self.sip
        # TYPE_S = self.type_es_press[_es]
        # TYPE_CTT = self.CTT_Type_s[TYPE_S]
        pos_g = ti.Vector([0.0, 0.0, 0.0])
        vel_g = ti.Vector([0.0, 0.0, 0.0])
        for _a1 in ti.static(range(3)):
            for _a2 in ti.static(range(3)):
                p = self.esN_pN_press[_es, _a1, _a2]
                pos_g += self.v_Gauss_sur[_a1, m] * self.v_Gauss_sur[_a2, n] * self.pos_p_s[p]
                vel_g += self.v_Gauss_sur[_a1, m] * self.v_Gauss_sur[_a2, n] * self.vel_p_s[p]


        base = ti.cast((pos_g - self.area_start) * self.inv_dx - 0.5, ti.i32)
        fx = (pos_g - self.area_start) * self.inv_dx - ti.cast(base, float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        f_g_cnt = ti.Vector([0.0, 0.0, 0.0])
        new_v_g = ti.Vector([0.0, 0.0, 0.0])
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    ix, iy, iz = base.x + i, base.y + j, base.z + k
                    if self.m_S[ix, iy, iz] > 0.0 :
                        NpI = w[i].x * w[j].y * w[k].z
                        f_g_cnt += NpI * self.f_S[ix, iy, iz] * self.m_gauss_press[g] / self.m_S[ix, iy, iz]
                        new_v_g += NpI * self.p_S[ix, iy, iz] / self.m_S[ix, iy, iz]


        for _a1 in ti.static(range(3)) :
            for _a2 in ti.static(range(3)) :
                p = self.esN_pN_press[_es, _a1, _a2]
                # print(p)
                self.f_p_cnt_s[p] += f_g_cnt * self.v_Gauss_sur[_a1, m] * self.v_Gauss_sur[_a2, n]
                # print(self.vel_p_s[p], self.m_gauss_press[g] / self.m_p_s[p] * (new_v_g - vel_g) * self.v_Gauss_sur[m] * self.v_Gauss_sur[n])
                # self.vel_p_s[p] += self.m_gauss_press[g] / self.m_p_s[p] * (new_v_g - vel_g) * self.v_Gauss_sur[m] * self.v_Gauss_sur[n]

    
        

    @ti.kernel
    def plus_pos_p(self) :
        for p in range(self.num_p_s + self.num_p_f_active[None]) :
            if p < self.num_p_s : 
                if self.sN_fix[p] == self.FREE :
                    self.vel_p_s[p] += self.dt * self.f_p_cnt_s[p] / self.m_p_s[p]
                    self.pos_p_s[p] += self.dt * self.vel_p_s[p]
                    if not(self.pos_p_s[p].x < self.BIG) :
                        self.divergence[None] = self.DIVERGENCE
                        # print("SOLID divergenced", self.pos_p_s_rest[p], self.pos_p_s[p])
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
        self.f_p_int_s.fill(0)
        self.f_p_cnt_s.fill(0)
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
            self.f_S[ix, iy, iz] = [0.0, 0.0, 0.0]
            self.norm_S[ix, iy, iz] = [0.0, 0.0, 0.0]
            # self.m_S[self.S0, ix, iy, iz] = 0.0
            # self.m_S[self.S1, ix, iy, iz] = 0.0
            # self.m_S[self.S2, ix, iy, iz] = 0.0
            # self.m_F[ix, iy, iz] = 0.0
            # self.p_S[self.S0, ix, iy, iz] = [0.0, 0.0, 0.0]
            # self.p_S[self.S1, ix, iy, iz] = [0.0, 0.0, 0.0]
            # self.p_S[self.S2, ix, iy, iz] = [0.0, 0.0, 0.0]
            # self.p_F[ix, iy, iz] = [0.0, 0.0, 0.0]
            # self.f_F[ix, iy, iz] = [0.0, 0.0, 0.0]
            # self.f_S[0, ix, iy, iz] = 0.0
            # self.f_S[1, ix, iy, iz] = 0.0
            # self.norm_CTT[0, ix, iy, iz] = [0.0, 0.0, 0.0]
            # self.norm_CTT[1, ix, iy, iz] = [0.0, 0.0, 0.0]
            
    
    def reduce_protruding(self) :
        if self.protruding[None] > 0.0 :
            self.protruding[None] -= self.dt * self.vel_add


    def whether_continue(self):
        if self.exist_edge[None] == self.EXIST :
            pass
            # self.export_Solid(self.dir_vtu + "/" + "SOLID_EDGE.vtu")
            # self.export_Fluid(self.dir_vtu + "/" + "FLUID_EDGE")
            # sys.exit("Error : Particles exist near the edge of the computational domain. Please extend the computational domain and restart the simulation.")
            # print("Error : Particles exist near the edge of the computational domain. Please extend the computational domain and restart the simulation.")
            
        if self.divergence[None] == self.DIVERGENCE:
            if self.EXPORT :
                self.export_Solid(self.dir_vtu + "/" + "SOLID_DIVERGED.vtu")
                self.export_Fluid(self.dir_vtu + "/" + "FLUID_DIVERGED")
            sys.exit("Error : The values diverged.")
            
                        
    @ti.kernel
    def plus_vel_p(self):
        for p in range(self.num_p_s) :
            Solid_P2Q._plus_vel_p(self, p)
            

    def main(self):
        print("roop start")
        while self.time_steps[None] < self.max_number :
        # while self.time_steps[None] < 2  :
            # if self.time_steps[None] % 100 == 0:
            #     print(self.time_steps[None])
            
            # print(self.time_steps[None])
                
            if self.time_steps[None] % self.output_span == 0:
                print(self.time_steps[None])
                
                if self.EXPORT:
                    self.export_Solid(self.dir_vtu + "/" + "SOLID{:05d}.vtu".format(self.output_times[None]))
                    self.export_Fluid(self.dir_vtu + "/" + "FLUID{:05d}".format(self.output_times[None]))
                    
                if self.time_steps[None] % self.output_span_numpy == 0:
                    if self.EXPORT and self.EXPORT_NUMPY :
                        self.export_numpy()
                        
                self.output_times[None] += 1
            
            if self.protruding[None] <= 0.0:
                if self.add_times[None] < self.num_add:
                    print("progress : add water")
                    self.add_f()
                    self.add_times[None] += 1
                    self.num_p_f_active[None] += self.num_p_f_add
                    self.protruding[None] = self.length_f_add
                else :
                    self.protruding[None] = - 1.0e-5
                    self.vel_add = 0.0
                    self.vel_add_vec = ti.Vector([0.0, 0.0, 0.0])
                    
                    
            self.cal_f_p_int_s_from_WEAK()
                
            # if self.ATTENUATION_s :
            #     self.cal_alpha_Dum()
                
            self.plus_vel_p()
            self.p2g()
            self.set_domain_edge()
            self.on_grid()
            self.diri_p_F()
            
            self.g2p()
            
            self.p_F.fill(0)
            self.update_p_F()
            self.diri_p_F()
            self.cal_L_p_f()
            self.cal_rho_sigma_p_f()
            self.plus_pos_p()
            self.clear()

            self.reduce_protruding()
            self.whether_continue()

            self.time_steps[None] += 1
            




