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



@ti.data_oriented
class addWater(Solid_MPM, Fluid_MPM, Solid_P1T):
# class addWater(Solid_MPM, Fluid_MPM, Solid_P2Q):
    def __init__(self,
            msh_s, msh_f_init, msh_f_add,
            dim, nip,
            ELE_s, SUR_s, ELE_f,
            young_s, nu_s, la_s, mu_s, rho_s,
            rho_f, mu_f, gamma_f, kappa_f, lambda_f, length_f_add,
            dt, dx, nx, ny, nz, gi,
            dx_mesh, dx_mesh_f,
            area_start, area_end,
            diri_area_start, diri_area_end,
            vel_add, vel_add_vec,
            num_add, max_number, output_span, add_span_time_step,
            ATTEMUATION_s, EXPORT, EXPORT_NUMPY, SEARCH,
            dir_vtu, dir_numpy,
            EXIST = 1,
            BIG = 1.0e20
        ):
        self.dim = dim
        self.nip = nip
        self.ELE_s, self.SUR_s, self.ELE_f = ELE_s, SUR_s, ELE_f
        self.young_s, self.nu_s, self.la_s, self.mu_s, self.rho_s = young_s, nu_s, la_s, mu_s, rho_s
        self.rho_f, self.mu_f, self.gamma_f, self.kappa_f, self.lambda_f, self.length_f_add = rho_f, mu_f, gamma_f, kappa_f, lambda_f, length_f_add
        self.dt, self.dx, self.nx, self.ny, self.nz, self.gi = dt, dx, nx, ny, nz, gi 
        self.dx_mesh, self.dx_mesh_f = dx_mesh, dx_mesh_f
        self.area_start, self.area_end = area_start, area_end
        self.vel_add, self.vel_add_vec = vel_add, vel_add_vec
        self.num_add, self.max_number, self.output_span, self.add_span_time_step = num_add, max_number, output_span, add_span_time_step
        self.diri_area_start, self.diri_area_end = diri_area_start, diri_area_end
        self.inv_dx = 1 / self.dx
        self.ATTENUATION_s, self.SEARCH = ATTEMUATION_s, SEARCH
        self.EXPORT, self.EXPORT_NUMPY = EXPORT, EXPORT_NUMPY
        self.dir_vtu, self.dir_numpy = dir_vtu, dir_numpy
        self.BIG = BIG
        self.EXIST, self.DIVERGENCE = 1, 1
        
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
                msh_s = msh_s,
                rho_s = self.rho_s,
                young_s = self.young_s,
                nu_s = self.nu_s,
                la_s = self.la_s,
                mu_s = self.mu_s,
                dt = self.dt,
                nip = self.nip,
                gi = self.gi
            )
        elif self.ELE_s == "tetra" :
            print("ELE_s", ELE_s)
            Solid_P1T.__init__(self, 
                msh_s = msh_s,
                rho_s = self.rho_s,
                young_s = self.young_s,
                nu_s = self.nu_s,
                la_s = self.la_s,
                mu_s = self.mu_s,
                dt = self.dt,
                gi = self.gi
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

        Solid_P2Q.set_taichi_field(self)
        Solid_MPM.set_taichi_field(self)
        Fluid_MPM.set_taichi_field(self, self.num_p_f_all)
        
    def init(self) :
        self.set_f_add(msh_f_add)
        self.set_f_init(msh_f_init)
        self.set_s_init(msh_s)
        self.get_S_fix()
        self.get_F_add(msh_f_add)
        self.get_es_press(msh_s)
        self.set_esN_pN_press()
        
        if self.ELE_s == "hexahedron27" :
            self.leg_weights_roots(self.nip)
            self.set_tN_pN_s()
            self.cal_Ja_Ref_s()
            
        self.cal_m_p_s()
        self.set_info(DATE, dir_export, SCHEME, SLIP, mesh_name_s, mesh_name_f_init, mesh_name_f_add, grav)
        self.export_info(dir_export + "/" + "Information")
        self.export_program(dir_export + "/" + "program.txt")
        self.export_calculation_domain(dir_export + "/" + "vtu" + "/" + "Domain")


    def set_f_add(self, msh_f_add):
        pos_f_add_np = np.zeros((self.num_p_f_add, self.dim), dtype=np.float64)
        for _f in range(self.num_node_ele_f) :
            f_arr = msh_f_add.cells_dict[self.ELE_f][:, _f]
            pos_f_add_np += msh_f_add.points[f_arr, :]
        pos_f_add_np /= self.num_node_ele_f
        self.pos_p_f_add.from_numpy(pos_f_add_np)
        self.m_p_f_add.fill(self.rho_f * (self.dx_mesh_f)**self.dim)
        self.rho_p_f_add.fill(self.rho_f)



    def set_f_init(self, msh_f_init):   
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
        m_p_f_np[:self.num_p_f_init] = self.rho_f * (self.dx_mesh_f)**self.dim
        self.m_p_f.from_numpy(m_p_f_np)
    
    # def cal_m_p_f(self) :
    #     for t in range(self.num_t_f_init + self.num_t_f_add) :
    #         if t < self.num_t_f_init :
    #             self._cal_m_p_f_init(t)
    #         else :
    #             self.cal_m_p_f_add(t - self.num_t_f_init)
                        

    def set_s_init(self, msh_s):
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
            self.rho_p_f[f] = self.rho_f



    def get_S_fix(self):
        S_fix_start_x = int((self.diri_area_start[0] - self.area_start[0]) * self.inv_dx - 0.5)
        S_fix_start_y = int((self.diri_area_start[1] - self.area_start[1]) * self.inv_dx - 0.5)
        S_fix_start_z = int((self.diri_area_start[2] - self.area_start[2]) * self.inv_dx - 0.5)
        
        S_fix_end_x = int((self.diri_area_end[0] - self.area_start[0]) * self.inv_dx - 0.5) + 2
        S_fix_end_y = int((self.diri_area_end[1] - self.area_start[1]) * self.inv_dx - 0.5) + 2
        S_fix_end_z = int((self.diri_area_end[2] - self.area_start[2]) * self.inv_dx - 0.5) + 2
        
        nx_fix = S_fix_end_x - S_fix_start_x + 1
        ny_fix = S_fix_end_y - S_fix_start_y + 1
        nz_fix = S_fix_end_z - S_fix_start_z + 1
        
        S_fix_np = np.zeros(0, dtype=np.int32)
        for _Ix in range(nx_fix):
            for _Iy in range(ny_fix):
                for _Iz in range(nz_fix):
                    Ix, Iy, Iz = _Ix + S_fix_start_x, _Iy + S_fix_start_y, _Iz + S_fix_start_z
                    IxIyIz = Iz * (self.nx * self.ny) + Iy * (self.nx) + Ix
                    S_fix_np = np.append(S_fix_np, IxIyIz)
        
        S_fix_np = np.unique(S_fix_np)
        self.num_S_fix = S_fix_np.shape[0]
        self.S_fix = ti.field(dtype=ti.i32, shape=self.num_S_fix)
        self.S_fix.from_numpy(S_fix_np)
        print("num_S_fix", self.num_S_fix)

    
    def get_F_add(self, msh_f_add):
        F_add_np = np.zeros(0, dtype=np.int32)
        for _p in range(msh_f_add.points.shape[0]):
            pos_p_this = msh_f_add.points[_p, :]
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

    def set_info(self, DATE, dir_export, SCHEME, SLIP, mesh_name_s, mesh_name_f_init, mesh_name_f_add, grav) :
        self.data = {
                "date" : DATE ,
                "dir_export" : dir_export,
                "Scheme" : SCHEME,
                "Slip" : SLIP,
                "dim" : self.dim,
                "mesh_name_s" : mesh_name_s,
                "mesh_name_f_init" : mesh_name_f_init,
                "mesh_name_f_add" : mesh_name_f_add,
                "Attenuation" : self.ATTENUATION_s,
                "max_number" : self.max_number,
                "output_span" : self.output_span,
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
                "mu_f" : self.mu_f,
                "lambda_f" : self.lambda_f,
                "kappa_f" : self.kappa_f,
                "rho_f" : self.rho_f,
                "gamma_f" : self.gamma_f,
                "nip" : self.nip,
                "grav" : grav
            }
    
    def export_info(self, dir):
        if self.EXPORT:
            s = pd.Series(self.data)
            s.to_csv(dir, header=False)

    def export_program(self, dir):
        if self.EXPORT :
            with open(__file__, mode="r", encoding="utf-8") as fr:
                prog = fr.read()
            with open(dir, mode="w") as fw:
                fw.write(prog)
                fw.flush()

    def export_calculation_domain(self, dir) :
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
                dir,
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
    def diri_norm_S(self):
        for _IxIyIz in range(self.num_S_fix):
            IxIyIz = self.S_fix[_IxIyIz]
            iz, ixiy = IxIyIz // (self.nx * self.ny), IxIyIz % (self.nx * self.ny)
            iy, ix = ixiy // self.nx, ixiy % self.nx
            self.norm_S[ix, iy, iz] = ti.Vector([0.0, 0.0, 0.0])
            
        for _IxIyIz in range(self.num_F_add):
            IxIyIz = self.F_add[_IxIyIz]
            iz, ixiy = IxIyIz // (self.nx * self.ny), IxIyIz % (self.nx * self.ny)
            iy, ix = ixiy // self.nx, ixiy % self.nx
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
    def plus_p_I_by_contact(self):
        for ix, iy, iz in ti.ndrange(
            (self.domain_edge[0, 0],self.domain_edge[0, 1]),
            (self.domain_edge[1, 0],self.domain_edge[1, 1]),
            (self.domain_edge[2, 0],self.domain_edge[2, 1])
            ):

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
                else :
                    self.p_S[ix, iy, iz] = p_I_S_tilde
                    self.p_F[ix, iy, iz] = p_I_F_tilde


    @ti.kernel
    def diri_p_I(self):
        for _IxIyIz in range(self.num_S_fix):
            IxIyIz = self.S_fix[_IxIyIz]
            iz, ixiy = IxIyIz // (self.nx*self.ny), IxIyIz % (self.nx*self.ny)
            iy, ix = ixiy // self.nx, ixiy % self.nx
            self.p_S[ix, iy, iz] = ti.Vector([0.0, 0.0, 0.0])
            
        for _IxIyIz in range(self.num_F_add):
            IxIyIz = self.F_add[_IxIyIz]
            iz, ixiy = IxIyIz // (self.nx*self.ny), IxIyIz % (self.nx*self.ny)
            iy, ix = ixiy // self.nx, ixiy % self.nx
            self.p_F[ix, iy, iz] = self.vel_add_vec * self.m_F[ix, iy, iz]


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
                if not(self.pos_p_s[p].x < self.BIG) :
                    self.divergence[None] = self.DIVERGENCE
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
            
    
    def reduce_protruding(self) :
        if self.protruding[None] > 0.0 :
            self.protruding[None] -= self.dt * self.vel_add


    def whether_continue(self):
        if self.exist_edge[None] == self.EXIST :
            sys.exit("Error : Particles exist near the edge of the computational domain. Please extend the computational domain and restart the simulation.")

        if self.divergence[None] == self.DIVERGENCE:
            sys.exit("Error : The values diverged.")
            
                        
    

    def main(self):
        print("roop start")
        while self.time_steps[None] < self.max_number :
        # while self.time_steps[None] < 1  :
            if self.time_steps[None] % 100 == 0:
                print(self.time_steps[None])
                
            if self.time_steps[None] % self.output_span == 0:
                print(self.time_steps[None])
                
                if self.EXPORT:
                    self.export_Solid(self.dir_vtu + "/" + "SOLID{:05d}.vtu".format(self.output_times[None]))
                    self.export_Fluid(self.dir_vtu + "/" + "FLUID{:05d}".format(self.output_times[None]))
                    if self.EXPORT_NUMPY :
                        self.export_numpy()
                    self.output_times[None] += 1
            
            if self.protruding[None] <= 0.0:
                if self.add_times[None] <= self.num_add:
                    self.add_f()
                    self.add_times[None] += 1
                    self.num_p_f_active[None] += self.num_p_f_add
                    self.protruding[None] = self.length_f_add
                    
                    
                
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

            self.reduce_protruding()
            self.whether_continue()

            self.time_steps[None] += 1
            
            # print("num_p_f_init", self.num_p_f_init)
            # print("num_p_f_add", self.num_p_f_add)
            # print("num_p_f_active", self.num_p_f_active[None])





if __name__ == '__main__':
    ti.init()

    USER = "Hashiguchi"
    USING_MACHINE = "CERVO"
    SCHEME = "MPM"
    ADD_INFO_LIST = False
    EXPORT = True
    EXPORT_NUMPY = True
    FOLDER_NAME = "MoyashiAddWaterP1T"
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
    
    
    ATTENUATION = False
    SLIP = True

    length_s, width_s, height_s = 18.0, 12.0, 63.0
    length_f_add, width_f_add, height_f_add = 4.5, 6.0, 6.0
    
    # ELE_s, SUR_s = "hexahedron27", "quad9"
    ELE_s, SUR_s = "tetra", "triangle"
    ELE_f, SUR_f = "hexahedron", "quad"

    if ELE_s == "hexahedron27" :
        mesh_name_s = "MoyashiTransfinite2"
        FOLDER_NAME = "MoyashiAddWaterP2"
        
    elif ELE_s == "tetra" :
        mesh_name_s = "MoyashiTetra"
        FOLDER_NAME = "MoyashiAddWaterP1T"
        
    mesh_name_f_init = "MoyashiWaterInit2"
    mesh_name_f_add = "MoyashiWaterAdd2"
    
    DATE = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')


    dir_export = "./consequence" + "/" + FOLDER_NAME + "/" + DATE 
    dir_vtu = dir_export + "/" + "vtu"
    dir_numpy = dir_export + "/" + "numpy"




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
    
    ChairBeindingObj = addWater(
    msh_s = msh_s,
    msh_f_init = msh_f_init,
    msh_f_add = msh_f_add,
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
    length_f_add = height_f_add,
    dt = dt,
    dx = dx,
    nx = nx,
    ny = ny,
    nz = nz,
    gi = gi,
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
    add_span_time_step = add_span_time_step,
    ATTEMUATION_s = ATTENUATION,
    EXPORT = EXPORT,
    EXPORT_NUMPY = EXPORT_NUMPY,
    SEARCH = SEARCH,
    dir_numpy = dir_numpy,
    dir_vtu = dir_vtu
)

    ChairBeindingObj.init()
    ChairBeindingObj.main()
