import numpy as np
import meshio
import taichi as ti
import datetime
import os
from pyevtk.hl import *

ti.init(arch=ti.gpu, default_fp=ti.f64, device_memory_fraction=0.9)

SLIP = True
msh_s = meshio.read('/home/hashiguchi/mpm_simulation/geometry/BendingArm/BendingArmLabelSize0.2-3.msh')
msh_f_add = meshio.read('/home/hashiguchi/mpm_simulation/geometry/BendingArm/addWaterR2.0L3.0Size0.2-3.msh')
msh_f_init = meshio.read('/home/hashiguchi/mpm_simulation/geometry/BendingArm/initWater4x4Size0.2-3.msh')


dx = 0.3
dt = 0.0003
rho_s = 4e1
young_s, nu_s = 4e6, 0.2

num_add = 100
vel_add = ti.Vector([-8.0, 0.0, 0.0])


export = True

FOLDER_NAME = "BendingArmWaterToWater"
FILE_NAME = "BendingArmWaterToWater"

if export:
    # FOLDER_NAME = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M') + FOLDER_NAME
    dr = '/home/hashiguchi/mpm_simulation/result/' + FOLDER_NAME + "/" + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    os.makedirs(dr, exist_ok=True)

#計算領域の設定
dim = 3
box_size = ti.Vector([76, 22, 78])
area_start = ti.Vector([-42, -11, -55])
inv_dx = 1 / dx
nx, ny, nz = int(box_size.x * inv_dx + 1), int(box_size.y * inv_dx + 1), int(box_size.z * inv_dx + 1)
mx = nx - 1
gi = ti.Vector([0.0, 0.0, 0.0])
diri_area_start = [31, -7.5, 0]
diri_area_end = [31, 7.5, 19.0]

#固体の物性値
la_s, mu_s = young_s * nu_s / ((1+nu_s) * (1-2*nu_s)) , young_s / (2 * (1+nu_s))
sound_s = np.sqrt((la_s + 2 * mu_s) / rho_s)
num_surface = 73
P_surface = 2

#流体の物性値
radius_f, long_f = 3.0, 3.0
rho_f = 0.9975e3
mu_f = 1.002e-3
lambda_f = 2/3 * mu_f
gamma_f = 7.0     ## water
kappa_f = 2.0e6
sound_f = np.sqrt(kappa_f / rho_f)

print("sound_s", sound_s)
print("sound_f", sound_f)

sound_max = sound_s if sound_s > sound_f else sound_f
dt_max = 0.1 * dx / sound_max

print("dt_max", dt_max)
print("dt", dt)



add_span_time = long_f / np.abs(vel_add.x)
add_span_time_step = int(add_span_time // dt) + 1

num_type = 2
max_number = add_span_time_step * num_add
output_span = max_number // 1000

print("output_span", output_span)

FLUID = 0
SOLID = 1
WAIT = 2

@ti.data_oriented
class BendingArm():
    def __init__(self):
        self.output_times = ti.field(dtype=ti.i32, shape=())
        self.add_times = ti.field(dtype=ti.i32, shape=())
        self.ELE_s, self.SUR_s = "tetra", "triangle"
        self.ELE_f, self.SUR_f = "tetra", "triangle"
        self.num_p_s = msh_s.points.shape[0]
        self.num_t_s, self.num_node_ele_s = msh_s.cells_dict[self.ELE_s].shape
        self.num_es_s, self.num_node_sur_s = msh_s.cells_dict[self.SUR_s].shape
        self.num_p_f_init, self.num_node_ele_f = msh_f_init.cells_dict[self.ELE_f].shape
        self.num_p_f_add = msh_f_add.cells_dict[self.ELE_f].shape[0]
        self.num_p = self.num_p_s + self.num_p_f_init + num_add * self.num_p_f_add
        
        self.StrainEnergy = ti.field(dtype=float, shape=(), needs_grad=True)
        self.m_p = ti.field(dtype=float, shape=self.num_p)
        self.rho_p = ti.field(dtype=float, shape=self.num_p)
        self.P_p = ti.field(dtype=float, shape=self.num_p)
        self.type_p = ti.field(dtype=ti.i32, shape=self.num_p)
        self.pos_p = ti.Vector.field(dim, dtype=float, shape=self.num_p, needs_grad=True)
        self.pos_s_rest = ti.Vector.field(dim, dtype=float, shape=self.num_p_s)
        self.d_pos_p = ti.Vector.field(dim, dtype=float, shape=self.num_p)
        self.vel_p = ti.Vector.field(dim, dtype=float, shape=self.num_p)
        self.f_p_ext = ti.Vector.field(dim, dtype=float, shape=self.num_p)
        self.C_p = ti.Matrix.field(dim, dim, dtype=float, shape=self.num_p)
        self.sigma_p = ti.Matrix.field(dim, dim, dtype=float, shape=self.num_p)
        self.epsilon_p = ti.Matrix.field(dim, dim, dtype=float, shape=self.num_p)
        self.L_p = ti.Matrix.field(dim, dim, dtype=float, shape=self.num_p)

        self.m_I = ti.field(dtype=float, shape=(nx, ny, nz, num_type))
        self.p_I = ti.Vector.field(dim, dtype=float, shape=(nx, ny, nz, num_type))
        self.f_F = ti.Vector.field(dim, dtype=float, shape=(nx, ny, nz))

        self.norm_S = ti.Vector.field(dim, dtype=float, shape=(nx, ny, nz))
        # pos_I = ti.Vector.field(dim, dtype=float, shape=(nx * ny * nz))

        self.pos_p_add = ti.Vector.field(dim, dtype=float, shape=self.num_p_f_add)
        self.rho_p_add = ti.field(dtype=float, shape=self.num_p_f_add)
        self.m_p_add = ti.field(dtype=float, shape=self.num_p_f_add)


        self.tN_pN = ti.field(dtype=ti.i32, shape=(self.num_t_s, self.num_node_ele_s))
        self.esN_pN = ti.field(dtype=ti.i32, shape=(self.num_es_s, self.num_node_sur_s))
        self.tN_pN.from_numpy(msh_s.cells_dict[self.ELE_s])
        self.esN_pN.from_numpy(msh_s.cells_dict[self.SUR_s])
        
        self.msh_points_f_init = ti.Vector.field(dim, dtype=float, shape=msh_f_init.points.shape[0])
        self.msh_points_f_init.from_numpy(msh_f_init.points)
        self.tN_pN_f_init = ti.field(dtype=ti.i32, shape=(msh_f_init.cells_dict[self.ELE_f].shape[0], msh_f_init.cells_dict[self.ELE_f].shape[1]))
        self.tN_pN_f_init.from_numpy(msh_f_init.cells_dict[self.ELE_f])
        

        print("num_t_s", self.num_t_s)
        print("num_p_s", self.num_p_s)
        print("num_es_s", self.num_es_s)
        print("num_p_f_add", self.num_p_f_add)
        print("num_p_f_init", self.num_p_f_init)
        print("num_p", self.num_p)
        
        self.set_sN_inf()
        self.pesN_pN = ti.field(dtype=ti.i32, shape=(self.num_pes, self.num_node_sur_s))
        self.set_pesN_pN()
        self.get_diri_IxIyIz_np()
        self.get_add_IxIyIz_np()
        
        self.pos_p.fill(-100)
        self.set_type_p()
        self.set_pos_p_s(msh_s.points)
        self.set_f_init()
        # self.set_f_init()
        self.set_f_add()
        self.cal_m_s()
        self.set_type_p()
        

    def set_sN_inf(self):
        self.sN_num_es = ti.field(dtype=ti.int32, shape=(num_surface))
        self.sN_P = ti.field(dtype=ti.int32, shape=(num_surface))
        self.num_pes = 0
        for s in range(num_surface):
            s_data = msh_s.cell_data['gmsh:physical'][s]
            self.sN_num_es[s] = s_data.shape[0]
            if s_data[0] == P_surface:
                self.sN_P[s] = 1
                self.num_pes += s_data.shape[0]

        print("sN_P", self.sN_P)
        print("num_pes", self.num_pes)


    @ti.kernel
    def set_pesN_pN(self):
        for es in range(self.num_es_s):
            pes_distin = 0
            es_distin = 0
            es_P = 0
            s = 0
            not_limite = True
            while not_limite:  
                if es < es_distin + self.sN_num_es[s]:
                    not_limite = False
                    es_P = self.sN_P[s]
                else:
                    es_distin += self.sN_num_es[s]
                    pes_distin += self.sN_num_es[s] if self.sN_P[s] else 0
                    s += 1
            
            if es_P:
                pesN = pes_distin + (es - es_distin)
                self.pesN_pN[pesN, 0] = self.esN_pN[es, 0]
                self.pesN_pN[pesN, 1] = self.esN_pN[es, 1]
                self.pesN_pN[pesN, 2] = self.esN_pN[es, 2]
                if pesN >= self.num_pes : print(pesN)
            

    def get_diri_IxIyIz_np(self):
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
        


    def get_add_IxIyIz_np(self):
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





    def export_Solid(self):
        points = self.pos_p.to_numpy()[:self.num_p_s]
        tetra = msh_s.cells_dict[self.ELE_s]
        cells = [
            (self.ELE_s, tetra)
        ]
        mesh_ = meshio.Mesh(
            points,
            cells,
            cell_data = {
                # "sigma_max" : [sigma_max.to_numpy()],
                # "sigma_mu" : [sigma_mu.to_numpy()],
                # "U_ele" : [U_ele.to_numpy()]
            }
        )
        mesh_.write(dr + "/" + FILE_NAME + "_SOLID{:04d}".format(self.output_times[None]) + ".vtu")
    


    def export_Fluid(self):
        num_f_end = self.num_p_s + self.num_p_f_init + self.add_times[None] * self.num_p_f_add
        pos_p_np = self.pos_p.to_numpy()[self.num_p_s:num_f_end,:]
        pos_p_x_np = pos_p_np[:,0].copy()
        pos_p_y_np = pos_p_np[:,1].copy()
        pos_p_z_np = pos_p_np[:,2].copy()
        P_p_np = self.P_p.to_numpy()[self.num_p_s:num_f_end]
        point_data = {"pressure": P_p_np.copy()}
        pointsToVTK(dr + "/" + FILE_NAME + "_FLUID{:04d}".format(self.output_times[None]), pos_p_x_np, pos_p_y_np, pos_p_z_np, data=point_data)
    

    def export_pos_diri_I(self):
        pos_diri_I_x_np = np.zeros((self.num_diri_I, dim), dtype=float)
        pos_diri_I_y_np = np.zeros((self.num_diri_I, dim), dtype=float)
        pos_diri_I_z_np = np.zeros((self.num_diri_I, dim), dtype=float)
        for _i in range(self.num_diri_I):
            ixiyiz = self.diri_I[_i]
            iz, ixiy = ixiyiz // (nx*ny), ixiyiz % (nx*ny)
            iy, ix = ixiy // nx, ixiy % nx
            pos_diri_I_x_np[_i] = ix * dx + area_start[0]
            pos_diri_I_y_np[_i] = iy * dx + area_start[1]
            pos_diri_I_z_np[_i] = iz * dx + area_start[2]
        pointsToVTK(dr + "/" + FILE_NAME + "_DIRI", pos_diri_I_x_np, pos_diri_I_y_np, pos_diri_I_z_np, data={})
    
    
    def export_pos_I(self):
        pos_I_x_np = np.zeros((nx*ny*nz, dim), dtype=float)
        pos_I_y_np = np.zeros((nx*ny*nz, dim), dtype=float)
        pos_I_z_np = np.zeros((nx*ny*nz, dim), dtype=float)
        for ixiyiz in range(nx * ny * nz):
            iz, ixiy = ixiyiz // (nx*ny), ixiyiz % (nx*ny)
            iy, ix = ixiy // nx, ixiy % nx
            pos_I_x_np[ixiyiz] = ix * dx + area_start[0]
            pos_I_y_np[ixiyiz] = iy * dx + area_start[1]
            pos_I_z_np[ixiyiz] = iz * dx + area_start[2]
        pointsToVTK(dr + "/" + FILE_NAME + "_POSI", pos_I_x_np, pos_I_y_np, pos_I_z_np, data={})
    


    @ti.kernel
    def cal_StrainEnergy(self):
        for t in range(self.num_t_s):
            a, b, c, d = self.tN_pN[t, 0], self.tN_pN[t, 1], self.tN_pN[t, 2], self.tN_pN[t, 3]
            Ref = ti.Matrix.cols([self.pos_s_rest[b] - self.pos_s_rest[a], self.pos_s_rest[c] - self.pos_s_rest[a], self.pos_s_rest[d] - self.pos_s_rest[a]])
            Crn = ti.Matrix.cols([self.pos_p[b] - self.pos_p[a], self.pos_p[c] - self.pos_p[a], self.pos_p[d] - self.pos_p[a]])
            F = Crn @ Ref.inverse()
            Vol = 1 / 6 * ti.abs(Ref.determinant())
            I1 = (F @ F.transpose()).trace()
            J = F.determinant()
            elementEnergy = 0.5 * mu_s * (I1 - dim) - mu_s * ti.log(J) + 0.5 * la_s * ti.log(J)**2
            self.StrainEnergy[None] += elementEnergy * Vol
        

    @ti.kernel 
    def cal_norm_S(self):
        for pes in range(self.num_pes):
            a, b, c = self.pesN_pN[pes, 0], self.pesN_pN[pes, 1], self.pesN_pN[pes, 2]
            pos_a, pos_b, pos_c = self.pos_p[a], self.pos_p[b], self.pos_p[c]
            vec_ab, vec_ac = pos_b - pos_a, pos_c - pos_a
            vec_ab_ac_cross = vec_ab.cross(vec_ac)
            norm_es_this = vec_ab_ac_cross.normalized()
            pos_w = (pos_a + pos_b + pos_c) / 3
            
            base = ti.cast((pos_w - area_start) * inv_dx - 0.5, ti.i32)
            fx = (pos_w - area_start) * inv_dx - ti.cast(base, float)
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    for k in ti.static(range(3)):
                        ix, iy, iz = base.x + i, base.y + j, base.z + k
                        self.norm_S[ix, iy, iz] += w[i].x * w[j].y * w[k].z * norm_es_this
        
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
    def cal_m_p_f_I(self):
        for p in range(self.num_p_s + self.num_p_f_init + self.add_times[None] * self.num_p_f_add):
            Type = self.type_p[p]
            if Type == WAIT : continue
            base = ti.cast((self.pos_p[p] - area_start) * inv_dx - 0.5, ti.i32)
            fx = (self.pos_p[p] - area_start) * inv_dx - ti.cast(base, float)
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
                        f_p_int = - self.pos_p.grad[p] if Type == SOLID else ti.Vector([0.0, 0.0, 0.0])
                        self.m_I[ix, iy, iz, Type] += NpI * self.m_p[p]
                        # self.p_I[ix, iy, iz, Type] += NpI * (self.m_p[p] * (self.vel_p[p] + self.C_p[p] @ dist) + dt * f_p_int)
                        self.p_I[ix, iy, iz, Type] += NpI * (self.m_p[p] * (self.vel_p[p] + self.C_p[p] @ dist) + dt * (f_p_int + self.f_p_ext[p]))
                        if Type == FLUID:
                            self.f_F[ix, iy, iz] += NpI * self.m_p[p] * gi
                            self.f_F[ix, iy, iz][0] += -self.m_p[p] / self.rho_p[p] * (self.sigma_p[p][0, 0] * dNpIdx[0] + self.sigma_p[p][0, 1] * dNpIdx[1] + self.sigma_p[p][0, 2] * dNpIdx[2])
                            self.f_F[ix, iy, iz][1] += -self.m_p[p] / self.rho_p[p] * (self.sigma_p[p][1, 0] * dNpIdx[0] + self.sigma_p[p][1, 1] * dNpIdx[1] + self.sigma_p[p][1, 2] * dNpIdx[2])
                            self.f_F[ix, iy, iz][2] += -self.m_p[p] / self.rho_p[p] * (self.sigma_p[p][2, 0] * dNpIdx[0] + self.sigma_p[p][2, 1] * dNpIdx[1] + self.sigma_p[p][2, 2] * dNpIdx[2])
                        
                        

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
            self.p_I[ix, iy, iz, FLUID] = vel_add * self.m_I[ix, iy, iz, FLUID]



    @ti.kernel
    def plus_vel_dpos_p(self):
        for p in range(self.num_p_s + self.num_p_f_init + self.add_times[None] * self.num_p_f_add):
            Type = self.type_p[p]
            if Type == WAIT : continue
            base = ti.cast((self.pos_p[p] - area_start) * inv_dx - 0.5, ti.i32)
            fx = (self.pos_p[p] - area_start) * inv_dx - ti.cast(base, float)
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
            new_C_p, new_vel_p, new_d_pos_p = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), ti.Vector([0.0, 0.0, 0.0]), ti.Vector([0.0, 0.0, 0.0])
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    for k in ti.static(range(3)):
                        ix, iy, iz = base.x + i, base.y + j, base.z + k
                        dist = (float(ti.Vector([i, j, k])) - fx) * dx
                        NpI = w[i].x * w[j].y * w[k].z
                        vel_this = self.p_I[ix, iy, iz, Type] / self.m_I[ix, iy, iz, Type]
                        vel_this = ti.Vector([0.0,0.0,0.0]) if self.m_I[ix,iy,iz,Type] == 0.0 else vel_this
                        new_C_p += 4 * inv_dx**2 * NpI * vel_this.outer_product(dist)
                        new_vel_p += NpI * vel_this
                        new_d_pos_p += NpI * vel_this * dt
            self.C_p[p] = new_C_p
            self.vel_p[p] = new_vel_p
            self.d_pos_p[p] = new_d_pos_p
                    
                    
    @ti.kernel
    def update_p_I(self):
        for _p in range(self.num_p_f_init + self.add_times[None] * self.num_p_f_add):
            p = _p + self.num_p_s
            Type = self.type_p[p]
            if Type == FLUID :
                base = ti.cast((self.pos_p[p] - area_start) * inv_dx - 0.5, ti.i32)
                fx = (self.pos_p[p] - area_start) * inv_dx - ti.cast(base, float)
                w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
                for i in ti.static(range(3)):
                    for j in ti.static(range(3)):
                        for k in ti.static(range(3)):
                            ix, iy, iz = base.x + i, base.y + j, base.z + k
                            dist = (float(ti.Vector([i, j, k])) - fx) * dx
                            NpI = w[i].x * w[j].y * w[k].z
                            self.p_I[ix, iy, iz, Type] += NpI * (self.m_p[p] * (self.vel_p[p] + self.C_p[p] @ dist))
                        
                    
                
    @ti.kernel
    def cal_L_p(self):
        for _p in range(self.num_p_f_init + self.add_times[None] * self.num_p_f_add):
            p = _p + self.num_p_s
            Type = self.type_p[p]
            if Type == FLUID:
                base = ti.cast((self.pos_p[p] - area_start) * inv_dx - 0.5, ti.i32)
                fx = (self.pos_p[p] - area_start) * inv_dx - ti.cast(base, float)
                w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2] 
                dw = [(fx - 1.5)  * inv_dx, -2 * (fx - 1) * inv_dx, (fx - 0.5) * inv_dx]
                new_L_p = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
                for i in ti.static(range(3)):
                    for j in ti.static(range(3)):
                        for k in ti.static(range(3)):
                            ix, iy, iz = base.x + i, base.y + j, base.z + k
                            vel_I_this = self.p_I[ix, iy, iz, Type] / self.m_I[ix, iy, iz, Type]
                            vel_I_this = [0.0, 0.0, 0.0] if self.m_I[ix, iy, iz, Type] == 0 else vel_I_this
                            dv = ti.Matrix([
                                [dw[i].x * w[j].y * w[k].z * vel_I_this.x, w[i].x * dw[j].y * w[k].z * vel_I_this.x, w[i].x * w[j].y * dw[k].z * vel_I_this.x],
                                [dw[i].x * w[j].y * w[k].z * vel_I_this.y, w[i].x * dw[j].y * w[k].z * vel_I_this.y, w[i].x * w[j].y * dw[k].z * vel_I_this.y],
                                [dw[i].x * w[j].y * w[k].z * vel_I_this.z, w[i].x * dw[j].y * w[k].z * vel_I_this.z, w[i].x * w[j].y * dw[k].z * vel_I_this.z]
                            ])
                            new_L_p += dv
                self.L_p[p] = new_L_p
                        
                        
    @ti.kernel
    def cal_rho_sigma_p(self):
        Iden = ti.Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        for _p in range(self.num_p_f_init + self.add_times[None] * self.num_p_f_add):
            p = _p + self.num_p_s
            Type = self.type_p[p]
            if Type == FLUID:
                tr_Dep = self.L_p[p].trace() * dt
                self.rho_p[p] /= 1 + tr_Dep
                P_this = kappa_f * ((self.rho_p[p] / rho_f)**gamma_f - 1)
                epsilon_dot = 0.5 * (self.L_p[p] + self.L_p[p].transpose())
                self.sigma_p[p] = 2 * mu_f * epsilon_dot + (lambda_f - P_this) * Iden
                self.P_p[p] = P_this
            
            
    @ti.kernel
    def plus_pos_p(self):
        for p in range(self.num_p_s + self.num_p_f_init + self.add_times[None] * self.num_p_f_add):
            self.pos_p[p] += self.d_pos_p[p]
                        

    @ti.kernel
    def set_pos_p_s(self, pos_p_rest_s : ti.types.ndarray()):
        for _p in range(self.num_p_s):
            p = _p
            for pd in ti.static(range(dim)):
                self.pos_p[p][pd] = pos_p_rest_s[_p, pd]
                self.pos_s_rest[p][pd] = pos_p_rest_s[_p, pd]
                
        
    @ti.kernel
    def set_f_init(self):
        for _p in range(self.num_p_f_init):
            p = _p + self.num_p_s
            a, b, c, d = self.tN_pN_f_init[_p, 0], self.tN_pN_f_init[_p, 1], self.tN_pN_f_init[_p, 2], self.tN_pN_f_init[_p, 3]
            pos_a, pos_b, pos_c, pos_d = self.msh_points_f_init[a], self.msh_points_f_init[b], self.msh_points_f_init[c], self.msh_points_f_init[d]
            vec_ab, vec_ac, vec_ad = pos_b - pos_a, pos_c - pos_a, pos_d - pos_a
            outer_ab_ac = [
                vec_ab[1] * vec_ac[2] - vec_ab[2] * vec_ac[1],
                vec_ab[2] * vec_ac[0] - vec_ab[0] * vec_ac[2],
                vec_ab[0] * vec_ac[1] - vec_ab[1] * vec_ac[0]
            ]
            dot_outer_ab_ac_vec_ad = outer_ab_ac[0] * vec_ad[0] + outer_ab_ac[1] * vec_ad[1] + outer_ab_ac[2] * vec_ad[2]
            Vol = 1 / 6 * ti.abs(dot_outer_ab_ac_vec_ad)
            self.m_p[p] = rho_f * Vol
            self.rho_p[p] = rho_f
            self.pos_p[p] = 0.25 * (pos_a + pos_b + pos_c + pos_d)
        
        
    def set_f_add(self):
        for p in range(self.num_p_f_add):
            a, b, c, d = msh_f_add.cells_dict[self.ELE_f][p, 0], msh_f_add.cells_dict[self.ELE_f][p, 1], msh_f_add.cells_dict[self.ELE_f][p, 2], msh_f_add.cells_dict[self.ELE_f][p, 3]
            pos_a, pos_b, pos_c, pos_d = msh_f_add.points[a, :], msh_f_add.points[b, :], msh_f_add.points[c, :], msh_f_add.points[d, :]
            vec_ab, vec_ac, vec_ad = pos_b - pos_a, pos_c - pos_a, pos_d - pos_a
            outer_ab_ac = [
                vec_ab[1] * vec_ac[2] - vec_ab[2] * vec_ac[1],
                vec_ab[2] * vec_ac[0] - vec_ab[0] * vec_ac[2],
                vec_ab[0] * vec_ac[1] - vec_ab[1] * vec_ac[0]
            ]
            dot_outer_ab_ac_vec_ad = outer_ab_ac[0] * vec_ad[0] + outer_ab_ac[1] * vec_ad[1] + outer_ab_ac[2] * vec_ad[2]
            Vol = 1 / 6 * ti.abs(dot_outer_ab_ac_vec_ad)
            self.pos_p_add[p] = ti.Vector((pos_a + pos_b + pos_c + pos_d) / 4)
            self.m_p_add[p] = rho_f * Vol
        
        
    @ti.kernel
    def add_f(self):
        for _p in range(self.num_p_f_add):
            p = _p + self.num_p_s + self.num_p_f_init + self.num_p_f_add * self.add_times[None]
            self.pos_p[p] = self.pos_p_add[_p]
            self.m_p[p] = self.m_p_add[_p]
            self.rho_p[p] = rho_f
            self.type_p[p] = FLUID
        
        
    @ti.kernel
    def cal_m_s(self):
        for t in range(self.num_t_s):
            a, b, c, d = self.tN_pN[t, 0], self.tN_pN[t, 1], self.tN_pN[t, 2], self.tN_pN[t, 3]
            pos_a, pos_b, pos_c, pos_d = self.pos_s_rest[a], self.pos_s_rest[b], self.pos_s_rest[c], self.pos_s_rest[d]
            matR = ti.Matrix.cols([pos_b - pos_a, pos_c - pos_a, pos_d - pos_a])
            Vol = 1 / 6 * ti.abs(matR.determinant()) 
            for _alpha in ti.static(range(self.num_node_ele_s)):
                alpha = self.tN_pN[t, _alpha]
                self.m_p[alpha] += rho_s * Vol / self.num_node_ele_s
                self.f_p_ext[alpha] += gi * rho_s * Vol / self.num_node_ele_s        
        
    @ti.kernel
    def set_type_p(self):
        for _p in range(self.num_p_s):
            p = _p
            self.type_p[p] = SOLID
        for _p in range(self.num_p_f_init):
            p = _p + self.num_p_s
            self.type_p[p] = FLUID
        for _p in range(self.num_p_f_add * num_add):
            p = _p + self.num_p_s + self.num_p_f_init
            self.type_p[p] = WAIT
        
    def main(self):
        print("roop start")
        for time_step in range(max_number):
            if time_step % 100 == 0:
                print(time_step)
            
            if time_step % add_span_time_step == 0:
                if self.add_times[None] <= num_add:
                    self.add_f()
                    self.add_times[None] += 1
                    
                    
            if time_step % output_span == 0:
                print(time_step)
                if export:
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
            self.cal_m_p_f_I()
            self.diri_norm_S()
            self.plus_p_I_by_contact()
            self.diri_p_I()
            
            self.plus_vel_dpos_p()
            
            self.p_I.fill(0)
            self.update_p_I()
            self.diri_p_I()
            self.cal_L_p()
            self.cal_rho_sigma_p()
            self.plus_pos_p()
            
BendingArmObj = BendingArm()

if __name__ == '__main__':
    BendingArmObj.main()
            
