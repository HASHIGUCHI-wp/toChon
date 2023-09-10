
import taichi as ti
import numpy as np
import sympy as sy
import meshio
import sys

@ti.data_oriented
class Solid_P1T:
    def __init__(
            self, 
            msh_s, rho_s, young_s, nu_s, la_s, mu_s, dt, nip = 0, dim = 3, gi = ti.Vector([0.0, 0.0, 0.0]), press_const = 0.0,
            INVERSE_NORM = True, ATTENUATION_s = False, PRESS_LABEL = 2, FIX = 1, BIG = 1.0e20, DIVERGENCE = 1.0
        ):
        self.dim = dim
        self.dt = dt
        self.rho_s = rho_s
        self.young_s = young_s
        self.nu_s = nu_s
        self.la_s = la_s
        self.mu_s = mu_s
        self.gi = gi
        self.press_const = press_const
        self.PRESS_LABEL = PRESS_LABEL
        self.INVERSE_NORM = INVERSE_NORM
        self.ATTENUATION_s = ATTENUATION_s
        self.BIG = BIG
        self.FIX = FIX
        self.DIVERGENCE = DIVERGENCE
        self.ELE_s, self.SUR_s = "tetra", "triangle"
        self.num_p_s = msh_s.points.shape[0]
        self.num_t_s, self.num_node_ele_s = msh_s.cells_dict[self.ELE_s].shape
        self.num_es_s, self.num_node_sur_s = msh_s.cells_dict[self.SUR_s].shape
        
    def set_taichi_field(self):
        self.m_p_s = ti.field(dtype=float, shape=self.num_p_s)
        self.sN_fix = ti.field(dtype=ti.i32, shape=self.num_p_s)
        self.pos_p_s = ti.Vector.field(self.dim, dtype=float, shape=self.num_p_s, needs_grad=True)
        self.pos_p_s_rest = ti.Vector.field(self.dim, dtype=float, shape=self.num_p_s)
        self.vel_p_s = ti.Vector.field(self.dim, dtype=float, shape=self.num_p_s)
        self.f_p_int_s = ti.Vector.field(self.dim, dtype=float, shape=self.num_p_s)
        self.f_p_ext_s = ti.Vector.field(self.dim, dtype=float, shape=self.num_p_s)
        self.C_p_s = ti.Vector.field(self.dim, dtype=float, shape=self.num_p_s)
        self.tN_pN_arr_s = ti.field(dtype=ti.i32, shape=(self.num_t_s, self.num_node_ele_s))
        self.esN_pN_arr_s = ti.field(dtype=ti.i32, shape=(self.num_es_s, self.num_node_sur_s))
        self.StrainEnergy = ti.field(dtype=float, shape=(), needs_grad=True)
        self.alpha_Dum = ti.field(dtype=float, shape=())
        self.Press_all = ti.field(dtype=float, shape=())
        self.divergence_s = ti.field(dtype=float, shape=())


    def set_data_solid(self) :
        self.data_solid = {
            "Attenuation" : self.ATTENUATION_s,
            "element" : self.ELE_s,
            "surface"  : self.SUR_s,
            "young_s" : self.young_s,
            "nu_s" : self.nu_s,
            "rho_s" : self.rho_s
        }

    def set_s_init(self, msh_s) :
        self.pos_p_s.from_numpy(msh_s.points)
        self.pos_p_s_rest.from_numpy(msh_s.points)
        self.tN_pN_arr_s.from_numpy(msh_s.cells_dict[self.ELE_s])
        self.esN_pN_arr_s.from_numpy(msh_s.cells_dict[self.SUR_s])


    def get_es_press(self, msh_s) :
        es_press_np = np.arange(0, self.num_es_s)[msh_s.cell_data['gmsh:physical'][0] == self.PRESS_LABEL]
        self.num_es_press = es_press_np.shape[0]
        self.es_press = ti.field(dtype=ti.i32, shape=self.num_es_press)
        self.es_press.from_numpy(es_press_np)
        self.esN_pN_press = ti.field(dtype=ti.i32, shape=(self.num_es_press, 3))


    @ti.kernel
    def set_esN_pN_press(self):
        for _es in range(self.num_es_press):
            es = self.es_press[_es]
            self.esN_pN_press[_es, 0] = self.esN_pN_arr_s[es, 0]
            self.esN_pN_press[_es, 1] = self.esN_pN_arr_s[es, 1]
            self.esN_pN_press[_es, 2] = self.esN_pN_arr_s[es, 2]


    
    @ti.kernel
    def cal_m_p_s(self):
        for t in range(self.num_t_s):
            a, b, c, d = self.tN_pN_arr_s[t, 0], self.tN_pN_arr_s[t, 1], self.tN_pN_arr_s[t, 2], self.tN_pN_arr_s[t, 3]
            pos_a, pos_b, pos_c, pos_d = self.pos_p_s_rest[a], self.pos_p_s_rest[b], self.pos_p_s_rest[c], self.pos_p_s_rest[d]
            matR = ti.Matrix.cols([pos_b - pos_a, pos_c - pos_a, pos_d - pos_a])
            Vol = 1 / 6 * ti.abs(matR.determinant()) 
            for _alpha in ti.static(range(self.num_node_ele_s)):
                alpha = self.tN_pN_arr_s[t, _alpha]
                self.m_p_s[alpha] += self.rho_s * Vol / self.num_node_ele_s
    

    def cal_Press_all(self): 
        self.Press_all[None] = self.press_const

    @ti.kernel
    def plus_f_p_ext_from_press(self):
        for _es in range(self.num_es_press):
            pass

    @ti.kernel
    def plus_f_ext_p_from_body_force(self):
        for t in range(self.num_t_s):
            a, b, c, d = self.tN_pN_arr_s[t, 0], self.tN_pN_arr_s[t, 1], self.tN_pN_arr_s[t, 2], self.tN_pN_arr_s[t, 3]
            pos_a, pos_b, pos_c, pos_d = self.pos_p_s_rest[a], self.pos_p_s_rest[b], self.pos_p_s_rest[c], self.pos_p_s_rest[d]
            matR = ti.Matrix.cols([pos_b - pos_a, pos_c - pos_a, pos_d - pos_a])
            Vol = 1 / 6 * ti.abs(matR.determinant()) 
            for _alpha in ti.static(range(self.num_node_ele_s)):
                alpha = self.tN_pN_arr_s[t, _alpha]
                self.f_p_ext_s[alpha] += self.rho_s * Vol / self.num_node_ele_s * self.gi

    @ti.kernel
    def cal_StrainEnergy(self):
        for t in range(self.num_t_s):
            a, b, c, d = self.tN_pN_arr_s[t, 0], self.tN_pN_arr_s[t, 1], self.tN_pN_arr_s[t, 2], self.tN_pN_arr_s[t, 3]
            Ref = ti.Matrix.cols([self.pos_p_s_rest[b] - self.pos_p_s_rest[a], self.pos_p_s_rest[c] - self.pos_p_s_rest[a], self.pos_p_s_rest[d] - self.pos_p_s_rest[a]])
            Crn = ti.Matrix.cols([self.pos_p_s[b] - self.pos_p_s[a], self.pos_p_s[c] - self.pos_p_s[a], self.pos_p_s[d] - self.pos_p_s[a]])
            F = Crn @ Ref.inverse()
            Vol = 1 / 6 * ti.abs(Ref.determinant())
            I1 = (F @ F.transpose()).trace()
            J = F.determinant()
            elementEnergy = 0.5 * self.mu_s * (I1 - self.dim) - self.mu_s * ti.log(J) + 0.5 * self.la_s * ti.log(J)**2
            self.StrainEnergy[None] += elementEnergy * Vol
            
            
    @ti.kernel
    def cal_f_p_int_s(self) :
        for s in range(self.num_p_s) :
            self.f_p_int_s[s] = - self.pos_p_s.grad[s]

    @ti.kernel
    def cal_alpha_Dum(self) :
        if self.ATTENUATION_s :
            uKu, uMu = 0.0, 0.0
            for p in range(self.num_p_s):
                if self.sN_fix[p] == self.FIX : continue
                u_this = self.pos_p_s[p] - self.pos_p_s_rest[p]
                uKu += u_this.dot(self.pos_p_s.grad[p])
                uMu += self.dim * self.m_p_s[p] * u_this.norm_sqr()
            self.alpha_Dum[None] = 2 * ti.sqrt(uKu / uMu) if uMu > 0.0 else 0.0


    @ti.kernel
    def plus_vel_pos_p(self):
        if self.ATTENUATION_s :
            beta = 0.5 * self.dt * self.alpha_Dum[None]
            for p in range(self.num_p_s):
                if self.sN_fix[p] == self.FIX : continue
                f_p_init = - self.pos_p_s.grad[p]
                self.vel_p_s[p] = (1 - beta) / (1 + beta) * self.vel_p_s[p] + self.dt * (self.f_p_ext_s[p] + f_p_init) / (self.m_p_s[p] * (1 + beta))
                self.pos_p_s[p] = self.pos_p_s[p] + self.dt * self.vel_p_s[p]
                if  not(self.pos_p_s[p].x < self.BIG) :
                    self.divergence_s[None] = self.DIVERGENCE
        else :
            for p in range(self.num_p_s):
                if self.sN_fix[p] == self.FIX : continue
                f_p_init = - self.pos_p_s.grad[p]
                self.vel_p_s[p] += self.dt * (self.f_p_ext_s[p] + f_p_init) / self.m_p_s[p]
                self.pos_p_s[p] += self.dt * self.vel_p_s[p]
                if  not(self.pos_p_s[p].x < self.BIG) :
                    self.divergence_s[None] = self.DIVERGENCE

    def whether_continue_s(self):
        if self.divergence_s[None] == self.DIVERGENCE :
            sys.exit("Error : The values diverged.")


    def export_Solid(self, dir):
        pos_p_rest_np = self.pos_p_s_rest.to_numpy()
        cells = [
            (self.ELE_s, self.tN_pN_arr_s.to_numpy())
        ]
        mesh_ = meshio.Mesh(
            pos_p_rest_np,
            cells,
            point_data = {
                "displacememt" : self.pos_p_s.to_numpy() - pos_p_rest_np
            },
            cell_data = {
                # "sigma_max" : [sigma_max.to_numpy()],
                # "sigma_mu" : [sigma_mu.to_numpy()],
                # "U_ele" : [U_ele.to_numpy()]
            }
        )
        mesh_.write(dir)
        
    @ti.kernel
    def cal_norm_S(self) :
        for _es in range(self.num_es_press):
            a, b, c = self.esN_pN_press[_es, 0], self.esN_pN_press[_es, 1], self.esN_pN_press[_es, 2]
            pos_a, pos_b, pos_c = self.pos_p_s[a], self.pos_p_s[b], self.pos_p_s[c]
            vec_ab, vec_ac = pos_b - pos_a, pos_c - pos_a
            vec_ab_ac_cross = vec_ab.cross(vec_ac)
            norm_es_this = vec_ab_ac_cross.normalized()
            pos_w = (pos_a + pos_b + pos_c) / 3
            
            base = ti.cast((pos_w - self.area_start) * self.inv_dx - 0.5, ti.i32)
            fx = (pos_w - self.area_start) * self.inv_dx - ti.cast(base, float)
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    for k in ti.static(range(3)):
                        ix, iy, iz = base.x + i, base.y + j, base.z + k
                        self.norm_S[ix, iy, iz] += w[i].x * w[j].y * w[k].z * norm_es_this
