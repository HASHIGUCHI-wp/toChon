import taichi as ti
import numpy as np
import sympy as sy
import meshio
import sys

@ti.data_oriented
class Solid_P2Q:
    def __init__(
            self, 
            msh_s, rho_s, young_s, nu_s, la_s, mu_s, dt, nip, dim = 3, gi = ti.Vector([0.0, 0.0, -9.81]), press_const = 0.0,
            INVERSE_NORM = True, ATTENUATION_s = False, PRESS_LABEL = 2, FIX = 1, BIG = 1.0e20, DIVERGENCE = 1.0
        ):
        self.dim = dim
        self.nip = nip
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
        self.ELE_s, self.SUR_s = "hexahedron27", "quad9"
        self.num_p_s = msh_s.points.shape[0]
        self.num_t_s, self.num_node_ele_s = msh_s.cells_dict[self.ELE_s].shape
        self.num_es_s, self.num_node_sur_s = msh_s.cells_dict[self.SUR_s].shape
        self.num_gauss = self.num_t_s * nip**dim
        

        self.m_p_s = ti.field(dtype=float, shape=self.num_p_s)
        self.sN_fix = ti.field(dtype=ti.i32, shape=self.num_p_s)
        self.pos_p_s = ti.Vector.field(dim, dtype=float, shape=self.num_p_s, needs_grad=True)
        self.pos_p_s_rest = ti.Vector.field(dim, dtype=float, shape=self.num_p_s)
        self.vel_p_s = ti.Vector.field(dim, dtype=float, shape=self.num_p_s)
        self.f_p_ext_s = ti.Vector.field(dim, dtype=float, shape=self.num_p_s)
        self.C_p_s = ti.Vector.field(dim, dtype=float, shape=self.num_p_s)
        self.Ja_Ref_s = ti.Matrix.field(dim, dim, dtype=float, shape=(self.num_t_s * nip**dim))
        self.tN_pN_arr_s = ti.field(dtype=ti.i32, shape=(self.num_t_s, self.num_node_ele_s))
        self.tN_pN_s = ti.field(dtype=ti.i32, shape=(self.num_t_s, 3, 3, 3))
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
        self.num_gauss_press = self.num_es_press * self.nip**2
        self.es_press = ti.field(dtype=ti.i32, shape=self.num_es_press)
        self.es_press.from_numpy(es_press_np)
        self.esN_pN_press = ti.field(dtype=ti.i32, shape=(self.num_es_press, 3, 3))


    @ti.kernel
    def set_esN_pN_press(self):
        for _es in range(self.num_es_press):
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
        self.gauss_x = ti.field(dtype=float, shape=self.nip)
        self.gauss_w = ti.field(dtype=float, shape=self.nip)

        self.v_Gauss = ti.field(dtype=float, shape=(3, self.nip))
        self.dv_Gauss = ti.field(dtype=float, shape=(3, self.nip))

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
            # tN_pN_arr = msh_s.cells_dict['hexahedron27'][t]
            self.tN_pN_s[t, 0, 2, 2] = self.tN_pN_arr_s[t,0]
            self.tN_pN_s[t, 0, 0, 2] = self.tN_pN_arr_s[t,1]
            self.tN_pN_s[t, 0, 0, 0] = self.tN_pN_arr_s[t,2]
            self.tN_pN_s[t, 0, 2, 0] = self.tN_pN_arr_s[t,3]

            self.tN_pN_s[t, 2, 2, 2] = self.tN_pN_arr_s[t,4]
            self.tN_pN_s[t, 2, 0, 2] = self.tN_pN_arr_s[t,5]
            self.tN_pN_s[t, 2, 0, 0] = self.tN_pN_arr_s[t,6]
            self.tN_pN_s[t, 2, 2, 0] = self.tN_pN_arr_s[t,7]

            self.tN_pN_s[t, 0, 1, 2] = self.tN_pN_arr_s[t,8]
            self.tN_pN_s[t, 0, 0, 1] = self.tN_pN_arr_s[t,9]
            self.tN_pN_s[t, 0, 1, 0] = self.tN_pN_arr_s[t,10]
            self.tN_pN_s[t, 0, 2, 1] = self.tN_pN_arr_s[t,11]

            self.tN_pN_s[t, 2, 1, 2] = self.tN_pN_arr_s[t,12]
            self.tN_pN_s[t, 2, 0, 1] = self.tN_pN_arr_s[t,13]
            self.tN_pN_s[t, 2, 1, 0] = self.tN_pN_arr_s[t,14]
            self.tN_pN_s[t, 2, 2, 1] = self.tN_pN_arr_s[t,15]

            self.tN_pN_s[t, 1, 2, 2] = self.tN_pN_arr_s[t,16]
            self.tN_pN_s[t, 1, 0, 2] = self.tN_pN_arr_s[t,17]
            self.tN_pN_s[t, 1, 0, 0] = self.tN_pN_arr_s[t,18]
            self.tN_pN_s[t, 1, 2, 0] = self.tN_pN_arr_s[t,19]

            self.tN_pN_s[t, 1, 2, 1] = self.tN_pN_arr_s[t,20]
            self.tN_pN_s[t, 1, 0, 1] = self.tN_pN_arr_s[t,21]
            self.tN_pN_s[t, 1, 1, 2] = self.tN_pN_arr_s[t,22]
            self.tN_pN_s[t, 1, 1, 0] = self.tN_pN_arr_s[t,23]

            self.tN_pN_s[t, 0, 1, 1] = self.tN_pN_arr_s[t,24]
            self.tN_pN_s[t, 2, 1, 1] = self.tN_pN_arr_s[t,25]
            self.tN_pN_s[t, 1, 1, 1] = self.tN_pN_arr_s[t,26]


    @ti.kernel
    def cal_Ja_Ref_s(self):
        for g in range(self.num_gauss):
            t, mnl = g // (self.nip**3), g % (self.nip**3)
            m, nl = mnl // (self.nip**2), mnl % (self.nip**2)
            n, l  = nl // self.nip, nl % self.nip
            for _a1 in ti.static(range(3)):
                for _a2 in ti.static(range(3)):
                    for _a3 in ti.static(range(3)):
                        a = self.tN_pN_s[t, _a1, _a2, _a3]
                        for pd in ti.static(range(self.dim)):
                            self.Ja_Ref_s[g][pd, 0] += self.dv_Gauss[_a1, m] * self.v_Gauss[_a2, n] * self.v_Gauss[_a3, l] * self.pos_p_s_rest[a][pd]
                            self.Ja_Ref_s[g][pd, 1] += self.v_Gauss[_a1, m] * self.dv_Gauss[_a2, n] * self.v_Gauss[_a3, l] * self.pos_p_s_rest[a][pd]
                            self.Ja_Ref_s[g][pd, 2] += self.v_Gauss[_a1, m] * self.v_Gauss[_a2, n] * self.dv_Gauss[_a3, l] * self.pos_p_s_rest[a][pd]

    
    @ti.kernel
    def cal_m_p_s(self):
        for g in range(self.num_gauss):
            t, mnl = g // (self.nip**3), g % (self.nip**3)
            m, nl = mnl // (self.nip**2), mnl % (self.nip**2)
            n, l  = nl // self.nip, nl % self.nip
            ja_ref = self.Ja_Ref_s[g]
            det_ja_ref = ja_ref.determinant()
            det_ja_ref = ti.abs(det_ja_ref)
            for _a1 in ti.static(range(3)):
                for _a2 in ti.static(range(3)):
                    for _a3 in ti.static(range(3)):
                        a = self.tN_pN_s[t, _a1, _a2, _a3]
                        self.m_p_s[a] += self.rho_s * self.v_Gauss[_a1, m] * self.v_Gauss[_a2, n] * self.v_Gauss[_a3, l] * self.gauss_w[m] * self.gauss_w[n] * self.gauss_w[l] * det_ja_ref

    
     
    @ti.kernel
    def cal_SUM_AREA(self):
        for _es in range(self.num_es_press):
            area = 0.0
            for m in ti.static(range(self.nip)):
                for n in ti.static(range(self.nip)):
                    k1, k2 = ti.Vector([0.0, 0.0, 0.0]), ti.Vector([0.0, 0.0, 0.0])
                    for _a1 in ti.static(range(3)):
                        for _a2 in ti.static(range(3)):
                            a = self.esN_pN_press[_es, _a1, _a2]
                            # print(self.pos_p[a])
                            k1 += self.dv_Gauss[_a1, m] * self.v_Gauss[_a2, n] * self.pos_p_s[a]
                            k2 += self.v_Gauss[_a1, m] * self.dv_Gauss[_a2, n] * self.pos_p_s[a]
                    k3 = k1.cross(k2)
                    J = k3.norm()
                    area += J * self.gauss_w[m] * self.gauss_w[n]
            self.SUM_AREA[None] += area
            print(area)

    def cal_Press_all(self): 
        self.Press_all[None] = self.press_const

    @ti.kernel
    def plus_f_p_ext_from_press(self):
        for g in range(self.num_gauss_press):
            _es, mn = g // (self.nip**2), g % (self.nip**2)
            m, n = mn // self.nip, mn % self.nip
            k1, k2 = ti.Vector([0.0, 0.0, 0.0]), ti.Vector([0.0, 0.0, 0.0])
            pos_a = ti.Vector([0.0, 0.0, 0.0])
            for _a1 in ti.static(range(3)):
                for _a2 in ti.static(range(3)):
                    a = self.esN_pN_press[_es, _a1, _a2]
                    pos_a += self.v_Gauss[_a1, m] * self.v_Gauss[_a2, n] * self.pos_p_s[a]
                    k1 += self.dv_Gauss[_a1, m] * self.v_Gauss[_a2, n] * self.pos_p_s[a]
                    k2 += self.v_Gauss[_a1, m] * self.dv_Gauss[_a2, n] * self.pos_p_s[a]
            k3 = k1.cross(k2)
            J = k3.norm()
            norm = k3.normalized()
            norm *= -1.0 if self.INVERSE_NORM else 1.0
            for _a1 in ti.static(range(3)):
                for _a2 in ti.static(range(3)):
                    a = self.esN_pN_press[_es, _a1, _a2]
                    self.f_p_ext_s[a] += self.Press_all[None] * norm * self.v_Gauss[_a1, m] * self.v_Gauss[_a2, n] * J * self.gauss_w[m] * self.gauss_w[n]
            # self.SUM_AREA[None] += J * self.gauss_w[m] * self.gauss_w[n]

    @ti.kernel
    def plus_f_ext_p_from_body_force(self):
        for g in range(self.num_gauss):
            t, mnl = g // (self.nip**3), g % (self.nip**3)
            m, nl = mnl // (self.nip**2), mnl % (self.nip**2)
            n, l  = nl // self.nip, nl % self.nip
            ja_ref = self.Ja_Ref_s[g]
            det_ja_ref = ja_ref.determinant()
            det_ja_ref = ti.abs(det_ja_ref)
            for _a1 in ti.static(range(3)):
                for _a2 in ti.static(range(3)):
                    for _a3 in ti.static(range(3)):
                        a = self.tN_pN_s[t, _a1, _a2, _a3]
                        self.m_p_s[a] += self.rho_s * self.v_Gauss[_a1, m] * self.v_Gauss[_a2, n] * self.v_Gauss[_a3, l] * self.gauss_w[m] * self.gauss_w[n] * self.gauss_w[l] * det_ja_ref
                        self.f_p_ext_s[a] += self.rho_s * self.gi * self.v_Gauss[_a1, m] * self.v_Gauss[_a2, n] * self.v_Gauss[_a3, l] * self.gauss_w[m] * self.gauss_w[n] * self.gauss_w[l] * det_ja_ref


    @ti.kernel
    def cal_StrainEnergy(self):
        for g in range(self.num_gauss):
            t, mnl = g // (self.nip**3), g % (self.nip**3)
            m, nl = mnl // (self.nip**2), mnl % (self.nip**2)
            n, l  = nl // self.nip, nl % self.nip
            ja_ref = self.Ja_Ref_s[g]
            det_ja_ref = ja_ref.determinant()
            det_ja_ref = ti.abs(det_ja_ref)
            inv_Ja_ref = ja_ref.inverse()
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
                        dNadx = inv_Ja_ref @ dNadt
                        FiJ += dNadx.outer_product(self.pos_p_s[a])
            I1 = (FiJ @ FiJ.transpose()).trace()
            J = FiJ.determinant()
            element_energy = 0.5 * self.mu_s * (I1 - self.dim) - self.mu_s * ti.log(J) + 0.5 * self.la_s * ti.log(J)**2
            self.StrainEnergy[None] += element_energy * self.gauss_w[m] * self.gauss_w[n] * self.gauss_w[l] * det_ja_ref

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


    def export_Solid(self, msh_s, dir):
    
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
        mesh_.write(dir)
