import taichi as ti
import numpy as np
import meshio
import sys
import datetime
import os
from Solid_P2Q import Solid_P2Q

ti.init(arch=ti.cpu, default_fp=ti.f64)

USING_MACHINE = "CERVO"

if USING_MACHINE == "MAC" :
    dir_mesh = "./mesh_file"
elif USING_MACHINE == "CERVO" :
    dir_mesh = "./mesh_file/Moyashi"
mesh_name = "MoyashiTransfiniteFillingSize1p5"

ATTENUATION_s = True
EXPORT = True
WEAK = True

if EXPORT:
    FOLDER_NAME = "MoyashiExpantionFilling"
    DATE = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    dir_export = "./consequence" + "/" + FOLDER_NAME + "/" + DATE
    os.makedirs(dir_export, exist_ok=True)
    

msh_s = meshio.read(dir_mesh + "/" + mesh_name + ".msh")
dx_mesh = 1.5
Length = 18.0
Width = 12.0
Height = 63.0
l1, l2 = 4.5, 7.5
w = 6.0
h1, h2 = 12.0, 45.0
num_t_s, num_es_s, num_l_s = "hexahedron27", "quad9", "line3"
num_t_s = msh_s.cells_dict[num_t_s].shape[0]



COL_SURFACE = 0
COL_VOLUME = 1


nip = 3
rho_s = 1068.72
rho_s = 40.0
young_s, nu_s = 82737.21, 0.36
young_s, nu_s = 4e5, 0.36
alpha_press = 1.5
la_s, mu_s = young_s * nu_s / ((1 + nu_s) * (1 - 2 * nu_s)) , young_s / (2 * (1 + nu_s))
eta1_s, eta2_s = 0.01 * la_s, 0.01 * mu_s
rho_a = 1.293
k_a = 1.4e5
sound_s = ti.sqrt((la_s + 2 * mu_s) / rho_s)
sound_a = ti.sqrt(k_a / rho_a)
sound_max = sound_s if sound_s > sound_a else sound_a
dt_max = 0.1 * dx_mesh / sound_max
dt = 6.0e-5
dt = 0.00019
gi = ti.Vector([0.0, 0.0, -9.81])

print("dt_max", dt_max)
print("dt", dt)

max_number = 200000
output_span = 100

# max_number = 200
# output_span = 1




@ti.data_oriented
class expansionFilling(Solid_P2Q):
    def __init__(self,
                msh_s,
                rho_s, young_s, nu_s, la_s, mu_s, eta1_s, eta2_s,
                rho_a, k_a,
                dt, max_number, output_span,
                ATTENUATION_s, WEAK, nip, gi
        ):
        self.msh_s = msh_s
        self.rho_s, self.young_s, self.nu_s, self.la_s, self.mu_s, self.eta1_s, self.eta2_s = rho_s, young_s, nu_s, la_s, mu_s, eta1_s, eta2_s
        self.rho_a, self.k_a = rho_a, k_a
        self.dt, self.max_number, self.output_span = dt, max_number, output_span
        self.ATTENUATION_s = ATTENUATION_s
        self.WEAK = WEAK
        self.nip = nip
        self.gi = gi
        self.SOLID, self.AIR = 0, 1

        Solid_P2Q.__init__(self,
            msh_s= msh_s,
            rho_s= rho_s,
            young_s= young_s,
            nu_s= nu_s,
            la_s= la_s,
            mu_s= mu_s,
            eta1_s= eta1_s,
            eta2_s= eta2_s,
            k_a = k_a,
            dt= dt,
            nip = nip,
            sip= 0,
            gi= gi,
            ATTENUATION_s= ATTENUATION_s,
            WEAK = WEAK
        )

        Solid_P2Q.set_taichi_field(self)
        Solid_P2Q.set_data_solid(self)
        Solid_P2Q.set_s_init(self)

        self.set_taichi_field()
        self.set_aN_a1a2a3()
        self.set_sN_fix()
        self.leg_weights_roots(self.nip)
        self.set_type_t()
        self.cal_Ja_Ref_s()
        self.cal_m_p_s()

        print(self.m_p_s.to_numpy().sum())
        vol_inner = w * (h1 * (l1 + l2) + h2 * l1)
        vol_all = Length * Width * Height
        print(self.rho_a * vol_inner + self.rho_s * (vol_all - vol_inner))

    
    def set_type_t(self) :
        self.type_t = ti.field(dtype=ti.i32, shape=self.num_t_s)
        self.type_t.from_numpy(self.msh_s.cell_data['gmsh:physical'][COL_VOLUME])
        self.t_solid_np = np.arange(0, self.num_t_s)[self.msh_s.cell_data['gmsh:physical'][COL_VOLUME] == self.SOLID]

    def set_taichi_field(self):
        self.time_step = ti.field(dtype=ti.i32, shape=())
        self.output_times = ti.field(dtype=ti.i32, shape=())
        self.P_act = ti.field(dtype=float, shape=())

    def set_sN_fix(self):
        sN_fix_np = (self.msh_s.points[:, 2] == 63.0).astype(np.int32)
        self.sN_fix.from_numpy(sN_fix_np)
        
    @ti.kernel
    def cal_m_p_s(self) :
        for g in range(self.num_gauss):
            t, mnl = g // (self.nip**3), g % (self.nip**3)
            m, nl = mnl // (self.nip**2), mnl % (self.nip**2)
            n, l  = nl // self.nip, nl % self.nip
            TYPE = self.type_t[t]
            rho = self.rho_s if TYPE == self.SOLID else self.rho_a
            ja_ref = self.Ja_Ref_s[g]
            det_ja_ref = ja_ref.determinant()
            det_ja_ref = ti.abs(det_ja_ref)
            for _a in ti.static(range(self.num_node_ele_s)):
                p = self.tN_pN_arr_s[t, _a]
                _a1, _a2, _a3 = self.aN_a1a2a3[_a, 0], self.aN_a1a2a3[_a, 1], self.aN_a1a2a3[_a, 2]
                self.m_p_s[p] += rho * self.v_Gauss[_a1, m] * self.v_Gauss[_a2, n] * self.v_Gauss[_a3, l] * self.gauss_w[m] * self.gauss_w[n] * self.gauss_w[l] * det_ja_ref

    

    def weather_continue(self) :
        if self.divergence_s[None] == self.DIVERGENCE :
            sys.exit("value is divergenced")

    def clear(self):
        self.f_p_int_s.fill(0)
        
    
    def cal_P_act(self) :
        self.P_act[None] = alpha_press * self.young_s
        
    
    @ti.kernel
    def cal_f_p_int_s_Weak(self) :
        for g in range(self.num_gauss) :
            t, mnl = g // (self.nip**3), g % (self.nip**3)
            m, nl = mnl // (self.nip**2), mnl % (self.nip**2)
            n, l  = nl // self.nip, nl % self.nip
            if self.type_t[t] == self.SOLID :
                self._cal_f_p_int_s_Weak_NeoHook(g, t, m, n, l)
            elif self.type_t[t] == self.AIR :
                self._cal_f_p_int_s_Weak_Air(g, t, m, n, l)
                
    
    def cal_alpha_Dum(self) :
        self.alpha_Dum[None] = 100.0
    
    
    def export_Solid(self, dir):
        cells = [
            (self.ELE_s, self.msh_s.cells_dict[self.ELE_s][self.t_solid_np, :])
        ]
        mesh_ = meshio.Mesh(
            self.msh_s.points,
            cells,
            point_data = {
                "displacememt" : self.pos_p_s.to_numpy() - self.msh_s.points
            },
            cell_data = {
                # "sigma_max" : [sigma_max.to_numpy()],
                # "sigma_mu" : [sigma_mu.to_numpy()],
                # "U_ele" : [U_ele.to_numpy()]
            }
        )
        mesh_.write(dir)
        
    def main(self) :
        while self.time_step[None] < self.max_number:
            if self.time_step[None] % self.output_span == 0 :
                print(self.time_step[None])
                if EXPORT:
                    self.export_Solid(dir_export + "/" + "SOLID{:05d}.vtu".format(self.output_times[None]))
                    self.output_times[None] += 1
                    
            self.cal_P_act()

            if self.WEAK_S :
                self.cal_f_p_int_s_Weak()
            else:
                with ti.Tape(self.StrainEnergy):
                    self.cal_StrainEnergy()

            self.cal_alpha_Dum()
            self.plus_vel_pos_p()
            self.weather_continue()
            self.clear()
            self.time_step[None] += 1




if __name__ == '__main__' :
    obj = expansionFilling(
        msh_s= msh_s,
        rho_s= rho_s,
        young_s= young_s,
        nu_s= nu_s,
        la_s= la_s,
        mu_s= mu_s,
        eta1_s= eta1_s,
        eta2_s= eta2_s,
        k_a= k_a,
        rho_a= rho_a,
        dt= dt,
        nip= nip,
        gi= gi,
        max_number= max_number,
        output_span= output_span,
        ATTENUATION_s= ATTENUATION_s,
        WEAK = WEAK
    )

    obj.main()
