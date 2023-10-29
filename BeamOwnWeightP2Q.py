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
    dir_mesh = "./mesh_file/Beam"
mesh_name = "Beam3DTramsfiniteSize0p25"

ATTENUATION_s = False
EXPORT = True
WEAK = True

if EXPORT:
    FOLDER_NAME = "BeamOwnWeight"
    DATE = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    dir_export = "./consequence" + "/" + FOLDER_NAME + "/" + DATE
    os.makedirs(dir_export, exist_ok=True)
    

msh_s = meshio.read(dir_mesh + "/" + mesh_name + ".msh")
nip = 3
rho_s = 1068.72
rho_s = 40.0
young_s, nu_s = 82737.21, 0.36
young_s, nu_s = 4e5, 0.36
la_s, mu_s = young_s * nu_s / ((1 + nu_s) * (1 - 2 * nu_s)) , young_s / (2 * (1 + nu_s))
eta1_s, eta2_s = 0.01 * la_s, 0.01 * mu_s
dx_mesh = 0.25
sound_s = ti.sqrt((la_s + 2 * mu_s) / rho_s)
dt_max = 0.1 * dx_mesh / sound_s
dt = 6.0e-5
dt = 0.00019
gi = ti.Vector([0.0, 0.0, -9.81])

print("dt_max", dt_max)
print("dt", dt)

max_number = 200000
output_span = 1000

COL_POINT = 0
COL_SURFACE = 1
COL_VOLUME = 2



@ti.data_oriented
class collisionBox(Solid_P2Q):
    def __init__(self,
                msh_s,
                rho_s, young_s, nu_s, la_s, mu_s, eta1_s, eta2_s,
                dt, max_number, output_span,
                ATTENUATION_s, WEAK, nip, gi
        ):
        self.msh_s = msh_s
        self.rho_s, self.young_s, self.nu_s, self.la_s, self.mu_s, self.eta1_s, self.eta2_s = rho_s, young_s, nu_s, la_s, mu_s, eta1_s, eta2_s
        self.dt, self.max_number, self.output_span = dt, max_number, output_span
        self.ATTENUATION_s = ATTENUATION_s
        self.WEAK = WEAK
        self.nip = nip
        self.gi = gi

        Solid_P2Q.__init__(self,
            msh_s= msh_s,
            rho_s= rho_s,
            young_s= young_s,
            nu_s= nu_s,
            la_s= la_s,
            mu_s= mu_s,
            eta1_s= eta1_s,
            eta2_s= eta2_s,
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
        self.cal_Ja_Ref_s()
        self.cal_m_p_s()

        print(self.m_p_s.to_numpy().sum())
        print(self.rho_s * 10)

        self.plus_f_ext_p_from_body_force()

        

    def set_taichi_field(self):
        self.time_step = ti.field(dtype=ti.i32, shape=())
        self.output_times = ti.field(dtype=ti.i32, shape=())

    def set_sN_fix(self):
        sN_fix_np = (self.msh_s.points[:, 0] == 0.0).astype(np.int32)
        self.sN_fix.from_numpy(sN_fix_np)

    def weather_continue(self) :
        if self.divergence_s[None] == self.DIVERGENCE :
            sys.exit("value is divergenced")

    def clear(self):
        self.f_p_int_s.fill(0)

        
    def main(self) :
        while self.time_step[None] < self.max_number:
            if self.time_step[None] % self.output_span == 0 :
                print(self.time_step[None])
                if EXPORT:
                    Solid_P2Q.export_Solid(self, dir_export + "/" + "SOLID{:05d}.vtu".format(self.output_times[None]))
                    self.output_times[None] += 1

            if self.WEAK_S :
                self.cal_f_p_int_s_from_WEAK()
            else:
                with ti.Tape(self.StrainEnergy):
                    self.cal_StrainEnergy()

            self.cal_alpha_Dum()
            self.plus_vel_pos_p()
            self.weather_continue()
            self.clear()
            self.time_step[None] += 1




if __name__ == '__main__' :
    obj = collisionBox(
        msh_s= msh_s,
        rho_s= rho_s,
        young_s= young_s,
        nu_s= nu_s,
        la_s= la_s,
        mu_s= mu_s,
        eta1_s= eta1_s,
        eta2_s= eta2_s,
        dt= dt,
        nip= nip,
        gi= gi,
        max_number= max_number,
        output_span= output_span,
        ATTENUATION_s= ATTENUATION_s,
        WEAK = WEAK
    )

    obj.main()
