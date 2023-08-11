import meshio
import taichi as ti
import numpy as np
import datetime
import os
import pandas as pd

ti.init(arch=ti.gpu, default_fp=ti.f64)

USING_MACHINE = "PC"
DEBUG = False
EXPORT = True
PRESS_TIME_CHANGE = "CONST"
FOLDER_NAME = "BendingArmAir"
DATE = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
PRESS_LABEL = 2
FIX = 1
Press = 2000

mesh_name = "bending ac_long_4x4_label"
if USING_MACHINE == "PC" :
    mesh_dir = "./mesh_file/"
    if EXPORT :
        export_dir = "./Consequence/" + FOLDER_NAME + "/" + DATE + "/"
        os.makedirs(export_dir, exist_ok=True)
        os.makedirs(export_dir + "vtu" + "/",  exist_ok=True)
elif USING_MACHINE == "ATLAS":
    mesh_dir = '/home/hashiguchi/mpm_simulation/geometry/BendingArm/'
    if EXPORT :
        export_dir = '/home/hashiguchi/mpm_simulation/result/' + FOLDER_NAME + "/" + DATE + "/"
        os.makedirs(export_dir, exist_ok=True)

dim = 3
err = 1.0e-5
dx_mesh = 0.2
dt = 6.0e-5

rho_s = 4e1
young_s, nu_s = 4e6, 0.2
la_s, mu_s = young_s * nu_s / ((1+nu_s) * (1-2*nu_s)) , young_s / (2 * (1+nu_s))
sound_s = np.sqrt((la_s + 2 * mu_s) / rho_s)
X_right = 56.0

dt_max = 0.1 * dx_mesh / sound_s

max_number = 100000
output_span = 100

print("dt_max", dt_max)
print("dt", dt)


msh_s = meshio.read(mesh_dir + mesh_name + ".msh")
num_es = msh_s.cells_dict['triangle'].shape[0]



if DEBUG: 
    print(np.arange(0, num_es)[msh_s.cell_data['gmsh:physical'][0] == 2])
    print(msh_s.cell_data['gmsh:physical'][0] == 2)
    print(msh_s.cell_data['gmsh:physical'][0].shape[0])
    print(msh_s.cells_dict['triangle'].shape[0])

@ti.data_oriented
class BendingArm():
    def __init__(self):
        self.output_times = ti.field(dtype=ti.i32, shape=())
        self.ELE, self.SUR = "tetra", "triangle"
        self.num_p_s = msh_s.points.shape[0]
        self.num_t_s, self.num_node_ele = msh_s.cells_dict[self.ELE].shape
        self.num_es_s, self.num_node_sur = msh_s.cells_dict[self.SUR].shape
        self.num_p = self.num_p_s
        
        self.StrainEnergy = ti.field(dtype=float, shape=(), needs_grad=True)
        self.m_p = ti.field(dtype=float, shape=self.num_p)
        self.vel_p = ti.Vector.field(dim, dtype=float, shape=self.num_p)
        self.pos_p = ti.Vector.field(dim, dtype=float, shape=self.num_p, needs_grad=True)
        self.pos_p_rest = ti.Vector.field(dim, dtype=float, shape=self.num_p)
        self.f_p_ext = ti.Vector.field(dim, dtype=float, shape=self.num_p)
        self.pN_fix = ti.field(dtype=ti.i32, shape=self.num_p)
        self.tN_pN = ti.field(dtype=ti.i32, shape=(self.num_t_s, self.num_node_ele))
        self.esN_pN = ti.field(dtype=ti.i32, shape=(self.num_es_s, self.num_node_sur))
        
        self.pos_p_rest.from_numpy(msh_s.points)
        self.pos_p.from_numpy(msh_s.points)
        self.tN_pN.from_numpy(msh_s.cells_dict[self.ELE])
        self.esN_pN.from_numpy(msh_s.cells_dict[self.SUR])
        
        
        self.get_es_press()
        self.set_pN_fix()
        self.cal_m_p()
        self.export_info()
        
    def get_es_press(self):
       es_press_np = np.arange(0, num_es)[msh_s.cell_data['gmsh:physical'][0] == PRESS_LABEL]
       self.num_es_press = es_press_np.shape[0]
       self.es_press = ti.field(dtype=ti.i32, shape=self.num_es_press)
       self.es_press.from_numpy(es_press_np)
    
    @ti.kernel
    def set_pN_fix(self):
        for p in range(self.num_p):
            pos_p_x = self.pos_p_rest[p].x
            if ti.abs(pos_p_x - X_right) < err :
                self.pN_fix[p] = FIX
                if DEBUG:
                    print(self.pos_p[p])
            
    
    @ti.kernel
    def cal_m_p(self) :
        for t in range(self.num_t_s):
            a, b, c, d = self.tN_pN[t, 0], self.tN_pN[t, 1], self.tN_pN[t, 2], self.tN_pN[t, 3]
            pos_a, pos_b, pos_c, pos_d = self.pos_p_rest[a], self.pos_p_rest[b], self.pos_p_rest[c], self.pos_p_rest[d]
            matR = ti.Matrix.cols([pos_b - pos_a, pos_c - pos_a, pos_d - pos_a])
            Vol = 1 / 6 * ti.abs(matR.determinant()) 
            self.m_p[a] += rho_s * Vol / self.num_node_ele
            self.m_p[b] += rho_s * Vol / self.num_node_ele
            self.m_p[c] += rho_s * Vol / self.num_node_ele
            self.m_p[d] += rho_s * Vol / self.num_node_ele
        
    def export_info(self):
        
        data = {
            "Scheme" : "FEM",
            "mesh_name" : mesh_name,
            "Press_Max" : Press,
            "Press_Time_Change" : PRESS_TIME_CHANGE,
            "max_number" : max_number,
            "output_span" : output_span,
            "date" : DATE ,
            "element" : self.ELE,
            "surface"  : self.SUR,
            "dt" : dt
        }
        
        s = pd.Series(data)
        s.to_csv(export_dir + "Information", header=False)
        
        print("num_p", self.num_p_s)
        print("num_t", self.num_t_s)
        print("num_es", self.num_es_s)
        print("num_es_press", self.num_es_press)
        
    def export_Solid(self):
        points = self.pos_p.to_numpy()
        print(points)
        cells = [
            (self.ELE, msh_s.cells_dict[self.ELE])
        ]
        mesh_ = meshio.Mesh(
            msh_s.points,
            cells,
            point_data = {
                "displacement" : points - msh_s.points
            },
            cell_data = {
                # "sigma_max" : [sigma_max.to_numpy()],
                # "sigma_mu" : [sigma_mu.to_numpy()],
                # "U_ele" : [U_ele.to_numpy()]
            }
        )
        mesh_.write(export_dir + "/" + "vtu" + "/" + "SOLID{:04d}".format(self.output_times[None]) + ".vtu")
        
    @ti.kernel
    def cal_StrainEnergy(self):
        for t in range(self.num_t_s):
            a, b, c, d = self.tN_pN[t,0], self.tN_pN[t,1], self.tN_pN[t,2], self.tN_pN[t,3]
            Ref = ti.Matrix.cols([self.pos_p_rest[b] - self.pos_p_rest[a], self.pos_p_rest[c] - self.pos_p_rest[a], self.pos_p_rest[d] - self.pos_p_rest[a]])
            Crn = ti.Matrix.cols([self.pos_p[b] - self.pos_p[a], self.pos_p[c] - self.pos_p[a], self.pos_p[d] - self.pos_p[a]])
            F = Crn @ Ref.inverse()
            Vol = 1 / 6 * ti.abs(Ref.determinant())
            I1 = (F @ F.transpose()).trace()
            J = F.determinant()
            elementEnergy = 0.5 * mu_s * (I1 - dim) - mu_s * ti.log(J) + 0.5 * la_s * ti.log(J)**2
            self.StrainEnergy[None] += elementEnergy * Vol
            
    @ti.kernel
    def cal_f_ext_p(self):
        for _es in range(self.num_es_press):
            es = self.es_press[_es]
            a, b, c = self.esN_pN[es, 0], self.esN_pN[es, 1], self.esN_pN[es, 2]
            pos_a, pos_b, pos_c = self.pos_p[a], self.pos_p[b], self.pos_p[c]
            vec_ab, vec_ac = pos_b - pos_a, pos_c - pos_a
            vec_ab_ac_cross = vec_ab.cross(vec_ac)
            area = 0.5 * ti.abs(ti.sqrt(vec_ab_ac_cross[0]**2 + vec_ab_ac_cross[1]**2 + vec_ab_ac_cross[2]**2))
            norm_es_this = vec_ab_ac_cross.normalized()
            self.f_p_ext[a] += - 1 / 3 * norm_es_this * Press * area
            self.f_p_ext[b] += - 1 / 3 * norm_es_this * Press * area
            self.f_p_ext[c] += - 1 / 3 * norm_es_this * Press * area
            
    @ti.kernel
    def plus_vel_pos_p(self):
        for p in range(self.num_p):
            if self.pN_fix[p] == FIX : continue
            f_p_init = - self.pos_p.grad[p]
            self.vel_p[p] += dt * (self.f_p_ext[p] + f_p_init) / self.m_p[p]
            self.pos_p[p] += dt * self.vel_p[p]


    def main(self):
        for time_step in range(max_number) :
            self.f_p_ext.fill(0)
            
            with ti.Tape(self.StrainEnergy):
                self.cal_StrainEnergy()
                
            self.cal_f_ext_p()
            self.plus_vel_pos_p()
            
            if time_step % output_span == 0 :
                print(time_step)
                self.export_Solid()
                self.output_times[None] += 1
            
BendingArmObj = BendingArm()

if __name__ == '__main__':
    BendingArmObj.main()
