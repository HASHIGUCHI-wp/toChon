import meshio
import taichi as ti
import numpy as np
import datetime
import os
import pandas as pd

USER = "hashiguchi"
USING_MACHINE = "PC"
DEBUG = False
EXPORT = True
ADD_INFO_LIST = True
ATTENUATION = True
SCHEME = "FEM"
PRESS_TIME_CHANGE = "TORIGO"
FOLDER_NAME = "BendingArmAir"
DATE = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
PRESS_LABEL = 2
FIX = 1
DONE = 1
PI = np.pi

    


mesh_name_s = "bsac_body_y3_gmsh_size0.75-0.9"
dx_mesh_max = 0.9
dx_mesh_min = 0.75

if USING_MACHINE == "PC" :
    ti.init(arch=ti.gpu, default_fp=ti.f64)
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

msh_s = meshio.read(mesh_dir + mesh_name_s + ".msh")


dim = 3
err = 1.0e-5


rho_s = 4e1
young_s, nu_s = 4e6, 0.2
la_s, mu_s = young_s * nu_s / ((1+nu_s) * (1-2*nu_s)) , young_s / (2 * (1+nu_s))
sound_s = np.sqrt((la_s + 2 * mu_s) / rho_s)
Z_right = 62.0

if SCHEME == "FEM" :
    alpha_press = 0.2
    Press = young_s * alpha_press
    Press_min = 0.0
    Press_max = alpha_press * young_s
    max_number = 300000
    output_span = 100
    period_step = int(max_number // 2)
    dt = 0.00005
    
elif SCHEME == "MPM" :
    dx = dx_mesh_max
    
    
    

dt_max = 0.1 * dx_mesh_min / sound_s



print("dt_max", dt_max)
print("dt", dt)




@ti.data_oriented
class Expansion():
    def __init__(self) -> None:
        self.output_times = ti.field(dtype=ti.i32, shape=())
        self.time_steps = ti.field(dtype=ti.i32, shape=())
        self.ELE, self.SUR = "tetra", "triangle"
        self.num_p_s = msh_s.points.shape[0]
        self.num_t_s, self.num_node_ele = msh_s.cells_dict[self.ELE].shape
        self.num_es_s, self.num_node_sur = msh_s.cells_dict[self.SUR].shape
        self.num_p = self.num_p_s
        
        self.StrainEnergy = ti.field(dtype=float, shape=(), needs_grad=True)
        self.pN_fix = ti.field(dtype=ti.i32, shape=self.num_p)
        self.m_p = ti.field(dtype=float, shape=self.num_p)
        self.vel_p = ti.Vector.field(dim, dtype=float, shape=self.num_p)
        self.pos_p = ti.Vector.field(dim, dtype=float, shape=self.num_p, needs_grad=True)
        self.pos_p_rest = ti.Vector.field(dim, dtype=float, shape=self.num_p)
        self.f_p_ext = ti.Vector.field(dim, dtype=float, shape=self.num_p)
        self.tN_pN = ti.field(dtype=ti.i32, shape=(self.num_t_s, self.num_node_ele))
        self.esN_pN = ti.field(dtype=ti.i32, shape=(self.num_es_s, self.num_node_sur))
        
        self.pos_p_rest.from_numpy(msh_s.points)
        self.pos_p.from_numpy(msh_s.points)
        self.tN_pN.from_numpy(msh_s.cells_dict[self.ELE])
        self.esN_pN.from_numpy(msh_s.cells_dict[self.SUR])
        
        self.alpha_Dum = ti.field(dtype=float, shape=())
        self.Press_Time = ti.field(dtype=float, shape=())
        
        self.set_pN_fix()
        self.get_es_press()
        self.cal_m_p()
        self.export_info()
        self.export_program()
        
        
    def get_es_press(self):
        es_press_np = np.arange(0, self.num_es_s)[msh_s.cell_data['gmsh:physical'][0] == PRESS_LABEL]
        self.num_es_press = es_press_np.shape[0]
        self.es_press = ti.field(dtype=ti.i32, shape=self.num_es_press)
        self.es_press.from_numpy(es_press_np)
        
        if DEBUG : 
            print(self.es_press)
            
    @ti.kernel
    def set_pN_fix(self):
        for p in range(self.num_p):
            pos_p_z = self.pos_p_rest[p].z
            if ti.abs(pos_p_z - Z_right) < err :
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
                "mu_s" : mu_s,
                "rho_s" : rho_s
            }
            
            if PRESS_TIME_CHANGE == "CONST" : 
                data_press = {
                    "Press_Time_Change" : PRESS_TIME_CHANGE,
                    "Press" : Press,
                    "alpha_press" : alpha_press
                }
            elif PRESS_TIME_CHANGE == "LINEAR" or PRESS_TIME_CHANGE == "TORIGO":
                data_press = {
                    "Press_Time_Change" : PRESS_TIME_CHANGE,
                    "Press_Max" : Press_max,   
                    "Press_Min" : Press_min, 
                    "alpha_press" : alpha_press,
                    "period_step" : period_step
                }
                
            data.update(data_press)
            
            s = pd.Series(data)
            s.to_csv(export_dir + "Information", header=False)
            
            if ADD_INFO_LIST :
                info_list = pd.read_csv(info_list_dir + "/" + "Info_list.csv", header=None, index_col=0)
                COL_NUM = 1
                while not(pd.isna(info_list.at["date", COL_NUM])) : COL_NUM += 1
                keys = list(data.keys())
                len_keys = len(keys)
                for k in range(len_keys): 
                    info_list.at[keys[k], COL_NUM] = data[keys[k]]
                    
                info_list.to_csv(info_list_dir + "/" + "Info_list.csv", header=None)
            
            
            
            print("num_p", self.num_p_s)
            print("num_t", self.num_t_s)
            print("num_es", self.num_es_s)
            print("num_es_press", self.num_es_press)
            
    def export_program(self):
        with open(__file__, mode="r") as fr:
                prog = fr.read()
        with open(export_dir + "/program.txt", mode="w") as fw:
            fw.write(prog)
            fw.flush()
    
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
            
    @ti.func
    def set_Press_time(self):
        if PRESS_TIME_CHANGE == "CONST" :
            self.Press_Time[None] = Press
        elif PRESS_TIME_CHANGE == "LINEAR" : 
            self.Press_Time[None] = Press_max if self.time_steps[None] >= period_step else self.time_steps[None] / period_step * (Press_max - Press_min) + Press_min
        elif PRESS_TIME_CHANGE == "TORIGO":
            self.Press_Time[None] = Press_max if self.time_steps[None] >= period_step else Press_max * 0.5 * (1 - ti.cos(PI * self.time_steps[None] / period_step))        
        
          
            
    @ti.kernel
    def cal_f_ext_p(self):
        self.set_Press_time()
        if DEBUG :
            print(self.Press_Time[None])
            
        for _es in range(self.num_es_press):
            es = self.es_press[_es]
            a, b, c = self.esN_pN[es, 0], self.esN_pN[es, 1], self.esN_pN[es, 2]
            pos_a, pos_b, pos_c = self.pos_p[a], self.pos_p[b], self.pos_p[c]
            vec_ab, vec_ac = pos_b - pos_a, pos_c - pos_a
            vec_ab_ac_cross = vec_ab.cross(vec_ac)
            area = 0.5 * ti.abs(ti.sqrt(vec_ab_ac_cross[0]**2 + vec_ab_ac_cross[1]**2 + vec_ab_ac_cross[2]**2))
            norm_es_this = vec_ab_ac_cross.normalized()
            self.f_p_ext[a] += - 1 / 3 * norm_es_this * self.Press_Time[None] * area
            self.f_p_ext[b] += - 1 / 3 * norm_es_this * self.Press_Time[None] * area
            self.f_p_ext[c] += - 1 / 3 * norm_es_this * self.Press_Time[None] * area
            
            
    @ti.kernel
    def cal_alpha_Dum(self):
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
                
        else :
            for p in range(self.num_p):
                if self.pN_fix[p] == FIX : continue
                f_p_init = - self.pos_p.grad[p]
                self.vel_p[p] += dt * (self.f_p_ext[p] + f_p_init) / self.m_p[p]
                self.pos_p[p] += dt * self.vel_p[p]
            
    def main_FEM(self):
        for time_step in range(max_number) :
            self.f_p_ext.fill(0)
            # self.V_norm.fill(0)
            # self.U_norm.fill(0)
            # self.FQ_norm.fill(0)
            
            with ti.Tape(self.StrainEnergy):
                self.cal_StrainEnergy()
            
            self.cal_f_ext_p()
            self.cal_alpha_Dum()
            self.plus_vel_pos_p()
            
            if time_step % output_span == 0 :
                print(time_step)
                # self.check_CVG()
                if EXPORT:
                    self.export_Solid()
                    self.output_times[None] += 1
                    
                # if self.CVG[None] == DONE : 
                #     break  
                
            self.time_steps[None] += 1
                
                
    def main(self):
        if SCHEME == "FEM" : 
            self.main_FEM()
    
ExpansionObj = Expansion()
    
if __name__ == '__main__' :
    ExpansionObj.main()
