import taichi as ti
import numpy as np
import meshio
import datetime
import os
from pyevtk.hl import *
import sympy as sy

ti.init(arch=ti.gpu, default_fp=ti.f64)

dim = 3

msh_s = meshio.read('./mesh_file/BendingArm/BenidngArmLabelSize0.2-3.msh')

num_p = msh_s.points.shape[0]
num_t = msh_s.cells_dict['tetra'].shape[0]
num_es = msh_s.cells_dict['triangle'].shape[0]

print("num_p", num_p)
print("num_t", num_t)
print("num_es", num_es)

export = True

FOLDER_NAME = "AirBendingArm"
FILE_NAME = "AirBendingArm"
if export:
    # FOLDER_NAME = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M') + FOLDER_NAME
    dr = './conseqence/' + FOLDER_NAME + "/" + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
    os.makedirs(dr, exist_ok=True)


dim = 3


box_size = ti.Vector([76, 30, 78])
area_start = ti.Vector([-42, -15, -50])

dx = 0.5
inv_dx = 1 / dx
nx, ny, nz = int(box_size.x * inv_dx + 1), int(box_size.y * inv_dx + 1), int(box_size.z * inv_dx + 1)


rho_s = 4e1
young, nu = 4e5, 0.3
la_s, mu_s = young * nu / ((1+nu) * (1-2*nu)) , young / (2 * (1+nu))
sound_s = ti.sqrt((la_s + 2 * mu_s) / rho_s)
x_min_s, x_max_s = -34.0, 34.0

Press_max = 10000.0
Press_max = 20000.0

max_number = 3000
period_max = 2000
output_span = max_number // 100
dt_max = 0.1 * dx / sound_s
dt = 0.0006

print("dt_max", dt_max)
print("dt", dt)

StrainEnergy = ti.field(dtype=float, shape=(), needs_grad=True)
pos_p = ti.Vector.field(dim, dtype=float, shape=num_p, needs_grad=True)
pos_p_rest = ti.Vector.field(dim, dtype=float, shape=num_p)
vel_p = ti.Vector.field(dim, dtype=float, shape=num_p)
f_p_ext = ti.Vector.field(dim, dtype=float, shape=num_p)
C_p = ti.Matrix.field(dim, dim, dtype=float, shape=num_p)
m_p = ti.field(dtype=float, shape=num_p)

m_I = ti.field(dtype=float, shape=(nx, ny, nz))
p_I = ti.Vector.field(dim, dtype=float, shape=(nx, ny, nz))



tN_pN = ti.field(dtype=ti.i32, shape=(num_t, 4))
esN_pN = ti.field(dtype=ti.i32, shape=(num_es, 3))
tN_pN.from_numpy(msh_s.cells_dict['tetra'])
esN_pN.from_numpy(msh_s.cells_dict['triangle'])

pos_p.from_numpy(msh_s.points)
pos_p_rest.from_numpy(msh_s.points)

num_es_inner = ti.field(dtype=ti.i32, shape=())
sum_area = ti.field(dtype=float, shape=())

PI = np.pi


num_surface = 73
P_surface = 2


sN_num_es = ti.field(dtype=ti.int32, shape=(num_surface))
sN_P = ti.field(dtype=ti.int32, shape=(num_surface))


def set_sN_inf():
    num_pes = 0
    for s in range(num_surface):
        s_data = msh_s.cell_data['gmsh:physical'][s]
        sN_num_es[s] = s_data.shape[0]
        if s_data[0] == P_surface:
            sN_P[s] = 1
            num_pes += s_data.shape[0]
    return num_pes

num_pes = set_sN_inf()

print("sN_P", sN_P)
print("num_pes", num_pes)

pesN_pN = ti.field(dtype=ti.i32, shape=(num_pes, 3))

@ti.kernel
def set_pesN_pN():
    for es in range(num_es):
        pes_distin = 0
        es_distin = 0
        es_P = 0
        s = 0
        not_limite = True
        while not_limite:  
            if es < es_distin + sN_num_es[s]:
                not_limite = False
                es_P = sN_P[s]
            else:
                es_distin += sN_num_es[s]
                pes_distin += sN_num_es[s] if sN_P[s] else 0
                s += 1
        
        if es_P:
            pesN = pes_distin + (es - es_distin)
            pesN_pN[pesN, 0] = esN_pN[es, 0]
            pesN_pN[pesN, 1] = esN_pN[es, 1]
            pesN_pN[pesN, 2] = esN_pN[es, 2]
            if pesN >= num_pes : print(pesN)
            
set_pesN_pN()


def get_diri_IxIyIz_np():
    diri_area_start = [31, -7.5, 0]
    diri_area_end = [31, 7.5, 19.0]
    
    diri_ix_s = int((diri_area_start[0] - area_start[0]) * inv_dx - 0.5)
    diri_iy_s = int((diri_area_start[1] - area_start[1]) * inv_dx - 0.5)
    diri_iz_s = int((diri_area_start[2] - area_start[2]) * inv_dx - 0.5)
    
    diri_ix_e = int((diri_area_end[0] - area_start[0]) * inv_dx - 0.5) + 2
    diri_iy_e = int((diri_area_end[1] - area_start[1]) * inv_dx - 0.5) + 2
    diri_iz_e = int((diri_area_end[2] - area_start[2]) * inv_dx - 0.5) + 2
    
    nx_diri = diri_ix_e - diri_ix_s + 1
    ny_diri = diri_iy_e - diri_iy_s + 1
    nz_diri = diri_iz_e - diri_iz_s + 1
    num_diri = nx_diri * ny_diri * nz_diri
    
    diri_IxIyIz_np = np.zeros(0, dtype=np.int32)
    for _Ix in range(nx_diri):
        for _Iy in range(ny_diri):
            for _Iz in range(nz_diri):
                Ix, Iy, Iz = _Ix + diri_ix_s, _Iy + diri_iy_s, _Iz + diri_iz_s
                IxIyIz = Iz * (nx * ny) + Iy * (nx) + Ix
                diri_IxIyIz_np = np.append(diri_IxIyIz_np, IxIyIz)
     
    return diri_IxIyIz_np

diri_IxIyIz_np = get_diri_IxIyIz_np()
diri_IxIyIz_np = np.unique(diri_IxIyIz_np)
num_diri_I = diri_IxIyIz_np.shape[0]

diri_I = ti.field(dtype=ti.i32, shape=num_diri_I)
diri_I.from_numpy(diri_IxIyIz_np)
print("num_diri_I", num_diri_I)

        

@ti.kernel
def cal_m_p():
    for t in range(num_t):
        a, b, c, d = tN_pN[t, 0], tN_pN[t, 1], tN_pN[t, 2], tN_pN[t, 3]
        pos_a, pos_b, pos_c, pos_d = pos_p_rest[a], pos_p_rest[b], pos_p_rest[c], pos_p_rest[d]
        matR = ti.Matrix.cols([pos_b - pos_a, pos_c - pos_a, pos_d - pos_a])
        Vol = 1 / 6 * ti.abs(matR.determinant()) 
        m_p[a] += 0.25 * rho_s * Vol
        m_p[b] += 0.25 * rho_s * Vol
        m_p[c] += 0.25 * rho_s * Vol
        m_p[d] += 0.25 * rho_s * Vol
        
@ti.kernel 
def cal_f_ext_p(time_step : int):
    for pes in range(num_pes):
        a, b, c = pesN_pN[pes, 0], pesN_pN[pes, 1], pesN_pN[pes, 2]
        pos_a, pos_b, pos_c = pos_p[a], pos_p[b], pos_p[c]
        center_ref = (pos_p_rest[a] + pos_p_rest[b] + pos_p_rest[c]) / 3
        # period = period_max * (center_ref.x - x_min_s) / (x_max_s - x_min_s)
        period = period_max * (x_max_s - center_ref.x) / (x_max_s - x_min_s)
        alpha = 0.5 * (1 - ti.cos(time_step / period * PI)) if time_step <= period else 1.0
        vec_ab, vec_ac = pos_b - pos_a, pos_c - pos_a
        vec_ab_ac_cross = vec_ab.cross(vec_ac)
        area = 0.5 * ti.abs(ti.sqrt(vec_ab_ac_cross[0]**2 + vec_ab_ac_cross[1]**2 + vec_ab_ac_cross[2]**2))
        norm_es_this = vec_ab_ac_cross.normalized()
        f_p_ext[a] += - 1 / 3 * norm_es_this * alpha * Press_max * area
        f_p_ext[b] += - 1 / 3 * norm_es_this * alpha * Press_max * area
        f_p_ext[c] += - 1 / 3 * norm_es_this * alpha * Press_max * area
        
        

@ti.kernel
def cal_StrainEnergy():
    for t in range(num_t):
        a, b, c, d = tN_pN[t,0], tN_pN[t,1], tN_pN[t,2], tN_pN[t,3]
        Ref = ti.Matrix.cols([pos_p_rest[b] - pos_p_rest[a], pos_p_rest[c] - pos_p_rest[a], pos_p_rest[d] - pos_p_rest[a]])
        Crn = ti.Matrix.cols([pos_p[b] - pos_p[a], pos_p[c] - pos_p[a], pos_p[d] - pos_p[a]])
        F = Crn @ Ref.inverse()
        Vol = 1 / 6 * ti.abs(Ref.determinant())
        I1 = (F @ F.transpose()).trace()
        J = F.determinant()
        elementEnergy = 0.5 * mu_s * (I1 - dim) - mu_s * ti.log(J) + 0.5 * la_s * ti.log(J)**2
        StrainEnergy[None] += elementEnergy * Vol


@ti.kernel
def cal_m_p_f_I():
    for p in range(num_p):
        base = ti.cast((pos_p[p] - area_start) * inv_dx - 0.5, ti.i32)
        fx = (pos_p[p] - area_start) * inv_dx - ti.cast(base, float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    ix, iy, iz = base.x + i, base.y + j, base.z + k
                    I = ti.Vector([i, j, k])
                    dist = (float(I) - fx) * dx
                    NpI = w[i].x * w[j].y * w[k].z
                    f_p_int = - pos_p.grad[p]
                    m_I[ix, iy, iz] += NpI * m_p[p]
                    p_I[ix, iy, iz] += NpI * (m_p[p] * (vel_p[p] + C_p[p] @ dist) + dt * (f_p_int + f_p_ext[p]))
                    
                    
@ti.kernel
def diri_p_I():
    for _IxIyIz in range(num_diri_I):
        IxIyIz = diri_I[_IxIyIz]
        iz, ixiy = IxIyIz // (nx*ny), IxIyIz % (nx*ny)
        iy, ix = ixiy // nx, ixiy % nx
        p_I[ix, iy, iz] = ti.Vector([0.0, 0.0, 0.0])
        
        
@ti.kernel
def plus_vel_dpos_p():
    for p in range(num_p):
        base = ti.cast((pos_p[p] - area_start) * inv_dx - 0.5, ti.i32)
        fx = (pos_p[p] - area_start) * inv_dx - ti.cast(base, float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        d_pos_p = ti.Vector([0.0, 0.0, 0.0])
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    ix, iy, iz = base.x + i, base.y + j, base.z + k
                    dist = (float(ti.Vector([i, j, k])) - fx) * dx
                    NpI = w[i].x * w[j].y * w[k].z
                    vel_this = p_I[ix, iy, iz] / m_I[ix, iy, iz]
                    vel_this = ti.Vector([0.0,0.0,0.0]) if m_I[ix,iy,iz] == 0.0 else vel_this
                    C_p[p] += 4 * inv_dx**2 * NpI * vel_this.outer_product(dist)
                    vel_p[p] += NpI * vel_this
                    d_pos_p += NpI * vel_this * dt
        pos_p[p] += d_pos_p
                    
                    
def export_Solid(outputTimes):
    points = np.zeros((num_p, 3), dtype=float)
    pos_p_np = pos_p.to_numpy()
    points = pos_p_np
    tetra = msh_s.cells_dict['tetra']
    cells = [
        ('tetra', tetra)
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
    mesh_.write(dr + "/" + FILE_NAME + "_SOLID" + str(outputTimes) + ".vtu")



cal_m_p()

output_times = 0

for time_step in range(max_number):
    m_I.fill(0)
    p_I.fill(0)
    f_p_ext.fill(0)
    
    with ti.Tape(StrainEnergy):
        cal_StrainEnergy()
    
    cal_f_ext_p(time_step)
    cal_m_p_f_I()
    diri_p_I()
    
    C_p.fill(0)
    vel_p.fill(0)
    plus_vel_dpos_p()
    
    if time_step % output_span == 0 :
        print(time_step)
        if export:
            export_Solid(output_times)
            output_times += 1
