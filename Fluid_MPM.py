import taichi as ti
import sys
from pyevtk.hl import *

@ti.data_oriented
class Fluid_MPM :
    def __init__(
        self,
        dim,
        rho_f, mu_f, gamma_f, kappa_f, lambda_f, dt, dx,
        nx, ny, nz,
        area_start = ti.Vector([0.0, 0.0, 0.0]), area_end = ti.Vector([1.0, 1.0, 1.0]),
        gi = ti.Vector([0.0, 0.0, 0.0]),
        EXIST = 1
    ) :
        self.dim = dim
        self.rho_f = rho_f
        self.mu_f = mu_f
        self.gamma_f = gamma_f
        self.kappa_f = kappa_f
        self.lambda_f = lambda_f
        self.dt = dt
        self.dx = dx
        self.inv_dx = 1 / dx
        self.area_start = area_start
        self.area_end = area_end
        self.gi = gi
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.EXIST = EXIST
        
    
    def set_taichi_field(self, num_p_f) :
        self.num_p_f = num_p_f
        self.m_p_f = ti.field(dtype=float, shape=self.num_p_f)
        self.rho_p_f = ti.field(dtype=float, shape=self.num_p_f)
        self.P_p_f = ti.field(dtype=float, shape=self.num_p_f)
        self.pos_p_f = ti.Vector.field(self.dim, dtype=float, shape=self.num_p_f)
        self.d_pos_p_f = ti.Vector.field(self.dim, dtype=float, shape=self.num_p_f)
        self.vel_p_f = ti.Vector.field(self.dim, dtype=float, shape=self.num_p_f)
        self.C_p_f = ti.Matrix.field(self.dim, self.dim, dtype=float, shape=self.num_p_f)
        self.sigma_p_f = ti.Matrix.field(self.dim, self.dim, dtype=float, shape=self.num_p_f)
        self.L_p_f = ti.Matrix.field(self.dim, self.dim, dtype=float, shape=self.num_p_f)
        
        self.m_F = ti.field(dtype=float, shape=(self.nx, self.ny, self.nz))
        self.p_F = ti.Vector.field(self.dim, dtype=float, shape=(self.nx, self.ny, self.nz))
        self.f_F = ti.Vector.field(self.dim, dtype=float, shape=(self.nx, self.ny, self.nz))
        
    @ti.kernel
    def p2g(self) :
        for f in range(self.num_p_f):
            self._p2g(f)
            
    @ti.kernel
    def g2g(self) :
        for f in range(self.num_p_f):
            self._g2p(f)
            
    @ti.kernel
    def update_p_F(self) :
        for f in range(self.num_p_f ):
            self._update_p_F(f)
            
    @ti.kernel
    def cal_L_p_f(self) :
        for f in range(self.num_p_f ):
            self._cal_L_p_f(f)
            
    @ti.kernel
    def cal_rho_sigma_p_f(self) :
        for f in range(self.num_p_f) :
            self.cal_rho_sigma_p_f(f)
            
        
    @ti.func
    def _p2g(self, f : ti.int32) :
        base = ti.cast((self.pos_p_f[f] - self.area_start) * self.inv_dx - 0.5, ti.i32)
        fx = (self.pos_p_f[f] - self.area_start) * self.inv_dx - ti.cast(base, float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        dw = [(fx - 1.5)  * self.inv_dx, -2 * (fx - 1) * self.inv_dx, (fx - 0.5) * self.inv_dx]
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    ix, iy, iz = base.x + i, base.y + j, base.z + k
                    I = ti.Vector([i, j, k])
                    dist = (float(I) - fx) * self.dx
                    NpI = w[i].x * w[j].y * w[k].z
                    dNpIdx = ti.Vector([dw[i].x * w[j].y * w[k].z, w[i].x * dw[j].y * w[k].z, w[i].x * w[j].y * dw[k].z])
                    self.m_F[ix, iy, iz] += NpI * self.m_p_f[f]
                    self.p_F[ix, iy, iz] += NpI * (self.m_p_f[f] * (self.vel_p_f[f] + self.C_p_f[f] @ dist))
                    self.f_F[ix, iy, iz] += NpI * self.m_p_f[f] * self.gi
                    self.f_F[ix, iy, iz][0] += - self.m_p_f[f] / self.rho_p_f[f] * (self.sigma_p_f[f][0, 0] * dNpIdx[0] + self.sigma_p_f[f][0, 1] * dNpIdx[1] + self.sigma_p_f[f][0, 2] * dNpIdx[2])
                    self.f_F[ix, iy, iz][1] += - self.m_p_f[f] / self.rho_p_f[f] * (self.sigma_p_f[f][1, 0] * dNpIdx[0] + self.sigma_p_f[f][1, 1] * dNpIdx[1] + self.sigma_p_f[f][1, 2] * dNpIdx[2])
                    self.f_F[ix, iy, iz][2] += - self.m_p_f[f] / self.rho_p_f[f] * (self.sigma_p_f[f][2, 0] * dNpIdx[0] + self.sigma_p_f[f][2, 1] * dNpIdx[1] + self.sigma_p_f[f][2, 2] * dNpIdx[2])
                    self.exist_Ix[ix], self.exist_Iy[iy], self.exist_Iz[iz] = self.EXIST, self.EXIST, self.EXIST
                    
    
    @ti.func
    def _g2p(self, f : ti.i32) :
        base = ti.cast((self.pos_p_f[f] - self.area_start) * self.inv_dx - 0.5, ti.i32)
        fx = (self.pos_p_f[f] - self.area_start) * self.inv_dx - ti.cast(base, float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_C_p, new_vel_p, new_d_pos_p = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), ti.Vector([0.0, 0.0, 0.0]), ti.Vector([0.0, 0.0, 0.0])
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    ix, iy, iz = base.x + i, base.y + j, base.z + k
                    dist = (float(ti.Vector([i, j, k])) - fx) * self.dx
                    NpI = w[i].x * w[j].y * w[k].z
                    vel_this = self.p_F[ix, iy, iz] / self.m_F[ix, iy, iz]
                    vel_this = ti.Vector([0.0, 0.0, 0.0]) if self.m_F[ix, iy, iz] == 0.0 else vel_this
                    new_C_p += 4 * self.inv_dx**2 * NpI * vel_this.outer_product(dist)
                    new_vel_p += NpI * vel_this
                    new_d_pos_p += NpI * vel_this * self.dt
        self.C_p_f[f] = new_C_p
        self.vel_p_f[f] = new_vel_p
        self.d_pos_p_f[f] = new_d_pos_p
        
        
    @ti.func
    def _update_p_F(self, f : int):
        base = ti.cast((self.pos_p_f[f] - self.area_start) * self.inv_dx - 0.5, ti.i32)
        fx = (self.pos_p_f[f] - self.area_start) * self.inv_dx - ti.cast(base, float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    ix, iy, iz = base.x + i, base.y + j, base.z + k
                    dist = (float(ti.Vector([i, j, k])) - fx) * self.dx
                    NpI = w[i].x * w[j].y * w[k].z
                    self.p_F[ix, iy, iz] += NpI * (self.m_p_f[f] * (self.vel_p_f[f] + self.C_p_f[f] @ dist))
                        
    
    @ti.func
    def _cal_L_p_f(self, f : int):
        base = ti.cast((self.pos_p_f[f] - self.area_start) * self.inv_dx - 0.5, ti.i32)
        fx = (self.pos_p_f[f] - self.area_start) * self.inv_dx - ti.cast(base, float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2] 
        dw = [(fx - 1.5)  * self.inv_dx, -2 * (fx - 1) * self.inv_dx, (fx - 0.5) * self.inv_dx]
        new_L_p_f = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    ix, iy, iz = base.x + i, base.y + j, base.z + k
                    vel_I_this = self.p_F[ix, iy, iz] / self.m_F[ix, iy, iz]
                    vel_I_this = [0.0, 0.0, 0.0] if self.m_F[ix, iy, iz] == 0 else vel_I_this
                    dv = ti.Matrix([
                        [dw[i].x * w[j].y * w[k].z * vel_I_this.x, w[i].x * dw[j].y * w[k].z * vel_I_this.x, w[i].x * w[j].y * dw[k].z * vel_I_this.x],
                        [dw[i].x * w[j].y * w[k].z * vel_I_this.y, w[i].x * dw[j].y * w[k].z * vel_I_this.y, w[i].x * w[j].y * dw[k].z * vel_I_this.y],
                        [dw[i].x * w[j].y * w[k].z * vel_I_this.z, w[i].x * dw[j].y * w[k].z * vel_I_this.z, w[i].x * w[j].y * dw[k].z * vel_I_this.z]
                    ])
                    new_L_p_f += dv
        self.L_p_f[f] = new_L_p_f
            
            
    @ti.func
    def _cal_rho_sigma_p_f(self, f : int):
        Iden = ti.Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        tr_Dep = self.L_p_f[f].trace() * self.dt
        self.rho_p_f[f] /= 1 + tr_Dep
        P_this = self.kappa_f * ((self.rho_p_f[f] / self.rho_f)**self.gamma_f - 1)
        epsilon_dot = 0.5 * (self.L_p_f[f] + self.L_p_f[f].transpose())
        self.sigma_p_f[f] = 2 * self.mu_f * epsilon_dot + (self.lambda_f - P_this) * Iden
        self.P_p_f[f] = P_this
        
    @ti.func
    def _plus_pos_p_f(self, f : int) :
        self.pos_p_f[f] += self.d_pos_p_f[f]
        if not(self.pos_p_f[f].x < self.BIG) : self.divergence[None] = self.DIVERGENCE
        
    
    def export_Fluid(self, dir):
        num_f_end = self.num_p_f_active[None]
        pos_p_np = self.pos_p_f.to_numpy()[:num_f_end, :]
        P_p_np = self.P_p_f.to_numpy()[:num_f_end]
        point_data = {"pressure": P_p_np.copy()}
        pointsToVTK(
            dir,
            pos_p_np[:, 0].copy(),
            pos_p_np[:, 1].copy(),
            pos_p_np[:, 2].copy(),
            data=point_data
        )
        
        
        
        
if __name__ == '__main__' :
    pass
