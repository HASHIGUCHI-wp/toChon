import taichi as ti
import Solid_P2Q as Solid_P2Q

@ti.data_oriented
class Solid_MPM():
    def __init__(
        self,
        num_p_s,
        dt, dx,
        nx, ny, nz,
        area_start = ti.Vector([0.0, 0.0, 0.0]), area_end = ti.Vector([1.0, 1.0, 1.0]),
        dim = 3, gi = ti.Vector([0.0, 0.0, 0.0])
    ):
        self.dim = dim
        self.num_p_s = num_p_s
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dt = dt
        self.dx = dx
        self.inv_dx = 1 / dx
        self.area_start = area_start
        self.area_end = area_end
        self.gi = gi
        self.nx = nx
        self.ny = ny
        self.nz = nz
        
        
    def set_taichi_field(self) :
        self.d_pos_p_s = ti.Vector.field(self.dim, dtype=float, shape=self.num_p_s)
        self.C_p_s = ti.Matrix.field(self.dim, self.dim, dtype=float, shape=self.num_p_s)
        
        self.m_S = ti.field(dtype=float, shape=(self.nx, self.ny, self.nz))
        self.p_S = ti.Vector.field(self.dim, dtype=float, shape=(self.nx, self.ny, self.nz))
        self.f_S = ti.Vector.field(self.dim, dtype=float, shape=(self.nx, self.ny, self.nz))
        self.norm_S = ti.Vector.field(self.dim, dtype=float, shape=(self.nx, self.ny, self.nz))
        
        
    
                        
                        
    @ti.kernel
    def p2g(self) :
        for s in range(self.num_p_s) :
            self._p2g(s)
        
    
    @ti.func
    def _p2g(self, s : ti.int32) :
        base = ti.cast((self.pos_p_s[s] - self.area_start) * self.inv_dx - 0.5, ti.i32)
        fx = (self.pos_p_s[s] - self.area_start) * self.inv_dx - ti.cast(base, float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        beta = 0.5 * self.dt * self.alpha_Dum[None] if self.ATTENUATION_s else 0.0
        f_p_int = - self.pos_p_s.grad[s]
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    ix, iy, iz = base.x + i, base.y + j, base.z + k
                    I = ti.Vector([i, j, k])
                    dist = (float(I) - fx) * self.dx
                    NpI = w[i].x * w[j].y * w[k].z
                    self.m_S[ix, iy, iz] += NpI * self.m_p_s[s]
                    # self.p_S[ix, iy, iz] += NpI * (self.m_p_s[s] * (self.vel_s[s] + self.C_p_s[s] @ dist) + self.dt * f_p_int)
                    self.p_S[ix, iy, iz] += NpI * ( (1 - beta) * self.m_p_s[s] * (self.vel_p_s[s] + self.C_p_s[s] @ dist) + self.dt * f_p_int) / (1 + beta)
                    self.exist_Ix[ix], self.exist_Iy[iy], self.exist_Iz[iz] = self.EXIST, self.EXIST, self.EXIST
        
    
    @ti.func
    def _g2p(self, s : ti.int32):
        base = ti.cast((self.pos_p_s[s] - self.area_start) * self.inv_dx - 0.5, ti.i32)
        fx = (self.pos_p_s[s] - self.area_start) * self.inv_dx - ti.cast(base, float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_C_p, new_vel_p, new_d_pos_p = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), ti.Vector([0.0, 0.0, 0.0]), ti.Vector([0.0, 0.0, 0.0])
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    ix, iy, iz = base.x + i, base.y + j, base.z + k
                    dist = (float(ti.Vector([i, j, k])) - fx) * self.dx
                    NpI = w[i].x * w[j].y * w[k].z
                    vel_this = self.p_S[ix, iy, iz] / self.m_S[ix, iy, iz]
                    vel_this = ti.Vector([0.0, 0.0, 0.0]) if self.m_S[ix, iy, iz] == 0.0 else vel_this
                    new_C_p += 4 * self.inv_dx**2 * NpI * vel_this.outer_product(dist)
                    new_vel_p += NpI * vel_this
                    new_d_pos_p += NpI * vel_this * self.dt
        self.C_p_s[s] = new_C_p
        self.vel_p_s[s] = new_vel_p
        self.d_pos_p_s[s] = new_d_pos_p
        
    
    
    
        
        
        
        
