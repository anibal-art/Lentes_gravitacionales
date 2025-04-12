import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from astropy import units as u
from astropy import constants as C
from scipy.integrate import simpson, dblquad
import jax
import jax.numpy as jnp
import scipy.integrate as integrate
from scipy.integrate import simpson
from jax import vmap
from jax.scipy.integrate import trapezoid

class dif_event_rate():
    def __init__(self,t,m,D,rs,rho_c,l_coord,b_coord, uT, is_mw):
        '''
        m (astropy quantity: M_sun): mass of the lens
        x (float): DL/DS
        D (astropy quantity: kpc) : Ds, distance to the source
        rs (astropy quantity: kpc)
        rho_c(astropy quantity: kpc)
        t(astropy quantity: hour)
        l_coord(float)
        b_coord(float)
        uT(float)
        u_min(float)
        '''
        self.m = m
        self.D = D 
        self.rs = rs
        self.rho_c = rho_c
        self.t = t
        self.l_coord = l_coord
        self.b_coord = b_coord
        self.uT = uT
        self.is_mw = is_mw 
        
    def rE(self,x):
        ''' 
        Radio de Einstein.
        Entradas:
        m: masa de la lente.
        x: D_OL/D distancia a la lente normalizada por la distancia a la fuente.
        D: Distancia a la fuente.
        thetaE = sqrt(4GM/C^2(1/DL - 1/DS))
        RE = DLsqrt(4GM/C^2(1/DL - 1/DS))
        RE = DLsqrt(4GM/C^2(DS-DL/DLDS))
        DL=xDS
        RE = DLsqrt(4GM/C^2 (DS-xDS/xDSDS))
        k=4G/C^2
        RE = sqrt(k M DS x (1-x))
        '''
        M, Ds = self.m, self.D
        k = 4*C.G/C.c**2
        arg = (k*M*Ds*x*(1-x)).decompose()
        rE = np.sqrt(arg).decompose()
        return rE
    
    def r(self,x):
        '''
        Distancia a la fuente.
        Entradas:
        x: D_OL/D distancia a la lente normalizada por la distancia a la fuente.
        '''
        if self.is_mw:
            R0 = 8.5*u.kpc
            b, l = self.b_coord*(np.pi/180), self.l_coord*(np.pi/180)
            R = np.sqrt((x*self.D)**2 +(R0)**2 - 2*R0*self.D*x*np.cos(b)*np.cos(l))
        else:
            R = self.D*(1-x)
        return R 
     
    def rho_NFW(self,x):
        '''
        Perfil de Navarro Frenk y White para los objetos del halo de la galaxia MilkyWay y M31.
        '''
        c = self.r(x)/self.rs
        bot = c*(1+c)**2
        return (self.rho_c/bot).decompose()
        
    def vc(self,x):
        """
        c = r/rs
        """
        c = self.r(x)/self.rs
        
        factor = -c/(1+c) + np.log(1+c) 
        Ml = 4*np.pi*factor*self.rho_c*self.rs**3 
        return np.sqrt(C.G*Ml/self.r(x)).decompose()
             
    def integrand(self,x,umin):
        '''
        Integrando de la tasa de eventos diferencial.
        El objetivo será La Gran Nube de Magallanes.
        '''
        
        rho = self.rho_NFW(x)
        c = self.r(x)/self.rs
        u_factor = np.sqrt(self.uT**2 - umin**2)
        vr = 2*self.rE(x)*u_factor/self.t
        Q = ((vr/self.vc(x))**2).decompose()
        exp_fac = np.exp(-Q)
        f = 1
        
        # print('vr',vr.to('m/s'))
        # print('u_factor',u_factor)        
        # print('Q',Q)
        # print('factor casi',(2*u.hour*u.hour*self.D*rho*exp_fac*Q*vr**2).decompose())
        
        return (2*f*u.hour*u.hour*self.D*(1/u_factor)*rho*exp_fac*Q*vr**2  /self.m ).decompose()

    def event_rate(self):
        '''
        Tasa de eventos diferencial.
        '''
        return integrate.dblquad(lambda x,umin: self.integrand(x,umin), 0, 1,
                                 0,1)[0]



jax.config.update("jax_enable_x64", True)
# Versión jax del integrando
def integrand_jax(umin, x, D, rs, rho_c, t, m, l_coord, b_coord, uT):
    R0 = 2.622825944267662e+20
    # Convertir grados a radianes
    b = b_coord * jnp.pi / 180
    l = l_coord * jnp.pi / 180

    # r(x)
    r = jnp.sqrt((x * D)**2 + R0**2 - 2 * R0 * D * x * jnp.cos(b) * jnp.cos(l))
    
    # rho_NFW
    # c = r / rs
    c = jnp.maximum(r / rs, 1e-12)
    rho = rho_c / (c * (1 + c)**2)

    # rE(x)
    G = 6.67430e-11  # SI
    c_light = 299792458.0  # m/s
    k = 4 * G / c_light**2  # SI units
    rE = jnp.sqrt(k * m * D * x * (1 - x))  # en m

    factor = -c / (1 + c) + jnp.log(1 + c)
    Ml = 4 * jnp.pi * factor * rho_c * rs**3
    # vc = jnp.sqrt(G * Ml / (r))  # pasar r de kpc a m para SI
    vc = jnp.sqrt(G * Ml / jnp.maximum(r, 1e-12))

    
    # integrando
    # u_factor = jnp.sqrt(uT**2 - umin**2)
    u_diff_sq = uT**2 - umin**2
    u_factor = jnp.sqrt(jnp.maximum(u_diff_sq, 0.0))

    # vr = 2 * rE * u_factor / t  # t en horas
    vr = 2 * rE * u_factor / jnp.maximum(t, 1e-12)

    Q = (vr / vc)**2
    exp_fac = jnp.exp(-Q)
    f = 1.0
    integrando = 2 * D * (1 / u_factor) * rho * exp_fac * Q * vr**2 / m
    # print(rs)
    # print('Ml',Ml)
    # print('vc',vc)
    # print('rE', rE)
    
    # print('u_factor',u_factor)
    # print('vr',vr)
    # print('Q',Q)
    # print('exp_fac',exp_fac)
    # print(type(t))
    # print(2 * D*t**2  * rho * exp_fac * Q * vr**2 )
    return integrando*3600**2


def integrand_jax_m31(umin, x, D, rs, rho_c, t, m, l_coord, b_coord, uT):
    R0 = 2.622825944267662e+20
    # Convertir grados a radianes
    b = b_coord * jnp.pi / 180
    l = l_coord * jnp.pi / 180

    # r(x)
    r = D*(1-x) 
    
    # rho_NFW
    c = r / rs
    rho = rho_c / (c * (1 + c)**2)

    # rE(x)
    G = 6.67430e-11  # SI
    c_light = 299792458.0  # m/s
    k = 4 * G / c_light**2  # SI units
    rE = jnp.sqrt(k * m * D * x * (1 - x))  # en m

    factor = -c / (1 + c) + jnp.log(1 + c)
    Ml = 4 * jnp.pi * factor * rho_c * rs**3
    vc = jnp.sqrt(G * Ml / (r))  # pasar r de kpc a m para SI
    
    # integrando
    u_factor = jnp.sqrt(uT**2 - umin**2)
    
    vr = 2 * rE * u_factor / (t)  # t en horas
    Q = (vr / vc)**2
    exp_fac = jnp.exp(-Q)
    f = 1.0
    integrando = 2 * D * (1 / u_factor) * rho * exp_fac * Q * vr**2 / m
    return integrando*3600**2



# Trapecio adaptado a JAX
# def trapz_jax(y, x, axis=-1):
#     dx = jnp.diff(x)
#     shape = [1] * y.ndim
#     shape[axis] = dx.shape[0]
#     dx = dx.reshape(shape)

#     y0 = jnp.take(y, indices=jnp.arange(y.shape[axis] - 1), axis=axis)
#     y1 = jnp.take(y, indices=jnp.arange(1, y.shape[axis]), axis=axis)

#     integrand = 0.5 * (y0 + y1) * dx
#     return jnp.sum(integrand, axis=axis)

# # Definición sin decorador
# def double_integral_2d(f, umin_range, x_range, num_u=6000, num_x=60000):
#     umin_vals = jnp.linspace(*umin_range, num=num_u)
#     x_vals = jnp.linspace(*x_range, num=num_x)
#     # print(x_vals)

#     U, X = jnp.meshgrid(umin_vals, x_vals, indexing='ij')
#     integrand_grid = f(U, X)

#     integral_u = trapz_jax(integrand_grid, umin_vals, axis=0)
#     return trapz_jax(integral_u, x_vals)

def double_integral_2d(f, u_bounds, x_bounds, num_u=1000, num_x=1000):
    u_vals = jnp.linspace(*u_bounds, num_u)
    x_vals = jnp.linspace(*x_bounds, num_x)

    # Vectorize inner integral over umin
    def inner_integral(x_val):
        f_vals = vmap(lambda u_val: f(u_val, x_val))(u_vals)
        return trapezoid(f_vals, u_vals)

    # Vectorize outer integral over x
    inner_vals = vmap(inner_integral)(x_vals)
    result = trapezoid(inner_vals, x_vals)
    return result

