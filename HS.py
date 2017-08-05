#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helmet Streamer Simulation Program
Created on Tue Aug  1 09:35:38 2017

@author: sven

Commentarios:
    - No uso cx,cy,cz por que creo que no es usado en HS.f
    - Qué es As en linea 610 de HS.f es as?
"""

import numpy as np
import matplotlib.pyplot as plt

#from matplotlib import pyplot, cm
#import mpl_toolkits.mplot3d.axes3d as p3
#import matplotlib.animation as animation

def initall_condition():

    return vx, vy, vz, ax, ay, az, bx, by, bz, ex, ey, ez
###############################################################################
#                                Parameter                                    #
###############################################################################

N_x, N_y, N_z, N_t = 90, 60, 60, 5
x_min, x_max, y_min, y_max, z_min, z_max = -0.5, 0.5, -0.5, 0.5, 0, 5
gamma = 5 / 3
mu = 1

###############################################################################
#                                   Main                                      #
###############################################################################

delta_x, delta_y, delta_z =  (x_max - x_min)/N_x, (y_max - y_min)/N_y, (z_max - z_min)/N_z

x, y, z = np.linspace(x_min, x_max, N_x), np.linspace(y_min, y_max, N_y), np.linspace(z_min, z_max, N_z)
X, Y, Z = np.meshgrid(y, x, z)

vx, vy, vz = np.zeros((N_t, N_x, N_y, N_z)), np.zeros((N_t, N_x, N_y, N_z)), np.zeros((N_t, N_x, N_y, N_z))
ax, ay, az = np.zeros((N_t, N_x, N_y, N_z)), np.zeros((N_t, N_x, N_y, N_z)), np.zeros((N_t, N_x, N_y, N_z))
bx, by, bz = np.zeros((N_t, N_x, N_y, N_z)), np.zeros((N_t, N_x, N_y, N_z)), np.zeros((N_t, N_x, N_y, N_z))
ex, ey, ez = np.zeros((N_t, N_x, N_y, N_z)), np.zeros((N_t, N_x, N_y, N_z)), np.zeros((N_t, N_x, N_y, N_z))
jx, jy, jz = np.zeros((N_t, N_x, N_y, N_z)), np.zeros((N_t, N_x, N_y, N_z)), np.zeros((N_t, N_x, N_y, N_z))
energy, eta, gauss = np.zeros((N_t, N_x, N_y, N_z)), np.zeros((N_t, N_x, N_y, N_z)), np.zeros((N_t, N_x, N_y, N_z))
p, rho, gauss = np.zeros((N_t, N_x, N_y, N_z)), np.zeros((N_t, N_x, N_y, N_z)), np.zeros((N_t, N_x, N_y, N_z))

#vx, vy, vz, ax, ay, az, bx, by, bz, ex, ey, ez = initall_condition()
if 1:
    x0 = 0
    x02 = np.zeros((N_x, N_y, N_z))
#    x, y, z = -0.5, -0.5, 0
    a, eta0, zcusp = 0.1, 0.0005, 4

    ho = 10
    a_s = 0.1
    c1 = 15
    c2 = 15 
    s1 = 0.8
    s2 = 0.4
    s3 = 0.2
    k1 = 1
    k2 = k1**(-(c1 + c2)*a_s)
    x01 = 0.0

#   while over all
#      z = dble(k)*dz
    
    p0 = s1**(-s2*Z)+s3 
    p = p0
    
    xsep = ((1/c1)*np.sqrt(2/p0)
    *np.arctanh(np.sqrt((p0 - k1*np.exp(-c1*a_s))/p0)))
#    *datanh(dsqrt((p0 - k1*np.exp(-c1*as))/p0),val)   
    
    p1 = np.exp(s2*Z)*s3+s1
    p2 = np.exp(a_s*c1+s2*Z)
    p3 = np.exp(s2*Z)
    p4 = np.exp(a_s*c1)
    p5 = np.sqrt((p3*s3+s1)/(2*p3))
    p6 = np.sqrt((-k1*p3+p3*p4*s3+p4*s1)/(p3*p4*s3+p4*s1))
    p7 = c1*p5*X-c1*p5*x0
    p8 = (np.sqrt((2*p3)/(p3*s3+s1))*np.arctanh(p6)*c1*p5 
          + np.sqrt((2*p3)/(p3*s3+s1))*np.arctanh(p6)*c2*p5 
          - c1*c2*p5*x0-c2*p7)
    
#    if(x .le. xsep) then X<xsep
    ay[0,X<xsep] = (-(2/c1)*np.log(np.cosh(c1*np.sqrt(p0[X<xsep]/2)
                    *(X[X<xsep] - 0)))+(1/c1)*np.log(p0[X<xsep]/k1))
    bx[0,X<xsep] = ((s1*s2*(2*np.cosh(p7[X<xsep])*p3[X<xsep]*p5[X<xsep]
                    -np.sinh(p7[X<xsep])*c1*p3[X<xsep]*s3*X[X<xsep] + np.sinh(p7[X<xsep])*c1*p3[X<xsep]*s3*x0
                    -np.sinh(p7[X<xsep])*c1*s1*X[X<xsep] + np.sinh(p7[X<xsep])*c1*s1*x0))/
                    (2*np.cosh(p7[X<xsep])*c1*p3[X<xsep]*p5[X<xsep]*(p3[X<xsep]*s3+s1)))
    bz[0,X<xsep] = (-2*np.sinh(p7[X<xsep])*p5[X<xsep])/np.cosh(p7[X<xsep])
    jy[0,X<xsep] = c1*p0[X<xsep]/(np.cosh(c1*X[X<xsep]*np.sqrt(p0[X<xsep]/2)))
#    else X>=xsep
    x02[X>=xsep] = (np.arctanh(np.sqrt((p0[X>=xsep] - k1*np.exp(-c1*a_s))/p0[X>=xsep]))*
            np.sqrt(2/p0[X>=xsep])*(1/c1 + 1/c2))
    ay[0,X>=xsep]= (-(2/c2)*np.log(np.cosh(c2*np.sqrt(p0[X>=xsep]/2)
                    *(X[X>=xsep] - x02[X>=xsep])))+(1/c2)*np.log(p0[X>=xsep]/k2))
    bx[0,X>=xsep] = ((s1*s2*(2*np.sqrt((2*p3[X>=xsep])/(p3[X>=xsep]*s3+s1))*
                    np.cosh(p8[X>=xsep]/c1)*c1*p3[X>=xsep]*p5[X>=xsep]*p6[X>=xsep]
                    + np.sqrt((2*p3[X>=xsep])/(p3[X>=xsep]*s3+s1))*
                    np.sinh(p8[X>=xsep]/c1)*c1*c2*p3[X>=xsep]*p6[X>=xsep]*s3*X[X>=xsep]
                    + np.sqrt((2*p3[X>=xsep])/(p3[X>=xsep]*s3+s1))*
                    np.sinh(p8[X>=xsep]/c1)*c1*c2*p6[X>=xsep]*s1*X[X>=xsep]
                    - 2*np.sinh(p8[X>=xsep]/c1)*c1*p3[X>=xsep]
                    -2*np.sinh(p8[X>=xsep]/c1)*c2*p3[X>=xsep]))/
                    (2*np.sqrt((2*p3[X>=xsep])/(p3[X>=xsep]*s3+s1))*
                     np.cosh(p8[X>=xsep]/c1)*c1*c2*p3[X>=xsep]*p5[X>=xsep]*p6[X>=xsep]*(p3[X>=xsep]*s3+s1)))
    bz[0,X>=xsep] = (2*np.sinh(p8[X>=xsep]/c1)*p5[X>=xsep])/np.cosh(p8[X>=xsep]/c1)
    jy[0,X>=xsep] = c2*p0[X>=xsep]/(np.cosh(c2*X[X>=xsep]*np.sqrt(p0[X>=xsep]/2)))
        
        
#        endif     für alle
    ax[:]  = 0
    az[:]  = 0
    by[:] = 0
    gauss[:] = 0
    rho[:] = p[:]*np.exp(-Z/ho)
    jx[:] = 0
    jz[:] = 0
    energy[:] = (p[:]/(gamma - 1)
                        + (bx[:]*bx[:]
                        + by[:]*by[:]
                        + bz[:]*bz[:])/(2*mu))
    eta[:] = (eta0*np.exp(-a*(Z-zcusp))*
                    np.tanh(abs(jy[:])))
#                       np.tanh(abs(dble(xjy(n,i,j,k))),val)
    ex[:] = 0
    ey[:] = eta[:]*jy[:]
    ez[:] = 0   



plt.contour(bx[0,30])
plt.yticks(range(20))

#p = demo_p()
#vx, vy ,vz = demo_v()
#Ax, Ay ,Az = demo_v()
#Bx, By ,Bz = demo_v()
#dBx_dt, dBy_dt, dBz_dt = demo_v()







###############################################################################
#                                  Cementerio                                 #
###############################################################################

#y, x, z = np.meshgrid(np.linspace(x_min, x_max, N_y),
#                      np.linspace(y_min, y_max, N_x),
#                      np.linspace(z_min, z_max, N_z))


