import sys
import numpy as np
from matplotlib import *
from make_grid import Grilla
from models_profiles import *
import os
from colossus.halo import concentration
from colossus.cosmology import cosmology  
params = {'flat': True, 'H0': 70.0, 'Om0': 0.25, 'Ob0': 0.044, 'sigma8': 0.8, 'ns': 0.95}
cosmology.addCosmology('MICE', params)
cosmo = cosmology.setCosmology('MICE')

#parameters

def make_shear_map(x,y,e_1,e_2,
                   ax = plt, cmap = 'plasma'):
                       
    e_x = np.zeros(len(x))
    e_y = np.zeros(len(x))
                           
    e_x[e_1 > 0.] -= e_1[e_1 > 0.]*np.sign(np.sin(theta[e_1 > 0.])*np.cos(theta[e_1 > 0.])) 
    e_x += e_2*np.cos(np.pi/4.)

    e_y[e_1 < 0.] -= e_1[e_1 < 0.]
    e_y += np.abs(e_2*np.sin(np.pi/4.))

    e_norm = np.sqrt(e_x**2 + e_y**2)
    
    mapq = ax.quiver(x,y,e_x/e_norm,e_y/e_norm,e_norm,
                      pivot='mid',cmap = cmap,
                      headlength=0, headwidth = 1,
                      norm=matplotlib.colors.LogNorm())
    
    return mapq


def make_map_from_model(M200=2.e14,c200=None,
                   z=0.2,e=0.5,
                   ncores=10,cosmo_params=params):

    if not c200:
        c200 = concentration.concentration(M200, '200c', z, model = 'diemer19')
        
    ############  MAKING A GRID
    g = Grilla(rangex = [-5,5],nbins = 10)
    x = g.x
    y = g.y
    r = g.r
    
    mask = (r>0.2)

    x,y = x[mask],y[mask]
    r   = r[mask]
    theta  = np.arctan2(y,x)
    
    ############  computing quantities

    DS1h   = Delta_Sigma_NFW_2h_parallel(r,z,M200 = M200,c200=c200,cosmo_params=cosmo_params,terms='1h',ncores=ncores)
    GT, GX = GAMMA_components_parallel(r,z,ellip=e,M200 = M200,c200=c200,cosmo_params=cosmo_params,terms='1h')
    DS2h   = Delta_Sigma_NFW_2h_parallel(r,z,M200 = M200,c200=c200,cosmo_params=cosmo_params,terms='2h',ncores=ncores)
    GT2h, GX2h = GAMMA_components_parallel(r,z,ellip=e,M200 = M200,c200=c200,cosmo_params=cosmo_params,terms='2h')

    f  = plt.figure(figsize=(12,12)) 
    gs = gridspec.GridSpec(3, 3, width_ratios=[1, 1,1.2])
    
    ax = []
    for j in range(9):
        ax += [plt.subplot(gs[j])]
        ax[j].axis([-5.2,5.2]*2)
    f.subplots_adjust(hspace=0,wspace=0)
    
    ######################
    # MONOPOLE
    ######################

    g0_1 = -1.*DS1h*np.cos(2.*theta)
    g0_2 = -1.*DS1h*np.sin(2.*theta)
    g0_1_2h = -1.*DS2h*np.cos(2.*theta)
    g0_2_2h = -1.*DS2h*np.sin(2.*theta)

    mq     = make_shear_map(x,y,g0_1,g0_2,ax=ax[0])
    mq_2h  = make_shear_map(x,y,g0_1_2h,g0_2_2h,ax=ax[3],cmap='viridis')
    mq_all = make_shear_map(x,y,g0_1+g0_1_2h,g0_2+g0_2_2h,ax=ax[6])
    
    ######################
    # QUADRUPOLE
    ######################

    gt2    = GT*np.cos(2.*theta)
    gx2    = GX*np.sin(2.*theta)
    gt2_2h = GT2h*np.cos(2.*theta)
    gx2_2h = GX2h*np.sin(2.*theta)

    g2_1 = -1.*gt2*np.cos(2.*theta) + gx2*np.sin(2.*theta)
    g2_2 = -1.*gt2*np.sin(2.*theta) - gx2*np.cos(2.*theta)
    g2_1_2h = -1.*gt2_2h*np.cos(2.*theta) + gx2_2h*np.sin(2.*theta)
    g2_2_2h = -1.*gt2_2h*np.sin(2.*theta) - gx2_2h*np.cos(2.*theta)

    qq     = make_shear_map(x,y,g2_1,g2_2,ax=ax[1])
    qq_2h  = make_shear_map(x,y,g2_1_2h,g2_2_2h,ax=ax[4],cmap='viridis')
    qq_all = make_shear_map(x,y,g2_1+g2_1_2h,g2_2+g2_2_2h,ax=ax[7])

    plt.setp(ax[1].get_yticklabels(), visible=False)
    plt.setp(ax[4].get_yticklabels(), visible=False)
    plt.setp(ax[7].get_yticklabels(), visible=False)
    
    ######################
    # TOTAL SHEAR MAP
    ######################
    gt = DS1h + GT*np.cos(2.*theta)
    gx = GX*np.sin(2.*theta)
    gt_2h = DS2h + GT2h*np.cos(2.*theta)
    gx_2h = GX2h*np.sin(2.*theta)

    g_1 = -1.*gt*np.cos(2.*theta) + gx*np.sin(2.*theta)
    g_2 = -1.*gt*np.sin(2.*theta) - gx*np.cos(2.*theta)
    g_1_2h = -1.*gt_2h*np.cos(2.*theta) + gx_2h*np.sin(2.*theta)
    g_2_2h = -1.*gt_2h*np.sin(2.*theta) - gx_2h*np.cos(2.*theta)

    tq     = make_shear_map(x,y,g2_1,g2_2,ax=ax[2])
    tq_2h  = make_shear_map(x,y,g2_1_2h,g2_2_2h,ax=ax[5],cmap='viridis')
    tq_all = make_shear_map(x,y,g2_1+g2_1_2h,g2_2+g2_2_2h,ax=ax[8])


    plt.setp(ax[5].get_yticklabels(), visible=False)
    plt.setp(ax[7].get_yticklabels(), visible=False)
    plt.setp(ax[8].get_yticklabels(), visible=False)

    
    
    mq.set_clim(np.quantile(np.abs(g_1),0.1),np.quantile(np.abs(g_1),0.9))
    qq.set_clim(np.quantile(np.abs(g_1),0.1),np.quantile(np.abs(g_1),0.9))
    tq.set_clim(np.quantile(np.abs(g_1),0.1),np.quantile(np.abs(g_1),0.9))
    mq_all.set_clim(np.quantile(np.abs(g_1),0.1),np.quantile(np.abs(g_1),0.9))
    qq_all.set_clim(np.quantile(np.abs(g_1),0.1),np.quantile(np.abs(g_1),0.9))
    tq_all.set_clim(np.quantile(np.abs(g_1),0.1),np.quantile(np.abs(g_1),0.9))
    mq_2h.set_clim(np.quantile(np.abs(g_1_2h),0.1),np.quantile(np.abs(g_1_2h),0.9))
    qq_2h.set_clim(np.quantile(np.abs(g_1_2h),0.1),np.quantile(np.abs(g_1_2h),0.9))
    tq_2h.set_clim(np.quantile(np.abs(g_1_2h),0.1),np.quantile(np.abs(g_1_2h),0.9))

    
    ax[0].set_ylabel('y [Mpc]')
    ax[3].set_ylabel('y [Mpc]')
    ax[6].set_ylabel('y [Mpc]')
    ax[6].set_xlabel('x [Mpc]')
    ax[7].set_xlabel('x [Mpc]')
    ax[8].set_xlabel('x [Mpc]')
    
    cbar = f.colorbar(tq)#,ticks=[40,80,120])
    # cbar.ax.set_yticklabels(['40','80','120'])
    cbar.set_label(u'$M_\odot$pc$^{-2}$', rotation=270)

    cbar = f.colorbar(tq_all)#,ticks=[40,80,120])
    # cbar.ax.set_yticklabels(['40','80','120'])
    cbar.set_label(u'$M_\odot$pc$^{-2}$', rotation=270)

    cbar = f.colorbar(tq_2h)#,ticks=[1,3,5])
    # cbar.ax.set_yticklabels(['10','30','50'])
    cbar.set_label(u'$M_\odot$pc$^{-2}$', rotation=270)
