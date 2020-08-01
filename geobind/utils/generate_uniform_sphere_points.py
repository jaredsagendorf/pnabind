# third-party modules
import numpy as np

def generateUniformSpherePoints(n, r=1):
    """Implemented from note by Markus Deserno: https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf"""
    alpha = 4.0*np.pi*r*r/n
    d = np.sqrt(alpha)
    
    m_nu = int(np.round(np.pi/d))
    d_nu = np.pi/m_nu
    d_phi = alpha/d_nu
    xp = []
    yp = []
    zp = []
    for m in range (0, m_nu):
        nu = np.pi*(m+0.5)/m_nu
        m_phi = int(np.round(2*np.pi*np.sin(nu)/d_phi))
        for n in range (0,m_phi):
            phi = 2*np.pi*n/m_phi
            xp.append(r*np.sin(nu)*np.cos(phi))
            yp.append(r*np.sin(nu)*np.sin(phi))
            zp.append(r*np.cos(nu))
    
    return np.stack([xp, yp ,zp], axis=-1)
