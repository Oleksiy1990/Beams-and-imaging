import numpy as np 

def lens_matrix(f):
    return np.matrix([[1,0],[-1/f,1]])

def lens_meniscus(rfront,rback,refr):
	return np.matrix([[1,0],[-(refr-1)*(1/rfront-1/rback),1]])

def mirror_matrix(radcurv):
	return np.matrix([[1,0],[2/radcurv,1]])

def space_matrix(dist,n):
    return np.matrix([[1,dist/n],[0,1]])

def q_param(lam,waist,rad):
    q = 1/(1/rad - lam*1j/(np.pi*waist**2))
    return q
def propagate_q(q_in,matr):
    q_out = (matr[0,0]*q_in+matr[0,1])/(matr[1,0]*q_in+matr[1,1])
    return q_out

def rayleigh_r(q):
    inv = 1/q
    rayl = -1/inv.imag
    return rayl
def waist_from_q(q,lam):
    inv = 1/q
    rayl = -1/inv.imag
    w = np.sqrt(lam*rayl/np.pi)
    return w