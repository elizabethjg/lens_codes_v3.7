import numpy as np


class Grilla:

    def __init__(self,rangex = [-10,10], rangey = None, nbins = 10):

        if not rangey:
            rangey = rangex
            
        xb = np.linspace(rangex[0],rangex[1],nbins)
        yb = np.linspace(rangey[0],rangey[1],nbins)
        
        x, y = np.meshgrid(xb,yb)
        
        y = y.flatten()
        x = x.flatten()
        
        theta  = np.arctan2(y,x)
        r = np.sqrt(x**2 + y**2)
        
        o = np.argsort(r)
        
        self.x = x[o]
        self.y = y[o]
        self.r = r[o]
        self.t = theta[o]
