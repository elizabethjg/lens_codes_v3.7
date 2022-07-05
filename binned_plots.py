import numpy as np
from scipy import stats
from matplotlib import pyplot as plt



def q_75(y):
    return np.quantile(y, 0.75)

def q_25(y):
    return np.quantile(y, 0.25)


def binned(x,y,nbins=10):
    
    
    bined = stats.binned_statistic(x,y,statistic='mean', bins=nbins)
    x_b = 0.5*(bined.bin_edges[:-1] + bined.bin_edges[1:])
    ymean     = bined.statistic

    bined = stats.binned_statistic(x,y,statistic='median', bins=nbins)
    x_b = 0.5*(bined.bin_edges[:-1] + bined.bin_edges[1:])
    q50     = bined.statistic
    
    bined = stats.binned_statistic(x,y,statistic=q_25, bins=nbins)
    q25     = bined.statistic

    bined = stats.binned_statistic(x,y,statistic=q_75, bins=nbins)
    q75     = bined.statistic

    bined = stats.binned_statistic(x,y,statistic='count', bins=nbins)
    N     = bined.statistic

    bined = stats.binned_statistic(x,y,statistic='std', bins=nbins)
    sigma = bined.statistic
    
    dig   = np.digitize(x,bined.bin_edges)
    mz    = np.ones(len(x))
    for j in range(nbins):
        mbin = dig == (j+1)
        mz[mbin] = y[mbin] >= q50[j]   
    mz = mz.astype(bool)
    return x_b,q50,q25,q75,mz,ymean,sigma/np.sqrt(N)
            

def make_plot2(X,Y,color='C0',nbins=20,plt=plt,label='',error = False,lw=1,lt='-'):
    x,q50,q25,q75,nada,ymean,ers = binned(X,Y,nbins)
    if error:
        plt.plot(x,ymean,lt,color=color,label=label,lw=lw)
        plt.fill_between(x,ymean+ers,ymean-ers,color=color,alpha=0.2)
    else:
        plt.plot(x,q50,lt,color=color,label=label,lw=lw)
        plt.fill_between(x,q75,q25,color=color,alpha=0.2)
