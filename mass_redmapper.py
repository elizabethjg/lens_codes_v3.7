from colossus.halo import mass_defs
import numpy as np
from colossus.halo import concentration
from colossus.cosmology import cosmology  
params = {'flat': True, 'H0': 70.0, 'Om0': 0.25, 'Ob0': 0.044, 'sigma8': 0.8, 'ns': 0.95}
cosmology.addCosmology('MICE', params)
cosmo = cosmology.setCosmology('MICE')

def logM200c(lambda_red,zmean):
		
	M200m = 2.21e14*(lambda_red/40.)**1.33
	c200m = concentration.concentration(M200m, '200m', zmean, model = 'diemer19')

	M200c = mass_defs.changeMassDefinition(M200m, c200m, zmean, '200m', '200c')[0]

	return np.log10(M200c)
