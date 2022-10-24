import numpy as np 
import pickle
import os 
import pandas as pd 
for final_model in ['32gf','64gf','128gf', '256gf', '32_64', '32_64_128', '32_64_128_256']:
	sweep=[]
	for final_wd in ['1e-6', '1e-5', '1e-4']:
		resdir=f'results/seer_cat/inaturalist18/lineareval/seer_{final_model}_wd{final_wd}'
		f = open(os.path.join(resdir, 'stats0.pkl'), 'rb')
		stats = pickle.load(f)
		sweep.append([float(stats['prec1_val'].values[-1]),
					float(stats['prec1_val'].values.max()),
					float(stats['prec1'].values[-1]), 
					len(stats), final_model, -1, -1, -1, float(final_wd),'inaturalist18'])

	sweep = pd.DataFrame(sweep)
	idx = np.argsort(sweep[0].values)
			
	print("%.3f, %.3f, %.3f, %d, %s, %d, %.3f, %d, %f, %s" % tuple(sweep.iloc[idx[-1]].values))

