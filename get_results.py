import pickle
import seaborn as sns; sns.set_theme()
from matplotlib import pyplot
with open ("analysis/stats_ti.txt", 'rb') as f: 
    stats = pickle.load(f)
sharp = []
for p in Product:
    sharp.append([p])
    for ti in stats[p]:
        if ti != 'pred':
            sharp[-1].append(abs(stats[p][ti][0][ti+'_ret']))
df = pd.DataFrame(sharp, columns = ['Product'] + list(stats[p].keys())[:-1])
df.set_index('Product', inplace = True)
pyplot.figure(figsize=(10, 10))
ax = sns.heatmap(df.T, linewidths=.5, annot=True)
s1 = ax.get_figure()
s1.savefig('HeatMap_ti.jpg',dpi=300,bbox_inches='tight')


with open ("analysis/stats_com.txt", 'rb') as f: 
    stats = pickle.load(f)
sharp = []
for p in Product:
    sharp.append([p])
    for ti in stats[p]:
        sharp[-1].append(abs(stats[p][ti][0][ti+'_ret']))
df = pd.DataFrame(sharp, columns = ['Product'] + list(stats[p].keys()))
df.set_index('Product', inplace = True)

pyplot.figure(figsize=(10, 10))
ax = sns.heatmap(df.T, linewidths=.5, annot=True)
s1 = ax.get_figure()
s1.savefig('HeatMap_pred.jpg',dpi=300,bbox_inches='tight')