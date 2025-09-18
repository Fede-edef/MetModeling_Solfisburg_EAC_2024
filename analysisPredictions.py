# File to run analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap as umap
from pandas import read_pickle
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import umap.umap_ as umap
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
import skbio
from scipy.spatial import distance
from scipy.spatial.distance import pdist, squareform
from scipy.spatial.distance import pdist, squareform
from skbio.stats.distance import permanova, DistanceMatrix
from skbio.stats.distance import anosim
import statistics


#Loading custom functions 
#PCoA with elipsis
def do_pcoa(pcoa_mat, distmat, col, cond1, cond2):
#     plt.close('all')
    df = pcoa_mat
    xa = df.columns[0]
    ya = df.columns[1]
    #df = df.merge(metadata, left_index = True, right_on = CMMR, how = 'left') #fede
    print(df[col].unique())
    g = sns.scatterplot(x = xa, y=ya, data = df, hue = col, size = 3, linewidth=.2, hue_order = [cond1, cond2],
                       palette=COLOR_B[:2][::-1])
    e1 = plot_point_cov(df[df[col] == cond1][[xa, ya]].values)
    e2 = plot_point_cov(df[df[col] == cond2][[xa, ya]].values)
    print(cond1, (df[col] == cond1).sum(), cond2, (df[col] == cond2).sum())
    e1.set_facecolor('none')
    e2.set_facecolor('none')
    e1.set_edgecolor(COLOR_B[1])
    e2.set_edgecolor(COLOR_B[0])
    #met = metadata.set_index(CMMR)[[col]].dropna() #fede
    met=df.dx #fede
    #filt_dm = distmat.loc[distmat.index.isin(met.index), distmat.columns.isin(met.index)] #fede
    filt_dm = pd.DataFrame(distmat)  #fede
    filt_dm.columns=df.index  #fede
    filt_dm.index=df.index  #fede
    c_dm = DistanceMatrix(filt_dm, ids = filt_dm.index)
    #print(anosim(c_dm, met,
       #col, permutations = int(1e4-1)))
    #print(permanova(c_dm, met,
       #col, permutations = int(1e4-1)))
    print(anosim(c_dm, met,
        permutations = int(1e4-1)))
    print(permanova(c_dm, met,
        permutations = int(1e4-1)))
    #return g
# https://stackoverflow.com/questions/12301071/multidimensional-confidence-intervals
from matplotlib.patches import Ellipse

def plot_point_cov(points, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma ellipse based on the mean and covariance of a point
    "cloud" (points, an Nx2 array).
    Parameters
    ----------
        points : An Nx2 array of the data points.
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.
    Returns
    -------
        A matplotlib ellipse artist
    """
    pos = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    return plot_cov_ellipse(cov, pos, nstd, ax, **kwargs)
def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the
    ellipse patch artist.
    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.
    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]
    if ax is None:
        ax = plt.gca()
    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
    ax.add_artist(ellip)
    return ellip



################################################ Loading NMPCs and labels #######################################

NMPCs = pd.read_csv('/home/federico/Documents/Quinn/unealthy/standard2.csv', index_col=0)
NMPCs = NMPCs.transpose()
labs= pd.read_csv('/home/federico/Documents/Quinn/BEids_for_metabolites2020Mar05 (copy).csv', index_col=0)

cases=labs.loc[labs.loc[:,'case_control']==1,'case_control']==1
cases=list(cases.index)
cases=[x.replace('-', '_') for x in list(cases)]

controls=labs.loc[labs.loc[:,'case_control']==0,'case_control']==0
controls=list(controls.index)
controls=[x.replace('-', '_') for x in list(controls)]

#Select only matched NMPCs
allObs= cases +controls
NMPCs=NMPCs.loc[NMPCs.index.isin(allObs), :]
#get rid of all 0s
NMPCs=NMPCs.loc[:, (NMPCs != NMPCs.iloc[0]).any()]
# drop the one sample where the NMPCs are all na (must have been an unsolvable model)
NMPCs = NMPCs.dropna(axis=0, how='all')
## remove columns (NMPCs) where all values are 0
NMPCs = NMPCs.loc[:, (NMPCs != 0).any(axis=0)]

## Code for log tranform, commented as we have signed NMPCs
#eps = NMPCs[NMPCs > 0].min(axis=1).sort_values().iloc[0] / 10
### use log(x+1) transform to reduce skew while preserving rank
#NMPCs_log_plus_eps = np.log10(NMPCs + eps)

# Setting mean treshold on computed values

for c in NMPCs.columns:
    if statistics.mean(NMPCs.loc[:, c])< 0.01:
       del NMPCs[c]



#setting at least to 10% of cases
atleast=round(len(cases)/100*10)



lst = []
for c in NMPCs.columns:
    cnt = 0
    x=NMPCs.loc[NMPCs.index.isin(cases), c]
    a = x.dropna()
    y=NMPCs.loc[NMPCs.index.isin(controls), c]
    b = y.dropna()

    if len(a) < atleast or len(b) < atleast:
        print('jump')
        #lst.append((c, [np.nan], [np.nan]))
        cnt = 1

    if np.sum(a!=0) < atleast or np.sum(b!=0) < atleast:
        print('jump')
        #lst.append((c, [np.nan], [np.nan]))
        cnt = 1

    if a.nunique() == 1 and b.nunique() == 1 and a.iloc[0] == b.iloc[0]:
        print('jump')
        #lst.append((c, [np.nan], [np.nan]))
        cnt = 1

    if cnt == 0:
        stat, p = mannwhitneyu(a, b, alternative='two-sided')[0:2]
        mDiff=np.median(b) - np.median(a) 
        if np.median(a) > np.median(b):
            stat = stat * -1
        lst.append((c, stat, p))



df = pd.DataFrame(lst)
df['q'] = multipletests(df[2], method='fdr_bh')[1]

sigDf=df[df['q']<0.1] 

#ok, we need to translate sigDf
tr=pd.read_table('/home/federico/Downloads/recon-store-metabolites-1 (3).tsv',index_col=0)

sigDfNam=sigDf[0].str.replace('EX_','')
sigDfNam=sigDfNam.str.replace('\[fe\]','')

sig=sigDfNam.tolist()
significant=tr.loc[tr.index.isin(sig),'fullName']
significantDf=pd.DataFrame(significant)
sigDf.index=significantDf.loc[:,'fullName']
sigDf.columns=['VMH_ID','Stat','p','q']
sigDf.to_csv('/home/federico/Documents/Quinn/analysisPY/sigHits.csv')

#sigDf.iloc[:,2]=round(sigDf.iloc[:,2],5)
#sigDf.iloc[:,3]=round(sigDf.iloc[:,3],5)

#Converting NMPCs into log scale

############################################################# Preparing data for volcano plot ##########################

lst = []
for c in NMPCs.columns:
    cnt = 0
    x=NMPCs.loc[NMPCs.index.isin(cases), c]
    a = x.dropna()
    y=NMPCs.loc[NMPCs.index.isin(controls), c]
    b = y.dropna()

    if len(a) < atleast or len(b) < atleast:
        print('jump')
        #lst.append((c, [np.nan], [np.nan]))
        cnt = 1

    if np.sum(a!=0) < atleast or np.sum(b!=0) < atleast:
        print('jump')
        #lst.append((c, [np.nan], [np.nan]))
        cnt = 1

    if a.nunique() == 1 and b.nunique() == 1 and a.iloc[0] == b.iloc[0]:
        print('jump')
        #lst.append((c, [np.nan], [np.nan]))
        cnt = 1

    if cnt == 0:
        stat, p = mannwhitneyu(a, b, alternative='two-sided')[0:2]
        mDiff=np.median(b) - np.median(a) #to comment to run original script
        #l2Fc=np.log2(np.median(b)/np.median(a))
        l2Fc = np.median(b) / np.median(a) #new definition
        #l2Fc = (np.median(b) - np.median(a))/ np.median(a) #old definition
        if np.median(a) > np.median(b):
            stat = stat * -1
        lst.append((c, stat, mDiff,l2Fc, p))



#df = pd.DataFrame(lst)
#df['q'] = multipletests(df[4], method='fdr_bh')[1]

df2 = pd.DataFrame(lst)
df2['q'] = multipletests(df2[4], method='fdr_bh')[1]
df2.columns=['VMH_ID','Stat','mDiff','Fc','p','q']
df2['-log(p)']= -np.log10(df2['p'])
df2['log2(fc)']= np.log2(df2['Fc'])
df2['-log(q)']= -np.log10(df2['q'])

#To find p value matching q value at 0.1 (before was 0.05)
#df2.sort_values('q')[['p','q']][50:60]
df2.sort_values('q')[['p','q']][40:60]


df3=df2
df3.sort_values('mDiff')[['mDiff']] [50:70]
#df3 = df3.drop(df3[(df3.mDiff < 0.1) & (df3.mDiff > -0.1)].index)
df3['Tal']= 1-(2*df3['Stat'])/(78*125)
df3['U1']=np.absolute(df3['Stat'])
df3['U1s']=np.absolute(df3['Stat'])/ (78*125)
df3['U1s']=1-(2*np.absolute(df3['Stat']))/ (78*125)

#To plot volcano
plt.close('all')
plt.figure(figsize=(8,8))
#plt.figure(figsize = (7,7))
sns.scatterplot(data=df3.sort_values(by='q'), x='U1s', y='-log(p)', #x='mDiff'
                s=120, palette=['darkgoldenrod', 'darkslategray'], legend=False)

#plt.ylabel("-log10(p)",  fontsize=20)
plt.ylabel("p-value",  fontsize=20)
plt.xlabel("Effect size", fontsize=20)
#plt.axhline(-np.log10(0.017), color='maroon', linewidth=3) #as it was before
plt.axhline(-np.log10(0.039), color='maroon', linewidth=3)
plt.axhline(-np.log10(0.05), color='black', linewidth=2)
plt.axvline(0.5, color='gray', linewidth=3)
plt.gca().xaxis.set_tick_params(width=3, length=10)
plt.gca().yaxis.set_tick_params(width=3, length=10)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.xlim([0.3, 0.7])
plt.scatter(0.588923076923077,1.47744384720017,marker='o', facecolors='none', edgecolors='r',s=150)
plt.scatter(0.594871794871795,1.63528603852118,marker='o', facecolors='none', edgecolors='r',s=150)
plt.scatter(0.390769230769231,2.04909732145944,marker='o', facecolors='none', edgecolors='r',s=150)
plt.scatter(0.368717948717949,2.77636528369129,marker='o', facecolors='none', edgecolors='r',s=150)

#Other top significant
plt.scatter(0.325128205128205,4.54812329984791,marker='o', facecolors='none', edgecolors='r',s=150)
#plt.scatter(0.336102564102564,4.06224632364609,marker='o', facecolors='none', edgecolors='r',s=150)
#plt.scatter(0.338666666666667,3.94980825007855,marker='o', facecolors='none', edgecolors='r',s=150)
plt.scatter(0.339282051282051,3.92364553254716,marker='o', facecolors='none', edgecolors='r',s=150)
plt.scatter(0.344717948717949,3.69643225643601,marker='o', facecolors='none', edgecolors='r',s=150)

plt.savefig("/home/federico/Documents/Quinn/analysisPY/outputVulc.png", dpi = 800)
plt.savefig("/home/federico/Documents/Quinn/analysisPY/outputVulc.svg", format="svg")

#Finding points at extreme
df3.sort_values('-log(p)')[['VMH_ID','mDiff','-log(p)','log2(fc)']] [70:92]

df3.sort_values('-log(p)')[['VMH_ID','mDiff','-log(p)','log2(fc)']] [85:92]
dfS=df3.sort_values('-log(p)')
dfS.to_csv("/home/federico/Documents/Quinn/analysisPY/dataset.csv")
#################################################################### Ordinations ########################################
#PCoA
# Distance Matrix
# Scaling mean = 0, var = 1
#eps = NMPCs[NMPCs > 0].min(axis=1).sort_values().iloc[0] / 10
## use log(x+1) transform to reduce skew while preserving rank
#NMPCs_log_plus_eps = np.log10(NMPCs + eps)

DF_standard = pd.DataFrame(StandardScaler().fit_transform(NMPCs),
                           index = NMPCs.index,
                           columns = NMPCs.columns)
#DF_standard=NMPCs
#DF_standard=NMPCs_log_plus_eps

distance_matrix = squareform(pdist(DF_standard.values, 'euclid')) #'euclid' 'cityblock' 'braycurtis'
my_pcoa = skbio.stats.ordination.pcoa(distance_matrix)

#Labelled PCoA scatter plot
#plt.scatter(my_pcoa.samples['PC1'],  my_pcoa.samples['PC2'])
#for i in range(len(DF_standard.index)):
  #plt.text(my_pcoa.samples.loc[str(i),'PC1'],  my_pcoa.samples.loc[str(i),'PC2'], DF_standard.index[i])
#plt.show()
#plt.savefig("/home/federico/Documents/Quinn/analysisPY/output.png", dpi = 800)

toMerge=labs.loc[:,'dx']
toMerge.index=[x.replace('-', '_') for x in list(toMerge.index)]
toMerge=toMerge[NMPCs.index]
toMerge=toMerge.to_frame()
merged2=pd.merge(DF_standard, toMerge, left_index=True, right_index=True)
merged2.dx.value_counts(dropna=False)

#plot with automated color labelling
ptb_colors = ['dodgerblue', 'darkorange']
ptb_colors = ['dodgerblue', 'firebrick']
ptb_colors = ['green','blue']

plt.close('all')
plt.figure()
fig = plt.figure(figsize = (3,3))
#sns.scatterplot(data=my_pcoa.samples, x='PC1', y='PC2', hue=list(merged2.dx.fillna('Missing_dx')), ax=plt.gca())
ax=sns.scatterplot(data=my_pcoa.samples, x='PC1', y='PC2', hue=list(merged2.dx), ax=plt.gca(), palette=ptb_colors)
ax.set(xlabel="PCo1 (19%)", ylabel = "PCo2 (15%)")
#sns.scatterplot(data=my_pcoa.samples, x='PC1', y='PC6', hue=list(merged2.dx), ax=plt.gca())
scatter_xlim = plt.gca().get_xlim()
scatter_ylim = plt.gca().get_ylim()
plt.savefig("/home/federico/Documents/Quinn/analysisPY/outputPCoA.png", dpi = 800)
plt.savefig("/home/federico/Documents/Quinn/analysisPY/outputPCoA.svg", format="svg")
#For post labeling: explained variance and permanova
#We need to remove NAs from labels
labsRecall=labs
labsRecall.index=[x.replace('-', '_') for x in list(labsRecall.index)]
intersect=NMPCs.index.intersection(labsRecall.index)

DF_standard2=DF_standard.loc[intersect,:]
merged2R=merged2.loc[intersect,:]

dm = squareform(pdist(DF_standard2.values, 'euclid'))
p_anova = permanova(DistanceMatrix(dm), merged2R.dx)
#p_anova = permanova(DistanceMatrix(dm), merged2.dx.fillna('Missing_dx'))

#Plotting boxplots of PCoA axes
ptb_colors = ['dodgerblue', 'darkorange']
#ptb_colors = ['dodgerblue', 'sandybrown']
my_pcoa.samples2=my_pcoa.samples

my_pcoa.samples2.index=NMPCs.index
mergedPCoA=pd.merge(my_pcoa.samples2, merged2.dx, left_index=True, right_index=True)

plt.close('all')
sns.reset_defaults()
fig = plt.figure(figsize = (1,4.5))
sns_plot=sns.boxplot(x='dx', y='PC2', data=mergedPCoA, palette=ptb_colors, showfliers=False, medianprops=dict(color='black'), whiskerprops=dict(color='black'), capprops=dict(color='black'))
sns_plot=sns.swarmplot(data=mergedPCoA, x = 'dx', y='PC2', palette=['gray', 'gray'] , s=3, dodge=True, alpha=1)
# fig = sns_plot.get_figure()
plt.ylim(scatter_ylim)
#plt.gca().get_xaxis().set_visible(False)
plt.gca().get_yaxis().set_visible(False)
#fig.savefig("/home/federico/Documents/Quinn/analysisPY/output.png",dpi=600)
fig.savefig("/home/federico/Documents/Quinn/analysisPY/output.svg", format="svg")
p = mannwhitneyu(mergedPCoA.loc[mergedPCoA.index.isin(cases), 'PC2'], mergedPCoA.loc[mergedPCoA.index.isin(controls), 'PC2'], alternative='two-sided')


plt.close('all')
#fig = plt.figure(figsize = (1,7))
fig = plt.figure(figsize = (1,4.5))
sns_plot=sns.boxplot(x='dx', y='PC1', data=mergedPCoA, palette=ptb_colors, showfliers=False, medianprops=dict(color='black'), whiskerprops=dict(color='black'), capprops=dict(color='black'))
sns_plot=sns.swarmplot(data=mergedPCoA, x = 'dx', y='PC1', palette=['gray', 'gray'] , s=3, dodge=True, alpha=1)
# fig = sns_plot.get_figure()
plt.ylim(scatter_xlim)
#plt.gca().get_xaxis().set_visible(False)
plt.gca().get_yaxis().set_visible(False)
fig.savefig("/home/federico/Documents/Quinn/analysisPY/output2.png",dpi=600)
fig.savefig("/home/federico/Documents/Quinn/analysisPY/output2.svg", format="svg")
p = mannwhitneyu(mergedPCoA.loc[mergedPCoA.index.isin(cases), 'PC1'], mergedPCoA.loc[mergedPCoA.index.isin(controls), 'PC1'], alternative='two-sided')


#Getting PcoA with ellipsis
dm2 = squareform(pdist(DF_standard2.values, 'euclid'))
my_pcoa2 = skbio.stats.ordination.pcoa(dm2)
pcoa_mat=my_pcoa2.samples
pcoa_mat.index=merged2R.index
metadata=merged2R.loc[:,'dx']
pcoa_mat=pd.merge(pcoa_mat,merged2R.loc[:,'dx'], left_index=True, right_index=True)

plt.close('all')
COLOR_B = ['darkorange','dodgerblue']
myPlot=do_pcoa(pcoa_mat, dm2, 'dx', 'HGD/EAC', 'Control')
plt.savefig("/home/federico/Documents/Quinn/analysisPY/output2.png", dpi = 800)

#uMAP
toMerge=labs.loc[:,'dx']
toMerge.index=[x.replace('-', '_') for x in list(toMerge.index)]
#Preparing joined dataframe
tempNMPCNam=NMPCs.columns
tempNMPCNam=tempNMPCNam.str.replace('EX_','')
tempNMPCNam=tempNMPCNam.str.replace('\[fe\]','')
NMPCs.columns=tempNMPCNam
toMerge2=NMPCs #[significant.index]
merged=pd.merge(toMerge2, toMerge, left_index=True, right_index=True)

ct1k_lra=merged.iloc[:,1:(len(merged.columns)-1)]

plt.close('all')
u = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    metric='braycurtis', #'braycurtis',
    random_state=42).fit_transform(ct1k_lra)


plt.scatter(
    u[:, 0],
    u[:, 1],
    c=[sns.color_palette()[x] for x in merged.dx.map({"neverBE":0, "advneo":1})])
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP of NMPCs', fontsize=24)

plt.xticks([])
_=plt.yticks([])
plt.legend([],[], frameon=False)
plt.savefig("/home/federico/Documents/Quinn/analysisPY/output.png", dpi = 800)




plt.close('all')
fig = plt.figure(figsize = (4.5,4.5))
kwargs = {'x': u[:,0], 'y':u[:,1]}

hu = csts_cts.OrigCST#ValOCT_CST
hu_race = md_cts.loc[ct1k_lra.index, 'race']

# sns.scatterplot(**kwargs, hue = hu_race, s=30, hue_order = sorted(hu_race.unique()), ax = plt.subplot(111))
sns.scatterplot(**kwargs, hue = hu.rename('CST'), s=30, palette = cst_colors, hue_order = sorted(hu.unique()), ax = plt.subplot(111))

plt.xticks([])
_=plt.yticks([])
plt.legend([],[], frameon=False)
plt.savefig('./1B.png', dpi = 800)
plt.savefig('./1B.pdf', dpi = 800)



fig = fig.get_figure()
fig.savefig("/home/federico/Documents/Quinn/analysisPY/output.png")


hu = csts_cts.OrigCST#ValOCT_CST
hu_race = md_cts.loc[ct1k_lra.index, 'race']


#################################################################### Boxplots ####################################

ptb_colors = ['firebrick', 'dodgerblue']
ptb_colors = ['dodgerblue', 'darkorange']
#Preparing labels
toMerge=labs.loc[:,'dx']
toMerge.index=[x.replace('-', '_') for x in list(toMerge.index)]
#Preparing joined dataframe
tempNMPCNam=NMPCs.columns
tempNMPCNam=tempNMPCNam.str.replace('EX_','')
tempNMPCNam=tempNMPCNam.str.replace('\[fe\]','')
NMPCs.columns=tempNMPCNam
toMerge2=NMPCs[significant.index]
merged=pd.merge(toMerge2, toMerge, left_index=True, right_index=True)

#Plotting
plt.close('all')
#sns_plot=sns.boxplot(x='dx', y='12dgr180', data=merged, palette=ptb_colors, showfliers=False, medianprops=dict(color='black'), whiskerprops=dict(color='black'), capprops=dict(color='black'))
sns_plot=sns.boxplot(x='dx', y='bhb', data=merged, palette=ptb_colors, showfliers=False, medianprops=dict(color='black'), whiskerprops=dict(color='black'), capprops=dict(color='black'))
sns_plot=sns.swarmplot(data=merged, x = 'dx', y = 'bhb', palette=['gray', 'gray'] , s=3, dodge=True, alpha=1)
fig = sns_plot.get_figure()
fig.savefig("/home/federico/Documents/Quinn/analysisPY/output_bhb.png",dpi=600)


#Standardize NMPCs and log transform them
eps = NMPCs[NMPCs > 0].min(axis=1).sort_values().iloc[0] / 10
## use log(x+1) transform to reduce skew while preserving rank
NMPCs_log_plus_eps = np.log10(NMPCs + eps)

DF_standardBx = pd.DataFrame(StandardScaler().fit_transform(NMPCs_log_plus_eps),
                           index = NMPCs.index,
                           columns = NMPCs.columns)


toMergeBx=labs.loc[DF_standardBx.index,'dx']
toMergeBx.index=[x.replace('-', '_') for x in list(toMergeBx.index)]
toMergeBx=toMergeBx.to_frame()
mergedBx=pd.merge(DF_standardBx, toMergeBx, left_index=True, right_index=True)
mergedBx.dx.value_counts(dropna=False)


#sns.boxplot(data=NMPCs.loc[:,'EX_12dgr180[fe]'], x = 'race', y = 'EX_tym[fe]', hue='PTB', palette=ptb_colors, showfliers=False, ax = axs, medianprops=dict(color='black'), whiskerprops=dict(color='black'), capprops=dict(color='black'))
#sns.boxplot(data=NMPCs.loc[:,'EX_12dgr180[fe]'], palette=ptb_colors)


#obut
plt.close('all')
fig = plt.figure(figsize = (2,4))
#fig.ylim([-4.5, 2.5])
sns_plot=sns.boxplot(x='dx', y='EX_2obut[fe]', data=mergedBx, palette=ptb_colors, showfliers=False, medianprops=dict(color='black'), whiskerprops=dict(color='black'), capprops=dict(color='black'))
sns_plot=sns.swarmplot(data=mergedBx, x = 'dx', y = 'EX_2obut[fe]', palette=['gray', 'gray'] , s=3, dodge=True, alpha=1)
sns_plot.set_ylim([-4.5, 3.1])
fig = sns_plot.get_figure()
fig.savefig("/home/federico/Documents/Quinn/analysisPY/2obut.svg", format="svg")


#lac
plt.close('all')
fig = plt.figure(figsize = (2,4))
#fig = plt.figure(figsize = (3,5))
sns_plot=sns.boxplot(x='dx', y='EX_lac_L[fe]', data=mergedBx, palette=ptb_colors, showfliers=False, medianprops=dict(color='black'), whiskerprops=dict(color='black'), capprops=dict(color='black'))
sns_plot=sns.swarmplot(data=mergedBx, x = 'dx', y = 'EX_lac_L[fe]', palette=['gray', 'gray'] , s=3, dodge=True, alpha=1)
sns_plot.set_ylim([-4.5, 3.1])
plt.gca().get_yaxis().set_visible(False)
fig = sns_plot.get_figure()
fig.savefig("/home/federico/Documents/Quinn/analysisPY/lac.svg", format="svg")

#but
plt.close('all')
#fig = plt.figure(figsize = (3,5))
fig = plt.figure(figsize = (2,4))
sns_plot=sns.boxplot(x='dx', y='EX_but[fe]', data=mergedBx, palette=ptb_colors, showfliers=False, medianprops=dict(color='black'), whiskerprops=dict(color='black'), capprops=dict(color='black'))
sns_plot=sns.swarmplot(data=mergedBx, x = 'dx', y = 'EX_but[fe]', palette=['gray', 'gray'] , s=3, dodge=True, alpha=1)
fig = sns_plot.get_figure()
fig.savefig("/home/federico/Documents/Quinn/analysisPY/but.svg", format="svg")

diag=mergedBx.loc[:,'EX_but[fe]']
diag.sort_values()
NMPCs.loc['OM_110','EX_but[fe]']
mergedBx.loc['OM_110','EX_but[fe]']

#re-doing boxplot for but

plt.close('all')
mergedBx2=mergedBx
mergedBx2.loc['OM_110','EX_but[fe]']=-4
fig = plt.figure(figsize = (2,4))
#fig = plt.figure(figsize = (3,5))
sns_plot=sns.boxplot(x='dx', y='EX_but[fe]', data=mergedBx2, palette=ptb_colors, showfliers=False, medianprops=dict(color='black'), whiskerprops=dict(color='black'), capprops=dict(color='black'))
sns_plot=sns.swarmplot(data=mergedBx2, x = 'dx', y = 'EX_but[fe]', palette=['gray', 'gray'] , s=3, dodge=True, alpha=1)
sns_plot.set_ylim([-4.5, 3.1])
plt.gca().get_yaxis().set_visible(False)
fig = sns_plot.get_figure()
fig.savefig("/home/federico/Documents/Quinn/analysisPY/but.svg", format="svg")

#trp
plt.close('all')
#fig = plt.figure(figsize = (3,5))
fig = plt.figure(figsize = (2,4))
sns_plot=sns.boxplot(x='dx', y='EX_trp_L[fe]', data=mergedBx, palette=ptb_colors, showfliers=False, medianprops=dict(color='black'), whiskerprops=dict(color='black'), capprops=dict(color='black'))
sns_plot=sns.swarmplot(data=mergedBx, x = 'dx', y = 'EX_trp_L[fe]', palette=['gray', 'gray'] , s=3, dodge=True, alpha=1)
sns_plot.set_ylim([-4.5, 3.1])
plt.gca().get_yaxis().set_visible(False)
fig = sns_plot.get_figure()
fig.savefig("/home/federico/Documents/Quinn/analysisPY/trp.svg", format="svg")

#To plot them all
for c in merged.columns:
    plt.close('all')
    sns_plot=sns.boxplot(x='dx', y=c, data=merged, palette=ptb_colors, showfliers=False, medianprops=dict(color='black'), whiskerprops=dict(color='black'), capprops=dict(color='black'))
    fig = sns_plot.get_figure()
    fig.savefig('/home/federico/Documents/Quinn/analysisPY/'+c+'.png', dpi=600)

plt.close('all')
fig,ax = plt.subplots(6,4, figsize=(15,20))
#cnt=0
cnt=24
#cnt=48
for j in range(len(ax)):
    if cnt == 44:
        break
    for i in range(len(ax[j])):
        if cnt == 44:
            break
        sns_plot = sns.boxplot(x='dx', y=merged.columns[cnt], data=merged, palette=ptb_colors, showfliers=False,
                               medianprops=dict(color='black'), whiskerprops=dict(color='black'),
                               capprops=dict(color='black'), ax=ax[j][i])
        #sns_plot.set_ylabel(merged.columns[cnt], fontsize=4)
        sns_plot.set(xlabel=None)
        sns_plot.set(ylabel=None)
        sns_plot.set_title(merged.columns[cnt], fontsize=12)
        sns_plot.tick_params(labelsize=7.5)
        cnt=cnt+1
fig.savefig('/home/federico/Documents/Quinn/analysisPY/test4.png', dpi=600)
