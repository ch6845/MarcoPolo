#!/usr/bin/env python
# coding: utf-8

# jupyter nbconvert 3_report_generator.ipynb --to script
# 
# conda activate pytorch;python 3_report_generator.py $WINDOW

# In[1]:


import os
import sys
print(os.getcwd());os.chdir('/data01/ch6845/MarcoPolo/')
print(os.getcwd())


# In[2]:


import pandas as pd
import numpy as np
from scipy.io import mmread

import matplotlib.pyplot as plt

import MarcoPolo.QQscore as QQ


# In[3]:


dataset_name_all=[
'Kohinbulk_filtered',
'HumanLiver_filtered',   
'Zhengmix8eq_filtered',
]

len(dataset_name_all)


# In[4]:


dataset_name_all=dataset_name_all+["TabulaAorta_filtered", #ok
"TabulaBladder_filtered", #ok
"TabulaBrainMyeloid_filtered", #ok
                  
"TabulaBrainNonMyeloid_filtered",#ok
"TabulaDiaphragm_filtered", #ok

"TabulaFat_filtered", #ok
"TabulaHeart_filtered", #ok

"TabulaKidney_filtered", #ok
"TabulaLargeIntestine_filtered", #ok
"TabulaLimbMuscle_filtered", #ok

"TabulaLiver_filtered", #ok
"TabulaLung_filtered", #ok

"TabulaMammaryGland_filtered", #ok
"TabulaMarrow_filtered", #ok

"TabulaPancreas_filtered", #ok
"TabulaSkin_filtered", #ok
                  
"TabulaSpleen_filtered", #ok
"TabulaThymus_filtered", #ok
                  
"TabulaTongue_filtered", #ok
"TabulaTrachea_filtered"] #ok


# In[5]:


if 'ipykernel' in sys.argv[0]:
    ipykernel=True
    dataset_name=dataset_name_all[0]
    output_mode='report'
else:
    ipykernel=False
    dataset_name=dataset_name_all[int(sys.argv[1])]
    if sys.argv[2]=='pub':
        output_mode='pub'
    elif sys.argv[2]=='test':
        output_mode='report'
    else:
        raise


# In[6]:


mode=2


# In[7]:


path='datasets/extract/{}'.format(dataset_name)
path


# In[8]:


print(path)


# In[9]:


if output_mode=='report':
    report_path='report/{}'.format(path.split('/')[-1])
    print(report_path)
elif output_mode=='pub':
    report_path='docs/{}'.format(path.split('/')[-1].replace('_filtered',''))
    report_path=report_path.replace('Kohinbulk','hESC')
    print(report_path)    


# In[10]:


exp_data=mmread('{}.data.counts.mm'.format(path)).toarray().astype(float)
with open('{}.data.col'.format(path),'r') as f: exp_data_col=[i.strip().strip('"') for i in f.read().split()]
with open('{}.data.row'.format(path),'r') as f: exp_data_row=[i.strip().strip('"') for i in f.read().split()]
assert exp_data.shape==(len(exp_data_row),len(exp_data_col))
assert len(set(exp_data_row))==len(exp_data_row)
assert len(set(exp_data_col))==len(exp_data_col)        

exp_data_meta=pd.read_csv('{}.metadatacol.tsv'.format(path),sep='\t')

cell_size_factor=pd.read_csv('{}.size_factor.tsv'.format(path),sep='\t',header=None)[0].values.astype(float)#.reshape(-1,1)

x_data_intercept=np.array([np.ones(exp_data.shape[1])]).transpose()
x_data_null=np.concatenate([x_data_intercept],axis=1)


# In[11]:


result_list,gamma_list_list=QQ.read_QQscore(path,[1,mode])


# In[12]:


gamma_list=gamma_list_list[-1]    

gamma_argmax_list=QQ.gamma_list_exp_data_to_gamma_argmax_list(gamma_list,exp_data)#gamma_argmax_list=QQ.gamma_list_to_gamma_argmax_list(gamma_list)
gamma_argmax_list,gamma_argmax_list.shape


# In[13]:


allscore=pd.read_csv('{}.MarcoPolo.{}.rank.tsv'.format(path,mode),index_col=0,sep='\t')


# # Gene DB

# In[14]:


#wget https://ftp.ncbi.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz

gene_info=pd.read_csv('datasets/Homo_sapiens.gene_info.gz',sep='\t')


# In[15]:


import sys

by='ID' if 'ENS' in exp_data_row[0] else 'name'

gene_info_select_list=[]

column_list=['Symbol','description','Other_designations','type_of_gene','dbXrefs']

for idx, query in enumerate(exp_data_row):
    if by=='ID':
        gene_info_select=gene_info[gene_info['dbXrefs'].str.contains(query,regex=False)]
    else:
        gene_info_select=gene_info[gene_info['Symbol'].str.lower()==query.lower()]
        if len(gene_info_select)==0:
            gene_info_select=gene_info[gene_info['Synonyms'].str.lower().str.contains(query.lower(),regex=False)]
        
    if len(gene_info_select)>=1:
        gene_info_select_list.append(gene_info_select[column_list].iloc[0])
    else:
        gene_info_select_list.append(pd.Series(index=column_list))
        print(query,len(gene_info_select))
    
    if idx%100==0:
        sys.stdout.write('\r%0.2f%%' % (100.0 * (idx/len(exp_data_row))))
        sys.stdout.flush()
gene_info_extract=pd.DataFrame(gene_info_select_list,index=np.arange(len(exp_data_row)))        


# In[16]:


allscore_munge=allscore.copy()
allscore_munge['Gene ID']=exp_data_row
assert len(gene_info_extract)==len(allscore_munge)
allscore_munge=allscore_munge.merge(right=gene_info_extract,left_index=True,right_index=True)


# In[17]:


allscore_munge[allscore_munge['Gene ID']=="ENSG00000168542"].iloc[:10]


# In[18]:


allscore_munge.sort_values('MarcoPolo_rank').iloc[:20,:20]


# In[19]:


allscore_munge.to_csv('{}.MarcoPolo.{}.rank.munge.tsv'.format(path,mode),sep='\t')


# In[20]:


allscore_munge['img']=allscore_munge.apply(lambda x: '<img src="plot_image/{idx}.png" alt="{idx}">'.format(dataset_name=dataset_name,idx=x.name),axis=1)

allscore_munge


# In[22]:


allscore_munge['Log2FC']=allscore_munge['lfc']/np.log10(2)

if output_mode=='report':

    allscore_munge=allscore_munge[[
                                'MarcoPolo_rank',
                                'Gene ID','Symbol',
                                'description', 'Other_designations', 'type_of_gene',
                                'Log2FC',
                                'MarcoPolo',
                                'QQratio', 'QQratio_rank',
                                'QQdiff', 'QQdiff_rank',
                                'votingscore', 'votingscore_rank',
                                'mean_0_all','mean_0_all_rank',
                                'PCvariance', 'PCvariance_rank',
                                'lfc', 'lfc_rank',
                                'minorsize','minorsize_rank',
                                'dbXrefs','img'
                               ]]

    allscore_munge[['Log2FC',
                    'QQratio', 
                    'QQdiff', 
                    'votingscore', 'votingscore_rank',
                    'mean_0_all',
                    'PCvariance',
                    'lfc']]=\
    allscore_munge[['Log2FC',
                    'QQratio',
                    'QQdiff', 
                    'votingscore', 'votingscore_rank',
                    'mean_0_all',
                    'PCvariance',
                    'lfc']].round(2)
    
elif output_mode=='pub':
    
    allscore_munge=allscore_munge[[
                                'MarcoPolo_rank',
                                'Gene ID','Symbol',
                                'description', 'Other_designations', 'type_of_gene',
                                'Log2FC',
                                'MarcoPolo',
                                'bimodalityscore_rank',
                                'votingscore_rank',
                                'proximityscore_rank',
                                'lfc', 'lfc_rank',
                                'minorsize','minorsize_rank',
                                'dbXrefs','img'
                               ]]

    allscore_munge[['Log2FC',
                    'lfc']]=\
    allscore_munge[['Log2FC',
                    'lfc']].round(2)    
    
else:
    raise


# In[ ]:


allscore_munge


# # file writing

# In[24]:


import os

try:
    os.mkdir('report')
except:
    print('already exists')
    
try:
    os.mkdir('docs')
except:
    print('already exists')    
    
try:
    os.mkdir('{}'.format(report_path))
except:
    print('already exists')
    
try:
    os.mkdir('{}/plot_image'.format(report_path))
except:
    print('already exists')

try:
    os.mkdir('{}/assets'.format(report_path))
except:
    print('already exists')

import shutil
#shutil.copy('report/template/index.html', 'report/{}/index.html'.format(dataset_name_path))
shutil.copy('report/template/assets/scripts.js', '{}/assets/scripts.js'.format(report_path))
shutil.copy('report/template/assets/styles.css', '{}/assets/styles.css'.format(report_path))
shutil.copy('report/template/assets/details_open.png', '{}/assets/details_open.png'.format(report_path))
shutil.copy('report/template/assets/details_close.png', '{}/assets/details_close.png'.format(report_path))
shutil.copy('report/template/assets/mp.png', '{}/assets/mp.png'.format(report_path))
shutil.copy('report/template/assets/mp_white.png', '{}/assets/mp_white.png'.format(report_path))
shutil.copy('report/template/assets/mp_white_large_font.png', '{}/assets/mp_white_large_font.png'.format(report_path))


# In[25]:


from jinja2 import Template

with open('report/template/index.html','r') as f:
    template_read=f.read()
template = Template(source=template_read)  

MarcoPolo_table=allscore_munge.sort_values("MarcoPolo_rank",ascending=True).set_index('MarcoPolo_rank').iloc[:1000]
MarcoPolo_table.index+=1
MarcoPolo_table=MarcoPolo_table.to_html(classes="table table-bordered",table_id='dataTable')

MarcoPolo_table=MarcoPolo_table.replace('<table ','<table width="100%" cellspacing="0" ')
template_rendered=template.render(MarcoPolo_table=MarcoPolo_table,num_gene=exp_data.shape[0],num_cell=exp_data.shape[1])

with open('{}/index.html'.format(report_path),'w') as f:
    f.write(template_rendered)


# In[26]:


#Porting of https://github.com/satijalab/seurat/blob/b51801bc4b1a66aed5456473c9fe0be884994c93/R/visualization.R#L2686
def DiscretePalette(n, palette=None):
    palettes={
                'alphabet':[
                  "#F0A0FF", "#0075DC", "#993F00", "#4C005C", "#191919", "#005C31",
                  "#2BCE48", "#FFCC99", "#808080", "#94FFB5", "#8F7C00", "#9DCC00",
                  "#C20088", "#003380", "#FFA405", "#FFA8BB", "#426600", "#FF0010",
                  "#5EF1F2", "#00998F", "#E0FF66", "#740AFF", "#990000", "#FFFF80",
                  "#FFE100", "#FF5005"
                ],
                'alphabet2':[
                  "#AA0DFE", "#3283FE", "#85660D", "#782AB6", "#565656", "#1C8356",
                  "#16FF32", "#F7E1A0", "#E2E2E2", "#1CBE4F", "#C4451C", "#DEA0FD",
                  "#FE00FA", "#325A9B", "#FEAF16", "#F8A19F", "#90AD1C", "#F6222E",
                  "#1CFFCE", "#2ED9FF", "#B10DA1", "#C075A6", "#FC1CBF", "#B00068",
                  "#FBE426", "#FA0087"
                ],
                'glasbey':[
                  "#0000FF", "#FF0000", "#00FF00", "#000033", "#FF00B6", "#005300",
                  "#FFD300", "#009FFF", "#9A4D42", "#00FFBE", "#783FC1", "#1F9698",
                  "#FFACFD", "#B1CC71", "#F1085C", "#FE8F42", "#DD00FF", "#201A01",
                  "#720055", "#766C95", "#02AD24", "#C8FF00", "#886C00", "#FFB79F",
                  "#858567", "#A10300", "#14F9FF", "#00479E", "#DC5E93", "#93D4FF",
                  "#004CFF", "#F2F318"
                ],
                'polychrome':[
                  "#5A5156", "#E4E1E3", "#F6222E", "#FE00FA", "#16FF32", "#3283FE",
                  "#FEAF16", "#B00068", "#1CFFCE", "#90AD1C", "#2ED9FF", "#DEA0FD",
                  "#AA0DFE", "#F8A19F", "#325A9B", "#C4451C", "#1C8356", "#85660D",
                  "#B10DA1", "#FBE426", "#1CBE4F", "#FA0087", "#FC1CBF", "#F7E1A0",
                  "#C075A6", "#782AB6", "#AAF400", "#BDCDFF", "#822E1C", "#B5EFB5",
                  "#7ED7D1", "#1C7F93", "#D85FF7", "#683B79", "#66B0FF", "#3B00FB"
                ],
                'stepped':[
                  "#990F26", "#B33E52", "#CC7A88", "#E6B8BF", "#99600F", "#B3823E",
                  "#CCAA7A", "#E6D2B8", "#54990F", "#78B33E", "#A3CC7A", "#CFE6B8",
                  "#0F8299", "#3E9FB3", "#7ABECC", "#B8DEE6", "#3D0F99", "#653EB3",
                  "#967ACC", "#C7B8E6", "#333333", "#666666", "#999999", "#CCCCCC"
                ]
            }
    if palette is None:
        if n<=26:
            palette="alphabet"
        elif n<=32:
            palette="glasbey"
        else:
            palette="polychrome"
    
    palette_array= palettes[palette]
    #print(len(palette_array))
    assert n<=len(palette_array), "Not enough colours in specified palette"

    return np.array(palette_array)[np.arange(n)]


# In[27]:


def exp_data_meta_transform(exp_data_meta):
    exp_data_meta_tSNE_center=exp_data_meta.groupby('phenoid').mean()[['tSNE_1','tSNE_2']]
    
    
    exp_data_meta_transformed=exp_data_meta.copy()
    for phenoid in exp_data_meta_transformed['phenoid'].unique():
        #min_dist=((exp_data_meta_tSNE_center-exp_data_meta_tSNE_center.loc[phenoid]).pow(2)).sum(axis=1).pow(0.5).nsmallest(2).iloc[-1]
        
        #print(exp_data_meta_tSNE_center)
        exp_data_meta_transformed.loc[exp_data_meta_transformed['phenoid']==phenoid,['tSNE_1','tSNE_2']]=        exp_data_meta_tSNE_center.loc[phenoid]+        (exp_data_meta_transformed.loc[exp_data_meta_transformed['phenoid']==phenoid,['tSNE_1','tSNE_2']]-exp_data_meta_tSNE_center.loc[phenoid])*        2/5
        
        
    return exp_data_meta_transformed

#exp_data_meta_transformed=exp_data_meta_transform(exp_data_meta)


# In[28]:


exp_data_meta_transformed=exp_data_meta.copy()
    
if 'HVGMarcoPolocoord' in dataset_name:
    newcoord=pd.read_csv('datasets/extract/{}.UMAP.HVGMarcoPolo.tsv'.format(dataset_name_input).replace('filtered','full'),sep='\t')
    
    assert np.all(exp_data_meta_transformed.index==newcoord.index)
    
    exp_data_meta_transformed[['tSNE_1','tSNE_2']]=newcoord[['UMAP_1','UMAP_2']]
    
if 'HVGMarcoPololabel' in dataset_name:
    newlabel=pd.read_csv('datasets/extract/{}.metadatacol.HVGMarcoPolo.tsv'.format(dataset_name_input).replace('filtered','full'),sep='\t')

    assert np.all(exp_data_meta_transformed.index==newlabel.index)
    exp_data_meta_transformed['phenoid']=newlabel['seurat_clusters']
    
if 'separate_coord' in dataset_name:
    exp_data_meta_transformed=exp_data_meta_transform(exp_data_meta_transformed)  


# In[29]:


if dataset_name=='Zhengmix8eq_filtered':
    newcoord=pd.read_csv('datasets/extract/Zhengmix8eq.tsne.MarcoPolodisp.2000.tsv',sep='\t')
    #newcoord['tSNE_2'][newcoord.index.str.contains('helper')]=newcoord['tSNE_2'][newcoord.index.str.contains('helper')]+40

    assert np.all(exp_data_meta_transformed.index==newcoord.index)
    exp_data_meta_transformed[['TSNE_1','TSNE_2']]=newcoord[['tSNE_1','tSNE_2']]   
    print('coord updated')


# In[30]:


report_path


# In[111]:


import seaborn as sns

plt.rcParams["figure.figsize"] = (16,16)
plt.rcParams["font.size"] = 10
plt.rcParams['font.family']='Arial'

fig = plt.figure(figsize=(10, 10))
gs=fig.add_gridspec(10,10)

ax=fig.add_subplot(gs[1:9,1:9])

#ax = fig.add_subplot(111)

plot_value=exp_data_meta_transformed['phenoid']
plot_value_unique=plot_value.unique().tolist()
plot_value_int=list(map(lambda x: plot_value_unique.index(x),plot_value))


#sns.scatterplot(x="tSNE_1", y="tSNE_2",hue=plot_value,style=np.array((list(range(0,2))*30))[plot_value_int],data=exp_data_meta,palette=plt.cm.rainbow if plot_value.dtype==int else None)#,)#,s=40,palette=plt.cm.rainbow)#,linewidth=0.3)    
scatterfig=sns.scatterplot(x="TSNE_1", y="TSNE_2",hue=plot_value,data=exp_data_meta_transformed,
                palette=DiscretePalette(len(plot_value_unique)).tolist() if plot_value.dtype==int else None,
               ax=ax,s=25,alpha=1,edgecolor='None'
               )#,)#,s=40,palette=plt.cm.rainbow)#,linewidth=0.3)    

#plt.legend('')
ax.get_legend().remove()
ax.set_ylabel(' ')
ax.set_xlabel(' ')

ax.set_xticks([])
ax.set_yticks([])
#ax.get_xaxis().set_ticks([])
#ax.get_xaxis().set_label('a')
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)
#ax.legend('')
#
plt.savefig('{}/plot_image/2D_Plot.png'.format(report_path),dpi=100,bbox_inches='tight')
plt.show()


# In[32]:


plt.rcParams["axes.prop_cycle"]


# In[33]:


DiscretePalette(len(plot_value_unique)).tolist()


# In[34]:


exp_data.shape,cell_size_factor.shape


# In[39]:


exp_data_corrected=(exp_data/cell_size_factor)


# In[123]:


#%matplotlib inline
import matplotlib.gridspec as gridspec
import seaborn as sns

#plt.rcParams["figure.figsize"] = (5*8,10)
plt.rcParams["font.size"] = 15
plt.rcParams['font.family']='Arial'
plt.ioff()

#for idx,(iter_idx,value) in enumerate(marker_criteria.iteritems()):
#for idx, (iter_idx,row) in enumerate(allscore[allscore['ismarker']==True].sort_values('all_rank').iterrows()):    
for count_idx, (iter_idx,row) in enumerate(allscore.sort_values('MarcoPolo',ascending=True).iloc[:].iterrows()):    
    
    if dataset_name!='Zhengmix8eq':
        phenoid_unique=exp_data_meta_transformed['phenoid'].unique()
    else:
        phenoid_unique=['b.cells',  'cd14.monocytes', 'cd56.nk',
                    'naive.cytotoxic','regulatory.t','cd4.t.helper', 'memory.t', 'naive.t']
    

    
    #subplot_size=(1,1+1+1+1+1)
    subplot_size=(1,1+1+1+1)
    
    
    #if idx==10*(subplot_size[0]*subplot_size[1]):
    #    break    
    

    #fig, axes = plt.subplots(*subplot_size)
    #fig = plt.figure(figsize=(3*10, 3*2)) 
    fig = plt.figure(figsize=(3*8, 6)) 
    gs=fig.add_gridspec(6,3*8)
    subplot_list=[fig.add_subplot(gs[0:6,0:6]),
                  fig.add_subplot(gs[0:6,6+2:6+2+6])]
    
    #gs = gridspec.GridSpec(subplot_size[0], subplot_size[1], width_ratios=[2, 1, 1, 1]) 
        
        
    #plt.subplots_adjust(wspace=0, hspace=0)
    
    #plt.text(.5,.95,'QQ:{:.1f}={}th Voting:{:d}={:d}th->{:d}'.format(row['QQratio'],int(row['QQratio_rank']),int(row['votingscore']),int(row['votingscore_rank']),int(row['all_rank']),idx),
    #                horizontalalignment='center',
    #                transform=ax.transAxes)
    

    exp_data_corrected_on=exp_data_corrected[iter_idx][gamma_argmax_list[iter_idx]==0]
    exp_data_corrected_off=exp_data_corrected[iter_idx][gamma_argmax_list[iter_idx]==1]
    
    bins_log=[np.power(1.2,i) for i in range(np.max([1,int(np.log(np.max([1,np.max(exp_data_corrected)]))/np.log(1.2))]))]
    bins_log_on=[np.power(1.1,i) for i in range(
        np.max([1,int(np.log(np.max([1,np.min(exp_data_corrected_on)]))/np.log(1.1))]),
        np.max([1,int(np.log(np.max([1,np.max(exp_data_corrected_on)]))/np.log(1.1))])
    
    )]
    bins_log_off=[np.power(1.2,i) for i in range(np.max([1,int(np.log(np.max([1,np.max(exp_data_corrected_off)]))/np.log(1.2))]))]
    
    
    
    
    
    for idx in range(2):
        #ax=plt.subplot(gs[idx%((subplot_size[0]*subplot_size[1]))])
        ax=subplot_list[idx]
        #ax=axes.flatten()[idx%((subplot_size[0]*subplot_size[1]))]
        #ax.set_axis_off()
        #ax.set_xticklabels([])
        #ax.set_yticklabels([])
        #ax.legend().remove()
        #ax.legend().set_visible(False)
        if idx==-1:
            plot_value=np.log10(1+exp_data_corrected[iter_idx])
            #plot_value=(plot_value-np.min(plot_value))/np.var(plot_value)
            #gamma_argmax_list[iter_idx]

            #ax.set_title(label=)
            #ax.title.set_text('QQ:{:.1f}={}th Voting:{:d}={:d}th->{:d}={:d}th'.format(row['QQscore'],int(row['QQscore_rank']),int(row['votingscore']),int(row['votingscore_rank']),int(row['all_rank']),idx))
            ax.title.set_text('tSNE')
            sns.scatterplot(x="TSNE_1", y="TSNE_2",label=None,legend=None,
                            hue=plot_value,data=exp_data_meta_transformed,ax=ax,s=15,linewidth=0.3,alpha=0.4,palette=plt.cm.Blues)#,palette=plt.cm.rainbow)#,linewidth=0.3)            
        if idx==0:

            #ax.set_title(label=)
            #ax.title.set_text('QQ:{:.1f}={}th Voting:{:d}={:d}th->{:d}={:d}th'.format(row['QQscore'],int(row['QQscore_rank']),int(row['votingscore']),int(row['votingscore_rank']),int(row['all_rank']),idx))
            
            plot_value=exp_data_meta_transformed['phenoid']
            plot_value_unique=plot_value.unique().tolist()
            plot_value_int=list(map(lambda x: plot_value_unique.index(x),plot_value))            
            
            s=10
            sns.scatterplot(x="TSNE_1", y="TSNE_2",hue=plot_value,data=exp_data_meta_transformed,
                            palette=DiscretePalette(len(plot_value_unique)).tolist() if plot_value.dtype==int else None,
                           ax=ax,alpha=0.3,edgecolor="None",
                            s=s
                           )
            sns.scatterplot(x="TSNE_1", y="TSNE_2",data=exp_data_meta_transformed.loc[gamma_argmax_list[iter_idx]==0],ax=ax,
                            edgecolor=[1,0,0,1],
                            facecolors="None",
                            linewidths=10,
                            s=s
                           )
            
            """
            plot_value=gamma_argmax_list[iter_idx]
            sns.scatterplot(x="tSNE_1", y="tSNE_2", label=None,legend=None,hue=plot_value,
                            data=data,ax=ax,s=15,linewidth=0.3,alpha=0.4)#,palette=plt.cm.rainbow)#,linewidth=0.3)            
            """            
            
            ax.title.set_text('On-Off in 2D plot')
            ax.get_legend().remove()
            
            ax.set_xlabel('Dim 1')            
            ax.set_ylabel('Dim 2',labelpad=-10)
            
            for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(1.5)              
            
        elif idx==1:
            ax.title.set_text('Expression of Cells')
            #bins_count,bins,patch=ax.hist(exp_data[iter_idx],bins=bins_log,color='black')
            ax.hist(exp_data_corrected_on,bins=bins_log,label='On',color=(1,0,0,0.8))
            ax.hist(exp_data_corrected_off,bins=bins_log,label='Off',color=(0.5,0.5,0.5,0.5))
            ax.set_xscale('log')
                        
            ax.set_ylabel('Cell Counts')                        
            ax.set_xlabel('Expression count (size factor corrected)')
            
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False) 
            
            leg=ax.legend(loc='upper left',
                                fontsize=15,
                                frameon=False,
                                bbox_to_anchor=(0.22, -0.15),
                                ncol=2,
                              handletextpad=0.2,
                                  columnspacing=1.3,
                                markerscale=2.5)  
            [rec.set_height(8) for rec in leg.get_patches()]
            [rec.set_width(15) for rec in leg.get_patches()]
            for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(2)             
            
            #sns.distplot(a=exp_data[iter_idx],kde=False,color='black',ax=ax)
        elif idx==2:
            ax.title.set_text('Expression of On Cells')
            ax.hist(exp_data_corrected_on,bins=bins_log_on,color=(1,0,0,0.8))
            #bins_count,bins,patch=ax.hist(exp_data_on,bins=bins_log,color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0])
            ax.set_xscale('log')
            
            ax.set_ylabel('Cell Counts')                        
            ax.set_xlabel('Expression count (size factor corrected)')
            
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)              
            
            #sns.distplot(a=exp_data[iter_idx][gamma_argmax_list[iter_idx]==0],kde=False,color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0],ax=ax)
        elif idx==3:
            ax.title.set_text('Expression of Off Cells')
            ax.hist(exp_data_corrected_off,bins=bins_log_off,color=(0.5,0.5,0.5,0.5))
            #bins_count,bins,patch=ax.hist(exp_data_off,bins=bins_log,color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1])
            ax.set_xscale('log')
            
            ax.set_ylabel('Cell Frequency')                        
            ax.set_xlabel('Expression count (size factor corrected)')
            
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)              
            
            #sns.distplot(a=exp_data[iter_idx][gamma_argmax_list[iter_idx]!=0],kde=False,color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1],ax=ax)
            #ax.axvline(x=1,linewidth=10)
            #plt.Line2D([0.5,0.5],[1,1], transform=fig.transFigure, color="black")
            #ax.axvspan(0.8, 0.9, transform=fig.transFigure,clip_on=False) 
    
    plt.savefig('{}/plot_image/{}.png'.format(report_path,iter_idx),dpi=60,bbox_inches='tight')
    #plt.show()
    plt.close(fig)
    
    if count_idx==1200:
        break


# In[121]:


report_path


# In[57]:


allscore.sort_values('MarcoPolo',ascending=True).iloc[:]


# In[60]:


allscore_munge.sort_values('MarcoPolo',ascending=True).iloc[:]


# In[62]:


[gamma_argmax_list[1]==0]


# In[44]:


exp_data_off


# In[ ]:





# In[ ]:


dsdsd


# # Old

# In[29]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.gridspec as gridspec
import seaborn as sns

#plt.rcParams["figure.figsize"] = (5*8,10)
plt.rcParams["font.size"] = 15
plt.rcParams['font.family']='Arial'
plt.ioff()

#for idx,(iter_idx,value) in enumerate(marker_criteria.iteritems()):
#for idx, (iter_idx,row) in enumerate(allscore[allscore['ismarker']==True].sort_values('all_rank').iterrows()):    
for count_idx, (iter_idx,row) in enumerate(allscore.sort_values('MarcoPolo',ascending=True).iterrows()):    
    
    if dataset_name!='Zhengmix8eq':
        phenoid_unique=exp_data_meta_transformed['phenoid'].unique()
    else:
        phenoid_unique=['b.cells',  'cd14.monocytes', 'cd56.nk',
                    'naive.cytotoxic','regulatory.t','cd4.t.helper', 'memory.t', 'naive.t']
    

    
    #subplot_size=(1,1+1+1+1+1)
    subplot_size=(1,1+1+1+1)
    
    
    #if idx==10*(subplot_size[0]*subplot_size[1]):
    #    break    
    

    #fig, axes = plt.subplots(*subplot_size)
    #fig = plt.figure(figsize=(3*10, 3*2)) 
    fig = plt.figure(figsize=(3*8, 3*2)) 
    gs = gridspec.GridSpec(subplot_size[0], subplot_size[1], width_ratios=[2, 1, 1, 1]) 
        
        
    #plt.subplots_adjust(wspace=0, hspace=0)
    
    #plt.text(.5,.95,'QQ:{:.1f}={}th Voting:{:d}={:d}th->{:d}'.format(row['QQratio'],int(row['QQratio_rank']),int(row['votingscore']),int(row['votingscore_rank']),int(row['all_rank']),idx),
    #                horizontalalignment='center',
    #                transform=ax.transAxes)
    
    
    
    
    for idx in range(subplot_size[1]):
        ax=plt.subplot(gs[idx%((subplot_size[0]*subplot_size[1]))])
        #ax=axes.flatten()[idx%((subplot_size[0]*subplot_size[1]))]
        #ax.set_axis_off()
        #ax.set_xticklabels([])
        #ax.set_yticklabels([])
        #ax.legend().remove()
        #ax.legend().set_visible(False)
        if idx==-1:
            plot_value=np.log10(1+exp_data[iter_idx])
            #plot_value=(plot_value-np.min(plot_value))/np.var(plot_value)
            #gamma_argmax_list[iter_idx]

            #ax.set_title(label=)
            #ax.title.set_text('QQ:{:.1f}={}th Voting:{:d}={:d}th->{:d}={:d}th'.format(row['QQscore'],int(row['QQscore_rank']),int(row['votingscore']),int(row['votingscore_rank']),int(row['all_rank']),idx))
            ax.title.set_text('tSNE')
            sns.scatterplot(x="TSNE_1", y="TSNE_2",label=None,legend=None,
                            hue=plot_value,data=exp_data_meta_transformed,ax=ax,s=15,linewidth=0.3,alpha=0.4,palette=plt.cm.Blues)#,palette=plt.cm.rainbow)#,linewidth=0.3)            
        if idx==0:

            #ax.set_title(label=)
            #ax.title.set_text('QQ:{:.1f}={}th Voting:{:d}={:d}th->{:d}={:d}th'.format(row['QQscore'],int(row['QQscore_rank']),int(row['votingscore']),int(row['votingscore_rank']),int(row['all_rank']),idx))
            
            plot_value=exp_data_meta_transformed['phenoid']
            plot_value_unique=plot_value.unique().tolist()
            plot_value_int=list(map(lambda x: plot_value_unique.index(x),plot_value))            
            
            s=10
            sns.scatterplot(x="TSNE_1", y="TSNE_2",hue=plot_value,data=exp_data_meta_transformed,
                            palette=DiscretePalette(len(plot_value_unique)).tolist() if plot_value.dtype==int else None,
                           ax=ax,alpha=0.3,edgecolor="None",
                            s=s
                           )
            sns.scatterplot(x="TSNE_1", y="TSNE_2",data=exp_data_meta_transformed.loc[gamma_argmax_list[iter_idx]==0],ax=ax,
                            edgecolor=[0,0,1,1],
                            facecolors="None",
                            linewidths=10,
                            s=s
                           )
            
            """
            plot_value=gamma_argmax_list[iter_idx]
            sns.scatterplot(x="tSNE_1", y="tSNE_2", label=None,legend=None,hue=plot_value,
                            data=data,ax=ax,s=15,linewidth=0.3,alpha=0.4)#,palette=plt.cm.rainbow)#,linewidth=0.3)            
            """            
            
            ax.title.set_text('On-Off in 2D Plot')
            ax.get_legend().remove()
            ax.set_ylabel('')
            ax.set_xlabel('')            
            
        elif idx==1:
            ax.title.set_text('Distribution of all cells')
            bins_count,bins,patch=ax.hist(exp_data[iter_idx],color='black')
            #sns.distplot(a=exp_data[iter_idx],kde=False,color='black',ax=ax)
        elif idx==2:
            ax.title.set_text('Distribution of "On" cells')
            exp_data_on=exp_data[iter_idx][gamma_argmax_list[iter_idx]==0]
            ax.hist(exp_data_on,bins=bins,color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0])
            #sns.distplot(a=exp_data[iter_idx][gamma_argmax_list[iter_idx]==0],kde=False,color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0],ax=ax)
        elif idx==3:
            ax.title.set_text('Distribution of "Off" cells')
            exp_data_off=exp_data[iter_idx][gamma_argmax_list[iter_idx]!=0]
            ax.hist(exp_data_off,bins=bins,color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1])
            #sns.distplot(a=exp_data[iter_idx][gamma_argmax_list[iter_idx]!=0],kde=False,color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1],ax=ax)
            #ax.axvline(x=1,linewidth=10)
            #plt.Line2D([0.5,0.5],[1,1], transform=fig.transFigure, color="black")
            #ax.axvspan(0.8, 0.9, transform=fig.transFigure,clip_on=False)          
    plt.savefig('{}/plot_image/{}.png'.format(report_path,iter_idx),dpi=100,bbox_inches='tight')
    #plt.show()
    plt.close(fig)
    #if count_idx==1:
    #    break    
    if count_idx==1200:
        break


# In[ ]:





# In[ ]:





# In[ ]:





# In[48]:


plt.hist(exp_data[2806])
iter_idx=2806

exp_data_off=exp_data[iter_idx][gamma_argmax_list[iter_idx]==1]
bins_log_off=[0]+[np.power(1.2,i) for i in range(int(np.log(np.max([1,np.max(exp_data_off)]))/np.log(1.2)))]


# In[49]:


bins_log_off


# In[44]:


allscore.sort_values('MarcoPolo',ascending=True)


# In[38]:


(int(np.log(np.max(1,np.max(exp_data))
           )/np.log(1.2)))


# In[40]:


np.max([1,np.max(exp_data)])


# In[ ]:




