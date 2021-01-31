#!/usr/bin/env python
# coding: utf-8

import pickle
import sys
import os
    
import numpy as np
import pandas as pd    

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from scipy.io import mmread
from sklearn.decomposition import PCA

import MarcoPolo.QQscore as QQ

import matplotlib.pyplot as plt

import shutil
from jinja2 import Template



def generate_report(input_path,output_path,output_mode='pub',gene_info_path='https://ftp.ncbi.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz',mode=2):
    path=input_path
    report_path=output_path
    
    
    print("------Loading dataset------")
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
    
    
    result_list,gamma_list_list,delta_log_list_list,beta_list_list=QQ.read_QQscore(path,[1,mode])
    
    gamma_list=gamma_list_list[-1]    

    gamma_argmax_list=QQ.gamma_list_exp_data_to_gamma_argmax_list(gamma_list,exp_data)#gamma_argmax_list=QQ.gamma_list_to_gamma_argmax_list(gamma_list)
    gamma_argmax_list,gamma_argmax_list.shape    
    
    allscore=pd.read_csv('{}.MarcoPolo.{}.rank.tsv'.format(path,mode),index_col=0,sep='\t')
    
    gene_info=pd.read_csv(gene_info_path,sep='\t')
    
    
    print("------Annotating genes------")

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
    
    
    allscore_munge=allscore.copy()
    allscore_munge['Gene ID']=exp_data_row
    assert len(gene_info_extract)==len(allscore_munge)
    allscore_munge=allscore_munge.merge(right=gene_info_extract,left_index=True,right_index=True)   
    #allscore_munge.to_csv('{}.MarcoPolo.{}.rank.munge.tsv'.format(path,mode),sep='\t')
    
    allscore_munge['img']=allscore_munge.apply(lambda x: '<img src="plot_image/{idx}.png" alt="{idx}">'.format(dataset_name=dataset_name,idx=x.name),axis=1)
    
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

    """
    try:
        os.mkdir('{}'.format(report_path),exist_ok=True)
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
    """

    #shutil.copy('report/template/index.html', 'report/{}/index.html'.format(dataset_name_path))
    
    os.makedirs('{}'.format(report_path),exist_ok=True)
    os.makedirs('{}/plot_image'.format(report_path),exist_ok=True)
    os.makedirs('{}/assets'.format(report_path),exist_ok=True)
    
    shutil.copy(os.path.join(os.path.dirname(__file__),'template/assets/scripts.js'), '{}/assets/scripts.js'.format(report_path))
    shutil.copy(os.path.join(os.path.dirname(__file__),'template/assets/styles.css'), '{}/assets/styles.css'.format(report_path))
    shutil.copy(os.path.join(os.path.dirname(__file__),'template/assets/details_open.png'), '{}/assets/details_open.png'.format(report_path))
    shutil.copy(os.path.join(os.path.dirname(__file__),'template/assets/details_close.png'), '{}/assets/details_close.png'.format(report_path))
    shutil.copy(os.path.join(os.path.dirname(__file__),'template/assets/mp.png'), '{}/assets/mp.png'.format(report_path))
    shutil.copy(os.path.join(os.path.dirname(__file__),'template/assets/mp_white.png'), '{}/assets/mp_white.png'.format(report_path))
    shutil.copy(os.path.join(os.path.dirname(__file__),'template/assets/mp_white_large_font.png'), '{}/assets/mp_white_large_font.png'.format(report_path))    
    
    
    

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
        
    
    print("------Drawing figures------")
    




