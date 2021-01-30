#!/usr/bin/env python
# coding: utf-8

import pickle

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import numpy as np
import pandas as pd
from scipy.io import mmread

import MarcoPolo.QQscore as QQ



def get_MarcoPolo(path,mode=2,voting_thres=0.7):
    # read scRNA data
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
    
    #read QQ
    result_list,gamma_list_list=QQ.read_QQscore(path,[1,mode])
    gamma_list=gamma_list_list[-1]
    
    gamma_argmax_list=QQ.gamma_list_exp_data_to_gamma_argmax_list(gamma_list,exp_data)
    
    
    # mean_0_all    
    mean_all=np.array([np.mean(exp_data[i,:]) for i in range(gamma_argmax_list.shape[0])])
    mean_0=np.array([np.mean(exp_data[i,gamma_argmax_list[i]==0]) for i in range(gamma_argmax_list.shape[0])])
    mean_other=np.array([np.mean(exp_data[i,gamma_argmax_list[i]!=0]) for i in range(gamma_argmax_list.shape[0])])
    mean_1=np.array([np.mean(exp_data[i,gamma_argmax_list[i]==1]) for i in range(gamma_argmax_list.shape[0])])    
    mean_0_all=mean_0-mean_all    
    
    # QQratio
    QQratio=result_list[0]['Q']/result_list[-1]['Q']
    
    # QQdiff
    QQdiff=(result_list[0]['Q'].values-result_list[-1]['Q'].values)/(+mean_all-np.log(mean_all))
    
    # QQdiffraw
    QQdiffraw=(result_list[0]['Q'].values-result_list[-1]['Q'].values)
    
    # voting score
    minorsize_list=np.sum(gamma_argmax_list==0,axis=1)
    minorsize_cliplist=QQ.gamma_argmax_list_to_minorsize_list_list(gamma_argmax_list)
    intersection_list=QQ.gamma_argmax_list_to_intersection_list(gamma_argmax_list)
    intersectioncount_thresholdcount=np.sum((intersection_list/minorsize_cliplist)>voting_thres,axis=1)
    

    
    
    ##########
    
    # lfc
    lfc=np.log10(np.array([np.mean(exp_data[i,gamma_argmax_list[i]==0],axis=0) for i in range(gamma_argmax_list.shape[0])])/                np.array([np.mean(exp_data[i,gamma_argmax_list[i]!=0],axis=0) for i in range(gamma_argmax_list.shape[0])]))      
    
    # PC Variance    
    exp_data_norm=np.log1p(10000*exp_data/exp_data.sum(axis=0))
    exp_data_norm_scale=(exp_data_norm-exp_data_norm.mean(axis=1).reshape(-1,1))/    exp_data_norm.std(axis=1).reshape(-1,1)
    exp_data_norm_scale[exp_data_norm_scale>10]=10


    from sklearn.decomposition import PCA
    pca = PCA(n_components=50)
    pca.fit(exp_data_norm_scale.T)
    exp_data_norm_scale_pc=pca.transform(exp_data_norm_scale.T)
    exp_data_norm_scale_pc.shape

    safe_var=lambda x: np.var(x[ (x<np.quantile(x,0.98)) & (x>np.quantile(x,0.02))])
    safe_var_axis=lambda x: np.apply_along_axis(func1d=safe_var, axis=0, arr=x)  
    
    #safe_std=lambda x: np.std(x[ (x<np.quantile(x,0.98)) & (x>np.quantile(x,0.02))])
    safe_std=lambda x: np.std(x)
    safe_std_axis=lambda x: np.apply_along_axis(func1d=safe_std, axis=0, arr=x)      
    
    """

    
    #PCvariance=np.array([np.var(exp_data_pc[gamma_argmax_list[i]==0,:5],axis=0).mean() for i in range(gamma_argmax_list.shape[0])])
    PCvariance=np.array([safe_var_axis(exp_data_pc[gamma_argmax_list[i]==0,:5]).mean() for i in range(gamma_argmax_list.shape[0])])       
    """

    #from sklearn.manifold import TSNE
    #exp_data_norm_scale_pc_tsne = TSNE(n_components=2).fit_transform(exp_data_norm_scale_pc[:,:5])
    #exp_data_norm_scale_pc_tsne.shape             
    exp_data_norm_scale_pc_top2varmean=np.array([exp_data_norm_scale_pc[gamma_argmax_list[i]==0,:2].var(axis=0).mean() for i in range(gamma_argmax_list.shape[0])])
    exp_data_norm_scale_pc_top3varmean=np.array([exp_data_norm_scale_pc[gamma_argmax_list[i]==0,:3].var(axis=0).mean() for i in range(gamma_argmax_list.shape[0])])
    exp_data_norm_scale_pc_top5varmean=np.array([exp_data_norm_scale_pc[gamma_argmax_list[i]==0,:5].var(axis=0).mean() for i in range(gamma_argmax_list.shape[0])])
    exp_data_norm_scale_pc_top10varmean=np.array([exp_data_norm_scale_pc[gamma_argmax_list[i]==0,:10].var(axis=0).mean() for i in range(gamma_argmax_list.shape[0])])
    
    
    exp_data_norm_scale_pc_top2varmean_safe=np.array([safe_var_axis(exp_data_norm_scale_pc[gamma_argmax_list[i]==0,:2]).mean() for i in range(gamma_argmax_list.shape[0])])
    exp_data_norm_scale_pc_top3varmean_safe=np.array([safe_var_axis(exp_data_norm_scale_pc[gamma_argmax_list[i]==0,:3]).mean() for i in range(gamma_argmax_list.shape[0])])
    exp_data_norm_scale_pc_top5varmean_safe=np.array([safe_var_axis(exp_data_norm_scale_pc[gamma_argmax_list[i]==0,:5]).mean() for i in range(gamma_argmax_list.shape[0])])
    exp_data_norm_scale_pc_top10varmean_safe=np.array([safe_var_axis(exp_data_norm_scale_pc[gamma_argmax_list[i]==0,:10]).mean() for i in range(gamma_argmax_list.shape[0])])

        
    exp_data_norm_scale_pc_top2stdmean=np.array([exp_data_norm_scale_pc[gamma_argmax_list[i]==0,:2].std(axis=0).mean() for i in range(gamma_argmax_list.shape[0])])
    exp_data_norm_scale_pc_top3stdmean=np.array([exp_data_norm_scale_pc[gamma_argmax_list[i]==0,:3].std(axis=0).mean() for i in range(gamma_argmax_list.shape[0])])
    exp_data_norm_scale_pc_top5stdmean=np.array([exp_data_norm_scale_pc[gamma_argmax_list[i]==0,:5].std(axis=0).mean() for i in range(gamma_argmax_list.shape[0])])
    exp_data_norm_scale_pc_top10stdmean=np.array([exp_data_norm_scale_pc[gamma_argmax_list[i]==0,:10].std(axis=0).mean() for i in range(gamma_argmax_list.shape[0])])        
    
    exp_data_norm_scale_pc_top2stdmean_safe=np.array([safe_std_axis(exp_data_norm_scale_pc[gamma_argmax_list[i]==0,:2]).mean() for i in range(gamma_argmax_list.shape[0])])
    exp_data_norm_scale_pc_top3stdmean_safe=np.array([safe_std_axis(exp_data_norm_scale_pc[gamma_argmax_list[i]==0,:3]).mean() for i in range(gamma_argmax_list.shape[0])])
    exp_data_norm_scale_pc_top5stdmean_safe=np.array([safe_std_axis(exp_data_norm_scale_pc[gamma_argmax_list[i]==0,:5]).mean() for i in range(gamma_argmax_list.shape[0])])
    exp_data_norm_scale_pc_top10stdmean_safe=np.array([safe_std_axis(exp_data_norm_scale_pc[gamma_argmax_list[i]==0,:10]).mean() for i in range(gamma_argmax_list.shape[0])])    
        
        
    try:
        markerrho=pd.read_csv('{}.markerrho.tsv'.format(path),index_col=0,sep='\t')
    except:
        print('markerrho does not exist')
        markerrho=pd.DataFrame([])  
        #except NameError:
    try:
        maxdiff=pd.read_csv('{}.maxdiff.tsv'.format(path),index_col=0,header=None,sep='\t')[1]
    except:
        print('maxdiff does not exist')
        maxdiff=np.zeros_like(QQratio.values)
        assert maxdiff.shape[0]==len(exp_data_row)        
  
    

    allscore=pd.DataFrame([QQratio.values,
                           QQdiff,
                           intersectioncount_thresholdcount,
                           exp_data_norm_scale_pc_top2varmean,exp_data_norm_scale_pc_top3varmean,exp_data_norm_scale_pc_top5varmean,exp_data_norm_scale_pc_top10varmean,
                           exp_data_norm_scale_pc_top2varmean_safe,exp_data_norm_scale_pc_top3varmean_safe,exp_data_norm_scale_pc_top5varmean_safe,exp_data_norm_scale_pc_top10varmean_safe,
                           exp_data_norm_scale_pc_top2stdmean,exp_data_norm_scale_pc_top3stdmean,exp_data_norm_scale_pc_top5stdmean,exp_data_norm_scale_pc_top10stdmean,
                           exp_data_norm_scale_pc_top2stdmean_safe,exp_data_norm_scale_pc_top3stdmean_safe,exp_data_norm_scale_pc_top5stdmean_safe,exp_data_norm_scale_pc_top10stdmean_safe,                           
                           lfc,mean_0_all,minorsize_list,
                           QQdiffraw,mean_all,
                           maxdiff,
                           list(map(lambda x: x in markerrho.columns,exp_data_row))
                          ],
                          index=['QQratio','QQdiff','votingscore',
                                 'PCvariance2old','PCvariance3old','PCvariance5old','PCvariance10old',
                                 'PCvariance2','PCvariance3','PCvariance5','PCvariance10',
                                 'PCstd2old','PCstd3old','PCstd5old','PCstd10old',
                                 'PCstd2','PCstd3','PCstd5','PCstd10',
                                 'lfc','mean_0_all','minorsize',
                                 'QQdiffraw','mean_all',
                                 'maxdiff','ismarker']).T

    allscore['QQratio_rank']=pd.Series(np.arange(allscore.shape[0]),index=allscore['QQratio'].sort_values(ascending=False).index).loc[allscore.index]
    allscore['QQdiff_rank']=pd.Series(np.arange(allscore.shape[0]),index=allscore['QQdiff'].sort_values(ascending=False).index).loc[allscore.index]
    allscore['votingscore_rank']=pd.Series(np.arange(allscore.shape[0]),index=allscore['votingscore'].sort_values(ascending=False).index).loc[allscore.index]

    allscore['PCvariance2old_rank']=pd.Series(np.arange(allscore.shape[0]),index=allscore['PCvariance2old'].sort_values(ascending=True).index).loc[allscore.index]
    allscore['PCvariance3old_rank']=pd.Series(np.arange(allscore.shape[0]),index=allscore['PCvariance3old'].sort_values(ascending=True).index).loc[allscore.index]
    allscore['PCvariance5old_rank']=pd.Series(np.arange(allscore.shape[0]),index=allscore['PCvariance5old'].sort_values(ascending=True).index).loc[allscore.index]
    allscore['PCvariance10old_rank']=pd.Series(np.arange(allscore.shape[0]),index=allscore['PCvariance10old'].sort_values(ascending=True).index).loc[allscore.index]    
    
    allscore['PCvariance2_rank']=pd.Series(np.arange(allscore.shape[0]),index=allscore['PCvariance2'].sort_values(ascending=True).index).loc[allscore.index]
    allscore['PCvariance3_rank']=pd.Series(np.arange(allscore.shape[0]),index=allscore['PCvariance3'].sort_values(ascending=True).index).loc[allscore.index]
    allscore['PCvariance5_rank']=pd.Series(np.arange(allscore.shape[0]),index=allscore['PCvariance5'].sort_values(ascending=True).index).loc[allscore.index]
    allscore['PCvariance10_rank']=pd.Series(np.arange(allscore.shape[0]),index=allscore['PCvariance10'].sort_values(ascending=True).index).loc[allscore.index]

    allscore['PCstd2old_rank']=pd.Series(np.arange(allscore.shape[0]),index=allscore['PCstd2old'].sort_values(ascending=True).index).loc[allscore.index]
    allscore['PCstd3old_rank']=pd.Series(np.arange(allscore.shape[0]),index=allscore['PCstd3old'].sort_values(ascending=True).index).loc[allscore.index]
    allscore['PCstd5old_rank']=pd.Series(np.arange(allscore.shape[0]),index=allscore['PCstd5old'].sort_values(ascending=True).index).loc[allscore.index]
    allscore['PCstd10old_rank']=pd.Series(np.arange(allscore.shape[0]),index=allscore['PCstd10old'].sort_values(ascending=True).index).loc[allscore.index]        
    
    allscore['PCstd2_rank']=pd.Series(np.arange(allscore.shape[0]),index=allscore['PCstd2'].sort_values(ascending=True).index).loc[allscore.index]
    allscore['PCstd3_rank']=pd.Series(np.arange(allscore.shape[0]),index=allscore['PCstd3'].sort_values(ascending=True).index).loc[allscore.index]
    allscore['PCstd5_rank']=pd.Series(np.arange(allscore.shape[0]),index=allscore['PCstd5'].sort_values(ascending=True).index).loc[allscore.index]
    allscore['PCstd10_rank']=pd.Series(np.arange(allscore.shape[0]),index=allscore['PCstd10'].sort_values(ascending=True).index).loc[allscore.index]    
    
    
    allscore['lfc_rank']=pd.Series(np.arange(allscore.shape[0]),index=allscore['lfc'].sort_values(ascending=False).index).loc[allscore.index]
    allscore['mean_0_all_rank']=pd.Series(np.arange(allscore.shape[0]),index=allscore['mean_0_all'].sort_values(ascending=False).index).loc[allscore.index]
    allscore['minorsize_rank']=pd.Series(np.arange(allscore.shape[0]),index=allscore['minorsize'].sort_values(ascending=False).index).loc[allscore.index]

    allscore['votingscore_rank'][allscore['votingscore']==0]=499999
    allscore['votingscore_rank'][allscore['votingscore']==1]=999999

    allscore['all_rank']=(1)*allscore['QQdiff_rank']+(2)*allscore['votingscore_rank']
    allscore['all_rank_rank']=pd.Series(np.arange(allscore.shape[0]),index=(-allscore['all_rank']).sort_values(ascending=False).index).loc[allscore.index]    

    
    allscore['votingscore_rank']=allscore[['QQratio_rank','mean_0_all_rank']].min(axis=1)
    allscore['votingscore_rank'][~(
            (allscore['lfc']>0.6)&
            (allscore['minorsize']>int(10))&
            (allscore['minorsize']<int(70/100*len(exp_data_col)))
            )]=len(allscore)    
    
    allscore['bimodalityscore_rank']=allscore[['QQratio_rank','mean_0_all_rank']].min(axis=1)
    allscore['bimodalityscore_rank'][~(
            (allscore['lfc']>0.6)&
            (allscore['minorsize']>int(10))&
            (allscore['minorsize']<int(70/100*len(exp_data_col)))
            )]=len(allscore)    
    
    
    allscore['proximityscore_rank']=allscore['PCstd2_rank'].copy()
    allscore['proximityscore_rank'][~(
            (allscore['lfc']>0.6)&
            (allscore['minorsize']>int(10))&
            (allscore['minorsize']<int(70/100*len(exp_data_col)))
            )]=len(allscore)       
    
    
    MarcoPolo_score=allscore[['votingscore_rank','proximityscore_rank','bimodalityscore_rank']].min(axis=1)
    
    
    allscore['MarcoPolo']=MarcoPolo_score
    allscore['MarcoPolo_rank']=pd.Series(np.arange(allscore.shape[0]),index=allscore['MarcoPolo'].sort_values(ascending=True).index).loc[allscore.index]
    
    
    return allscore
    




