import os
import time

import pandas as pd

print(os.getcwd());os.chdir('/data01/ch6845/MarcoPolo/')
print(os.getcwd())


import MarcoPolo.QQscore as QQ

import MarcoPolo.summarizer as summarizer

dataset_original_all=[
'Kohinbulk_filtered',
'HumanLiver_filtered',
'Zhengmix8eq_filtered'
]

dataset_tabula_all=["TabulaAorta_filtered", #ok
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

dataset_simul_all=[]
for ncells_total in [1000,2000,5000,10000]:
    for prop in ['1e-2','5e-3','1e-3','5e-4']:
        for i in range(1,10+1):
            dataset_simul_all.append('Simul_{}_{}_{}_filtered'.format(ncells_total,prop,i))
            
# conda activate MarcoPolo; for i in {0..160};do python marco.py $i 0;done            
dataset_name_all=dataset_simul_all

import sys

idx= int(sys.argv[1])
gpu_idx= int(sys.argv[2])
print(idx,gpu_idx)

if idx%6==gpu_idx:
    dataset_name=dataset_name_all[idx]

    print(dataset_name)
    path='datasets/extract/{}'.format(dataset_name)
    start_time=time.time()
    QQscore_result=QQ.save_QQscore(path=path,device='cuda:{}'.format(idx%6),num_cluster_list=[1,2],num_thread=6)
    end_time=time.time()
    print('\n',end_time-start_time)
    pd.DataFrame([end_time-start_time]).to_csv('{}.QQscore.runtime.tsv'.format(path),index=None,header=None)
