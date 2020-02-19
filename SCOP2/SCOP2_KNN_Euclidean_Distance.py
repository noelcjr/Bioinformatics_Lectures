# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 20:25:03 2016

@author: noel
"""
import pandas as pd
import numpy as np
from collections import Counter

#file_path = '/home/noel/Projects/Job_Interviews/2ndGenome/scop2_processed_data.csv'
#df_pdb = pd.read_csv(file_path)

# I will get rid of small proteins because they do not have secondary structure info
###############################################################################
file_path = '/home/noel/Projects/Bioinformatics/SCOP2/files/'
SCOP2_sf_represeq = pd.read_csv(file_path+"scop_sf_represeq_lib20200117_fa.csv")
SCOP2_fa_represeq = pd.read_csv(file_path+"scop_fa_represeq_lib20200117_fa.csv")

SCOP2_fa_represeq = SCOP2_fa_represeq.set_index("FA-DOMID")
SCOP2_sf_represeq = SCOP2_sf_represeq.set_index("FA-DOMID")

SCOP2_cla = pd.read_csv(file_path+"scop_cla.csv", index_col="FA-DOMID")
SCOP2_cla.head()
SCOP2_des = pd.read_csv(file_path+"scop_des.csv")
SCOP2_des.head()

len(SCOP2_cla.CL[SCOP2_cla.CL == 1000000])            #  'alpha'           #4351  # 156
len(SCOP2_cla.CL[SCOP2_cla.CL == 1000001])            #  'beta'            #4509  # 52 
len(SCOP2_cla.CL[SCOP2_cla.CL == 1000002])            # 'alpha/beta'       #5012  # 224
len(SCOP2_cla.CL[SCOP2_cla.CL == 1000003])            # 'alpha+beta'       #5815  # 236         
len(SCOP2_cla.CL[SCOP2_cla.CL == 1000004])            #  'beta'            #1312  # 7
SCOP2_cla.shape
####################   KNN Euclidian Distance #################################
def euclidean_distance(np1, np2):
    return np.linalg.norm(np1-np2)
#euclidean_distance(np.array([1,2,3,4]),np.array([1,3,4,8]))
def predict(current_df,unknown, k = 3):
    '''
    Input:
        unknown  == four attributes of an unknown flower
        k        == the number of neighbors used
    Output:
        A prediction of the species of flower (str)
    '''
    distances = [(euclidean_distance(unknown, row[:-1]),row[-1]) for row in current_df]
    nearest = sorted(distances)[:k]

    return Counter([n[1] for n in nearest]).most_common(1)[0][0]
###############################################################################
prot_struct_aa_type = ['hydrophobic', 'charged', 'polar', 'other']
aa_count_df = SCOP2_fa_represeq.loc[SCOP2_cla.index,prot_struct_aa_type]
aa_count_df.head()
aa_count_df.insert(4,"CL",list(SCOP2_cla.CL),True)
aa_count_df.head()
aa_count_df = aa_count_df.reset_index()
del aa_count_df['FA-DOMID']
aa_count_df = aa_count_df.drop(aa_count_df.other[aa_count_df.other > 0].index)
len(aa_count_df.other[aa_count_df.other > 0])
aa_count_df = aa_count_df.reset_index()
del aa_count_df['other']
del aa_count_df['index']

np.random.seed(10)
for i in range(10):
    print(np.random.choice(range(10), 10, replace=False))
    
total_cv_results = []
for k in [20]: #1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]:
    print("k size: ",k)
    results = []
    np.random.seed(10)
    for i in range(5):
        print(i," cross correlation")
        l = np.random.choice(range(20966), 20966, replace=False)
        current_df = aa_count_df.values[l]
        predictions = np.array([predict(current_df[0:20000],row[:-1],k) for row in current_df[20000:]])
        actual = np.array([row[-1] for row in current_df[20000:]])
        results.append(np.mean(predictions == actual))
    total_cv_results.append((k,sum(results)/len(results)))
    print(k,sum(results)/len(results))

# k size:   1 0.4229813664596273
# k size:   2 0.4229813664596273
# k size:   3 0.4341614906832298
# k size:   4 0.4492753623188405
# k size:   5 0.4641821946169772
# k size:   6 0.4623188405797102
# k size:   7 0.4623188405797102
# k size:   8 0.4714285714285714
# k size:   9 0.4732919254658385
# k size:  10 0.4712215320910973
# k size:  11 0.47184265010351967
# k size:  12 0.4708074534161491
# k size:  13 0.47474120082815735
# k size:  14 0.47722567287784684
# k size:  15 0.4809523809523809
# k size:  16 0.4795031055900621
# k size:  17 0.479296066252588
# k size:  18 0.47536231884057967
# k size:  19 0.47763975155279503
# k size:  20
prot_struct_aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 
                  'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
aa_count_df = SCOP2_fa_represeq.loc[SCOP2_cla.index,prot_struct_aa]
aa_count_df.head()
aa_count_df.insert(20,"CL",list(SCOP2_cla.CL),True)
aa_count_df.head()
aa_count_df = aa_count_df.reset_index()
del aa_count_df['FA-DOMID']
# Warning running the loop below willl take a few hours.

total_cv_results = []
for k in [1,2,3,4,5,6,7,8,9,10]:
    print("k size: ",k)
    results = []
    np.random.seed(10)
    for i in range(5):
        print(i," cross correlation")
        l = np.random.choice(range(20966), 20966, replace=False)
        current_df = aa_count_df.values[l]
        predictions = np.array([predict(current_df[0:20500],row[:-1],k) for row in current_df[20500:]])
        actual = np.array([row[-1] for row in current_df[20500:]])
        results.append(np.mean(predictions == actual))
    total_cv_results.append((k,sum(results)/len(results)))
    print(k,sum(results)/len(results))

# best result (10, 0.61884057971014494) in old SCOP2

# k size:  1   0.6540772532188841
# k size:  2   0.6540772532188841
# k size:  3   0.6472103004291846
# k size:  4   0.6484978540772532
# k size:  5   0.6472103004291845
# k size:  6   0.6476394849785407
# k size:  7   0.6433476394849785
# k size:  8   0.6420600858369099
# k size:  9   0.6369098712446352
# k size: 10   0.6399141630901287

aa_count_df = SCOP2_fa_represeq.loc[SCOP2_cla.index,prot_struct_aa]
aa_count_df.head()
aa_count_df.insert(20,"TP",list(SCOP2_cla.TP),True)
aa_count_df.head()
aa_count_df = aa_count_df.reset_index()
del aa_count_df['FA-DOMID']

total_cv_results = []
for k in [1,2,3,4,5,6,7,8,9,10]:
    results = []
    np.random.seed(10)
    for i in range(5):
        print(i," cross correlation")
        l = np.random.choice(range(20966), 20966, replace=False)
        current_df = aa_count_df.values[l]
        predictions = np.array([predict(current_df[0:20500],row[:-1],k) for row in current_df[20500:]])
        actual = np.array([row[-1] for row in current_df[20500:]])
        results.append(np.mean(predictions == actual))
    total_cv_results.append((k,sum(results)/len(results)))
    print("k size: ",k,sum(results)/len(results))

# k size:  1 0.9776824034334763
# k size:  2 0.9776824034334763
# k size:  3 0.9789699570815451
# k size:  4 0.9798283261802576
# k size:  5 0.9793991416309012
# k size:  6 0.9798283261802576
# k size:  7 0.9789699570815451
# k size:  8 0.9793991416309014
# k size:  9 0.9789699570815451
# k size:  10 0.9781115879828327


