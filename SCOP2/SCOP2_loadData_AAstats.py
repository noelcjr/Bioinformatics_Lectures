#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 14:42:53 2020

@author: noel
"""
import requests
import pandas as pd
import numpy as np
"""
As an alternative, we can use the API provided along with the text files.
"""
r = requests.get('http://scop.mrc-lmb.cam.ac.uk/api/term/1000001')
r.text
term = r.json()
"""
I found API dictionaries not to be as clear as tables.

1. We could import all the information from SCOP2 from text files provided for 
downloading.
"""
file_path = '/home/noel/Projects/Bioinformatics/SCOP2/files/'
cols = ["FA-DOMID", "FA-PDBID", "FA-PDBREG", "FA-UNIID", "FA-UNIREG", 
        "SF-DOMID", "SF-PDBID", "SF-PDBREG", "SF-UNIID", "SF-UNIREG", "SCOPCLA"]
SCOP2_cla = pd.read_csv(file_path+"scop-cla-20200117.txt",sep=" ",header = 6,
                        names=cols)
SCOP2_cla.shape
SCOP2_cla.isnull().sum()
SCOP2_cla.head()
SCOP2_cla = SCOP2_cla.set_index("FA-DOMID")
SCOP2_cla.head()
TP = []
CL = []
CF = []
SF = []
FA = []
for i in SCOP2_cla.SCOPCLA:
    entry = i.split(",")
    TP.append(entry[0].split("=")[1])
    CL.append(entry[1].split("=")[1])
    CF.append(entry[2].split("=")[1])
    SF.append(entry[3].split("=")[1])
    FA.append(entry[4].split("=")[1])
del SCOP2_cla["SCOPCLA"]
data = {"TP":TP,"CL":CL,"CF":CF,"SF":SF,"FA":FA}
SCOP2_cla.insert(9,"TP", TP, True)
SCOP2_cla.insert(10,"CL", CL, True)
SCOP2_cla.insert(11,"CF", CF, True)
SCOP2_cla.insert(12,"SF", SF, True)
SCOP2_cla.insert(13,"FA", FA, True)

SCOP2_cla.to_csv(file_path+"scop_cla.csv")

cols = ["NODE_ID", "NODE_NAME"]
with open(file_path+"scop-des-20200117.txt", 'r') as f:
    count_lines = 1
    SCOP2_des = pd.DataFrame(columns=cols)
    for line in f:
        if count_lines > 6:
            des_dict = {}
            line_list = line.split()
            des_dict[cols[0]] = line_list[0]
            des_dict[cols[1]] = ' '.join(line_list[1:])
            SCOP2_des = SCOP2_des.append(des_dict, ignore_index=True)
        count_lines += 1
SCOP2_des.shape
SCOP2_des.isnull().sum()
SCOP2_des.head()
SCOP2_des = SCOP2_des.set_index("NODE_ID")
SCOP2_des.head()

SCOP2_des.to_csv(file_path+"scop_des.csv")
# NOTE the following two CSV files are loaded. They have already been produced 
#      and they take a few minutes to run. To save time we just load them now.
#      to produce them, run the commented section after the next two lines.
"""   
SCOP2_sf_represeq = pd.read_csv(file_path+"scop_sf_represeq_lib20200117_fa.csv")
SCOP2_fa_represeq = pd.read_csv(file_path+"scop_fa_represeq_lib20200117_fa.csv")

SCOP2_fa_represeq = SCOP2_fa_represeq.set_index("FA-DOMID")
SCOP2_sf_represeq = SCOP2_sf_represeq.set_index("FA-DOMID")
# TODO if do not have the CSV files above run the following:
"""
with open(file_path+"scop_fa_represeq_lib20200117.fa", 'r') as f:
    sequences = []
    col_names = []
    seq_line = "null"
    for line in f:
        if line[0] == ">":
            if seq_line == "null":
                seq_line = line+":"
                for i in line.split():
                    if i[0] == ">":
                        col_names.append("FA-DOMID")
                    else:
                        col_names.append(i.split("=")[0])
                col_names.append("Sequence")
            else:
                sequences.append(seq_line)
                seq_line = line+":"
        else:
            seq_line += line
    sequences.append(seq_line)
col_names
SCOP2_fa_represeq =  pd.DataFrame(columns=col_names)          
for i in sequences:
    fa_dict = {}
    for j in i.split('\n')[:-1]:
        if j[0] == ">":
            cols = j[1:].split(" ")
            fa_dict[col_names[0]] = cols[0]
            fa_dict[col_names[1]] = cols[1].split("=")[1]
            fa_dict[col_names[2]] = cols[2].split("=")[1]
            fa_dict[col_names[3]] = cols[3].split("=")[1]
        elif j[0] == ":":
            fa_dict[col_names[4]] = j[1:]
    SCOP2_fa_represeq = SCOP2_fa_represeq.append(fa_dict, ignore_index=True)
SCOP2_fa_represeq = SCOP2_fa_represeq.set_index("FA-DOMID")

with open(file_path+"scop_sf_represeq_lib20200117.fa", 'r') as f:
    sequences = []
    col_names = []
    seq_line = "null"
    for line in f:
        if line[0] == ">":
            if seq_line == "null":
                seq_line = line+":"
                for i in line.split():
                    if i[0] == ">":
                        col_names.append("FA-DOMID")
                    else:
                        col_names.append(i.split("=")[0])
                col_names.append("Sequence")
            else:
                sequences.append(seq_line)
                seq_line = line+":"
        else:
            seq_line += line
    sequences.append(seq_line)
col_names
SCOP2_sf_represeq =  pd.DataFrame(columns=col_names)          
for i in sequences:
    sf_dict = {}
    for j in i.split('\n')[:-1]:
        if j[0] == ">":
            cols = j[1:].split(" ")
            sf_dict[col_names[0]] = cols[0]
            sf_dict[col_names[1]] = cols[1].split("=")[1]
            sf_dict[col_names[2]] = cols[2].split("=")[1]
            sf_dict[col_names[3]] = cols[3].split("=")[1]
        elif j[0] == ":":
            sf_dict[col_names[4]] = j[1:]
    SCOP2_sf_represeq = SCOP2_sf_represeq.append(sf_dict, ignore_index=True)
SCOP2_sf_represeq = SCOP2_sf_represeq.set_index("FA-DOMID")   

aa = 'ACDEFGHIKLMNPQRSTVWYBXZUO'
aa3 =  ["ALA","CYS","ASP","GLU","PHE","GLY","HIS","ILE","LYS","LEU","MET", \
        "ASN","PRO","GLN","ARG","SER","THR","VAL","TRP","TYR","ASX","GLX","B",
        "X","Z","U","O"]

aa_d = {"A":"ALA","C":"CYS","D":"ASP","E":"GLU","F":"PHE","G":"GLY","H":"HIS", \
        "I":"ILE","K":"LYS","L":"LEU","M":"MET","N":"ASN","P":"PRO","Q":"GLN", \
        "R":"ARG","S":"SER","T":"THR","V":"VAL","W":"TRP","Y":"TYR","B":"ASX",
        "Z":"GLX","X":"X","U":"U","O":"O"}

aa_types = {"hydrophobic":"GAVFPMILW","charged":"DEKRH","charged+":"DE",
            "charged-":"KRH","polar":"STYCNQ","other":"BZXUO"}

SCOP2_fa_represeq.head()

aa_c = {}
for i in aa:
    aa_c[i] = np.zeros(SCOP2_fa_represeq.shape[0])
    
aa_t = {}
for i in aa_types.keys():
    aa_t[i] = np.zeros(SCOP2_fa_represeq.shape[0])

count = 0
for i in SCOP2_fa_represeq.index:
    for j in SCOP2_fa_represeq.loc[i,"Sequence"]:
        aa_c[j][count] += 1
        if j in aa_types["hydrophobic"]:
            aa_t["hydrophobic"][count] += 1
        if j in aa_types["charged"]:
            aa_t["charged"][count] += 1
        if j in aa_types["charged+"]:
            aa_t["charged+"][count] += 1
        if j in aa_types["charged-"]:
            aa_t["charged-"][count] += 1
        if j in aa_types["polar"]:
            aa_t["polar"][count] += 1
        if j in aa_types["other"]:
            aa_t["other"][count] += 1        
    count += 1

for i in aa_c.keys():
    SCOP2_fa_represeq[i] = list(aa_c[i])
    SCOP2_fa_represeq = SCOP2_fa_represeq.astype({i:int})

for i in aa_t.keys():
    SCOP2_fa_represeq[i] = list(aa_t[i])
    SCOP2_fa_represeq = SCOP2_fa_represeq.astype({i:int})

SCOP2_fa_represeq.to_csv(file_path+"scop_fa_represeq_lib20200117_fa.csv")

SCOP2_sf_represeq.head()
#SCOP2_sf_represeq = SCOP2_sf_represeq.set_index("FA-DOMID")
aa_c = {}
for i in aa:
    aa_c[i] = np.zeros(SCOP2_sf_represeq.shape[0])
    
aa_t = {}
for i in aa_types.keys():
    aa_t[i] = np.zeros(SCOP2_sf_represeq.shape[0])

count = 0
for i in SCOP2_sf_represeq.index:
    for j in SCOP2_sf_represeq.loc[i,"Sequence"]:
        aa_c[j][count] += 1
        if j in aa_types["hydrophobic"]:
            aa_t["hydrophobic"][count] += 1
        if j in aa_types["charged"]:
            aa_t["charged"][count] += 1
        if j in aa_types["charged+"]:
            aa_t["charged+"][count] += 1
        if j in aa_types["charged-"]:
            aa_t["charged-"][count] += 1
        if j in aa_types["polar"]:
            aa_t["polar"][count] += 1
        if j in aa_types["other"]:
            aa_t["other"][count] += 1        
    count += 1

for i in aa_c.keys():
    SCOP2_sf_represeq[i] = list(aa_c[i])
    SCOP2_sf_represeq = SCOP2_sf_represeq.astype({i:int})

for i in aa_t.keys():
    SCOP2_sf_represeq[i] = list(aa_t[i])
    SCOP2_sf_represeq = SCOP2_sf_represeq.astype({i:int})
SCOP2_sf_represeq.to_csv(file_path+"scop_sf_represeq_lib20200117_fa.csv")

# Ok so far. TODO scop-des-20200117.txt
