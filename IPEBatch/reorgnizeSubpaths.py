#encoding=utf-8

"""
As there are too many subpaths between some node pairs, we need to filter the subpaths with max path number.
"""

import numpy
import random
# import theano
# from theano import tensor
# import dataProcessTools
import ConfigParser
import string, os, sys, time

SEED = 123
random.seed(SEED)

cf = ConfigParser.SafeConfigParser()
cf.read("pythonParamsConfig") # the config file
main_dir=cf.get("param", "root_dir")  # the main dir
maxPathNum=cf.get("param", "maxPathNum") # the max number of subpaths between node pairs
 
subpaths_train_1k_file=main_dir+'/subpathsSaveFile1k' 
subpaths_train_10k_file=main_dir+'/subpathsSaveFile10k' 
subpaths_train_100k_file=main_dir+'/subpathsSaveFile100k' 
subpaths_test_file=main_dir+'/subpathsSaveFileTest' 
 
subpaths_train_1k_file_new=main_dir+'/subpathsSaveFile1k_new' 
subpaths_train_10k_file_new=main_dir+'/subpathsSaveFile10k_new' 
subpaths_train_100k_file_new=main_dir+'/subpathsSaveFile100k_new' 
subpaths_test_file_new=main_dir+'/subpathsSaveFileTest_new' 
 


def readSubpathsFileAndFilter(subpathsFile, maxNum, subpathSaveFile):
    output = open(subpathSaveFile, 'w') 
    map={} 
    with open(subpathsFile) as f:
        for l in f:
            tmp0=l.strip()
            tmp=l.strip().split('\t')
            if len(tmp)<=0:
                continue
            key=tmp[0]+'-'+tmp[1]
            if key in map: 
                map[key].add(tmp0)
            else: 
                sets=set()
                sets.add(tmp0)
                map[key]=sets
    for key in map: 
        sets=map[key] 
        if len(sets)<=maxNum: 
            for path in sets:
                output.write(path+'\n')
                output.flush()
        else: 
            list=random.sample(sets, maxNum) 
            for path in list:
                output.write(path+'\n')
                output.flush()
    f.close()
    f=None
    output.close()
    output=None
                
    

if __name__=='__main__':
    
    readSubpathsFileAndFilter(subpaths_train_1k_file, maxPathNum, subpaths_train_1k_file_new)
    readSubpathsFileAndFilter(subpaths_train_10k_file, maxPathNum, subpaths_train_10k_file_new)
    readSubpathsFileAndFilter(subpaths_train_100k_file, maxPathNum, subpaths_train_100k_file_new)
    readSubpathsFileAndFilter(subpaths_test_file, maxPathNum, subpaths_test_file_new)
    



