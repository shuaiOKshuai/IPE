#encoding=utf-8

"""
As there may be too many sequences(interactive paths structures) for the test data, you can filter the valid sequences out for test data, to save space.
"""

import numpy

import ConfigParser
import string, os, sys, time


cf = ConfigParser.SafeConfigParser()
cf.read("pythonParamsConfig")
main_dir=cf.get("param", "root_dir") 
sequences_test_file=main_dir+'/sequencesSaveFileTest' 
test_file=main_dir+'/test'
dest_folder=main_dir+'/testSplitFiles/' 


def readAllTuplesMapFromTestFile(test_file):
    map={}
    line=0 
    with open(test_file) as f:
        for l in f:
            tmp=l.strip().split()
            if len(tmp)<=0:
                continue
            sets=set()
            for i in range(1,len(tmp)):
                sets.add(tmp[0]+'-'+tmp[i])
                sets.add(tmp[i]+'-'+tmp[0])
            map[line]=sets 
            line+=1
    f.close()
    f=None
    return map
    
def readSequencesAndSave(sequenceFile, dest_folder, map):
    sequences={}
    with open(sequenceFile) as f:
        for l in f:
            tmp0=l.strip() 
            tmp1=tmp0.split('&') 
            if len(tmp0)<=0:
                continue
            if tmp1[0] in sequences: 
                sequences[tmp1[0]].add(tmp0)
            else: 
                sets=set()
                sets.add(tmp0)
                sequences[tmp1[0]]=sets
    f.close()
    f=None
    print 'Finish read data from sequence file... time ==',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    
    for i in range(len(map)): 
        filepath=dest_folder+bytes(i) 
        output = open(filepath, 'w')
        
        pairsSet=map[i] 
        for pair in pairsSet: 
            if pair in sequences: 
                for seq in sequences[pair]:
                    output.write(seq+'\n')
        output.close()
        output=None
        

if __name__=='__main__':
    
    print 'Start to read all pairs from test file ... time ==',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    map=readAllTuplesMapFromTestFile(test_file)
    print 'Start to read sequences and save them to the files ... time ==',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    readSequencesAndSave(sequences_test_file, dest_folder, map)
    print 'Final finished ... time ==',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    