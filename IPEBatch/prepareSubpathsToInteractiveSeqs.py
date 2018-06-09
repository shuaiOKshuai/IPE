#encoding=utf-8

"""
We leverage this file for Interactive paths Generation.
"""

import numpy
import random
import dataProcessTools
import ConfigParser
import string, os, sys, time
import SplitTestDatasetsIntoMultiFiles

SEED = 123
random.seed(SEED)


cf = ConfigParser.SafeConfigParser()
cf.read("pythonParamsConfig")

main_dir=cf.get("param", "root_dir") # the main dir
dataset_name=cf.get("param", "dataset_name") # dataset name
class_name=cf.get("param", "class_name") # class name
sequences_file=main_dir+'/'+dataset_name+'/sequencesSaveFile' # sequences file
sequences_file_train=main_dir+'/'+dataset_name+'/sequencesSaveFileTrain' # sequences file for train
subpaths_file=main_dir+'/'+dataset_name+'/subpathsSaveFile' # subpaths file
maxlen_subpaths=cf.getint("param", "maxlen_subpaths") # max length for subpaths
repeatTimes=cf.getint("param", "repeatTimes") # repeat times
repeatProportion=cf.getfloat("param", "repeatProportion") # repeat proportion
maxPathNum=cf.getint("param", "maxPathNum") # max num for paths

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
    return map
                

def getAlltuplesBatch(rootdir):
    folder=rootdir+'/'
    tuples=set()
    path=folder+'train_1k' 
    with open(path) as f:
        for l in f:
            tmp=l.strip().split()
            if len(tmp)<=0:
                continue
            tuples.add(tmp[0]+'-'+tmp[1])
            tuples.add(tmp[1]+'-'+tmp[0])
            tuples.add(tmp[0]+'-'+tmp[2])
            tuples.add(tmp[2]+'-'+tmp[0])
    f.close()
    f=None
    # training data 100
    path=folder+'train_10k' 
    with open(path) as f:
        for l in f:
            tmp=l.strip().split()
            if len(tmp)<=0:
                continue
            tuples.add(tmp[0]+'-'+tmp[1])
            tuples.add(tmp[1]+'-'+tmp[0])
            tuples.add(tmp[0]+'-'+tmp[2])
            tuples.add(tmp[2]+'-'+tmp[0])
    f.close()
    f=None
    # training data 1000
    path=folder+'train_100k' 
    with open(path) as f:
        for l in f:
            tmp=l.strip().split()
            if len(tmp)<=0:
                continue
            tuples.add(tmp[0]+'-'+tmp[1])
            tuples.add(tmp[1]+'-'+tmp[0])
            tuples.add(tmp[0]+'-'+tmp[2])
            tuples.add(tmp[2]+'-'+tmp[0])
    f.close()
    f=None
    # test data
    path=folder+'test' 
    with open(path) as f:
        for l in f:
            tmp=l.strip().split()
            if len(tmp)<=0:
                continue
            for j in range(1,len(tmp)):
                tuples.add(tmp[0]+'-'+tmp[j])
                tuples.add(tmp[j]+'-'+tmp[0])
    f.close()
    f=None
    return tuples

def getAlltuples(rootdir, datasetName, relationName):
    folder=rootdir+'/'+datasetName+'.splits/'
    tuples=set()
    folder_train10=folder+'train.10/'
    for i in range(1,6):
        path=folder_train10+'train_'+relationName+'_'+bytes(i) 
        with open(path) as f:
            for l in f:
                tmp=l.strip().split()
                if len(tmp)<=0:
                    continue
                tuples.add(tmp[0]+'-'+tmp[1])
                tuples.add(tmp[1]+'-'+tmp[0])
                tuples.add(tmp[0]+'-'+tmp[2])
                tuples.add(tmp[2]+'-'+tmp[0])
        f.close()
        f=None
    # training data 100
    folder_train100=folder+'train.100/'
    for i in range(1,6):
        path=folder_train100+'train_'+relationName+'_'+bytes(i) 
        with open(path) as f:
            for l in f:
                tmp=l.strip().split()
                if len(tmp)<=0:
                    continue
                tuples.add(tmp[0]+'-'+tmp[1])
                tuples.add(tmp[1]+'-'+tmp[0])
                tuples.add(tmp[0]+'-'+tmp[2])
                tuples.add(tmp[2]+'-'+tmp[0])
        f.close()
        f=None
    # training data 1000
    folder_train1000=folder+'train.1000/'
    for i in range(1,6):
        path=folder_train1000+'train_'+relationName+'_'+bytes(i) 
        with open(path) as f:
            for l in f:
                tmp=l.strip().split()
                if len(tmp)<=0:
                    continue
                tuples.add(tmp[0]+'-'+tmp[1])
                tuples.add(tmp[1]+'-'+tmp[0])
                tuples.add(tmp[0]+'-'+tmp[2])
                tuples.add(tmp[2]+'-'+tmp[0])
        f.close()
        f=None
    # test data
    folder_test=folder+'test/'
    for i in range(1,6):
        path=folder_test+'test_'+relationName+'_'+bytes(i) 
        with open(path) as f:
            for l in f:
                tmp=l.strip().split()
                if len(tmp)<=0:
                    continue
                for j in range(1,len(tmp)):
                    tuples.add(tmp[0]+'-'+tmp[j])
                    tuples.add(tmp[j]+'-'+tmp[0])
        f.close()
        f=None
    return tuples

def getAlltuplesForTraining(rootdir, datasetName, relationName):
    folder=rootdir+'/'+datasetName+'.splits/'
    tuples=set()
    folder_train10=folder+'train.10/'
    for i in range(1,6):
        path=folder_train10+'train_'+relationName+'_'+bytes(i) 
        with open(path) as f:
            for l in f:
                tmp=l.strip().split()
                if len(tmp)<=0:
                    continue
                tuples.add(tmp[0]+'-'+tmp[1])
                tuples.add(tmp[1]+'-'+tmp[0])
                tuples.add(tmp[0]+'-'+tmp[2])
                tuples.add(tmp[2]+'-'+tmp[0])
        f.close()
        f=None
    # training data 100
    folder_train100=folder+'train.100/'
    for i in range(1,6):
        path=folder_train100+'train_'+relationName+'_'+bytes(i) 
        with open(path) as f:
            for l in f:
                tmp=l.strip().split()
                if len(tmp)<=0:
                    continue
                tuples.add(tmp[0]+'-'+tmp[1])
                tuples.add(tmp[1]+'-'+tmp[0])
                tuples.add(tmp[0]+'-'+tmp[2])
                tuples.add(tmp[2]+'-'+tmp[0])
        f.close()
        f=None
    # training data 1000
    folder_train1000=folder+'train.1000/'
    for i in range(1,6):
        path=folder_train1000+'train_'+relationName+'_'+bytes(i) 
        with open(path) as f:
            for l in f:
                tmp=l.strip().split()
                if len(tmp)<=0:
                    continue
                tuples.add(tmp[0]+'-'+tmp[1])
                tuples.add(tmp[1]+'-'+tmp[0])
                tuples.add(tmp[0]+'-'+tmp[2])
                tuples.add(tmp[2]+'-'+tmp[0])
        f.close()
        f=None
    return tuples

def getAlltuplesForTestFromOneFile(filePath):
    tuples=set()
    # test data
    with open(filePath) as f:
        for l in f:
            tmp=l.strip().split()
            if len(tmp)<=0:
                continue
            for j in range(1,len(tmp)):
                tuples.add(tmp[0]+'-'+tmp[j])
                tuples.add(tmp[j]+'-'+tmp[0])
    f.close()
    f=None
    return tuples

def getAlltuplesForSingleDirection(rootdir, datasetName, relationName):
    folder=rootdir+'/'+datasetName+'.splits/'
    tuples=set()
    folder_train10=folder+'train.10/'
    for i in range(1,6):
        path=folder_train10+'train_'+relationName+'_'+bytes(i) 
        with open(path) as f:
            for l in f:
                tmp=l.strip().split()
                if len(tmp)<=0:
                    continue
                tuples.add(tmp[0]+'-'+tmp[1])
                tuples.add(tmp[0]+'-'+tmp[2])
        f.close()
        f=None
    # training data 100
    folder_train100=folder+'train.100/'
    for i in range(1,6):
        path=folder_train100+'train_'+relationName+'_'+bytes(i) 
        with open(path) as f:
            for l in f:
                tmp=l.strip().split()
                if len(tmp)<=0:
                    continue
                tuples.add(tmp[0]+'-'+tmp[1])
                tuples.add(tmp[0]+'-'+tmp[2])
        f.close()
        f=None
    # training data 1000
    folder_train1000=folder+'train.1000/'
    for i in range(1,6):
        path=folder_train1000+'train_'+relationName+'_'+bytes(i) 
        with open(path) as f:
            for l in f:
                tmp=l.strip().split()
                if len(tmp)<=0:
                    continue
                tuples.add(tmp[0]+'-'+tmp[1])
                tuples.add(tmp[0]+'-'+tmp[2])
        f.close()
        f=None
    # test data
    folder_test=folder+'test/'
    for i in range(1,6):
        path=folder_test+'test_'+relationName+'_'+bytes(i) 
        with open(path) as f:
            for l in f:
                tmp=l.strip().split()
                if len(tmp)<=0:
                    continue
                for j in range(1,len(tmp)):
                    tuples.add(tmp[0]+'-'+tmp[j])
#                     tuples.add(tmp[j]+'-'+tmp[0])
        f.close()
        f=None
    return tuples

def generateDependencyBySubpathsAndSave(tuples, subpathsMap, dependencySaveFile, repeatTimes):
    output = open(dependencySaveFile, 'w') 
    for tuple in tuples: 
        arr=tuple.strip().split('-') 
        start=int(arr[0]) 
        end=int(arr[1]) 
        if tuple not in subpathsMap: 
            continue
        subpaths=subpathsMap[tuple] 
        for time in range(repeatTimes): 
            random.shuffle(subpaths)
            nodesCountMap={} 
            pathsNodesMap={} 
            nodesPathsMap={} 
            for i in range(len(subpaths)):
                path=subpaths[i] 
                for node in path:
                    if node==start: 
                        continue
                    if node in nodesCountMap: 
                        nodesCountMap[node]+=1
                    else: 
                        nodesCountMap[node]=1
                    if i in pathsNodesMap: 
                        pathsNodesMap[i].add(node)
                    else: 
                        pathsNodesMap[i]=set()
                        pathsNodesMap[i].add(node)
                    if node in nodesPathsMap: 
                        nodesPathsMap[node].add(i) 
                    else: 
                        nodesPathsMap[node]=set()
                        nodesPathsMap[node].add(i)
            
            preSucMap={} 
            for node in nodesCountMap:
                if nodesCountMap[node]>1: 
                    for path in pathsNodesMap:
                        if node in pathsNodesMap[path]:
                            pSet=set()
                            sSet=set()
                            flag=True 
                            curNodeIndex=-1 
                            for index in range(len(subpaths[path])): 
                                n=subpaths[path][index] 
                                if flag and n!=node: 
                                    pSet.add(n)
                                elif flag and n==node: 
                                    pSet.add(n)
                                    flag=False 
                                    curNodeIndex=index
                                else: 
                                    sSet.add(n)
                            if path in preSucMap: 
                                if node==start: 
                                    preSucMap[path][node]=[pSet, sSet, -1, curNodeIndex] 
                                else: 
                                    preSucMap[path][node]=[pSet, sSet, subpaths[path][curNodeIndex-1], curNodeIndex] 
                            else: 
                                preSucMap[path]={} 
                                if node==start: 
                                    preSucMap[path][node]=[pSet, sSet, -1] 
                                else: 
                                    preSucMap[path][node]=[pSet, sSet, subpaths[path][curNodeIndex-1], curNodeIndex] 
            
            dependency=[]
            dependencyMap={} 
            for p in range(len(subpaths)): 
                if p not in preSucMap: 
                    continue
                repeatNodes=preSucMap[p] 
                for node in repeatNodes: 
                    for index in nodesPathsMap[node]: 
                        if p<index or (p>index and len(preSucMap[p][node][1].intersection(preSucMap[index][node][0]))==0):
                            if p in dependencyMap: 
                                if node in dependencyMap[p]: 
                                    dependencyMap[p][node][index]=preSucMap[index][node][2] 
                                else: 
                                    dependencyMap[p][node]={}
                                    dependencyMap[p][node][index]=preSucMap[index][node][2]
                            else: 
                                dependencyMap[p]={}
                                dependencyMap[p][node]={}
                                dependencyMap[p][node][index]=preSucMap[index][node][2]
            index=0
            curMap={} 
            for p in range(len(subpaths)): 
                curMap[p]=0
            subpathsIndex=[] 
            hasTraverseMap={} 
            hasFinishPaths=set() 
            flag=False 
            while len(hasFinishPaths)!=len(curMap): 
                for p in curMap:
                    if len(hasFinishPaths)==len(curMap): 
                        break
                    if p in hasFinishPaths: 
                        continue
                    curNode=curMap[p] 
                    for i in range(curNode,len(subpaths[p])): 
                        flag=True 
                        node=subpaths[p][i] 
                        if p in dependencyMap and node in dependencyMap[p]: 
                            for p2 in dependencyMap[p][node]: 
                                if p2 not in hasTraverseMap or (p2 in hasTraverseMap and dependencyMap[p][node][p2] not in hasTraverseMap[p2]): 
                                    flag=False
                                    break
                        if flag: 
                            if len(subpathsIndex)<=p: 
                                subpathsIndex.append([])
                            subpathsIndex[p].append([node, index])
                            index+=1
                            if p not in hasTraverseMap: 
                                hasTraverseMap[p]=set()
                            hasTraverseMap[p].add(node)
                            if i==len(subpaths[p])-1: 
                                curMap[p]=-1 
                                hasFinishPaths.add(p) 
                            else: 
                                curMap[p]=i+1
                        else: 
                            break
            str=tuple+'&'
            for p in range(len(subpathsIndex)):
                for n in subpathsIndex[p]:
                    str+=bytes(n[0])+'-'+bytes(n[1])+' '
                str+='#'
            str+='&'
            for p1 in dependencyMap:
                for n1 in dependencyMap[p1]:
                    for p2 in dependencyMap[p1][n1]:
                        str+=bytes(p1)+':'+bytes(n1)+'<-'+bytes(p2)+':'+bytes(dependencyMap[p1][n1][p2])+' '
            str+='\n'
            output.write(str)
            output.flush()
    output.close()
    output=None
    

def generateDependencyBySubpathsThenRemoveAndSave(tuples, subpathsMap, dependencySaveFile, repeatTimes, proportion):
    output = open(dependencySaveFile, 'w') 
    repeatNumber=0
    tupleIndex=0 
    print 'The number of tuples is ',len(tuples)
    for tuple in tuples: 
        tupleIndex+=1
        if tupleIndex%2000000==0: 
            print '+',
        if tupleIndex%50000000==0: 
            print ' '
        arr=tuple.strip().split('-') 
        start=int(arr[0]) 
        end=int(arr[1]) 
        if tuple not in subpathsMap: 
            continue
        subpaths=subpathsMap[tuple] 
        if repeatTimes>0:
            repeatNumber=repeatTimes
        else:
            repeatNumber=proportion*len(subpaths)
        for time in range(repeatNumber): 
            random.shuffle(subpaths)
            nodesCountMap={} 
            pathsNodesMap={} 
            nodesPathsMap={} 
            for i in range(len(subpaths)):
                path=subpaths[i] 
                for node in path:
                    if node==start: 
                        continue
                    if node in nodesCountMap: 
                        nodesCountMap[node]+=1
                    else: 
                        nodesCountMap[node]=1
                    if i in pathsNodesMap: 
                        pathsNodesMap[i].add(node)
                    else: 
                        pathsNodesMap[i]=set()
                        pathsNodesMap[i].add(node)
                    if node in nodesPathsMap: 
                        nodesPathsMap[node].add(i) 
                    else: 
                        nodesPathsMap[node]=set()
                        nodesPathsMap[node].add(i)
            
            preSucMap={} 
            for node in nodesCountMap:
                if nodesCountMap[node]>1: 
                    for path in pathsNodesMap:
                        if node in pathsNodesMap[path]: 
                            pSet=set()
                            sSet=set()
                            flag=True 
                            curNodeIndex=-1 
                            for index in range(len(subpaths[path])): 
                                n=subpaths[path][index] 
                                if flag and n!=node: 
                                    pSet.add(n)
                                elif flag and n==node: 
                                    pSet.add(n)
                                    flag=False 
                                    curNodeIndex=index
                                else: 
                                    sSet.add(n)
                            if path in preSucMap: 
                                if node==start: 
                                    preSucMap[path][node]=[pSet, sSet, -1, curNodeIndex] 
                                else: 
                                    preSucMap[path][node]=[pSet, sSet, subpaths[path][curNodeIndex-1], curNodeIndex] 
                            else: 
                                preSucMap[path]={} 
                                if node==start: 
                                    preSucMap[path][node]=[pSet, sSet, -1] 
                                else: 
                                    preSucMap[path][node]=[pSet, sSet, subpaths[path][curNodeIndex-1], curNodeIndex] 
            
            dependencyMap={} 
            for p in range(len(subpaths)): 
                if p not in preSucMap: 
                    continue
                repeatNodes=preSucMap[p] 
                for node in repeatNodes: 
                    for index in nodesPathsMap[node]: 
                        if p==index: 
                            continue
                        if p not in dependencyMap: 
                            dependencyMap[p]={}
                        if node not in dependencyMap[p]: 
                            dependencyMap[p][node]={}
                        dependencyMap[p][node][index]=preSucMap[index][node][2] 
            
            index=0
            curMap={} 
            for p in range(len(subpaths)): 
                curMap[p]=0
            subpathsIndex=[] 
            hasTraverseMap={} 
            hasFinishPaths=set() 
            flag=False 
            count=0 
            while len(hasFinishPaths)!=len(curMap): 
                count=0 
                for p in curMap: 
                    if len(hasFinishPaths)==len(curMap): 
                        break
                    if p in hasFinishPaths: 
                        continue
                    curNode=curMap[p] 
                    for i in range(curNode,len(subpaths[p])): 
                        flag=True 
                        node=subpaths[p][i] 
                        if p in dependencyMap and node in dependencyMap[p]: 
                            for p2 in dependencyMap[p][node]: 
                                if p2 not in hasTraverseMap or (p2 in hasTraverseMap and dependencyMap[p][node][p2] not in hasTraverseMap[p2]): 
                                    flag=False
                                    break
                        if flag: 
                            if len(subpathsIndex)<=p: 
                                subpathsIndex.append([])
                            subpathsIndex[p].append([node, index])
                            index+=1
                            if p not in hasTraverseMap: 
                                hasTraverseMap[p]=set()
                            hasTraverseMap[p].add(node)
                            if i==len(subpaths[p])-1: 
                                curMap[p]=-1 
                                hasFinishPaths.add(p) 
                            else: 
                                curMap[p]=i+1
                            count=0 
                        else: 
                            count+=1
                            break
                if count>0 and count>=len(curMap)-len(hasFinishPaths): 
                    for p in curMap: 
                        if len(hasFinishPaths)==len(curMap): 
                            break
                        if p in hasFinishPaths: 
                            continue
                        curNode=curMap[p] 
                        if len(subpathsIndex)<=p: 
                                subpathsIndex.append([])
                        node=subpaths[p][curNode] 
                        subpathsIndex[p].append([node, index]) 
                        index+=1
                        if p not in hasTraverseMap: 
                            hasTraverseMap[p]=set()
                        hasTraverseMap[p].add(node)
                        if curNode==len(subpaths[p])-1: 
                            curMap[p]=-1 
                            hasFinishPaths.add(p) 
                        else: 
                            curMap[p]=curNode+1
                        nodesDependMap={} 
                        for dependP in dependencyMap[p][node]: 
                            if dependP not in hasTraverseMap or dependencyMap[p][node][dependP] not in hasTraverseMap[dependP]: 
                                nodesDependMap[dependP]=dependencyMap[p][node][dependP]
                        for dependP in nodesDependMap: 
                            dependencyMap[p][node].pop(dependP) 
                        break
                        
            str=tuple+'&'
            for p in range(len(subpathsIndex)):
                for n in subpathsIndex[p]:
                    str+=bytes(n[0])+'-'+bytes(n[1])+' '
                str+='#'
            str+='&'
            for p1 in dependencyMap:
                for n1 in dependencyMap[p1]:
                    for p2 in dependencyMap[p1][n1]:
                        str+=bytes(p1)+':'+bytes(n1)+'<-'+bytes(p2)+':'+bytes(dependencyMap[p1][n1][p2])+' '
            str+='\n'
            output.write(str)
            output.flush()
    output.close()
    output=None

def readAllSequencesFromFile(sequenceFile):
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
    return sequences

def saveSequencesToOneFile(sequences, dest_file, pairsSet):
    output = open(dest_file, 'w')
    for key in sequences: 
        if key in pairsSet: 
            for seq in sequences[key]:
                output.write(seq+'\n')
                output.flush()
    output.close()
    output=None
    

if __name__=='__main__':
    
    print 'Filter all subpaths ...... time ==',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    readSubpathsFileAndFilter(subpaths_file, maxPathNum, subpaths_file+'_new')
    subpaths_file=subpaths_file+'_new'
    
    print 'Get all tuples ...... time ==',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    tuples=getAlltuples(main_dir, dataset_name, class_name)
    print 'Load all subpaths ...... time ==',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    subpathsMap=dataProcessTools.loadAllSubPathsByTyplesRemoveRepeatPaths(subpaths_file, tuples, maxlen_subpaths)
    print 'Generate the sequences ...... time ==',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    generateDependencyBySubpathsThenRemoveAndSave(tuples, subpathsMap, sequences_file, repeatTimes, repeatProportion)
    print 'Finished the sequences, time ==',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    
    sequences=readAllSequencesFromFile(sequences_file)
    
    trainTuples=getAlltuplesForTraining(main_dir, dataset_name, class_name)
    saveSequencesToOneFile(sequences, sequences_file_train, trainTuples)
    
    index='1'
    testFile=main_dir+'/'+dataset_name+'.splits/test/test_'+class_name+'_'+index
    sequenceFileTest=main_dir+'/'+dataset_name+'/sequencesSaveFileTest'+index
    tuples=getAlltuplesForTestFromOneFile(testFile)
    saveSequencesToOneFile(sequences, sequenceFileTest, tuples)
    
    index='2'
    testFile=main_dir+'/'+dataset_name+'.splits/test/test_'+class_name+'_'+index
    sequenceFileTest=main_dir+'/'+dataset_name+'/sequencesSaveFileTest'+index
    tuples=getAlltuplesForTestFromOneFile(testFile)
    saveSequencesToOneFile(sequences, sequenceFileTest, tuples)
    
    index='3'
    testFile=main_dir+'/'+dataset_name+'.splits/test/test_'+class_name+'_'+index
    sequenceFileTest=main_dir+'/'+dataset_name+'/sequencesSaveFileTest'+index
    tuples=getAlltuplesForTestFromOneFile(testFile)
    saveSequencesToOneFile(sequences, sequenceFileTest, tuples)
    
    index='4'
    testFile=main_dir+'/'+dataset_name+'.splits/test/test_'+class_name+'_'+index
    sequenceFileTest=main_dir+'/'+dataset_name+'/sequencesSaveFileTest'+index
    tuples=getAlltuplesForTestFromOneFile(testFile)
    saveSequencesToOneFile(sequences, sequenceFileTest, tuples)
    
    index='5'
    testFile=main_dir+'/'+dataset_name+'.splits/test/test_'+class_name+'_'+index
    sequenceFileTest=main_dir+'/'+dataset_name+'/sequencesSaveFileTest'+index
    tuples=getAlltuplesForTestFromOneFile(testFile)
    saveSequencesToOneFile(sequences, sequenceFileTest, tuples)
    
    print 'Final finished, time ==',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    
    
    