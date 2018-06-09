#encoding=utf-8

"""
We leverage this file for Interactive paths Generation.
"""

import numpy
import random
import dataProcessTools
import ConfigParser
import string, os, sys, time

SEED = 123
random.seed(SEED)


cf = ConfigParser.SafeConfigParser()
cf.read("pythonParamsConfig")

main_dir=cf.get("param", "root_dir") # the main dir
dataset_name=cf.get("param", "dataset_name") # dataset name
class_name=cf.get("param", "class_name") # class name
subpaths_file=cf.get("param", "subpaths_file") # subpaths file
sequences_file=cf.get("param", "sequences_file") # sequences file
maxlen_subpaths=cf.getint("param", "maxlen_subpaths") # max length for subpaths
repeatTimes=cf.getint("param", "repeatTimes") # repeat times
repeatProportion=cf.getfloat("param", "repeatProportion") # repeat proportion


def getAlltuples(rootdir, datasetName, relationName):
    """
    """
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

def getAlltuplesForSingleDirection(rootdir, datasetName, relationName):
    """
    """
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
#                 tuples.add(tmp[1]+'-'+tmp[0])
                tuples.add(tmp[0]+'-'+tmp[2])
#                 tuples.add(tmp[2]+'-'+tmp[0])
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
#                 tuples.add(tmp[1]+'-'+tmp[0])
                tuples.add(tmp[0]+'-'+tmp[2])
#                 tuples.add(tmp[2]+'-'+tmp[0])
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
#                 tuples.add(tmp[1]+'-'+tmp[0])
                tuples.add(tmp[0]+'-'+tmp[2])
#                 tuples.add(tmp[2]+'-'+tmp[0])
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
    """
    """
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
    """
    """
    output = open(dependencySaveFile, 'w') 
    repeatNumber=0
    tupleIndex=0 
    print 'The number of tuples is ',len(tuples)
    for tuple in tuples: 
        tupleIndex+=1
        if tupleIndex%500==0: 
            print '+',
        if tupleIndex%20000==0: 
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


if __name__=='__main__':
    
    print 'Get all tuples ...... time ==',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    tuples=getAlltuples(main_dir, dataset_name, class_name)
    print 'Load all subpaths ...... time ==',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    subpathsMap=dataProcessTools.loadAllSubPathsByTyplesRemoveRepeatPaths(subpaths_file, tuples, maxlen_subpaths)
    print 'Generate the sequences ...... time ==',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    generateDependencyBySubpathsThenRemoveAndSave(tuples, subpathsMap, sequences_file, repeatTimes, repeatProportion)
    print 'Finished the sequences, time ==',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    