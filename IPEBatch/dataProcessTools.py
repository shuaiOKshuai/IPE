#encoding=utf-8

import numpy
import theano

# Set the random number generators' seeds for consistency
SEED = 123
numpy.random.seed(SEED)

def get_minibatches_idx(n, minibatch_size, shuffle=False):
    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)


def getTrainingData(trainingDataFile):
    data=[] 
    pairs=[] 
    with open(trainingDataFile) as f:
        for l in f:
            tmp=l.strip().split()
            if len(tmp)<=0:
                continue
            arr=[]
            arr.append(tmp[0]+'-'+tmp[1])
            arr.append(tmp[1]+'-'+tmp[0])
            arr.append(tmp[0]+'-'+tmp[2])
            arr.append(tmp[2]+'-'+tmp[0])
            pairs.append(arr) 
            tmp=[int(x) for x in tmp] 
            data.append(tmp)
            
    return data,pairs

def getWordsEmbeddings(wordsEmbeddings_path):
    size=0
    dimension=0
    wemb=[]
    with open(wordsEmbeddings_path) as f:
        for l in f:
            arr=l.strip().split()
            if len(arr)==2: 
                size=int(arr[0])
                dimension=int(arr[1])
                wemb=numpy.zeros((size,dimension)).astype(theano.config.floatX)  # @UndefinedVariable 
                continue
            id=int(arr[0]) 
            for i in range(0,dimension):
                wemb[id][i]=float(arr[i+1])
    return wemb,dimension,size

def getTypesEmbeddings(typesEmbeddings_path):
    size=0
    dimension=0
    typeEmb=[]
    with open(typesEmbeddings_path) as f:
        for l in f:
            arr=l.strip().split()
            if len(arr)==2: 
                size=int(arr[0])
                dimension=int(arr[1])
                typeEmb=numpy.zeros((size,dimension)).astype(theano.config.floatX)  # @UndefinedVariable 
                continue
            id=int(arr[0]) 
            for i in range(0,dimension):
                typeEmb[id][i]=float(arr[i+1])
    return typeEmb,dimension,size

def loadAllSubPaths(subpaths_file,maxlen=1000):
    map={}
    with open(subpaths_file) as f:
        for l in f: 
            splitByTab=l.strip().split('\t')
            key=splitByTab[0]+'-'+splitByTab[1] 
            sentence=[int(y) for y in splitByTab[2].split()[:]] 
            if len(sentence)>maxlen: 
                continue
            if key in map: 
                map[key].append(sentence)
            else: 
                tmp=[]
                tmp.append(sentence)
                map[key]=tmp
    return map

def loadAllSubPathsByTyples(subpaths_file, tuples, maxlen=1000):
    map={}
    with open(subpaths_file) as f:
        for l in f: 
            splitByTab=l.strip().split('\t')
            key=splitByTab[0]+'-'+splitByTab[1]
            sentence=[int(y) for y in splitByTab[2].split()[:]] 
            if len(sentence)>maxlen: 
                continue
            if key not in tuples: 
                continue
            if key in map: 
                map[key].append(sentence)
            else: 
                tmp=[]
                tmp.append(sentence)
                map[key]=tmp
    return map

def loadAllSubPathsByTyplesRemoveRepeatPaths(subpaths_file, tuples, maxlen=1000):
    map={}
    with open(subpaths_file) as f:
        for l in f: 
            splitByTab=l.strip().split('\t')
            key=splitByTab[0]+'-'+splitByTab[1] 
            sentence=[int(y) for y in splitByTab[2].split()[:]] 
            if len(sentence)>maxlen:
                continue
            if key not in tuples: 
                continue
            if key in map: 
                map[key].add(splitByTab[2])
            else: 
                tmp=set()
                tmp.add(splitByTab[2])
                map[key]=tmp
    result={}
    for key in map:
        result[key]=[]
        for path in map[key]:
            result[key].append([int(y) for y in path.split()[:]])
    return result

def readAllSequencesFromFile(dependencySaveFile):
    map={} 
    with open(dependencySaveFile) as f:
        for l in f:
            tmp=l.strip().split('&') 
            if tmp[0] not in map: 
                map[tmp[0]]=[]
            maxLen=0 
            sequence=[]
            pathsTmp=tmp[1].replace('#','\t')
            paths=pathsTmp.strip().split('\t') 
            for path in paths: 
                pathList=[]
                nodes=path.strip().split(' ') 
                for node in nodes:
                    ns=node.strip().split('-') 
                    id0=int(ns[0]) # node id
                    id1=int(ns[1]) 
                    pathList.append([id0, id1]) 
                    if id1>maxLen: 
                        maxLen=id1
                        
                sequence.append(pathList)
            if len(tmp[2])==0: 
                dependenciesList=[]
                arr=[sequence, dependenciesList, maxLen+1]
                continue
            dependencies=tmp[2].strip().split(' ') 
            dependenciesList=[]
            for depend in dependencies: 
                dep=depend.strip().split('<-') 
                depList=[]
                d0=dep[0].strip().split(':') 
                d1=dep[1].strip().split(':') 
                depList.append(int(d0[0]))
                depList.append(int(d0[1]))
                depList.append(int(d1[0]))
                depList.append(int(d1[1]))
                dependenciesList.append(depList)
            arr=[sequence, dependenciesList, maxLen+1] 
            map[tmp[0]].append(arr)
    
    f.close()
    f=None
    return map
                

def discountForPathlength(beta, length):
    return numpy.exp(-beta*length)

def prepareDataForTraining(trainingDataTriples, trainingDataPairs, sequences_map, alpha, beta, gamma):
    n_triples=len(trainingDataTriples)
    sequencesNum=0
    maxlen=0
    for tuples in trainingDataPairs: 
        for tuple in tuples: 
            if tuple in sequences_map: 
                for sequence in sequences_map[tuple]: 
                    sequencesNum+=1
                    if sequence[2]>maxlen: 
                        maxlen=sequence[2]
    trainingParis=numpy.zeros([n_triples,4,2]).astype('int64')
    sequences_matrix=numpy.zeros([sequencesNum, maxlen]).astype('int64')
    dependency_matrix=numpy.zeros([sequencesNum, maxlen, maxlen]).astype(theano.config.floatX)  # @UndefinedVariable
    dependWeight_matrix=numpy.zeros([sequencesNum, maxlen, maxlen]).astype(theano.config.floatX)  # @UndefinedVariable
    sequencesLen_vector=numpy.zeros([sequencesNum]).astype('int64')
    discountSeq_matrix=numpy.zeros([sequencesNum, maxlen]).astype(theano.config.floatX)  # @UndefinedVariable
    discountForEachNode_matrix=numpy.zeros([sequencesNum, maxlen]).astype(theano.config.floatX)  # @UndefinedVariable
    
    currentIndex=0 
    
    for index in range(len(trainingDataPairs)): 
        tuples=trainingDataPairs[index] 
        for tupleIndex in range(len(tuples)): 
            tuple=tuples[tupleIndex]
            trainingParis[index][tupleIndex][0]=currentIndex 
            if tuple in sequences_map: 
                for sequence in sequences_map[tuple]: 
                    map={}
                    subpaths=sequence[0]
                    for i in range(len(subpaths)):
                        subpath=subpaths[i] 
                        length=len(subpath) 
                        pathDiscount=discountForPathlength(beta, length)
                        subpathMap={}
                        for j in range(len(subpath)):
                            node=subpath[j] 
                            subpathMap[node[0]]=node[1] 
                            sequences_matrix[currentIndex][node[1]]=node[0] 
                            discountForEachNode_matrix[currentIndex][node[1]]=discountForPathlength(gamma, j) 
                            if j>0: 
                                dependency_matrix[currentIndex][node[1]][subpath[j-1][1]]=1. 
                                dependWeight_matrix[currentIndex][node[1]][subpath[j-1][1]]=alpha 
                                discountSeq_matrix[currentIndex][node[1]]=pathDiscount 
                            if j==0: 
                                dependency_matrix[currentIndex][node[1]][node[1]]=1. 
                                dependWeight_matrix[currentIndex][node[1]][node[1]]=1. 
                                discountSeq_matrix[currentIndex][node[1]]=pathDiscount 
                                
                        map[i]=subpathMap 
                    sequencesLen_vector[currentIndex]=sequence[2] 
                    for depend in sequence[1]: 
                        dependency_matrix[currentIndex][map[depend[0]][depend[1]]][map[depend[2]][depend[3]]]=1.
                        dependWeight_matrix[currentIndex][map[depend[0]][depend[1]]][map[depend[2]][depend[3]]]=1.-alpha
                    currentIndex+=1 
                trainingParis[index][tupleIndex][1]=currentIndex 
            else: 
                trainingParis[index][tupleIndex][1]=currentIndex 
    indexSet=set()
    for i in range(len(trainingParis)):
        if trainingParis[i][0][0]==trainingParis[i][1][1] or trainingParis[i][2][0]==trainingParis[i][3][1]: 
            indexSet.add(i)
    trainingParis_new=numpy.zeros([n_triples-len(indexSet),4,2]).astype('int64')
    j=0 
    for i in range(len(trainingParis)):
        if i not in indexSet:
            trainingParis_new[j]=trainingParis[i]
            j+=1
    
    return trainingParis_new, sequences_matrix, dependency_matrix, dependWeight_matrix, sequencesLen_vector, discountSeq_matrix, discountForEachNode_matrix
    
    
def prepareDataForTrainingBatch(trainingDataTriples, trainingDataPairs, sequences_map, alpha, beta, gamma):
    trainingDataPairsValid=[] 
    for tuples in trainingDataPairs: 
        if (tuples[0] not in sequences_map and tuples[1] not in sequences_map) or (tuples[2] not in sequences_map and tuples[3] not in sequences_map): 
            continue
        trainingDataPairsValid.append(tuples) 
    n_triples=len(trainingDataPairsValid)
    sequencesNum=0
    maxlen=0
    for tuples in trainingDataPairsValid: 
        for tuple in tuples: 
            if tuple in sequences_map: 
                for sequence in sequences_map[tuple]: 
                    sequencesNum+=1
                    if sequence[2]>maxlen: 
                        maxlen=sequence[2]
    trainingParis=numpy.zeros([n_triples,4,2]).astype('int64')
    sequences_matrix=numpy.zeros([maxlen,sequencesNum]).astype('int64')
    masks_matrix=numpy.zeros([maxlen,sequencesNum]).astype(theano.config.floatX)  # @UndefinedVariable
    dependency_matrix=numpy.zeros([maxlen, sequencesNum, maxlen]).astype(theano.config.floatX)  # @UndefinedVariable
    dependWeight_matrix=numpy.zeros([maxlen, sequencesNum, maxlen]).astype(theano.config.floatX)  # @UndefinedVariable
    sequencesLen_vector=numpy.zeros([sequencesNum]).astype('int64')
    discountSeq_matrix=numpy.zeros([maxlen, sequencesNum]).astype(theano.config.floatX)  # @UndefinedVariable
    discountForEachNode_matrix=numpy.zeros([maxlen, sequencesNum]).astype(theano.config.floatX)  # @UndefinedVariable
    groups_tensor3=numpy.zeros([2, len(trainingDataPairsValid), sequencesNum]).astype(theano.config.floatX)  # @UndefinedVariable
    
    currentIndex=0 
    for index in range(len(trainingDataPairsValid)): 
        tuples=trainingDataPairsValid[index] 
        for tupleIndex in range(len(tuples)): 
            tuple=tuples[tupleIndex]
            trainingParis[index][tupleIndex][0]=currentIndex 
            if tuple in sequences_map: 
                for sequence in sequences_map[tuple]:
                    map={}
                    subpaths=sequence[0] 
                    for i in range(len(subpaths)):
                        subpath=subpaths[i] 
                        length=len(subpath) 
                        pathDiscount=discountForPathlength(beta, length) 
                        subpathMap={}
                        for j in range(len(subpath)):
                            node=subpath[j] 
                            subpathMap[node[0]]=node[1] 
                            sequences_matrix[node[1]][currentIndex]=node[0] 
                            masks_matrix[node[1]][currentIndex]=1. 
                            discountForEachNode_matrix[node[1]][currentIndex]=discountForPathlength(gamma, j) 
                            if j>0: 
                                dependency_matrix[node[1]][currentIndex][subpath[j-1][1]]=1. 
                                dependWeight_matrix[node[1]][currentIndex][subpath[j-1][1]]=alpha 
                                discountSeq_matrix[node[1]][currentIndex]=pathDiscount 
                            if j==0: 
                                dependency_matrix[node[1]][currentIndex][node[1]]=1. 
                                dependWeight_matrix[node[1]][currentIndex][node[1]]=1.
                                discountSeq_matrix[node[1]][currentIndex]=pathDiscount 
                                
                        map[i]=subpathMap 
                    sequencesLen_vector[currentIndex]=sequence[2] 
                    for depend in sequence[1]: 
                        dependency_matrix[map[depend[0]][depend[1]]][currentIndex][map[depend[2]][depend[3]]]=1. 
                        dependWeight_matrix[map[depend[0]][depend[1]]][currentIndex][map[depend[2]][depend[3]]]=1.-alpha 
                    if tupleIndex==0 or tupleIndex==1: 
                        groups_tensor3[0][index][currentIndex]=1. 
                    else: 
                        groups_tensor3[1][index][currentIndex]=1. 
                    
                    currentIndex+=1 
                trainingParis[index][tupleIndex][1]=currentIndex 
            else: 
                trainingParis[index][tupleIndex][1]=currentIndex 
    indexSet=set()
    for i in range(len(trainingParis)):
        if trainingParis[i][0][0]==trainingParis[i][1][1] or trainingParis[i][2][0]==trainingParis[i][3][1]: 
            indexSet.add(i)
    trainingParis_new=numpy.zeros([n_triples-len(indexSet),4,2]).astype('int64')
    j=0 
    for i in range(len(trainingParis)):
        if i not in indexSet:
            trainingParis_new[j]=trainingParis[i]
            j+=1
    
    return trainingParis_new, sequences_matrix, dependency_matrix, dependWeight_matrix, sequencesLen_vector, discountSeq_matrix, discountForEachNode_matrix, masks_matrix, groups_tensor3 
    
                    
def prepareDataForTest(query, candidate, sequences_map, alpha, beta, gamma):
    key1=bytes(query)+'-'+bytes(candidate)
    key2=bytes(candidate)+'-'+bytes(query)
    if key1 not in sequences_map and key2 not in sequences_map:
        return None,None,None,None,None,None
    sequences=[]
    if key1 in sequences_map:
        sequences.extend(sequences_map[key1])
    if key2 in sequences_map:
        sequences.extend(sequences_map[key2])
    if len(sequences)==0: 
        return None,None,None,None,None,None
    maxlen=0
    for sequence in sequences:
        if sequence[2]>maxlen: 
            maxlen=sequence[2]
    sequencesNum=len(sequences) 
    sequences_matrix=numpy.zeros([sequencesNum, maxlen]).astype('int64')
    dependency_matrix=numpy.zeros([sequencesNum, maxlen, maxlen]).astype(theano.config.floatX)  # @UndefinedVariable
    dependWeight_matrix=numpy.zeros([sequencesNum, maxlen, maxlen]).astype(theano.config.floatX)  # @UndefinedVariable
    sequencesLen_vector=numpy.zeros([sequencesNum]).astype('int64')
    discountSeq_matrix=numpy.zeros([sequencesNum, maxlen]).astype(theano.config.floatX)  # @UndefinedVariable
    discountForEachNode_matrix=numpy.zeros([sequencesNum, maxlen]).astype(theano.config.floatX)  # @UndefinedVariable
    for index in range(len(sequences)):
        sequence=sequences[index]
        map={}
        subpaths=sequence[0] 
        for i in range(len(subpaths)):
            subpath=subpaths[i] 
            length=len(subpath)
            pathDiscount=discountForPathlength(beta, length)
            subpathMap={}
            for j in range(len(subpath)):
                node=subpath[j] 
                subpathMap[node[0]]=node[1] 
                sequences_matrix[index][node[1]]=node[0] 
                discountForEachNode_matrix[index][node[1]]=discountForPathlength(gamma, j) 
                if j>0: 
                    dependency_matrix[index][node[1]][subpath[j-1][1]]=1. 
                    dependWeight_matrix[index][node[1]][subpath[j-1][1]]=alpha 
                    discountSeq_matrix[index][node[1]]=pathDiscount 
                if j==0: 
                    dependency_matrix[index][node[1]][node[1]]=1. 
                    dependWeight_matrix[index][node[1]][node[1]]=1. 
                    discountSeq_matrix[index][node[1]]=pathDiscount 
            map[i]=subpathMap 
        sequencesLen_vector[index]=sequence[2]
        for depend in sequence[1]: 
            dependency_matrix[index][map[depend[0]][depend[1]]][map[depend[2]][depend[3]]]=1.
            dependWeight_matrix[index][map[depend[0]][depend[1]]][map[depend[2]][depend[3]]]=1.-alpha
    return sequences_matrix, dependency_matrix, dependWeight_matrix, sequencesLen_vector, discountSeq_matrix, discountForEachNode_matrix
    

def prepareDataForTestBatch(query, candidates, sequences_map, alpha, beta, gamma):
    sequencesNum=0
    maxlen=0
    for candidate in candidates:
        key1=bytes(query)+'-'+bytes(candidate)
        key2=bytes(candidate)+'-'+bytes(query)
        if key1 in sequences_map:
            sequencesNum+=len(sequences_map[key1]) 
            for sequence in sequences_map[key1]: 
                if maxlen<sequence[2]:
                    maxlen=sequence[2]
        if key2 in sequences_map:
            sequencesNum+=len(sequences_map[key2]) 
            for sequence in sequences_map[key2]: 
                if maxlen<sequence[2]: 
                    maxlen=sequence[2]
    sequences_matrix=numpy.zeros([maxlen, sequencesNum]).astype('int64')
    masks_matrix=numpy.zeros([maxlen, sequencesNum]).astype(theano.config.floatX)  # @UndefinedVariable
    dependency_matrix=numpy.zeros([maxlen, sequencesNum, maxlen]).astype(theano.config.floatX)  # @UndefinedVariable
    dependWeight_matrix=numpy.zeros([maxlen, sequencesNum, maxlen]).astype(theano.config.floatX)  # @UndefinedVariable
    sequencesLen_vector=numpy.zeros([sequencesNum]).astype('int64')
    discountSeq_matrix=numpy.zeros([maxlen, sequencesNum]).astype(theano.config.floatX)  # @UndefinedVariable
    discountForEachNode_matrix=numpy.zeros([maxlen, sequencesNum]).astype(theano.config.floatX)  # @UndefinedVariable
    groups_tensor=numpy.zeros([len(candidates), sequencesNum]).astype(theano.config.floatX)  # @UndefinedVariable
    currentIndex=0 
    for tupleIndex in range(len(candidates)):
        candidate=candidates[tupleIndex]
        key1=bytes(query)+'-'+bytes(candidate)
        key2=bytes(candidate)+'-'+bytes(query)
        sequences=[]
        if key1 in sequences_map:
            sequences.extend(sequences_map[key1])
        if key2 in sequences_map:
            sequences.extend(sequences_map[key2])
        for index in range(len(sequences)):
            sequence=sequences[index]
            map={}
            subpaths=sequence[0] 
            for i in range(len(subpaths)):
                subpath=subpaths[i] 
                length=len(subpath) 
                pathDiscount=discountForPathlength(beta, length)
                subpathMap={}
                for j in range(len(subpath)):
                    node=subpath[j] 
                    subpathMap[node[0]]=node[1] 
                    sequences_matrix[node[1]][currentIndex]=node[0] 
                    masks_matrix[node[1]][currentIndex]=1. 
                    discountForEachNode_matrix[node[1]][currentIndex]=discountForPathlength(gamma, j) 
                    if j>0: 
                        dependency_matrix[node[1]][currentIndex][subpath[j-1][1]]=1. 
                        dependWeight_matrix[node[1]][currentIndex][subpath[j-1][1]]=alpha 
                        discountSeq_matrix[node[1]][currentIndex]=pathDiscount 
                    if j==0: 
                        dependency_matrix[node[1]][currentIndex][node[1]]=1. 
                        dependWeight_matrix[node[1]][currentIndex][node[1]]=1. 
                        discountSeq_matrix[node[1]][currentIndex]=pathDiscount 
                map[i]=subpathMap 
            sequencesLen_vector[currentIndex]=sequence[2] 
            for depend in sequence[1]: 
                dependency_matrix[map[depend[0]][depend[1]]][currentIndex][map[depend[2]][depend[3]]]=1.
                dependWeight_matrix[map[depend[0]][depend[1]]][currentIndex][map[depend[2]][depend[3]]]=1.-alpha
            groups_tensor[tupleIndex][currentIndex]=1. 
            currentIndex+=1
    
    return sequences_matrix, dependency_matrix, dependWeight_matrix, sequencesLen_vector, discountSeq_matrix, discountForEachNode_matrix, masks_matrix, groups_tensor
    

if __name__=='__main__':
    dependencySaveFile='D:/test/link-prediction/saveFile'
    sequences_map=readAllSequencesFromFile(dependencySaveFile)
    print sequences_map
    alpha=0.2
    beta=0.1
    gamma=0.05
    trainingDataTriples=[[5,9,19]]
    trainingDataPairs=[['5-9','9-5','5-19','19-5']]
    
    sequences_matrix, dependency_matrix, dependWeight_matrix, sequencesLen_vector, discountSeq_matrix, discountForEachNode_matrix=prepareDataForTest(5, 9, sequences_map, alpha, beta, gamma)
    print 'sequences_matrix=',sequences_matrix
    print 'dependency_matrix=',dependency_matrix
    print 'dependWeight_matrix=',dependWeight_matrix
    print 'sequencesLen_vector=',sequencesLen_vector
    print 'discountSeq_matrix=',discountSeq_matrix
    print 'discountForEachNode_matrix=',discountForEachNode_matrix
    print '------------------'
    print discountSeq_matrix[0]
    print '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
    
    candidates=[9,19]
    sequences_matrix, dependency_matrix, dependWeight_matrix, sequencesLen_vector, discountSeq_matrix, discountForEachNode_matrix, masks_matrix, groups_tensor=prepareDataForTestBatch(5, candidates, sequences_map, alpha, beta, gamma)
    print 'sequences_matrix=',sequences_matrix
    print 'dependency_matrix=',dependency_matrix
    print 'dependWeight_matrix=',dependWeight_matrix
    print 'sequencesLen_vector=',sequencesLen_vector
    print 'discountSeq_matrix=',discountSeq_matrix
    print 'discountForEachNode_matrix=',discountForEachNode_matrix
    print 'masks_matrix=',masks_matrix
    print 'groups_tensor=',groups_tensor
    
    