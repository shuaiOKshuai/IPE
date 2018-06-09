#encoding=utf-8
'''
'''

import numpy
import theano

# Set the random number generators' seeds for consistency
SEED = 123
numpy.random.seed(SEED)

def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """
    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):# 
        minibatches.append(idx_list[minibatch_start:# 
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)# 


def getTrainingData(trainingDataFile):
    '''
    :type string
    :param trainingDataFile 
    '''
    data=[] # 
    pairs=[] # 
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
    """
    :type String
    :param wordsEmbeddings_path
    """
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
#                 wemb[id][i]=theano.config.floatX(arr[i+1])  # @UndefinedVariable
                wemb[id][i]=float(arr[i+1])
    return wemb,dimension,size

def getTypesEmbeddings(typesEmbeddings_path):
    """
    :type String
    :param wordsEmbeddings_path，
    """
    size=0
    dimension=0
    typeEmb=[]
    with open(typesEmbeddings_path) as f:
        # 
        for l in f:
            arr=l.strip().split()
            if len(arr)==2: #
                size=int(arr[0])
                dimension=int(arr[1])
                # 
                typeEmb=numpy.zeros((size,dimension)).astype(theano.config.floatX)  # @UndefinedVariable #
                continue
            # 
            id=int(arr[0]) # 
            for i in range(0,dimension):
#                 wemb[id][i]=theano.config.floatX(arr[i+1])  # @UndefinedVariable
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
    """
    """
    map={}
    with open(subpaths_file) as f:
        for l in f: # 对于每一行，执行以下操作
            splitByTab=l.strip().split('\t')
            key=splitByTab[0]+'-'+splitByTab[1] # 组合成key,startId-endId
            sentence=[int(y) for y in splitByTab[2].split()[:]] # index=2是这条sub-path
#             sentence=[y for y in splitByTab[2].split()[:]] # index=2是这条sub-path
#             sentence=[[y] for y in sentence] # 将这个横向vector变成列式
#             sentence=numpy.asarray(sentence) # 将sentence变化成matrix的形式，可以直接使用
            if len(sentence)>maxlen: # 如果长度超过了maxlen，则舍弃
                continue
            # 检查这个tuple是否合格
            if key not in tuples: # 如果不合格，则扔掉
                continue
            if key in map: # 如果结果中已经有这个key了，则直接添加
                map[key].append(sentence)
            else: # 如果还没有这个key，则新建一个[]，然后添加进去
                tmp=[]
                tmp.append(sentence)
                map[key]=tmp
    return map

def loadAllSubPathsByTyplesRemoveRepeatPaths(subpaths_file, tuples, maxlen=1000):
    """
    """
    map={}
    with open(subpaths_file) as f:
        for l in f: # 
            splitByTab=l.strip().split('\t')
            key=splitByTab[0]+'-'+splitByTab[1] # 
            sentence=[int(y) for y in splitByTab[2].split()[:]] # 
            if len(sentence)>maxlen: # 
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
    """
    """
    map={} # 
    with open(dependencySaveFile) as f:
        for l in f:
            tmp=l.strip().split('&') # 
            if tmp[0] not in map: # 
                map[tmp[0]]=[]
            maxLen=0 # 
            sequence=[]
            pathsTmp=tmp[1].replace('#','\t')
            paths=pathsTmp.strip().split('\t') # 
            for path in paths: # 
                pathList=[]
                nodes=path.strip().split(' ') # 
                for node in nodes:
                    ns=node.strip().split('-') # 
                    id0=int(ns[0]) # node id
                    id1=int(ns[1]) # 
                    pathList.append([id0, id1]) # 
                    if id1>maxLen: #
                        maxLen=id1
                        
                sequence.append(pathList)
            if len(tmp[2])==0: # 
                dependenciesList=[]
                arr=[sequence, dependenciesList, maxLen+1]
                continue
            
            dependencies=tmp[2].strip().split(' ') # 
            dependenciesList=[]
            for depend in dependencies: #
                dep=depend.strip().split('<-') # 
                depList=[]
                d0=dep[0].strip().split(':') # 
                d1=dep[1].strip().split(':') # 
                depList.append(int(d0[0]))
                depList.append(int(d0[1]))
                depList.append(int(d1[0]))
                depList.append(int(d1[1]))
                dependenciesList.append(depList)
            # 
            arr=[sequence, dependenciesList, maxLen+1] # 
            map[tmp[0]].append(arr)
    
    # 首先关闭资源
    f.close()
    f=None
    return map
                

def discountForPathlength(beta, length):
    """计算discount"""
    return numpy.exp(-beta*length)

def prepareDataForTraining(trainingDataTriples, trainingDataPairs, sequences_map, alpha, beta, gamma):
    """
        
    """
    n_triples=len(trainingDataTriples)
    sequencesNum=0
    maxlen=0
    for tuples in trainingDataPairs: # 
        for tuple in tuples: # 
            if tuple in sequences_map: #
                for sequence in sequences_map[tuple]: # 
                    sequencesNum+=1
                    if sequence[2]>maxlen: #
                        maxlen=sequence[2]
    trainingParis=numpy.zeros([n_triples,4,2]).astype('int64')
    sequences_matrix=numpy.zeros([sequencesNum, maxlen]).astype('int64')
    dependency_matrix=numpy.zeros([sequencesNum, maxlen, maxlen]).astype(theano.config.floatX)  # @UndefinedVariable
    dependWeight_matrix=numpy.zeros([sequencesNum, maxlen, maxlen]).astype(theano.config.floatX)  # @UndefinedVariable
    sequencesLen_vector=numpy.zeros([sequencesNum]).astype('int64')
    discountSeq_matrix=numpy.zeros([sequencesNum, maxlen]).astype(theano.config.floatX)  # @UndefinedVariable
    discountForEachNode_matrix=numpy.zeros([sequencesNum, maxlen]).astype(theano.config.floatX)  # @UndefinedVariable
    
    currentIndex=0 # 
    for index in range(len(trainingDataPairs)): # 
        tuples=trainingDataPairs[index] # 
        for tupleIndex in range(len(tuples)): # 
            tuple=tuples[tupleIndex]
            trainingParis[index][tupleIndex][0]=currentIndex #
            if tuple in sequences_map: # 
                for sequence in sequences_map[tuple]: 
                    map={}
                    subpaths=sequence[0] # 
                    for i in range(len(subpaths)):
                        subpath=subpaths[i] # 
                        length=len(subpath) # 
                        pathDiscount=discountForPathlength(beta, length)
                        subpathMap={}
                        for j in range(len(subpath)):
                            node=subpath[j] # 
                            subpathMap[node[0]]=node[1] 
                            
                            sequences_matrix[currentIndex][node[1]]=node[0] 
                            discountForEachNode_matrix[currentIndex][node[1]]=discountForPathlength(gamma, j) 
                            if j>0: 
                                dependency_matrix[currentIndex][node[1]][subpath[j-1][1]]=1. 
                                dependWeight_matrix[currentIndex][node[1]][subpath[j-1][1]]=alpha 
                                discountSeq_matrix[currentIndex][node[1]]=pathDiscount 
                            if j==0: #
                                dependency_matrix[currentIndex][node[1]][node[1]]=1. # 
                                dependWeight_matrix[currentIndex][node[1]][node[1]]=1. # 
                                discountSeq_matrix[currentIndex][node[1]]=pathDiscount # 
                                
                        map[i]=subpathMap # 
                    sequencesLen_vector[currentIndex]=sequence[2] # 
                    for depend in sequence[1]: 
                        dependency_matrix[currentIndex][map[depend[0]][depend[1]]][map[depend[2]][depend[3]]]=1.
                        dependWeight_matrix[currentIndex][map[depend[0]][depend[1]]][map[depend[2]][depend[3]]]=1.-alpha
                    currentIndex+=1 # 
                trainingParis[index][tupleIndex][1]=currentIndex # 
            else: # 
                trainingParis[index][tupleIndex][1]=currentIndex # 
    indexSet=set()
    for i in range(len(trainingParis)):
        if trainingParis[i][0][0]==trainingParis[i][1][1] or trainingParis[i][2][0]==trainingParis[i][3][1]: # 如果不合格，则记录下来这个index
            indexSet.add(i)
    trainingParis_new=numpy.zeros([n_triples-len(indexSet),4,2]).astype('int64')
    j=0 # 
    for i in range(len(trainingParis)):
        if i not in indexSet:
            trainingParis_new[j]=trainingParis[i]
            j+=1
    
    return trainingParis_new, sequences_matrix, dependency_matrix, dependWeight_matrix, sequencesLen_vector, discountSeq_matrix, discountForEachNode_matrix
                        
def prepareDataForTest(query, candidate, sequences_map, alpha, beta, gamma):
    """
    """
    key1=bytes(query)+'-'+bytes(candidate)
    key2=bytes(candidate)+'-'+bytes(query)
    if key1 not in sequences_map and key2 not in sequences_map:
        return None,None,None,None,None,None
    sequences=[]
    if key1 in sequences_map:
        sequences.extend(sequences_map[key1])
    if key2 in sequences_map:
        sequences.extend(sequences_map[key2])
    if len(sequences)==0: # 
        return None,None,None,None,None,None
    maxlen=0
    for sequence in sequences:
        if sequence[2]>maxlen: 
            maxlen=sequence[2]
    sequencesNum=len(sequences) #
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
            subpath=subpaths[i] # 
            length=len(subpath) # 
            pathDiscount=discountForPathlength(beta, length)
            subpathMap={}
            for j in range(len(subpath)):
                node=subpath[j] # 
                subpathMap[node[0]]=node[1] # 
                sequences_matrix[index][node[1]]=node[0] # 
                discountForEachNode_matrix[index][node[1]]=discountForPathlength(gamma, j) # 
                if j>0: # 
                    dependency_matrix[index][node[1]][subpath[j-1][1]]=1. # 
                    dependWeight_matrix[index][node[1]][subpath[j-1][1]]=alpha # 
                    discountSeq_matrix[index][node[1]]=pathDiscount # 
                if j==0: # 
                    dependency_matrix[index][node[1]][node[1]]=1. # 
                    dependWeight_matrix[index][node[1]][node[1]]=1. # 
                    discountSeq_matrix[index][node[1]]=pathDiscount # 
            map[i]=subpathMap # 
        sequencesLen_vector[index]=sequence[2] # 
        for depend in sequence[1]: # 
            dependency_matrix[index][map[depend[0]][depend[1]]][map[depend[2]][depend[3]]]=1.
            dependWeight_matrix[index][map[depend[0]][depend[1]]][map[depend[2]][depend[3]]]=1.-alpha
    return sequences_matrix, dependency_matrix, dependWeight_matrix, sequencesLen_vector, discountSeq_matrix, discountForEachNode_matrix
    

if __name__=='__main__':
    dependencySaveFile='D:/test/link-prediction/saveFile'
    sequences_map=readAllSequencesFromFile(dependencySaveFile)
    print sequences_map
    alpha=0.2
    beta=0.1
    gamma=0.05
    trainingDataTriples=[[5,9,1]]
    trainingDataPairs=[['5-9','9-5','5-1','1-5']]
    trainingParis_new, sequences_matrix, dependency_matrix, dependWeight_matrix, sequencesLen_vector, discountSeq_matrix, discountForEachNode_matrix=prepareDataForTraining(trainingDataTriples, trainingDataPairs, sequences_map, alpha, beta, gamma)
#     print 'trainingParis_new=',trainingParis_new
#     print 'sequences_matrix=',sequences_matrix
#     print 'dependency_matrix=',dependency_matrix
#     print 'dependWeight_matrix=',dependWeight_matrix
#     print 'sequencesLen_vector=',sequencesLen_vector
#     print 'discountSeq_matrix=',discountSeq_matrix
#     print 'discountForEachNode_matrix=',discountForEachNode_matrix
#     print '------------------'
#     print discountSeq_matrix[0]
    
    sequences_matrix, dependency_matrix, dependWeight_matrix, sequencesLen_vector, discountSeq_matrix, discountForEachNode_matrix=prepareDataForTest(5, 9, sequences_map, alpha, beta, gamma)
    print 'trainingParis_new=',trainingParis_new
    print 'sequences_matrix=',sequences_matrix
    print 'dependency_matrix=',dependency_matrix
    print 'dependWeight_matrix=',dependWeight_matrix
    print 'sequencesLen_vector=',sequencesLen_vector
    print 'discountSeq_matrix=',discountSeq_matrix
    print 'discountForEachNode_matrix=',discountForEachNode_matrix
    print '------------------'
    print discountSeq_matrix[0]