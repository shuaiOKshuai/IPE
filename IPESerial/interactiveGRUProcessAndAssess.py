#encoding=utf-8
'''
'''

import numpy
import theano
from theano import tensor
from collections import OrderedDict
import dataProcessTools
import toolsFunction
import evaluateTools
import interactiveGRUProcessModel

def load_params(path, params):
    """
    """
    pp = numpy.load(path) 
    for kk, vv in params.items():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params


def get_interactiveGRUModel(
                      
                   model_params_path='', 
                     word_dimension=0, 
                     type_dimension=0,
                     dimension=0,
                     attention_dimension=0,
                     alpha=0.1,
                     beta=0.1,
                     gamma=0.1,
                      ):
    """
    """
    model_options = locals().copy()
    
    tparams = OrderedDict()
    tparams['W_z']=None
    tparams['W_r']=None
    tparams['W_h']=None
    tparams['U_z']=None
    tparams['U_r']=None
    tparams['U_h']=None
    tparams['b_z']=None
    tparams['b_r']=None
    tparams['b_h']=None
    tparams['type_eta']=None
    tparams['w']=None
    tparams=load_params(model_params_path, tparams) 
    
    sequences_matrix, dependency_matrix, dependWeight_matrix, sequencesLen_vector, discountSeq_matrix, discountForEachNode_matrix, wordsEmbeddings, typesEmbeddings, value=interactiveGRUProcessModel.interactiveGRUProcessModel(model_options, tparams)
    func=theano.function([sequences_matrix, dependency_matrix, dependWeight_matrix, sequencesLen_vector, discountSeq_matrix, discountForEachNode_matrix, wordsEmbeddings, typesEmbeddings], value, on_unused_input='ignore') 
    return func 


def compute_path2vec(
                     wordsEmbeddings=None, 
                     wordsEmbeddings_path='None', 
                     typesEmbeddings=None,
                     typesEmbeddings_path='None',
                     word_dimension=0,
                     type_dimension=0,
                     dimension=0,
                     attention_dimension=0,
                     wordsSize=0,
                     subpaths_map=None, 
                     subpaths_file='',
                     sequences_map=None, 
                     sequences_file='',
                     maxlen_subpaths=1000, 
                     maxlen=100,  
                     alpha=0,
                     beta=0,
                     gamma=0,
                     
                     test_data_file='', 
                     top_num=10,
                     ideal_data_file='', 
                     func=None, 
                   ):
    model_options = locals().copy()
    
    
    if wordsEmbeddings is None:
        if wordsEmbeddings_path is not None: 
            wordsEmbeddings,dimension,wordsSize=dataProcessTools.getWordsEmbeddings(wordsEmbeddings_path)
        else: 
            exit(0) 
    if typesEmbeddings is None: 
        if typesEmbeddings_path is not None: 
            typesEmbeddings,type_dimension,wordsSize=dataProcessTools.getTypesEmbeddings(typesEmbeddings_path)
        else: 
            exit(0) 
            
    sequences_data=dataProcessTools.readAllSequencesFromFile(sequences_file)

    errCount=0 

    line_count=0 
    test_map={} 
    print 'Compute MAP and nDCG for file ',test_data_file
    with open(test_data_file) as f:
        for l in f: 
            arr=l.strip().split()
            query=int(arr[0]) 
            map={} 
            for i in range(1,len(arr)): 
                candidate=int(arr[i]) 
                sequences_matrix, dependency_matrix, dependWeight_matrix, sequencesLen_vector, discountSeq_matrix, discountForEachNode_matrix=dataProcessTools.prepareDataForTest(query, candidate, sequences_data, alpha, beta, gamma)
                if sequences_matrix is None or len(sequences_matrix)==0: 
                    map[candidate]=-1000. 
                    errCount+=1
                else: 
                    value=func(sequences_matrix, dependency_matrix, dependWeight_matrix, sequencesLen_vector, discountSeq_matrix, discountForEachNode_matrix,wordsEmbeddings,typesEmbeddings)
                    map[candidate]=value
            
            tops_in_line=toolsFunction.mapSortByValueDESC(map, top_num)
            test_map[line_count]=tops_in_line 
            line_count+=1 
                
    
    line_count=0
    ideal_map={}
    with open(ideal_data_file) as f: 
        for l in f: 
            arr=l.strip().split()
            arr=[int(x) for x in arr] 
            ideal_map[line_count]=arr[1:] 
            line_count+=1 
    
    MAP=evaluateTools.get_MAP(top_num, ideal_map, test_map)
    MnDCG=evaluateTools.get_MnDCG(top_num, ideal_map, test_map)
    
    print 'errCount =',errCount
    return MAP,MnDCG
    
    
    
    
    