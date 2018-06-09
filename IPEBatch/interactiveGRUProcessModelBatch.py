#encoding=utf-8

import numpy
import theano
from theano import tensor
import toolsFunction
import interactiveGRUModelBatch

def interactiveGRUProcessModel(model_options,tparams):
    sequences_matrix=tensor.matrix('sequences_matrix',dtype='int64') # shape=maxlen*#sequences
    masks_matrix=tensor.matrix('masks_matrix', dtype=theano.config.floatX) # @UndefinedVariable # 
    dependency_matrix=tensor.tensor3('dependency_matrix', dtype=theano.config.floatX) # @UndefinedVariable # shape=maxLen*#sequences*maxLen
    dependWeight_matrix=tensor.tensor3('dependWeight_matrix', dtype=theano.config.floatX) # @UndefinedVariable # shape=maxLen*#sequences*maxLen
    sequencesLen_vector=tensor.vector('sequencesLen_vector', dtype='int64') # shape=#sequence*0
    discountSeq_matrix=tensor.matrix('discountSeq_matrix', dtype=theano.config.floatX) # @UndefinedVariable # shape=maxlen*#sequence
    discountForEachNode_matrix=tensor.matrix('discountForEachNode_matrix', dtype=theano.config.floatX) # @UndefinedVariable # shape=maxLen*#sequence 
    
    wordsEmbeddings=tensor.matrix('wordsEmbeddings',dtype=theano.config.floatX)  # @UndefinedVariable # #(words)*word_dimension
    typesEmbeddings=tensor.matrix('typesEmbeddings', dtype=theano.config.floatX)  # @UndefinedVariable # shape=#(words)*type_dim
    
    groups_tensor=tensor.matrix('groups_tensor', dtype=theano.config.floatX)  # @UndefinedVariable 
    
    # shape=seqNum*dim
    embs=interactiveGRUModelBatch.interactiveGRUModel(model_options, tparams, sequences_matrix, masks_matrix, dependency_matrix, dependWeight_matrix, discountSeq_matrix, discountForEachNode_matrix, wordsEmbeddings, typesEmbeddings)
    group_embs=groups_tensor[:,:,None]*embs + ((1.-groups_tensor)*(-1000))[:,:,None] 
    group_embs=group_embs.max(axis=1) # shape=tupleNum*dim
    values=tensor.dot(group_embs, tparams['w']) # shape=tupleNum
    
    return sequences_matrix, dependency_matrix, dependWeight_matrix, sequencesLen_vector, discountSeq_matrix, discountForEachNode_matrix, wordsEmbeddings, typesEmbeddings, masks_matrix, groups_tensor, values
        
def discountModel(alpha,length):
    return tensor.exp(alpha*length*(-1))