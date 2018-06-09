#encoding=utf-8

import numpy
import theano
from theano import tensor
import interactiveGRUModelBatch

def interactiveGRULearning(model_options,tparams):
    trainingParis=tensor.tensor3('trainingParis',dtype='int64') # 3D tensor,shape=#(triples)*4*2
    sequences_matrix=tensor.matrix('sequences_matrix',dtype='int64') # shape=maxlen*#sequences
    masks_matrix=tensor.matrix('masks_matrix', dtype=theano.config.floatX) # @UndefinedVariable # 
    dependency_matrix=tensor.tensor3('dependency_matrix', dtype=theano.config.floatX) # @UndefinedVariable # shape=maxLen*#sequences*maxLen
    dependWeight_matrix=tensor.tensor3('dependWeight_matrix', dtype=theano.config.floatX) # @UndefinedVariable # shape=maxLen*#sequences*maxLen
    sequencesLen_vector=tensor.vector('sequencesLen_vector', dtype='int64') # shape=#sequence*0
    discountSeq_matrix=tensor.matrix('discountSeq_matrix', dtype=theano.config.floatX) # @UndefinedVariable # shape=maxlen*#sequence
    discountForEachNode_matrix=tensor.matrix('discountForEachNode_matrix', dtype=theano.config.floatX) # @UndefinedVariable # shape=maxLen*#sequence 
    
    wordsEmbeddings=tensor.matrix('wordsEmbeddings', dtype=theano.config.floatX)  # @UndefinedVariable # #(words)*word_dimension
    typesEmbeddings=tensor.matrix('typesEmbeddings', dtype=theano.config.floatX)  # @UndefinedVariable # shape=#(words)*type_dim
    
    groups_tensor=tensor.tensor3('groups_tensor', dtype=theano.config.floatX)  # @UndefinedVariable 
    
    # shape=seqNum*dim
    embs=interactiveGRUModelBatch.interactiveGRUModel(model_options, tparams, sequences_matrix, masks_matrix, dependency_matrix, dependWeight_matrix, discountSeq_matrix, discountForEachNode_matrix, wordsEmbeddings, typesEmbeddings)
    
    groups1=groups_tensor[0] 
    groups2=groups_tensor[1] 
    group_embs1=groups1[:,:,None]*embs + ((1.-groups1)*(-1000.))[:,:,None] # shape=tupleNum*seqNum*dim
    embs1=group_embs1.max(axis=1) # shape=tupleNum*dim
    group_embs2=groups2[:,:,None]*embs + ((1.-groups2)*(-1000.))[:,:,None] # shape=tupleNum*seqNum*dim
    embs2=group_embs2.max(axis=1) # shape=tupleNum*dim
    
    param=model_options['objective_function_param'] 
    lossVector=-tensor.log(tensor.nnet.sigmoid(param*(tensor.dot(embs1,tparams['w'])-tensor.dot(embs2,tparams['w'])))) # shape=tuplesNum
    loss=lossVector.sum()
    
    cost=loss
    
    cost+=model_options['decay']*(tparams['W_z'] ** 2).sum()
    cost+=model_options['decay']*(tparams['W_r'] ** 2).sum()
    cost+=model_options['decay']*(tparams['W_h'] ** 2).sum()
    cost+=model_options['decay']*(tparams['U_z'] ** 2).sum()
    cost+=model_options['decay']*(tparams['U_r'] ** 2).sum()
    cost+=model_options['decay']*(tparams['U_h'] ** 2).sum()
    cost+=model_options['decay']*(tparams['b_z'] ** 2).sum()
    cost+=model_options['decay']*(tparams['b_r'] ** 2).sum()
    cost+=model_options['decay']*(tparams['b_h'] ** 2).sum() 
    cost+=model_options['decay']*(tparams['type_eta'] ** 2).sum()
    cost+=model_options['decay']*(tparams['w'] ** 2).sum()
    return trainingParis, sequences_matrix, dependency_matrix, dependWeight_matrix, sequencesLen_vector, discountSeq_matrix, discountForEachNode_matrix, wordsEmbeddings, typesEmbeddings, masks_matrix, groups_tensor, cost

def discountModel(alpha,length):
    return tensor.exp(alpha*length*(-1))
    