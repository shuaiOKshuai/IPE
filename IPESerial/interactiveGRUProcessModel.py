#encoding=utf-8
'''
'''

import numpy
import theano
from theano import tensor
import toolsFunction
import interactiveGRUModel


def interactiveGRUProcessModel(model_options,tparams):
    """
    """
    sequences_matrix=tensor.matrix('sequences_matrix',dtype='int64') 
    dependency_matrix=tensor.tensor3('dependency_matrix', dtype=theano.config.floatX) # @UndefinedVariable 
    dependWeight_matrix=tensor.tensor3('dependWeight_matrix', dtype=theano.config.floatX) # @UndefinedVariable 
    sequencesLen_vector=tensor.vector('sequencesLen_vector', dtype='int64') 
    discountSeq_matrix=tensor.matrix('discountSeq_matrix', dtype=theano.config.floatX) # @UndefinedVariable 
    discountForEachNode_matrix=tensor.matrix('discountForEachNode_matrix', dtype=theano.config.floatX) # @UndefinedVariable 
    
    wordsEmbeddings=tensor.matrix('wordsEmbeddings',dtype=theano.config.floatX)  # @UndefinedVariable 
    typesEmbeddings=tensor.matrix('typesEmbeddings', dtype=theano.config.floatX)  # @UndefinedVariable 
    
    def _generateEmbForSequence(index):
        length=sequencesLen_vector[index] 
        sequence=sequences_matrix[index][:length] 
        dependency=dependency_matrix[index][:length, :length] 
        dependWeight=dependWeight_matrix[index][:length, :length] 
        discountSeq=discountSeq_matrix[index][:length] 
        discountForEachNode=discountForEachNode_matrix[index][:length] 
        emb=interactiveGRUModel.interactiveGRUModel(model_options, tparams, sequence, dependency, dependWeight, discountSeq, discountForEachNode, wordsEmbeddings, typesEmbeddings)
        return emb
    
    rval,update=theano.scan(
                            _generateEmbForSequence, 
                            sequences=tensor.arange(sequences_matrix.shape[0]), 
                            )
    emb=rval.max(axis=0)
    
    value=tensor.dot(emb,tparams['w'])
    
    return sequences_matrix, dependency_matrix, dependWeight_matrix, sequencesLen_vector, discountSeq_matrix, discountForEachNode_matrix, wordsEmbeddings, typesEmbeddings, value
        
def discountModel(alpha,length):
    """
    """
    return tensor.exp(alpha*length*(-1))