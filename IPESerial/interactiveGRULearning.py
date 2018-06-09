#encoding=utf-8
'''

'''

import numpy
import theano
from theano import tensor
import interactiveGRUModel

def interactiveGRULearning(model_options,tparams):
    """
    """
    trainingParis=tensor.tensor3('trainingParis',dtype='int64') # 
    sequences_matrix=tensor.matrix('sequences_matrix',dtype='int64') # 
    dependency_matrix=tensor.tensor3('dependency_matrix', dtype=theano.config.floatX) # @UndefinedVariable #
    dependWeight_matrix=tensor.tensor3('dependWeight_matrix', dtype=theano.config.floatX) # @UndefinedVariable # 
    sequencesLen_vector=tensor.vector('sequencesLen_vector', dtype='int64') # 
    discountSeq_matrix=tensor.matrix('discountSeq_matrix', dtype=theano.config.floatX) # @UndefinedVariable # 
    discountForEachNode_matrix=tensor.matrix('discountForEachNode_matrix', dtype=theano.config.floatX) # @UndefinedVariable #
    
    wordsEmbeddings=tensor.matrix('wordsEmbeddings',dtype=theano.config.floatX)  # @UndefinedVariable #
    typesEmbeddings=tensor.matrix('typesEmbeddings', dtype=theano.config.floatX)  # @UndefinedVariable #
    
    def _generateEmbForSequence(index):
        length=sequencesLen_vector[index] 
        sequence=sequences_matrix[index][:length] 
        dependency=dependency_matrix[index][:length, :length]
        dependWeight=dependWeight_matrix[index][:length, :length] 
        discountSeq=discountSeq_matrix[index][:length]
        discountForEachNode=discountForEachNode_matrix[index][:length] 
        emb=interactiveGRUModel.interactiveGRUModel(model_options, tparams, sequence, dependency, dependWeight, discountSeq, discountForEachNode, wordsEmbeddings, typesEmbeddings)
        return emb
    
    def _generateEmbForTuple(start, end):
        rval,update=theano.scan(
                                _generateEmbForSequence,
                                sequences=tensor.arange(start,end),
                                )
        emb=rval.max(axis=0)
        return emb
    
    def _processTuple(tuplePair, lossSum): 
        start=tuplePair[0][0] 
        end=tuplePair[1][1]
        emb1=_generateEmbForTuple(start, end) 
        
        start=tuplePair[2][0] 
        end=tuplePair[3][1]
        emb2=_generateEmbForTuple(start, end) 
        
        loss=0
        param=model_options['objective_function_param'] 
        if model_options['objective_function_method']=='sigmoid': 
            loss=-tensor.log(tensor.nnet.sigmoid(param*(tensor.dot(emb1,tparams['w'])-tensor.dot(emb2,tparams['w'])))) # sigmoid
        else: 
            value=param + tensor.dot(emb2,tparams['w']) - tensor.dot(emb1,tparams['w'])
            loss=value*(value>0)
        
        return loss+lossSum
        
    
    rval,update=theano.scan(
                            _processTuple,
                            sequences=trainingParis, 
                            outputs_info=tensor.constant(0., dtype=theano.config.floatX), # @UndefinedVariable 
                            )
    cost=rval[-1]
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
    return trainingParis, sequences_matrix, dependency_matrix, dependWeight_matrix, sequencesLen_vector, discountSeq_matrix, discountForEachNode_matrix, wordsEmbeddings, typesEmbeddings, cost

def discountModel(alpha,length):
    """
    """
    return tensor.exp(alpha*length*(-1))
    