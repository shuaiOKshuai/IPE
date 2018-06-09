#encoding=utf-8
'''
'''

import numpy
import theano
from theano import tensor

def interactiveGRUModel(model_options, tparams, sequence, dependency, dependWeight, discountSeq, discountForEachNode, wemb, types):
    """
    """
    proj=wemb[sequence] # shape= length * word_dim
    typesInfo=types[sequence] # shape= length * type_dim
    
    # hArr的shape= length * dim
    
    
    def _step(index,hArr):
        
        # The following two lines are the method aggregating path heterogeneity, distance awareness and node heterogeneity.
        # If you don not need some steps, you can change the calculation of hi_sum.
        typesBias=tensor.nnet.sigmoid(tensor.dot(dependency[index][:, None]*typesInfo, tparams['type_eta'])) # shape=length*0
        hi_sum=(hArr * dependency[index][:,None] * discountForEachNode[:,None] * dependWeight[index][:,None] * typesBias[:,None]).max(axis=0) # shape=dim*0
        
        # If you only want to aggregate the outputs from its predecessors directly, you can do as follows
        # hi_sum= hArr * dependency[index][:,None]
        # or you want to take the distance awareness into consideration, you can do as follows
        # hi_sum= (hArr * dependency[index][:,None] * discountForEachNode[:,None])
        # Like the above examples, you can customize your own models.
        
        # shape=dim*0
        z_t=tensor.nnet.sigmoid(tensor.dot(tparams['W_z'], proj[index]) + tensor.dot(tparams['U_z'], hi_sum) + tparams['b_z'])
        # shape=dim*0
        r_t=tensor.nnet.sigmoid(tensor.dot(tparams['W_r'], proj[index]) + tensor.dot(tparams['U_r'], hi_sum) + tparams['b_r'])
        # shape=dim*0
        _h_t=tensor.tanh(tensor.dot(tparams['W_h'], proj[index]) + tensor.dot(tparams['U_h'], hi_sum*r_t) + tparams['b_h'])
        # shape=dim
        h_t=(1.-z_t)*hi_sum + z_t*_h_t # 
        
        hArr=tensor.set_subtensor(hArr[index], h_t)
        
        return hArr
    
    rval, update=theano.scan(
                             _step,
                             sequences=tensor.arange(sequence.shape[0]),
                            outputs_info=tensor.zeros((sequence.shape[0], model_options['dimension']), dtype=theano.config.floatX),# @UndefinedVariable 
                             )
    embs=rval[-1] 
    discountEmbs=embs*discountSeq[:,None] # shape=length*dim
    emb=discountEmbs.max(axis=0) # shape=dim*0
    return emb # 
    
def numpy_floatX(data):
    return numpy.asarray(data, dtype=theano.config.floatX)  # @UndefinedVariable
    