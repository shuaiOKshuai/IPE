#encoding=utf-8

import numpy
import theano
from theano import tensor

def interactiveGRUModel(model_options, tparams, sequencesM, masks, dependencies, dependWeight, discountSeq, discountForEachNode, wemb, types):
    proj=wemb[sequencesM] # shape= len * num * dim
    typesInfo=types[sequencesM.T] 
    discountForEachNode_T=discountForEachNode.T 
    discountSeq_T=discountSeq.T 
    masks_T=masks.T 
    
    def _step(index,hArr):
        
        # The following two lines are the method aggregating path heterogeneity, distance awareness and node heterogeneity.
        # If you don not need some steps, you can change the calculation of hi_sum.
        
        # type bias sigmoid(types dot eta)
        #                                        shape=num*length shape=num*length*type_dim  shape=type_dim
        typesBias=tensor.nnet.sigmoid(tensor.dot(dependencies[index][:,:,None]*typesInfo, tparams['type_eta'])) # shape=num*length
        # shape=num*len*gru_dim    shape=num*len            shape=num*len                   shape=num*len                  shape=num*len
        hi_sum=(hArr * dependencies[index][:,:,None] * discountForEachNode_T[:,:,None] * dependWeight[index][:,:,None] * typesBias[:,:,None]).max(axis=1) # shape=num*gru_dim
        
        # If you only want to aggregate the outputs from its predecessors directly, you can do as follows
        # hi_sum= hArr * dependencies[index][:,:,None]
        # or you want to take the distance awareness into consideration, you can do as follows
        # hi_sum= (hArr * dependencies[index][:,:,None] * discountForEachNode_T[:,:,None])
        # Like the above examples, you can customize your own models.
        
        # shape=num*gru_dim               shape=num*dim  shape=dim*gru_dim     shape=num*gru_dim shape=gru_dim*gru_dim
        z_t=tensor.nnet.sigmoid(tensor.dot(proj[index], tparams['W_z']) + tensor.dot(hi_sum, tparams['U_z']) + tparams['b_z']) # shape=num*gru_dim
        # shape=num*gru_dim           
        r_t=tensor.nnet.sigmoid(tensor.dot(proj[index], tparams['W_r']) + tensor.dot(hi_sum, tparams['U_r']) + tparams['b_r']) # shape=num*gru_dim
        # shape=num*gru_dim                                                shape=num*gru_dim
        _h_t=tensor.tanh(tensor.dot(proj[index], tparams['W_h']) + tensor.dot(hi_sum*r_t, tparams['U_h']) + tparams['b_h']) # shape=num*gru_dim
        # shape=num*gru_dim
        h_t=(1.-z_t)*hi_sum + z_t*_h_t 
        
        hArr=tensor.set_subtensor(hArr[:,index,:], h_t)
        
        return hArr
    
    rval, update=theano.scan(
                             _step,
                             sequences=tensor.arange(sequencesM.shape[0]),
                            outputs_info=tensor.zeros((sequencesM.shape[1], sequencesM.shape[0], model_options['dimension']), dtype=theano.config.floatX),# @UndefinedVariable 
                             )
    embs=rval[-1] 
    discountEmbs=embs*discountSeq_T[:,:,None] # shape=num*length*dim
    # shape=num*length*dim
    discountEmbs=discountEmbs+((1.-masks_T)*(-10000.))[:,:,None]
    embs=discountEmbs.max(axis=1) # shape=num*dim
    return embs 
    
def numpy_floatX(data):
    return numpy.asarray(data, dtype=theano.config.floatX)  # @UndefinedVariable
    