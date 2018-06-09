#encoding=utf-8
'''
'''

import dataProcessTools
import numpy
import theano
from theano import tensor
from theano import config
from collections import OrderedDict
import time
import six.moves.cPickle as pickle  # @UnresolvedImport
import interactiveGRULearning


# Set the random number generators' seeds for consistency
SEED = 123
numpy.random.seed(SEED)

def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)  # @UndefinedVariable

def gradientDescentGroup(learning_rate,grads,tparams,fourPairs, subPaths_matrix, subPaths_mask, subPaths_lens, wemb,cost):
    """
    """
    update=[(shared,shared-learning_rate*g) for g,shared in zip(grads,tparams.values())]
    func=theano.function([fourPairs, subPaths_matrix, subPaths_mask, subPaths_lens, wemb],cost,updates=update,on_unused_input='ignore')
    return func

def adadelta(lr, tparams, grads, trainingParis, sequences_matrix, dependency_matrix, dependWeight_matrix, sequencesLen_vector, discountSeq_matrix, discountForEachNode_matrix, wordsEmbs, typesEmbs, cost):
    """
    An adaptive learning rate optimizer
    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """
    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]
    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]
    f_grad_shared = theano.function([trainingParis, sequences_matrix, dependency_matrix, dependWeight_matrix, sequencesLen_vector, discountSeq_matrix, discountForEachNode_matrix, wordsEmbs, typesEmbs], cost, updates=zgup + rg2up,
                                    on_unused_input='ignore',
                                    name='adadelta_f_grad_shared')
    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2)) 
             for ru2, ud in zip(running_up2, updir)] 
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)] 
    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update

        
def ortho_weight(ndim):
    """
    """
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(config.floatX)  # @UndefinedVariable

def init_params_weight(row,column):
    """
    """
    W = numpy.random.rand(row, column) #
    W = W*2.0-1.0 # 
    return W.astype(config.floatX)  # @UndefinedVariable


def init_sharedVariables(options):
    """
    """
    print 'init shared Variables......'
    params = OrderedDict()
    W_z=init_params_weight(options['dimension'],options['word_dimension'])
    W_r=init_params_weight(options['dimension'],options['word_dimension'])
    W_h=init_params_weight(options['dimension'],options['word_dimension'])
    
    U_z=init_params_weight(options['dimension'],options['dimension'])
    U_r=init_params_weight(options['dimension'],options['dimension'])
    U_h=init_params_weight(options['dimension'],options['dimension'])
    
    b_z=numpy.random.rand(options['dimension'], ).astype(config.floatX)-0.5  # @UndefinedVariable # 
    b_r=numpy.random.rand(options['dimension'], ).astype(config.floatX)-0.5  # @UndefinedVariable # 
    b_h=numpy.random.rand(options['dimension'], ).astype(config.floatX)-0.5  # @UndefinedVariable # 
    
    type_eta=numpy.random.rand(options['type_dimension'], ).astype(config.floatX)-0.5  # @UndefinedVariable # 
    
    w=numpy.random.rand(options['dimension'], ).astype(config.floatX)-0.5  # @UndefinedVariable # 
    
    
    params['W_z']=W_z
    params['W_r']=W_r
    params['W_h']=W_h
    params['U_z']=U_z
    params['U_r']=U_r
    params['U_h']=U_h
    params['b_z']=b_z
    params['b_r']=b_r
    params['b_h']=b_h
    params['type_eta']=type_eta
    
    params['w']=w
    
    return params

def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.items():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams
    
def unzip(zipped):
    """
    """
    new_params = OrderedDict()
    for kk, vv in zipped.items():
        new_params[kk] = vv.get_value()
    return new_params



main_dir='D:/dataset/test/icde2016_metagraph/'
def interactiveGRUTraining(
                     trainingDataFile=main_dir+'facebook.splits/train.10/train_classmate_1', 
                     wordsEmbeddings=None, 
                     wordsEmbeddings_path=main_dir+'facebook/nodesFeatures', 
                     typesEmbeddings=None,
                     typesEmbeddings_path='',
                     word_dimension=22, 
                     type_dimension=20,
                     dimension=64, 
                     attention_dimension=12,
                     wordsSize=1000000, 
                     subpaths_map=None, 
                     subpaths_file=main_dir+'facebook/subpathsSaveFile',
                     sequences_map=None, 
                     sequences_file='',
                     maxlen_subpaths=1000, 
                     maxlen=100,  # Sequence longer then this get ignored 
                     batch_size=1, 
                     is_shuffle_for_batch=False, 
                     alpha=0.1, 
                     beta=0.1,
                     gamma=0.1,
                     objective_function_method='hinge-loss', 
                     objective_function_param=0,  
                     lrate=0.0001, 
                     max_epochs=10, 
                     
                     dispFreq=5, 
                     saveFreq=5, 
                     saveto=main_dir+'facebook/path2vec-modelParams.npz', 
                     
                     
                     decay=0.01, 
                     
                     ):
    """
    """
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
    
    
    trainingData,trainingPairsData=dataProcessTools.getTrainingData(trainingDataFile)
    allBatches=dataProcessTools.get_minibatches_idx(len(trainingData), batch_size, is_shuffle_for_batch)

    sequences_data=dataProcessTools.readAllSequencesFromFile(sequences_file)
    
    params=init_sharedVariables(model_options) 
    tparams=init_tparams(params) 
    print 'Generate models ......'
    trainingParis, sequences_matrix, dependency_matrix, dependWeight_matrix, sequencesLen_vector, discountSeq_matrix, discountForEachNode_matrix, wordsEmbs, typesEmbs, cost=interactiveGRULearning.interactiveGRULearning(model_options, tparams)
    
    print 'Generate gradients ......'
    grads=tensor.grad(cost,wrt=list(tparams.values()))
    print 'Using Adadelta to generate functions ......'
    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update=adadelta(lr, tparams, grads, trainingParis, sequences_matrix, dependency_matrix, dependWeight_matrix, sequencesLen_vector, discountSeq_matrix, discountForEachNode_matrix, wordsEmbs, typesEmbs, cost)
    
    print 'Start training models ......'
    best_p = None
    history_cost=[] 
    
    start_time = time.time() 
    print 'start time ==',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(start_time))
    uidx=0 # update index
    for eidx in range(max_epochs):
        for _, batch in allBatches: 
            uidx += 1
            trainingDataForBatch=[trainingData[i] for i in batch] 
            trainingPairsForBatch=[trainingPairsData[i] for i in batch] 
            trainingParis_data, sequences_matrix_data, dependency_matrix_data, dependWeight_matrix_data, sequencesLen_vector_data, discountSeq_matrix_data, discountForEachNode_matrix_data=dataProcessTools.prepareDataForTraining(trainingDataForBatch, trainingPairsForBatch, sequences_data, alpha, beta, gamma)
            cost=f_grad_shared(trainingParis_data, sequences_matrix_data, dependency_matrix_data, dependWeight_matrix_data, sequencesLen_vector_data, discountSeq_matrix_data, discountForEachNode_matrix_data, wordsEmbeddings, typesEmbeddings)
            f_update(lrate)
            if numpy.isnan(cost) or numpy.isinf(cost):
                print('bad cost detected: ', cost)
                return 
            if numpy.mod(uidx, dispFreq) == 0:
                print 'Epoch =', eidx, ',  Update =', uidx, ',  Cost =', cost
            if saveto and numpy.mod(uidx, saveFreq) == 0:
                print('Saving...')
                if best_p is not None: 
                    params = best_p
                else: 
                    params = unzip(tparams)
                numpy.savez(saveto, history_errs=history_cost, **params)
                pickle.dump(model_options, open('%s.pkl' % saveto, 'wb'), -1)
                print('Done')
    end_time = time.time() 
    print 'end time ==',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(end_time))
    print 'Training finished! Cost time == ', end_time-start_time,' s'
            
            
    