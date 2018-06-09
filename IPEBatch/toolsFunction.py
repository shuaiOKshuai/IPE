#encoding=utf-8

import numpy
import theano
from theano import tensor
from theano.ifelse import ifelse

def mapSortByValueDESC(map,top):
    if top>len(map): 
        top=len(map)
    items=map.items() 
    backitems=[[v[1],v[0]] for v in items]  
    backitems.sort(reverse=True) 
    e=[ backitems[i][1] for i in range(top)]  
    return e


def mapSortByValueASC(map,top):
    if top>len(map): 
        top=len(map)
    items=map.items() 
    backitems=[[v[1],v[0]] for v in items]  
    backitems.sort() 
    e=[ backitems[i][1] for i in range(top)]  
    return e


