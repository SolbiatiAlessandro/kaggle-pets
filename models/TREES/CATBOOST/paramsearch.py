from sortedcontainers import SortedList
import copy
import collections
import numpy as np
from itertools import product,chain
import pandas
from sklearn.model_selection import KFold
import catboost as cb

''' a class for doing grid search on a set of parameters provided in a dict. 'pdict' should be a dictionary like the following:
pdict = {'depth':[1,2], 'iterations':[250,100,500], 'thread_count':4}

when grid_search is called it will return an iterator that provides samples from the dictionary e.g.
{'depth':1, 'iterations':250, 'thread_count':4}
{'depth':2, 'iterations':250, 'thread_count':4}
{'depth':1, 'iterations':100, 'thread_count':4}
etc.
after calling an iteration of grid_search, you need to test the classifier and run 'register_result'
This will update the internal list of results, so that the next call to grid_search will use the best
parameters for all the parameters not currently being updated.

grid_search can be provided a list e.g. grid_search(['depth']) this will use the current best parameters for all
the other arguments and only search over 'depth'. You can then call e.g. grid_search(['iterations']) and it will use
the best depth found previously and cycle through all the 'iterations'. Searching incrementally can be much faster
than doing a full grid search, but may miss the global optimum. '''
class paramsearch:
    def __init__(self,pdict):    
        self.pdict = {}
        # if something is not passed in as a sequence, make it a sequence with 1 element
        #   don't treat strings as sequences
        for a,b in pdict.items():
            if isinstance(b, collections.Sequence) and not isinstance(b, str): self.pdict[a] = b
            else: self.pdict[a] = [b]
        # our results are a sorted list, so the best score is always the final element
        self.results = SortedList()       
                    
    def grid_search(self,keys=None):
        # do grid search on only the keys listed. If none provided, do all
        if keys==None: keylist = self.pdict.keys()
        else: keylist = keys
 
        listoflists = [] # this will be list of lists of key,value pairs
        for key in keylist: listoflists.append([(key,i) for i in self.pdict[key]])
        for p in product(*listoflists):
            # do any changes to the current best parameter set
            if len(self.results)>0: template = self.results[-1][1]
            else: template = {a:b[0] for a,b in self.pdict.items()}
            # if our updates are the same as current best, don't bother
            if self.equaldict(dict(p),template): continue
            # take the current best and update just the ones to change
            yield self.overwritedict(dict(p),template)
                              
    def equaldict(self,a,b):
        for key in a.keys(): 
            if a[key] != b[key]: return False
        return True            
                              
    def overwritedict(self,new,old):
        old = copy.deepcopy(old)
        for key in new.keys(): old[key] = new[key]
        return old            
    
    # save a (score,params) pair to results. Since 'results' is a sorted list,
    #   the best score is always the final element. A small amount of noise is added
    #   because sorted lists don't like it when two scores are exactly the same    
    def register_result(self,result,params):
        self.results.add((result+np.random.randn()*1e-10,params))    
        
    def bestscore(self):
        return self.results[-1][0]
        
    def bestparam(self):
        return self.results[-1][1]
        
