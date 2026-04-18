import numpy as np

class Markov:
    _graph = np.zeros((0,0)) 

    def __init__(self, n:int = 0):
        if n: 
            self._graph = np.zeros((n,n))

    def _change_size(self, n:int):
        size = self._graph.shape[0]
        s    = size + n

        new__graph = np.zeros((s, s))

        if s > size: s = size 

        new__graph[:s, :s] = self._graph[:s, :s]

        self._graph = new__graph        

    def add_state(self, n:int = 1):
        self._change_size(n)

    def remove_state(self, n:int = 1):
        self._change_size(-n)     

    def add_transition(self, init:int, targ:int, prob:float):
        if prob > 1 or prob < 0:
            raise Exception("Probability not valid")
        if init > self._graph.shape[0] or targ > self._graph.shape[0]:
            raise Exception("Origin or destiny node do not exist")

        self._graph[init][targ] = prob

    def __str__(self):
        return f"{self._graph}"
