
import numpy as np


def line_search():

def wolf():

def update(Bk, pk, qk):
   #Broyden family
   sk = Bk @ qk
   tk = sk.T @ qk
   miuk = pk.T @ qk
   vk = (1/miuk) * pk - (1/tk) * sk
   return Bk + (1/miuk)*pk@pk.T -(1/tk)*sk@sk.T + tk*vk@vk.T #return Bk+1

    #Secant equation
    #v = pk -  Bk @ qk
    #return Bk - (v @ v.T)/(v @ qk) #B(k+1)

def BFGS():
