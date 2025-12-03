import numpy as np

def initialize_const(K, B: list[int] = [], fs: list[int] = [], Pt = [], t: list[int] = [], Nr: list[int] = [], Nt:list[int] = [], N:list[int] = [], L: list[int] = [], snr_db: list[int] = [], desired_CNR: list[int] = [], epsilon:list[int] = []):                   
    
    K = K
    
    NR = Nr # no. of receiver antennas at BS
    assert len(Nt) == K
    NT = np.array(Nt) # no. of transmit antennas 
    # n = LT
    n = np.array(N) # blocklength/channel uses of each user: (K,)
    L = np.array(L)
    assert all(n%L == np.array([0]*K)) # check if n is divisble by L
    T = n//L # Channel uses per coherant interval.
    
    dk = np.min((NR, NT), axis = 0)
    assert len(Pt) == K
    Pt = np.array(Pt)
    assert len(fs) == K
    fs = np.array(fs)
            
    assert len(snr_db) == K
    snr_db = np.array(snr_db)
    
    assert len(n) == len(fs)
    latency = n/fs
    
    assert len(B) == K
    B = np.array(B)
    
    assert len(epsilon) == K
    epsilon = np.array(epsilon)
    
    constants = {"K":K, "Pt": Pt, "fs": fs, "NR":NR, "NT": NT,"n":n, "L": L, "T": T, "snr_db":snr_db, "desired_CNR":desired_CNR, "latency": latency, "B": B, "epsilon": epsilon, "dk": dk}
    return constants