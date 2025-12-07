import numpy as np

# ------------------ Utility to initialize system constants --------------------
def initialize_system_params(K, B: list[int] = [], fs: list[int] = [], Pt = [], Nr: list[int] = [], Nt:list[int] = [], N:list[int] = [], T: list[int] = [], snr_db: list[float] = [], desired_CNR: list[float] = [], epsilon:list[float] = []):                   
    
    K = K
    
    NR = Nr # no. of receiver antennas at BS
    assert len(Nt) == K
    NT = np.array(Nt) # no. of transmit antennas 
    # n = LT
    n = np.array(N) # blocklength/channel uses of each user: (K,)
    T = np.array(T) # blocks over which channel is constant
    assert all(n%T == np.array([0]*K)) # check if n is divisble by L
    L = n//T # NO. of coherant interval.
    
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
    
    dk = np.min((NR, NT), axis = 0)
    
    constants = {"K":K, "Pt": Pt, "fs": fs, "NR":NR, "NT": NT,"n":n, "L": L, "T": T, "snr_db":snr_db, "desired_CNR":desired_CNR, "latency": latency, "B": B, "epsilon": epsilon, "dk": dk}
    return constants
