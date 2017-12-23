import scipy.io as sio
import lemo.support as S
mat = sio.loadmat('./input_data/Q.mat')
Q = mat['Q']
Q = S.variable(Q)