import scipy.io as sio
import lemo.support as S
mat = sio.loadmat('./input_data/A.mat')
A = mat['A']
A = S.variable(A)

mat = sio.loadmat('./input_data/b.mat')
b = mat['b']
b = S.variable(b.reshape(-1))

mat = sio.loadmat('./input_data/P0.mat')
P0 = mat['P0']
P0 = S.variable(P0)

mat = sio.loadmat('./input_data/P1.mat')
P1 = mat['P1']
P1 = S.variable(P1)

mat = sio.loadmat('./input_data/P2.mat')
P2 = mat['P2']
P2 = S.variable(P2)

mat = sio.loadmat('./input_data/q0.mat')
q0 = mat['q0']
q0 = S.variable(q0.reshape(-1))

mat = sio.loadmat('./input_data/q1.mat')
q1 = mat['q1']
q1 = S.variable(q1.reshape(-1))

mat = sio.loadmat('./input_data/q2.mat')
q2 = mat['q2']
q2 = S.variable(q2.reshape(-1))

mat = sio.loadmat('./input_data/r1.mat')
r1 = mat['r1']
r1 = S.variable(r1.reshape(-1))

mat = sio.loadmat('./input_data/r2.mat')
r2 = mat['r2']
r2 = S.variable(r2.reshape(-1))










