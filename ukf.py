import numpy as np
import Quaternion


def sigma_points(P,Q):
    n = P.shape[0]
    s = np.linalg.cholesky(Q)
    s = s*np.sqrt(0.1*n)
    W = np.hstack((s,-s))
    return W

def WtoX(W,previous_state_x):
    qkminus1 = previous_state_x[:4]
    # W is 6*12
    qwvector = W
    #omegakminus1 = previous_state_x[4:]
    #omegaW = W[3:,:]
    X = np.zeros((7,4))
    for i in range(6):
        qw = Quaternion.vectoquat(qwvector[:,i])
        qkminus1qw = Quaternion.multiply_quaternion(qkminus1,qw)
        #omegakmius1plusomegaW = omegakminus1+omegaW[:,i]
        #X[:,i] = np.hstack((qkminus1qw, omegakmius1plusomegaW))
        X[i,:] = qkminus1qw
    X[6,:] = previous_state_x
    return X

def XtoY(X, deltat, omegak):
    Y = np.zeros((7,4))
    norm_omegak = np.linalg.norm(omegak)
    if norm_omegak == 0:
        qdelta = np.array([1,0,0,0])
    else:
        edelta = omegak*deltat
        qdelta = Quaternion.vectoquat(edelta)
    for i in range(7):
        Y[i,:] = Quaternion.multiply_quaternion(X[i,:], qdelta)
    return Y

def meanY(Y, previous_state_x):
    for j in range(100):
        evec = np.zeros((7,3))
        for i in range(7):
            prev_inv = Quaternion.inverse_quaternion(previous_state_x)
            ei = Quaternion.multiply_quaternion(Y[i,:], prev_inv)
            ei = Quaternion.normalize_quaternion(ei)
            eve = Quaternion.quattovec(ei)
            if np.linalg.norm(eve)==0:
                evec[i,:] = np.zeros((3,))
            else:
                evec[i,:] = (-np.pi + np.remainder(np.linalg.norm(eve)+np.pi,2*np.pi))/np.linalg.norm(eve)*eve
                #evec[i,:] = eve/np.linalg.norm(eve)
        ei_avg = np.mean(evec, axis=0)

        previous_state_x = Quaternion.normalize_quaternion(Quaternion.multiply_quaternion(Quaternion.vectoquat(ei_avg), previous_state_x))
        if np.linalg.norm(ei_avg) < 0.01:
            # print(j)
            break

    return previous_state_x, evec

def calculatepk(W):
    pk = np.zeros((3,3))
    for i in range(7):
        pk = pk + W[i,:][:,None]*W[i,:][None,:]
    pk = pk/7
    return pk

def YtoZ(Y):
    Z = np.zeros((7,3))
    quatg = np.array([0,0,0,1])
    for i in range(7):
        qk = Y[i,:]
        qkinv = Quaternion.inverse_quaternion(qk)
        prod = Quaternion.multiply_quaternion(Quaternion.multiply_quaternion(qkinv,quatg),qk)
        Z[i,:] = Quaternion.quattovec(prod)
    #print(Z.T)
    return Z

def pzz(Z, zmean):
    pzz = np.zeros((3,3))
    Z_diff = Z - zmean
    for i in range(7):
        pzz = pzz + Z_diff[i,:][:,None]*Z_diff[i,:][None,:]
    pzz = pzz/7
    return pzz

def pvv(pzz, R):
    return pzz + R

def pxz(W, Z, zmean):
    pxz = np.zeros((3,3))
    Z_diff = Z-zmean
    for i in range(7):
        pxz = pxz + W[i,:][:,None]*Z_diff[i,:][None,:]
    pxz = pxz/7
    return pxz

def kalman_gain(pxz, pvv):
    return np.dot(pxz,np.linalg.inv(pvv))

def innovation(zmean, actual):
    return actual/np.linalg.norm(actual) - zmean

def update_state(pk,kk,v, ymean):
    return Quaternion.multiply_quaternion(Quaternion.vectoquat(np.dot(kk,v)),ymean)

def update_cov(pk, kk, pvv):
    return pk - np.dot(np.dot(kk,pvv),kk.T)

# if __name__ == '__main__':
#
#     P = np.eye(6,6)
#     Q = np.ones((6,6))
#     R = np.ones((6,6))
#     W = sigma_points(P,Q)
#     #print(W)
#     previous_state_x = 0.5*np.ones((7,))
#     X = WtoX(W, previous_state_x)
#     # for i in range(X.shape[1]):
#     #     print(X[i,:])
#     Y = XtoY(X, 0.01)
#     ymean, rwvector = meanY(Y)
#     # print(ymean)
#     W = YtoW(Y, ymean)
#     # print(W)
#     # print(ymean)
#     pk = calculatepk(W)
#     Z = YtoZ(Y,np.array([0,0,9.8]))
#     # for i in range(Z.shape[1]):
#     #     print(Z[:,i])
#     zmean  = np.mean(Z, axis = 1)
#     #vdummy is the actual value
#     actualdummy = np.array([10, 10, 10, 1, 1, 1])
#     v = innovation(zmean, actualdummy)
#     pzz = pzz(Z, zmean)
#     pvv = pvv(pzz, R)
#     pxz = pxz(W, Z, zmean)
#     kk = kalman_gain(pxz, pvv)
#     update = update(kk, v)
#     update_fin = np.zeros((7,))
#     update_fin[:4] = Quaternion.vectoquat(update[:3])
#     update_fin[4:] = update[3:]
#     new_state_quat = ymean + update_fin
#     pk = update_cov(pk, kk, pvv)
#     # print(pk)
#     print(new_state_quat)
#     # print(kk)
#     # for i in range(pzz.shape[1]):
#     #     print(pzz[:,i])
#     # print(Z)
