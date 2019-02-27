import numpy as np

def add_quaternion(q1, q2):
    return (q1+q2)

def multiply_quaternion(q1, q2):
    ans = np.zeros((4,)).astype('float64')
    ans[0] = q2[0]*q1[0] - q2[1]*q1[1] - q2[2]*q1[2] - q2[3]*q1[3]
    ans[1] = q2[0]*q1[1] + q2[1]*q1[0] - q2[2]*q1[3] + q2[3]*q1[2]
    ans[2] = q2[0]*q1[2] + q2[1]*q1[3] + q2[2]*q1[0] - q2[3]*q1[1]
    ans[3] = q2[0]*q1[3] - q2[1]*q1[2] + q2[2]*q1[1] + q2[3]*q1[0]
    return ans

def scale_quaternion(q1,s):
    ans = np.zeros((4,))
    ans[0] = q1[0]*s
    ans[1] = q1[1]*s
    ans[2] = q1[2]*s
    ans[3] = q1[3]*s
    return ans

def normalize_quaternion(q1):
    return q1/np.linalg.norm(q1)

def conjugate_quaternion(q1):
    return -q1

def divide_quaternion(q1,q2):
    ans = np.zeros((4,))
    norm = q2[0]**2+q2[1]**2+q2[2]**2+q2[3]**2
    ans[0] = (q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3])/norm
    ans[1] = (q1[1]*q2[0] - q1[0]*q2[1] - q1[3]*q2[2] + q1[2]*q2[3])/norm
    ans[2] = (q1[2]*q2[0] + q1[3]*q2[1] - q1[0]*q2[2] - q1[1]*q2[3])/norm
    ans[3] = (q1[3]*q2[0] - q1[2]*q2[1] + q1[1]*q2[2] - q1[0]*q2[3])/norm
    ans = ans
    return ans

def inverse_quaternion(q1):
    ans = np.zeros((4,))
    ans[0] = q1[0]
    ans[1:] = -q1[1:]
    return ans/np.linalg.norm(q1)

def quattovec(q1):
    if q1[0]==1:
        return np.zeros((3,))
    ans = np.zeros((3,))
    theta = 2*np.arccos(q1[0])
    ans = (theta/np.sin(theta/2))*q1[1:]
    return ans

def rottheta(q1, q2):
    q12 = normalize_quaternion(multiply_quaternion(q2, inverse_quaternion(q1)))
    theta = 2*np.arccos(q12[0])
    return q12, theta

def vectoquat(v):
    v = v/2
    norm = np.linalg.norm(v)
    if norm==0:
        return np.array([1,0,0,0])
    quart = np.zeros((4,))
    quart[0] = np.cos(norm)
    quart[1] = (v[0]/norm)*np.sin(norm)
    quart[2] = (v[1]/norm)*np.sin(norm)
    quart[3] = (v[2]/norm)*np.sin(norm)
    return quart

def quattorot(q):
    q = normalize_quaternion(q)
    qrot = np.zeros((3,3))
    qrot[0,1] = -q[3]
    qrot[0,2] = q[2]
    qrot[1,2] = -q[1]
    qrot[1,0] = q[3]
    qrot[2,0] = -q[2]
    qrot[2,1] = q[1]

    rot = np.identity(3) + 2*np.dot(qrot,qrot) + 2*np.array(q[0])*qrot
    return rot
