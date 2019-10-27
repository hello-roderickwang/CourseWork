import sys
import numpy as np
import matplotlib.pyplot as plt

def kalman_filter(x, P, measurement, R, motion, Q, F, H):
    # Update x, P based on measurement m
    print('size of measurement:', len(measurement))
    print('size of x:', len(x), '*', len(x[0]))
    y = np.matrix(measurement).T - H * x
    S = H * P * H.T + R
    K = P * H.T * S.I
    x = x + K * y
    I = np.matrix(np.eye(F.shape[0]))
    P = (I - K * H) * P
    # Predict x, P based on motion
    x = F * x + motion
    P = F * P * F.T + Q
    return x, P

if __name__ == "__main__":
    
    # Retrive file name for input data
    if(len(sys.argv) < 5):
        print ("Four arguments required: python kalman2d.py [datafile] [x1] [x2] [lambda]")
        exit()
    
    filename = sys.argv[1]
    x10 = float(sys.argv[2])
    x20 = float(sys.argv[3])
    scaler = float(sys.argv[4])

    # Read data
    lines = [line.rstrip('\n') for line in open(filename)]
    data = []
    for line in range(0, len(lines)):
        data.append(list(map(float, lines[line].split(' '))))

    # Print out the data
    print ("The input data points in the format of 'k [u1, u2, z1, z2]', are:")
    for it in range(0, len(data)):
        print (str(it + 1) + ": ", end='')
        print (*data[it])

    # Kalman Filter
    x = np.matrix(np.array(data)[:,:2])
    P = np.matrix(np.eye(2)) * scaler
    N = len(data)
    w = np.random.randn(len(x))/1000000
    v = np.random.randn(len(x))/1000000
    true_x = np.array(np.array(data)[:,0:1])
    # true_y = np.array(np.array(data)[:,1])
    observed_x = np.array(np.array(data)[:,2])
    observed_y = np.array(np.array(data)[:,3])
    print('observed_x:', observed_x)
    print('v:', v)
    for j in range(len(x)):
        observed_x[j] = observed_x[j] + v[j]
        observed_y[j] = observed_y[j] + v[j]
    plt.plot(observed_x, observed_y, 'ro')
    result = []
    R = (0.01*0.02)-(0.005*0.005)
    Q = np.matrix('''
        0.0001 0.00002;
        0.00002 0.0001''')
    F = np.matrix(np.eye(2))
    H = np.matrix(np.eye(2))
    i = 0
    observed = np.array([observed_x, observed_y])
    print('observed:', observed)
    print('size of observed:', len(observed), '*', len(observed[0]))
    for m in range(len(x)):
        if m == 0:
            x = x.T
        x, P = kalman_filter(x, P, observed[:, m], R, w[i], Q, F, H)
        i = i + 1
        result.append((x[:2]).tolist())
    kalman_x, kalman_y = zip(*result)
    plt.plot(kalman_x, kalman_y, 'g-')
    plt.show()