

def mactrick(m,gamma_x):
    """ Use Cholesky decomposition to get first column of C

    Args:
        m (int) : length of time series
        gamma_x (float array) : first column of covariance matrix

    Returns:
        h : array of float with impulse response
    """

    U = np.zeros((m,2))  # C = U’*U
    V = np.zeros((m,2))
    h = np.zeros(m)

    #--- define the generators u and v
    U[:,0] = gamma_x/math.sqrt(gamma_x[0])
    V[1:m,0] = U[1:m,0]
    h[m-1] = U[m-1,0]

    k_old =0;
    k_new =1;
    for k in range(0,m-1):
        sin_theta = V[k+1,k_old]/U[k,k_old]
        cos_theta = math.sqrt(1.0-pow(sin_theta,2))
        U[k+1:m,k_new] = ( U[k:m-1,k_old] - sin_theta*V[k+1:m,k_old])/cos_theta
        V[k+1:m,k_new] = (-sin_theta*U[k:m-1,k_old] + V[k+1:m,k_old])/cos_theta
        h[m-1-k] = U[m-1,k_new]

        k_old = 1-k_old
        k_new = 1-k_new

    return h


def mactrick2(m,gamma_x):
    """ Use Cholesky decomposition to get first column of C

    Args:
        m (int) : length of time series
        gamma_x (float array) : first column of covariance matrix

    Returns:
        h : array of float with impulse response
    """

    U = np.zeros((m,2))  # C = U’*U
    V = np.zeros((m,2))
    h = np.zeros(m)

    #--- define the generators u and v
    U[:,0] = gamma_x/math.sqrt(gamma_x[0])
    V[1:m,0] = U[1:m,0]
    h[m-1] = U[m-1,0]

    k_old =0;
    k_new =1;
    for k in range(0,m-1):
        sin_theta = V[k+1,k_old]/U[k,k_old]
        cos_theta = math.sqrt(1.0-pow(sin_theta,2))
        U[k+1:m,k_new] = ( U[k:m-1,k_old] - sin_theta*V[k+1:m,k_old])/cos_theta
        V[k+1:m,k_new] = (-sin_theta*U[k:m-1,k_old] + V[k+1:m,k_old])/cos_theta
        h[m-1-k] = U[m-1,k_new]

        k_old = 1-k_old
        k_new = 1-k_new

    return h
