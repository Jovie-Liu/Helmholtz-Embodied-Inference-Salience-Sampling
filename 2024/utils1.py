import numpy as np

def parameter_initialization(init_type,n_dz):
    """
    Arguments:
    init_type -- "zero" or "random", "zero" assigns 0 to all parameters, "random" samples from standard Gaussian
    n_dz -- number of neurons for each layer, numpy array of shape (n+1,m), where m is the number of instantiation layers, 
    n is the maximum number of inserted layers between adjacent instantiation layers
    
    Returns:
    Phi -- Recognition parameter set, Python dictionary of length m-1 with each key-value pair being a parameter matrix of 
    shape (n_z{i+1}, n_zi+1), where the last column represents bias b's
    Theta -- Generative parameter set, Python dictionary of length m with each key-value pair being a parameter matrix of 
    shape (n_z{i}, n_z{i+1}+1), where the last column represents bias b's
    """
    Phi = {}
    Theta = {}
    m = n_dz.shape[1]
    
    if init_type == "zero":
        for i in range(m-1):
            n = np.where(n_dz[1:,i] != 0)[0].size  # number of inserted layers between i and i+1
            if n == 0:
                Phi["Phi_" + str(i) + str(i+1)] = np.zeros((n_dz[0,i+1],n_dz[0,i]+1))
                Theta["Theta_" + str(i+1) + str(i)] = np.zeros((n_dz[0,i],n_dz[0,i+1]+1))
            else:
                for j in range(1,n+1):
                    Phi["Phi_" + str(i) + str(i+1) + "_" + str(j)] = np.zeros((n_dz[j,i],n_dz[j-1,i]+1))
                    Theta["Theta_" + str(i+1) + str(i) + "_" + str(j)] = np.zeros((n_dz[j-1,i],n_dz[j,i]+1))
                Phi["Phi_" + str(i) + str(i+1) + "_" + str(j+1)] = np.zeros((n_dz[0,i+1],n_dz[j,i]+1))
                Theta["Theta_" + str(i+1) + str(i) + "_" + str(j+1)] = np.zeros((n_dz[j,i],n_dz[0,i+1]+1))
    elif init_type == "random":
        for i in range(m-1):
            n = np.where(n_dz[1:,i] != 0)[0].size
            if n == 0:
                Phi["Phi_" + str(i) + str(i+1)] = np.random.randn(n_dz[0,i+1],n_dz[0,i]+1)
                Theta["Theta_" + str(i+1) + str(i)] = np.random.randn(n_dz[0,i],n_dz[0,i+1]+1)
            else:
                for j in range(1,n+1):
                    Phi["Phi_" + str(i) + str(i+1) + "_" + str(j)] = np.random.randn(n_dz[j,i],n_dz[j-1,i]+1)
                    Theta["Theta_" + str(i+1) + str(i) + "_" + str(j)] = np.random.randn(n_dz[j-1,i],n_dz[j,i]+1)
                Phi["Phi_" + str(i) + str(i+1) + "_" + str(j+1)] = np.random.randn(n_dz[0,i+1],n_dz[j,i]+1)
                Theta["Theta_" + str(i+1) + str(i) + "_" + str(j+1)] = np.random.randn(n_dz[j,i],n_dz[0,i+1]+1)
    
    else:
        raise Exception("Wrong Init Type")
        
    return Phi, Theta

def sigmoid(x):
    y = 1/(1+np.exp(-x))
    return y

def wake_sample(n_dz,d0,value_set,Phi,activation_type,bias):
    """
    Arguments:
    n_dz -- number of neurons for each layer, numpy array of shape (n+1,m), where m is the number of instantiation layers, 
    n is the maximum number of inserted layers between adjacent instantiation layers
    d0 -- input pattern, numpy array of shape (n_d, 1)
    value_set -- list or array [a,b], where a is the positive outcome and b is the negative outcome of a Bernoulli experiment
    Phi -- Recognition parameter set, Python dictionary of length m-1 with each key-value pair being a parameter matrix of 
    shape (n_z{i+1}, n_zi+1), where the last column represents bias b's
    activation_type -- we provide 2 choices of activation functions: tanh(x) and sigmoid(x)
    bias -- list or array [instantiation bias, MLP bias], taking binary value in {True, False}. For example, [False,True] means 
    no instantiation bias but has MLP bias
    
    Returns:
    Alpha_Q -- assignment of each neuron (binary value), Python dictionary of length m-1 with each key-value pair being 
    a numpy array of shape (n_dz[0,i], 1),i = 0,...m-1
    """
    
    m = n_dz.shape[1]
    S = d0  # assignment of each layer
    Alpha_Q = {"z0":d0}
    inst_bias = bias[0]
    mlp_bias = bias[1]
    a = value_set[0]
    b = value_set[1]
    
    for i in range(m-2):
        n = np.where(n_dz[1:,i] != 0)[0].size  # number of inserted layers between i and i+1
        if n == 0:
            phi = Phi["Phi_" + str(i) + str(i+1)]
            if inst_bias == True:
                q = sigmoid(np.matmul(phi,np.append(S,[[1]], axis=0)))
            else:
                q = sigmoid(np.matmul(phi[:,:-1],S))
            S = ((q > np.random.rand(len(q),1)).astype(int))*(a-b)+b   # rejection sampling as a or b
            Alpha_Q["z"+str(i+1)] = S
        else:
            g = S
            for j in range(1,n+1):
                phi = Phi["Phi_" + str(i) + str(i+1) + "_" + str(j)]
                if activation_type == "sigmoid":
                    if mlp_bias == True:
                        g = sigmoid(np.matmul(phi,np.append(g,[[1]], axis=0)))*(a-b)+b  # scale to [b,a]
                    else:
                        g = sigmoid(np.matmul(phi[:,:-1],g))*(a-b)+b
                elif activation_type == "tanh":
                    if mlp_bias == True:
                        g = np.tanh(np.matmul(phi,np.append(g,[[1]], axis=0)))*(a-b)/2+(a+b)/2 # scale to [b,a]
                    else:
                        g = np.tanh(np.matmul(phi[:,:-1],g))*(a-b)/2+(a+b)/2
                    
            phi = Phi["Phi_" + str(i) + str(i+1) + "_" + str(j+1)]
            if inst_bias == True:
                q = sigmoid(np.matmul(phi,np.append(g,[[1]], axis=0)))
            else:
                q = sigmoid(np.matmul(phi[:,:-1],g))
            S = ((q > np.random.rand(len(q),1)).astype(int))*(a-b)+b
            Alpha_Q["z"+str(i+1)] = S
    Alpha_Q["z"+str(m-1)] = [[1]]
        
    return Alpha_Q

def sleep_sample(n_dz,value_set,Theta,activation_type,bias):
    """
    Arguments:
    n_dz -- number of neurons for each layer, numpy array of shape (n+1,m), where m is the number of instantiation layers, 
    n is the maximum number of inserted layers between adjacent instantiation layers
    value_set -- list or array [a,b], where a is the positive outcome and b is the negative outcome of a Bernoulli experiment
    Theta -- Generative parameter set, Python dictionary of length m with each key-value pair being a parameter matrix of 
    shape (n_z{i}, n_z{i+1}+1), where the last column represents bias b's
    activation_type -- we provide 2 choices of activation functions: tanh(x) and sigmoid(x)
    bias -- list or array [instantiation bias, MLP bias,data bias], taking binary value in {True, False}. For example, 
    [False,True,True] means no instantiation bias but has MLP bias and data bias
    
    Returns:
    Alpha_P -- assignment of each neuron (binary value), Python dictionary of length m-1 with each key-value pair being 
    a numpy array of shape (n_dz[0,i], 1),i = m-1,...,0
    """
    m = n_dz.shape[1]
    inst_bias = bias[0]
    mlp_bias = bias[1]
    data_bias = bias[2]
    a = value_set[0]
    b = value_set[1]
    S = [[1]]
    Alpha_P = {"z"+str(m-1):S}
    
    
    for i in range(m-1,0,-1):
        n = np.where(n_dz[1:,i-1] != 0)[0].size  # number of inserted layers between i and i-1
        if n == 0:
            theta = Theta["Theta_" + str(i) + str(i-1)]
            if i > 1:
                if inst_bias == True:
                    p = sigmoid(np.matmul(theta,np.append(S,[[1]], axis=0))) #
                else:
                    p = sigmoid(np.matmul(theta[:,:-1],S))
                S = ((p > np.random.rand(len(p),1)).astype(int))*(a-b)+b   # rejection sampling as a or b
                Alpha_P["z"+str(i-1)] = S
            else:
                if data_bias == True:
                    p = sigmoid(np.matmul(theta,np.append(S,[[1]], axis=0)))
                else:
                    p = sigmoid(np.matmul(theta[:,:-1],S))
                S = ((p > np.random.rand(len(p),1)).astype(int))*(a-b)+b   # rejection sampling as a or b
                Alpha_P["z"+str(i-1)] = S
        else:
            g = S
            for j in range(n+1,1,-1):
                theta = Theta["Theta_" + str(i) + str(i-1) + "_" + str(j)]
                if activation_type == "sigmoid":
                    if mlp_bias == True:
                        g = sigmoid(np.matmul(theta,np.append(g,[[1]], axis=0)))*(a-b)+b  # scale to [b,a]
                    else:
                        g = sigmoid(np.matmul(theta[:,:-1],g))*(a-b)+b
                elif activation_type == "tanh":
                    if mlp_bias == True:
                        g = np.tanh(np.matmul(theta,np.append(g,[[1]], axis=0)))*(a-b)/2+(a+b)/2 # scale to [b,a]
                    else:
                        g = np.tanh(np.matmul(theta[:,:-1],g))*(a-b)/2+(a+b)/2
                    
            theta = Theta["Theta_" + str(i) + str(i-1) + "_" + str(j-1)]
            
            if i > 1:
                if inst_bias == True:
                    p = sigmoid(np.matmul(theta,np.append(g,[[1]], axis=0)))
                else:
                    p = sigmoid(np.matmul(theta[:,:-1],g))
                S = ((p > np.random.rand(len(p),1)).astype(int))*(a-b)+b   # rejection sampling as a or b
                Alpha_P["z"+str(i-1)] = S
            else:
                if data_bias == True:
                    p = sigmoid(np.matmul(theta,np.append(g,[[1]], axis=0)))
                else:
                    p = sigmoid(np.matmul(theta[:,:-1],g))
                S = ((p > np.random.rand(len(p),1)).astype(int))*(a-b)+b   # rejection sampling as a or b
                Alpha_P["z"+str(i-1)] = S
            
    return Alpha_P

def multi_Bernoulli_update(x,y,parameter_set,lr,value_set,activation_type,bias):
    """
    Arguments:
    x -- input instantiation layer, numpy array of shape (n,1)
    y -- target instantiation layer, numpy array of shape (m,1)
    parameter_set -- parameters from x to y. Python dictionary of length l+1, l is the number of inserted layers. 
    The keys are ordered sequentially from layer x to y.
    lr -- learning rate, decimals
    value_set -- list or array [a,b], where a is the positive outcome and b is the negative outcome of a Bernoulli experiment
    activation_type -- we provide 2 choices of activation functions: tanh(x) and sigmoid(x)
    bias -- list or array [instantiation (data) bias, MLP bias], taking binary value in {True, False}. For example, 
    [False,True] means no instantiation bias but has MLP (data) bias
    
    Returns:
    parameter_set -- updated parameters
    loss -- value of loss function before updating, a number
    """
    
    inst_bias = bias[0]
    mlp_bias = bias[1]
    a = value_set[0]
    b = value_set[1]
    l = len(parameter_set)
    keys = [*parameter_set]
    G = {'z0': x}
    g = x
    
    for i in range(l-1):
        phi = parameter_set[keys[i]]
        if activation_type == "sigmoid":
            if mlp_bias == True:
                g = sigmoid(np.matmul(phi,np.append(g,[[1]], axis=0)))*(a-b)+b  # scale to [b,a]
            else:
                g = sigmoid(np.matmul(phi[:,:-1],g))*(a-b)+b
        elif activation_type == "tanh":
            if mlp_bias == True:
                g = np.tanh(np.matmul(phi,np.append(g,[[1]], axis=0)))*(a-b)/2+(a+b)/2 # scale to [b,a]
            else:
                g = np.tanh(np.matmul(phi[:,:-1],g))*(a-b)/2+(a+b)/2
        G['z'+str(i+1)] = g

    phi = parameter_set[keys[l-1]]
    if inst_bias == True:
        q = sigmoid(np.matmul(phi,np.append(g,[[1]], axis=0)))
    else:
        q = sigmoid(np.matmul(phi[:,:-1],g))
        
    # derivatives
    u = q - (y-b)/(a-b)
    loss = np.sum(np.abs(u))  # for visulization
    for i in range(l-1,0,-1):
        phi = parameter_set[keys[i]][:,:-1]
        dz = np.matmul(phi.T,u)
        z = G['z'+str(i)]
        parameter_set[keys[i]] -= lr * np.outer(u,np.append(z,[[1]], axis=0))
        if activation_type == "sigmoid":
            u = dz * z * (1-z) * (a-b)
        elif activation_type == "tanh":
            u = dz * (1-z**2) * (a-b)/2
            
    parameter_set[keys[0]] -= lr * np.outer(u,np.append(x,[[1]], axis=0))
    
    return parameter_set,loss

def wake_update_delta(Phi,Alpha_P,lr,n_dz,value_set,activation_type,bias):
    """
    Arguments:
    Phi -- Recognition parameter set, Python dictionary of length m-1 with each key-value pair being a parameter matrix of 
    shape (n_z{i+1}, n_zi+1), where the last column represents bias b's
    Alpha_P -- assignment of each neuron (binary value), Python dictionary of length m with each key-value pair being 
    a numpy array of shape (n_dz[0,i], 1),i = m-1,...,0
    lr -- learning rate, decimals
    
    n_dz -- number of neurons for each layer, numpy array of shape (n+1,m), where m is the number of instantiation layers, 
    n is the maximum number of inserted layers between adjacent instantiation layers
    value_set -- list or array [a,b], where a is the positive outcome and b is the negative outcome of a Bernoulli experiment
    activation_type -- we provide 2 choices of activation functions: tanh(x) and sigmoid(x)
    bias -- list or array [instantiation bias, MLP bias], taking binary value in {True, False}. For example, [False,True] means 
    no instantiation bias but has MLP bias
    
    Returns:
    Phi -- Updated recognition parameter set, Python dictionary of length m-1 with each key-value pair being a parameter matrix of 
    shape (n_z{i+1}, n_zi+1), where the last column represents bias b's
    Loss -- numpy array of length m-1; the first m-2 values are layer loss, the last term is the total loss
    """
    m = n_dz.shape[1]
    Loss = np.zeros(m)
    for i in range(m-2):
        n = np.where(n_dz[1:,i] != 0)[0].size  # number of inserted layers between i and i+1
        if n == 0:
            parameter_set = {"Phi_" + str(i) + str(i+1): Phi["Phi_" + str(i) + str(i+1)]}
        else:
            parameter_set = {k: Phi[k] for k in ["Phi_" + str(i) + str(i+1) + "_" + str(j) for j in range(1,n+2)]}
            
        x = Alpha_P['z'+str(i)]
        y = Alpha_P['z'+str(i+1)]
        parameter_set,loss = multi_Bernoulli_update(x,y,parameter_set,lr,value_set,activation_type,bias)
        Loss[i] = loss
        Loss[-1] += loss
        for k in [*parameter_set]:
            Phi[k] = parameter_set[k]
        
    return Phi,Loss

def sleep_update_delta(Theta,Alpha_Q,lr,n_dz,value_set,activation_type,bias):
    """
    Arguments:
    Theta -- Generative parameter set, Python dictionary of length m with each key-value pair being a parameter matrix of 
    shape (n_z{i}, n_z{i+1}+1), where the last column represents bias b's
    Alpha_Q -- Recognition assignment of each neuron (binary value), Python dictionary of length m with each key-value pair being 
    a numpy array of shape (n_z, 1)
    lr -- learning rate, decimals
    
    n_dz -- number of neurons for each layer, numpy array of shape (n+1,m), where m is the number of instantiation layers, 
    n is the maximum number of inserted layers between adjacent instantiation layers
    value_set -- list or array [a,b], where a is the positive outcome and b is the negative outcome of a Bernoulli experiment
    activation_type -- we provide 2 choices of activation functions: tanh(x) and sigmoid(x)
    bias -- list or array [instantiation bias, MLP bias], taking binary value in {True, False}. For example, [False,True] means 
    no instantiation bias but has MLP bias
    
    Returns:
    Theta -- Updated generative parameter set, Python dictionary of length m with each key-value pair being a parameter matrix of 
    shape (n_z{i}, n_z{i+1}+1), where the last column represents bias b's
    Loss -- numpy array of length m; the first m-1 values are layer loss, the last term is the total loss
    """
    
    m = n_dz.shape[1]
    inst_bias = bias[0]
    mlp_bias = bias[1]
    data_bias = bias[2]
    
    Loss = np.zeros(m)
    bias = [inst_bias,mlp_bias]
    for i in range(m-1,1,-1):
        n = np.where(n_dz[1:,i-1] != 0)[0].size  # number of inserted layers between i and i+1
        if n == 0:
            parameter_set = {"Theta_" + str(i) + str(i-1): Theta["Theta_" + str(i) + str(i-1)]}
        else:
            parameter_set = {k: Theta[k] for k in ["Theta_" + str(i) + str(i-1) + "_" + str(j) for j in range(n+1,0,-1)]}
            
        x = Alpha_Q['z'+str(i)]
        y = Alpha_Q['z'+str(i-1)]
        parameter_set,loss = multi_Bernoulli_update(x,y,parameter_set,lr,value_set,activation_type,bias)
        Loss[i-1] = loss
        Loss[-1] += loss
        for k in [*parameter_set]:
            Theta[k] = parameter_set[k]
            
    bias = [data_bias,mlp_bias]     
    n = np.where(n_dz[1:,0] != 0)[0].size  # number of inserted layers between 0 and 1
    if n == 0:
        parameter_set = {"Theta_10": Theta["Theta_10"]}
    else:
        parameter_set = {k: Theta[k] for k in ["Theta_10_"+ str(j) for j in range(n+1,0,-1)]}

    x = Alpha_Q['z1']
    y = Alpha_Q['z0']
    parameter_set,loss = multi_Bernoulli_update(x,y,parameter_set,lr,value_set,activation_type,bias)
    Loss[0] = loss
    Loss[-1] += loss
    for k in [*parameter_set]:
        Theta[k] = parameter_set[k]

    return Theta,Loss

def well_formed_generate(n,value_set):
    """
    Well-formedness rules:
        1. Start with 1
        2. Forbid 00100 (no 100, 001 on the boundary)
        3. Forbid 0000
        
    Arguments:
    n -- length of input layer (single data point)
    value_set -- list or array [a,b], where a is the positive outcome and b is the negative outcome of a Bernoulli experiment
    
    Returns:
    well_formed_set -- a dataset obeys the well-formedness rules, numpy array of shape (n,n_data), n_data is the number of datapoints 
    in the generated dataset
    """
    
    well_formed_set = np.zeros([1,n])
    well_formed_set[0,0] = 1

    for i in range(1,n):
        for j in range(np.shape(well_formed_set)[0]):
            if i == 2 and np.array_equal(well_formed_set[j,i-2:i], [1,0]):
                well_formed_set[j,i] = 1
            elif i > 3 and np.array_equal(well_formed_set[j,i-3:i], [0,0,0]):
                well_formed_set[j,i] = 1
            elif i > 3 and np.array_equal(well_formed_set[j,i-4:i], [0,0,1,0]):
                well_formed_set[j,i] = 1
            else:
                well_formed_set = np.append(well_formed_set, well_formed_set[j:j+1,:], axis=0)
                well_formed_set[j,i] = 1

    ind = np.array([], dtype=np.int8)
    for i in range(well_formed_set.shape[0]):
        if np.array_equal(well_formed_set[i,-3:], [0,0,1]):
            ind = np.append(ind,i)

    well_formed_set = np.delete(well_formed_set,ind,0)
    well_formed_set = np.transpose(well_formed_set)
    a = value_set[0]
    b = value_set[1]
    well_formed_set = well_formed_set*(a-b)+b
    
    return well_formed_set

def random_generate(k,n,n_data,value_set):
    """
    The dataset is generated in a favor of Bayesian mixure of Gaussians. Given k mixture Gaussian components, we sample their 
    means u_1...u_k uniformly from [0,1]. Then we randomly assign each data to one of the components, and sample from its 
    Gaussian distribution (u_k, sigma). sigma is a hyperparameter, we default it to 1.
    
    The "Bayesian mixure of Gaussians" generation is just a way to generate dataset with non-singular distributions. The 
    generated data distribution is not identified with the mixure of Gaussian distributions that generated it. In other words, 
    the data is treated as sole evidence without any prior on how it's been generated thus its reconstruction is not convolved 
    with it's generative distributions, which is a major difference from varietional inference.
        
    Arguments:
    k -- number of Gaussian components
    n -- length of input layer (single data point)
    n_data -- number of datapoints to generate
    value_set -- list or array [a,b], where a is the positive outcome and b is the negative outcome of a Bernoulli experiment
    
    Returns:
    random_set -- generated dataset, numpy array of shape (n,n_data), n_data is the number of datapoints in the generated dataset
    """
    
    u = np.random.rand(n,k)
    c = np.random.randint(k, size=(n_data,))
    data_mean = u[:,c]
    prob = np.random.randn(n,n_data) + data_mean
    random_set = (prob>0.5).astype(int)
    
    a = value_set[0]
    b = value_set[1]
    random_set = random_set *(a-b)+b
    
    return random_set

def all_comb(n, value_set):
    """
    All combinations of possible datapoints. 2^n
        
    Arguments:
    n -- length of input layer (single data point)
    value_set -- list or array [a,b], where a is the positive outcome and b is the negative outcome of a Bernoulli experiment
    
    Returns:
    entire_set -- a set containing all possible datapoints the input could be, numpy array of shape (n,2^n), 
    2^n is the number of all possible combinations of n binary neurons
    """
    a = value_set[0]
    b = value_set[1]
    
    entire_set = np.zeros((2,n))
    entire_set[0,0] = 1
    for i in range(1,n):
        for j in range(entire_set.shape[0]):
            entire_set = np.append(entire_set, entire_set[j:j+1,:], axis=0)
            entire_set[j,i] = 1
    entire_set = entire_set*(a-b)+b
    entire_set = np.transpose(entire_set)
    
    return entire_set

def reorder_all_comb(entire_set,dataset):
    """
    This function reorders the entire set with respect to the generated (or given) dataset. Since we are dealing with 
    categorical distributions, to visualize the result better, we put the datapoints contained in the dataset 
    (let's say k datapoints) to the first k columns of the entire_set, which are followed by false instances in subsequent
    columns.
        
    Arguments:
    entire_set -- a set containing all possible datapoints the input could be, numpy array of shape (n,2^n), 
    2^n is the number of all possible combinations of n binary neurons
    dataset -- generated (or given) dataset, numpy array of shape (n,n_data), n_data is the number of datapoints.
    
    Returns:
    reordered_set -- entire_set reordered as columns 0-k represents valid instances contained in the dataset, 
    columns k-2^n represents false instances not in the dataset. numpy array of shape (n,2^n)
    k -- number of distinct datapoints in dataset, integer
    """
    
    dataset = np.unique(dataset, axis=1)
    reordered_set = np.zeros(entire_set.shape)
    
    k = dataset.shape[1]
    reordered_set[:,:k] = dataset
    r = k
    for i in range(entire_set.shape[1]):
        flag = 0
        for j in range(k):
            if np.array_equal(entire_set[:,i], dataset[:,j]):
                flag = 1
                break
        if flag == 0:
            reordered_set[:,r] = entire_set[:,i]
            r += 1
    return reordered_set

def reorder_pseudo_Gaussian(entire_set,dataset):
    """
    This function reorders the entire set with respect to the generated (or given) dataset. Since we are dealing with 
    categorical distributions, to visualize the result better, we put the datapoints contained in the dataset 
    (let's say k datapoints) to the first k columns of the entire_set, which are followed by false instances in subsequent
    columns.
        
    Arguments:
    entire_set -- a set containing all possible datapoints the input could be, numpy array of shape (n,2^n), 
    2^n is the number of all possible combinations of n binary neurons
    dataset -- generated (or given) dataset, numpy array of shape (n,n_data), n_data is the number of datapoints.
    
    Returns:
    reordered_set -- entire_set reordered as columns 0-k represents valid instances contained in the dataset, 
    columns k-2^n represents false instances not in the dataset. numpy array of shape (n,2^n)
    k -- number of distinct datapoints in dataset, integer
    """
    
    values,counts = np.unique(dataset, axis=1, return_counts = True)
    reordered_set = np.zeros(entire_set.shape)
    
    order = np.append(np.argsort(counts)[::2],np.argsort(counts)[1::2][::-1])
    dataset_arranged = values[:,order]

    k = counts.size
    reordered_set[:,:k] = dataset_arranged
    r = k
    for i in range(entire_set.shape[1]):
        flag = 0
        for j in range(k):
            if np.array_equal(entire_set[:,i], dataset_arranged[:,j]):
                flag = 1
                break
        if flag == 0:
            reordered_set[:,r] = entire_set[:,i]
            r += 1
    return reordered_set

def generate(n_sample,n_dz,value_set,Theta,activation_type,bias):
    """
    Arguments:
    n_sample -- number of samples
    n_dz -- number of neurons for each layer, numpy array of shape (n+1,m), where m is the number of instantiation layers, 
    n is the maximum number of inserted layers between adjacent instantiation layers
    value_set -- list or array [a,b], where a is the positive outcome and b is the negative outcome of a Bernoulli experiment
    Theta -- Generative parameter set, Python dictionary of length m with each key-value pair being a parameter matrix of 
    shape (n_z{i}, n_z{i+1}+1), where the last column represents bias b's
    activation_type -- we provide 2 choices of activation functions: tanh(x) and sigmoid(x)
    bias -- list or array [instantiation bias, MLP bias,data bias], taking binary value in {True, False}. For example, 
    [False,True,True] means no instantiation bias but has MLP bias and data bias
    
    Returns:
    generation -- generated instances after training, numpy array of shape (n,n_data), n is the length of input layer, 
    n_data is the number of datapoints generated
    """
    generation = np.zeros((n_dz[0,0],n_sample))
    for i in range(n_sample):
        Alpha_P = sleep_sample(n_dz,value_set,Theta,activation_type,bias)
        generation[:,i:i+1] = Alpha_P['z0']
    return generation

def metrics(generation,reordered_set,dataset):
    """
    Arguments:
    generation -- generated instances after training, numpy array of shape (n,n_sample), n is the length of input layer, 
    n_sample is the number of datapoints generated
    reordered_set -- entire_set reordered as columns 0-k represents valid instances contained in the dataset, 
    columns k-2^n represents false instances not in the dataset. numpy array of shape (n,2^n)
    dataset -- numpy array of shape (n,n_data), n_data is the number of datapoints.
    
    Returns:
    distribution -- assigned category for generated samples based on reordered set, numpy array of shape (n_sample, )
    data_dist -- assigned category for dataset  based on reordered set, numpy array of shape (n_data, )
    statistics -- python dictionary with keys:
        percent -- percentage of positive instances
        n_fn -- number of false negative samples, missing evidence
        FN -- position of false negative samples, numpy array of shape (k-n_fn, )
        n_fp -- number of false positive samples, outliers
        FP -- position and counts of false positive samples, numpy array of shape (2,n_fp)
    MSE -- mean squared error between the generation Q and the data evidence P on the support of P (on positive instances only).
    """
    n_sample = generation.shape[1]
    n_data = dataset.shape[1]
    distribution = np.zeros((n_sample, ),dtype = int)
    for i in range(n_sample):
        for j in range(reordered_set.shape[1]):
            if np.array_equal(generation[:,i], reordered_set[:,j]):
                distribution[i] = j
                break
    values_t, counts_t = np.unique(distribution, return_counts=True)
    
    data_dist = np.zeros((n_data, ),dtype = int)
    for i in range(n_data):
        for j in range(reordered_set.shape[1]):
            if np.array_equal(dataset[:,i], reordered_set[:,j]):
                data_dist[i] = j
                break
    values_d, counts_d  = np.unique(data_dist, return_counts=True)
    k = counts_d.size
    
    
    # statistics
    percent = np.sum(counts_t[values_t < k])/n_sample
    n_fn = k-values_t[values_t < k].size
    FN = np.zeros((n_fn,),dtype = int)
    dist_positive = np.array([values_t[values_t < k], counts_t[values_t < k]])
    s = 0
    values_t[values_t < k]
    dist_values = np.append(np.append(-1,values_t[values_t < k]),k)   # append 0 and k in the range
    
    for i in range(dist_values.size-1):
        diff = dist_values[i+1] - dist_values[i]
        for j in range(1,diff):
            FN[s] = dist_values[i]+j
            dist_positive = np.append(dist_positive, np.array([[dist_values[i]+j],[0]]),axis = 1)
            s += 1
    dist_positive = np.unique(dist_positive,axis = 1)
    n_fp = values_t[values_t >= k].size
    FP = np.array([values_t[values_t >= k], counts_t[values_t >= k]])
    statistics = {'percent': percent, 'FN': FN, 'n_fn':n_fn, 'FP': FP, 'n_fp':n_fp}
    
    # metric 2: distribution difference. Since our ditributions are discrete, we calculate a mean squared error (MSE) between 
    #           the generation Q and the data evidence P on the support of P (on positive instances only).
    
    counts_t = counts_t/n_sample*n_data  # distribution in the same scale as dataset
    MSE = np.sum((dist_positive[1,:]/n_sample*n_data - counts_d)**2)/k
    ABS_Error = np.abs(dist_positive[1,:]/n_sample*n_data - counts_d).sum()/k
    
    return distribution,data_dist,statistics, MSE,ABS_Error