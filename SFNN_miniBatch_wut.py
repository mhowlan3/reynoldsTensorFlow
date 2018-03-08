
# coding: utf-8

# In[1]:

## Load modules
import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import os
cwd = os.getcwd() # current working directory
import sys
sys.path.append(os.path.abspath(cwd))
import utils


# In[2]:

import sys
sys.path.append(os.path.abspath(cwd))


# In[3]:

## Load data
#data_dir = "cycle_10bins_conditions_none"#"2sec_data_avg" # name of the data folder

# Loading the train, dev, test data from specified folder
#os.chdir(cwd+"/"+data_dir+"/train")
#x_train = genfromtxt("x_train.csv", delimiter=',')
#y_train = genfromtxt("y_train.csv", delimiter=',')


# In[4]:

# data (num_samples, features)
basedir = '/home/mhowland/Dropbox/Research/Stanford/Lele/AWS_2016-06_Multi-Height_Time_Series/'
windFile = 'Firewheel_FWL'+str(1)+'_mast_file_tsec_72-months.csv'
windFileDev = 'Firewheel_FWL'+str(2)+'_mast_file_tsec_72-months.csv'
def importWindFile(basedir, windFile):
    data = np.loadtxt(basedir+windFile, delimiter=',', skiprows=7)
    ws = np.zeros((data.shape[0],5))
    time = np.linspace(0., 10.*data.shape[0],data.shape[0])
    nt = ws.shape[0]
    hour = data[:,3]
    month = data[:,1]
    for i  in range(0,5):
        ind = 5*(i+1)
        ws[:,i] = data[:,ind]
    return ws, hour, time, nt, month
ws, hour, time, nt, month = importWindFile(basedir, windFile)
wsDev, _, _, _, _ = importWindFile(basedir, windFile)


# In[5]:

# Full dataset
Nt = ws.shape[0]
# Physical parameters
horizon = 18
dataHeight = 0 # index of the height we want
# Initialize
xFull = np.zeros((horizon-1,Nt))
yFull = np.zeros((Nt,1))
xFullDev = np.zeros((horizon-1,Nt))
yFullDev = np.zeros((Nt,1))
for i in range(horizon,Nt):
    xFull[:,i-horizon] = ws[i-horizon:i-1,dataHeight]
    yFull[i-horizon] = ws[i,dataHeight]
    xFullDev[:,i-horizon] = wsDev[i-horizon:i-1,dataHeight]
    yFullDev[i-horizon] = wsDev[i,dataHeight]
# Select the dev set
randomSeed = int(np.round(np.random.rand(1)*Nt))
n_dev = int(np.round(Nt*0.01)) # use 1% of the data for dev
#x_dev = xFull[:,randomSeed:randomSeed+n_dev]
#y_dev = yFull[randomSeed:randomSeed+n_dev]
#x_dev = x_dev.T
# Set the test data as the rest of it
#x_train = np.append(xFull[:,0:randomSeed-1],xFull[:,randomSeed+n_dev+1:Nt],axis=1)
#y_train = np.append(yFull[0:randomSeed-1],yFull[randomSeed+n_dev+1:Nt],axis=0)
#x_train = x_train.T

## Instead use an entirely different MET tower
x_dev = xFullDev[:,0:n_dev]
y_dev = yFullDev[0:n_dev]
x_dev = x_dev.T
x_train = xFull#[:,0:n_dev*30]
y_train = yFull#[0:n_dev*30]
x_train = x_train.T
time = time[0:n_dev]
# Make a multiple of the minibatch size
newNtrain = int(np.floor(x_train.shape[0])/256)
x_train = x_train[0:newNtrain*256,:]
newNdev = int(np.floor(x_dev.shape[0])/256)
x_dev = x_dev[0:newNdev*256,:]

print(x_train.shape)
print(x_dev.shape)


# In[6]:

# Train set
#nt = 1000
#horizon = 18
#n_train = nt-horizon
#x_train = np.zeros((horizon-1,n_train))
#y_train = np.zeros((n_train,1))
#dataHeight = 0 # index of the height we want
#for i in range(horizon,nt):
#    x_train[:,i-horizon] = ws[i-horizon:i-1,dataHeight]
#    y_train[i-horizon] = ws[i,dataHeight]
#x_train = x_train.T
# Dev set
#randomSeed = 180 # days in the future from the train set
#randomSeed = randomSeed*24*6 # dev is some amount of day in the future
#nd = 100
#n_dev = nd-horizon
#x_dev = np.zeros((horizon-1,n_dev))
#y_dev = np.zeros((n_dev,1))
#for i in range(horizon,nd):
#    x_dev[:,i-horizon] = ws[randomSeed+i-horizon:randomSeed+i-1,dataHeight]
#    y_dev[i-horizon] = ws[randomSeed+i,dataHeight]
#x_dev = x_dev.T


# In[7]:

#print(x_all.shape)
#x_train = x_all[10:110,10:20] # (100,10)
data_size, features = x_train.shape


# In[8]:

def normalize_data(data):
    mu = np.mean(data,axis=0) # compute the mean along axis = 0 (num_samples for raw data)
    cov = np.std(data,axis=0) # using std instead of variance seems to be best
    return mu, cov # returning the normalizations for the data

x_mu, x_cov = normalize_data(x_train) # x_train is (features x num_samples)
y_mu, y_cov = normalize_data(y_train) # x_train is (features x num_samples)

#y_mu, y_cov = normalize_data(y_train)

X_train = ((x_train - x_mu)/x_cov).T # still in (features, data_samples)
Y_train = np.squeeze(y_train)
X_dev = ((x_dev - x_mu)/x_cov).T # still in (features, data_samples)
Y_dev = np.squeeze(y_dev)

print(X_train.shape)
print(Y_train.shape)


# In[9]:

def initialize_parameters(num_hid_layers, size_hid_layers, n_x, output_size, num_filt):
    parameters = {}
    total_layers = num_hid_layers
    
    n_x_orig = n_x/num_filt
    
    F1 = tf.get_variable("F1", [n_x_orig, 1, 1], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    parameters['F' + str(1)] = F1
    F2 = tf.get_variable("F2", [n_x_orig, 1, 1], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    parameters['F' + str(2)] = F2
    F3 = tf.get_variable("F3", [n_x_orig, 1, 1], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    parameters['F' + str(3)] = F3
    F4 = tf.get_variable("F4", [n_x_orig, 1, 1], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    parameters['F' + str(4)] = F4

    
    for l in range(1,total_layers+1):
        if l == 1:
            a = size_hid_layers
            b = n_x
        elif l == total_layers:
            a = output_size
            b = size_hid_layers
        else:
            a = size_hid_layers
            b = size_hid_layers
            
        parameters['w' + str(l)] = tf.get_variable('w'+str(l), [a, b], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
        parameters['b' + str(l)] = tf.get_variable('b'+str(l), [a,1], initializer = tf.zeros_initializer())   
    return parameters


# In[10]:

def forward_prop(X,parameters, total_layers,  m, n_x, keep_prob=1., num_filts = 1):
    #total_layers = len(parameters)//2
    layer_outputs = {}
    
    # Get filters
    #F1 = parameters['F1']
    #t = X.get_shape().as_list()
    #m = t[0]
    #n_x = t[1]
    
    #num_filts = 4
    X = tf.transpose(X)
    X = tf.reshape(X, [m, n_x, 1])
    #X = tf.reshape(X, [m, n_x, 1, 1])
    for ll in range(1,num_filts+1):
        #print(tf.shape(X[0:m,(l-1)*n_x:l*n_x+1, :]))
        #print(tf.shape(parameters['F' + str(l)]))
        X = tf.concat([X, tf.nn.conv1d(X[0:m,(ll-1)*n_x:ll*n_x+1, :], parameters['F' + str(ll)], stride = 1, padding = 'SAME')], axis=1)
    X = tf.squeeze(tf.squeeze(X))
    X = tf.reshape(X, [m, n_x*(num_filts+1)])
    X = tf.transpose(X)
    layer_outputs['A0'] = X
    
    for l in range(1,total_layers+1):
        layer_outputs['Z' + str(l)] = tf.matmul(parameters['w' + str(l)],layer_outputs['A' + str(l-1)])+parameters['b' + str(l)]
        layer_outputs['A' + str(l)] = tf.nn.relu(layer_outputs['Z' + str(l)])
        layer_outputs['A' + str(l)] = tf.nn.dropout(layer_outputs['A' + str(l)],keep_prob)
    
    return layer_outputs['Z' + str(total_layers)]


# In[11]:

# DID NOT WRITE! MUST CITE COURSE:
# https://github.com/JudasDie/deeplearning.ai/blob/master/Convolutional%20Neural%20Networks/week1/cnn_utils.py
def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[1]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:,permutation]
    shuffled_Y = Y[permutation]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = int(np.floor(m/mini_batch_size)) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:,k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:,num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches, permutation


# In[33]:

#def model(X_train, Y_train, X_dev, Y_dev, learning_rate, beta, k_p,
#         num_epochs, num_hid_layers, size_hid_layers, minibatch_size, print_interval=10,plotting = True,
#         reg_type='L2'):
def model(X_train, Y_train, X_dev, Y_dev, learning_rate, beta, k_p, reg_type,
          num_epochs, num_hid_layers, size_hid_layers, minibatch_size, log_dir, print_loss=True, 
          plotting = True, print_interval=10, plot_interval=100, parallel=False, queue=0):

    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    losses = []                                        # To keep track of the cost
    train_errs = []                                        # To keep track of the cost
    epochs = [] # keep track of epoch number
    dev_errs = []                                        # To keep track of the cost
    eps = 10**-8.
    trainMiniY = []
    devMiniY = []
    seed = 3
    train_preds = []
    mDev = X_dev.shape[1]
    dev_preds = []
    devBatchY = []
    permutations = []
    dev_predsOut = np.zeros((mDev,))
    Y_dev_out = np.zeros((mDev,))
    
    O = 1 # output size?
    
    # Successive filters
    num_filt = 4
    
    X = tf.placeholder(tf.float32,[n_x,None])
    Y = tf.placeholder(tf.float32,[None])
    keep_prob = tf.placeholder(tf.float32)
    
    #parameters = initialize_parameters(num_hid_layers, size_hid_layers, n_x, O)
    #out = forward_prop(X, parameters)
    parameters = initialize_parameters(num_hid_layers, size_hid_layers, n_x*(num_filt+1), O, num_filt)
    out = forward_prop(X, parameters, num_hid_layers, minibatch_size, n_x, num_filts = num_filt)
    
    #loss = tf.reduce_mean(tf.squared_difference(out, Y)) # L2 loss --> not good for our problem
    loss = tf.reduce_mean(tf.losses.absolute_difference(tf.squeeze(Y),tf.squeeze(out))) # L1 loss
    
    # Loss function with L2 Regularization
    if reg_type == "L2":
        for l in range(1,num_hid_layers+1):
            if l == 1:
                regularizers = tf.nn.l2_loss(parameters['w' + str(l)]) 
            else:
                regularizers = regularizers + tf.nn.l2_loss(parameters['w' + str(l)])

        loss = tf.reduce_mean(loss + beta * regularizers)
    elif reg_type == "L1":
        print("Add L1 regularization here")
    
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss) # Optimizer, change the learning rate here

    init = tf.global_variables_initializer() # When init is run later (session.run(init)),
    
    # Start logging file for this particular model
    if print_loss:
        file = open(log_dir,'a') 
        file.write('START OF NEW MODEL\n') 
        file.write("learning rate = %f, hidden layers = %d, hidden units = %d, epochs = %d \n"% (learning_rate, num_hid_layers, size_hid_layers, num_epochs))
        file.write("L2 beta = %f, Dropout Keep Prob = %f \n"% (beta, k_p)) 
        file.close() 
        
        
    with tf.Session() as sess: # starting tf session --> all computation on tf graph in this with struct
        sess.run(init)
        for epoch in range(num_epochs+1):
            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches, _ = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                _ , temp_cost = sess.run([optimizer,loss], feed_dict={X: minibatch_X, Y: minibatch_Y})
                # Minibatch stats
                train_pred = sess.run(out, feed_dict={X: minibatch_X})
                #train_err = np.divide(np.mean(abs(train_pred - minibatch_Y)),np.mean(abs(minibatch_Y))) # absolute error
                #train_errs.append(train_err)
                train_preds = np.append(train_preds, train_pred)
                trainMiniY = np.append(trainMiniY, minibatch_Y)
                minibatch_cost += temp_cost / num_minibatches
            train_err = np.divide(np.mean(abs(train_preds - trainMiniY)),np.mean(abs(trainMiniY)))
            train_errs.append(train_err)
            loss_val = minibatch_cost
            losses.append(loss_val)
            
            if epoch % (num_epochs/print_interval) == 0:
                print("Loss: ",loss_val)
                # Output the predictions
                # Loop over dev set
                num_minibatches = int(mDev / minibatch_size)
                seed = seed + 1
                minibatches, permutations = random_mini_batches(X_dev, Y_dev, minibatch_size, seed)
                dev_preds = []
                devBatchY = []
                for minibatch in minibatches:
                    # Select a minibatch
                    (minibatch_X, minibatch_Y) = minibatch
                    _ , temp_cost = sess.run([optimizer,loss], feed_dict={X: minibatch_X, Y: minibatch_Y})
                    dev_pred = sess.run(out, feed_dict={X: minibatch_X})
                    dev_preds = np.append(dev_preds, dev_pred)
                    devBatchY = np.append(devBatchY, minibatch_Y)
                dev_err = np.divide(np.mean(abs(dev_preds - devBatchY)),np.mean(abs(devBatchY))) # absolute error
                dev_errs.append(dev_err)
                dev_predsOut[permutations] = dev_preds
                Y_dev_out[permutations] = devBatchY
                #fig = plt.figure(num=None, dpi=200, facecolor='w', edgecolor='k'); ax = plt.gca()
                #plt.plot(np.squeeze(devBatchY), '-')
                #plt.plot(np.squeeze(dev_preds), '-')
                epochs.append(epoch)
                if print_loss:
                    file = open(log_dir,'a') 
                    file.write("Epoch %d loss: %f \n" % (epoch, loss_val) )
                    file.write("Train error: %f \n" % (train_err)) 
                    file.write("Dev error: %f \n" % (dev_err))
                    file.close() 
                print('Train error: ', train_err)
                print('Dev error: ', dev_err)
        if plotting:
            # Plot percent errors during iterations
            fig = plt.figure(num=None, dpi=200, facecolor='w', edgecolor='k'); ax = plt.gca()
            plt.plot(np.squeeze(train_errs))
            ax.set_yscale('log')
            plt.ylabel('Percent error')
            plt.title("Error for learning rate = " + str(learning_rate))
            plt.show()
    ind = np.argmin(dev_errs)
    min_dev = dev_errs[ind]
    min_epoch = epochs[ind] 
    
    results = train_errs, Y_train, train_preds, Y_dev_out, dev_predsOut, dev_errs, min_dev, min_epoch
    #results = train_err, dev_err, test_err, min_dev, min_epoch
    return results


# In[47]:

learning_rate = 0.0005#0.0005
num_epochs = 5000 # total number of epochs to iterate through
print_interval = 100 # number of prints per total run
minibatch_size = 256
num_hid_layers = 10
size_hid_layers = 128
beta=0
reg_type = 'L2N'
k_p = 1.
log_dir = 'testwut'

#results=model(X_train, Y_train, X_dev, Y_dev, learning_rate, num_epochs, num_hid_layers, size_hid_layers, minibatch_size, print_interval, plotting=False)
results=model(X_train, Y_train, X_dev, Y_dev, learning_rate, beta, k_p, reg_type,
          num_epochs, num_hid_layers, size_hid_layers, minibatch_size, log_dir, print_loss=True, 
          plotting = True, print_interval=print_interval, plot_interval=100, parallel=False, queue=0)


# In[48]:

nt = results[3].shape[0]
shift_err = np.divide(np.mean(abs(results[3][0:nt-1] - results[3][1:nt])),np.mean(abs(results[3][1:nt])))
print(shift_err, results[5][-1])


# In[49]:

## Plot Results
plotData = True
plotPred = True
if plotData==True:
    # Train error
    train_err = results[0]
    fig = plt.figure(num=None, dpi=200, facecolor='w', edgecolor='k'); ax = plt.gca()
    plt.plot(np.squeeze(train_err))
    ax.set_yscale('log')
    plt.ylabel('Train percent error')
    plt.title("Error for learning rate = " + str(learning_rate))
    plt.show()
    # Dev error
    dev_err = results[5]
    fig = plt.figure(num=None, dpi=200, facecolor='w', edgecolor='k'); ax = plt.gca()
    plt.plot(np.squeeze(dev_err))
    ax.set_yscale('log')
    plt.ylabel('Dev percent error')
    plt.title("Error for learning rate = " + str(learning_rate))
    plt.show()
    if plotPred == True:
        # Predictions vs train
        #Y_train = results[1]
        #Y_pred = results[2]
        #fig = plt.figure(num=None, dpi=200, facecolor='w', edgecolor='k'); ax = plt.gca()
        #plt.plot(np.squeeze(Y_train[1:1000]))
        #plt.plot(np.squeeze(Y_pred[1:1000]))
        #plt.ylabel('Wind speed, [m/s]')
        #plt.show()
        # Predictions vs dev
        Y_dev = results[3]
        Y_shift = results[3][1:nt]
        dev_pred = results[4]
        fig = plt.figure(num=None, dpi=200, facecolor='w', edgecolor='k'); ax = plt.gca()
        time = np.linspace(0,10*float(Y_dev.shape[0])/(60. * 24.), Y_dev.shape[0])
        print(Y_dev.shape, Y_shift.shape, dev_pred.shape)
        plt.plot(time,np.squeeze(Y_dev), '-')
        plt.plot(time[1:nt],np.squeeze(Y_dev[0:nt-1]), '.')
        plt.plot(time,np.squeeze(dev_pred), 'o')
        plt.ylabel('Wind speed, [m/s]')
        plt.xlabel('Time, [days]')
        plt.xlim( (5, 6) ) 
        plt.show()


# In[30]:
#
#dev_err = np.divide(np.mean(abs(Y_dev - dev_pred)),np.mean(abs(Y_dev)))
#print(dev_err)
#print(results[5])
#
#
## In[ ]:
#
## Does creating a log file in each call to the model function mess up the main log file?
#
#def hyperparamSearch(X_train, Y_train, X_dev, Y_dev, lr_rng, num_hid_layers_rng, beta_rng, k_p_rng, reg_type,
#                 size_hid_layers_rng, num_sims, num_epochs, minibatch_size, log_dir, parallel, cores):
#    # compute random values within the ranges for each param of length num_sims
#    num_features, data_length = X_train.shape 
#    num_params = 5
#    np.random.seed(13) # set seed for rand
#    lower_bounds = [lr_rng[0],num_hid_layers_rng[0],size_hid_layers_rng[0],beta_rng[0],k_p_rng[0]]
#    upper_bounds = [lr_rng[1],num_hid_layers_rng[1],size_hid_layers_rng[1],beta_rng[1],k_p_rng[1]]
#    sample_size = [num_sims, num_params] # num_sims x number of params in search
#    samples_params = np.random.uniform(lower_bounds, upper_bounds, sample_size)
#
#    # modifying the initial random parameters
#    lr_samples = 10**samples_params[:,0] # log scale
#    hl_samples = samples_params[:,1].astype(int) # rounded down to nearest int
#    hu_samples = (samples_params[:,2]*num_features).astype(int) # base of 10 neurons used for each level
#    beta = samples_params[:,3]
#    k_p = samples_params[:,4]
#    
#    # save the data for the ranges used to the main sim file
#    utils.logging.info("lr_rng = "+str(lr_rng)+" hidden layers rng = "+str(num_hid_layers_rng)+" hidden units rng = "+str(size_hid_layers_rng)+" num sims = %d", num_sims)
#    log_name = "lr_rng = "+str(lr_rng)+" hidden layers rng = "+str(num_hid_layers_rng)+" hidden units rng = "+str(size_hid_layers_rng)+" num sims = "+ str(num_sims)
#    fileStr = os.path.join(cwd+"/logOut/"+log_dir,log_name+'.log')
#    print(fileStr)
#    file = open(fileStr,'w') 
#    file.write('START OF NEW HYPERPARAM SWEEP\n') 
#    file.close() 
#
#    
#    results = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
#    
#    if parallel: # Need cores 
#        print("parallelizing the training") # add parallel ability for specified number of cores similar to funct above
#        results = multi_sim(X_train, Y_train, X_dev, Y_dev, X_test, Y_test, lr_samples, beta, k_p, reg_type,
#                            num_epochs, hl_samples, hu_samples, minibatch_size, log_dir, True, False, 10, 
#                            100, True, cores)
#        print(results)
#    else:
#        for i in range(len(lr_samples)):
#            results2 = model(X_train, Y_train, X_dev, Y_dev, lr_samples[i], beta[i], k_p[i], reg_type,
#                             num_epochs, hl_samples[i], hu_samples[i], minibatch_size, 
#                             fileStr, print_loss=False, plotting = False, print_interval=10, 
#                             plot_interval=100, parallel=False, queue=0) # call model funct            temp_results = np.array([lr_samples[i], hl_samples[i], hu_samples[i], beta[i], k_p[i], num_epochs, train_err, dev_err, test_err, min_epoch, min_dev])
#            train_err, Y_train, train_pred, Y_dev, dev_pred, dev_err, min_dev, min_epoch = results2
#            temp_results = np.array([lr_samples[i], hl_samples[i], hu_samples[i], beta[i], k_p[i], num_epochs, train_err[-1], dev_err[-1], min_epoch, min_dev])
#            #utils.set_logger(os.path.join(cwd+"/"+log_dir,log_dir+'.log')) # reset logger to main log file
#            #utils.logging.info("START OF NEW MODEL")
#            #utils.logging.info("learning rate = %f, hidden layers = %d, hidden units = %d, beta = %f, keep_prob = %f, epochs = %d, reg_type = %s", lr_samples[i], hl_samples[i], hu_samples[i], beta[i], k_p[i], num_epochs, reg_type) # add other hyperparams
#            #utils.logging.info("Train Err = %f, Dev Err = %f, Min Dev Err = %f, Min Epoch = %d", train_err[-1], dev_err[-1], min_dev, min_epoch) # add other hyperparams
#            file = open(fileStr,'a') 
#            file.write('START OF NEW MODEL\n') 
#            file.write("learning rate = %f, hidden layers = %d, hidden units = %d, beta = %f, keep_prob = %f, epochs = %d, reg_type = %s \n"% (lr_samples[i], hl_samples[i], hu_samples[i], beta[i], k_p[i], num_epochs, reg_type))
#            file.write("Train Err = %f, Dev Err = %f, Min Dev Err = %f, Min Epoch = %d \n"% (train_err[-1], dev_err[-1], min_dev, min_epoch))
#            file.close() 
#            print(results, temp_results)
#            results = np.vstack((results,temp_results))# get all results in a list
#        
#        # results contain an array of the parameters and then the resulting errors
#        results = results[1:,:] # get rid of placeholder row
#        results= results[results[:,-1].argsort()] # sort by the lowest dev error
#        file = open(fileStr,'a') 
#        file.write('RESULTS\n') 
#        file.write(str(results))
#        file.close() 
#        #utils.logging.info("RESULTS")
#        #utils.logging.info(str(results))
#    return results
#
#
## In[ ]:
#
#log_dir = ""
#y_ind = 0 # which of the metabolic vectors to predict
#learning_rate = 0.008#0.0005
#num_epochs = 10 # total number of epochs to iterate through
#print_interval = 10 # number of prints
#plot_interval = 10 # data points in the plots
#minibatch_size = 10 # NOT IMPLEMENTED YET
#num_hid_layers = 3
#size_hid_layers = 10
#printing = True # True prints the losses and train/test errors as the network is trained
#plotting = True
#num_sims = 10 # number of simulations run for the hyperparamSearch
#
## Here we are assuming we're not normalizing the Y data
#Y_train = y_train[:,y_ind].T
#Y_dev = y_dev[:,y_ind].T
#
##train_err, dev_err, test_err = model(X_train, Y_train, X_dev, Y_dev, X_test, Y_test, learning_rate, 
#    #num_epochs, num_hid_layers, size_hid_layers, minibatch_size, log_dir, printing, plotting, print_interval, plot_interval)
#
## define the ranges for the hyperparam search
#lr_rng = [-4.,-2.] # range of lr will be done with these values use as 10^r for log scale
#num_hid_layers_rng = [3.0,7.0] # these will be rounded down to nearest Int so add 1 more to value desired for upper bound
#size_hid_layers_rng = [1.0, 4.0] # These values are multiplied by the number of input features
#beta_rng = [0.0, 0.03] # regularization coefficient range
#k_p_rng = [0.8, 1.0] # dropout "keep probability" range
#reg_type = "L2" # set to type of desired regularization "L2", "none" (I don't think L1 is necessary since we don't want feature selection)
#parallel = False
#cores = 2 # number of cores to run parallel trials
#if parallel:
#    num_sims = cores # make this match for now to test it
## run a hyperparamSearch
#results = hyperparamSearch(X_train, Y_train, X_dev, Y_dev, lr_rng, num_hid_layers_rng, beta_rng, k_p_rng, reg_type,
#                 size_hid_layers_rng, num_sims, num_epochs, minibatch_size, log_dir, parallel, cores)
#
#
## In[ ]:
#


