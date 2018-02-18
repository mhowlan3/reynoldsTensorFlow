
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
import utils


# In[2]:


import sys
sys.path.append(os.path.abspath(cwd))


# In[3]:


# data (num_samples, features)
basedir = cwd
windFile = '/Firewheel_FWL'+str(1)+'_mast_file_tsec_72-months.csv'
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


# In[4]:


#print(ws.shape,hour[:,np.newaxis].shape)
#data = np.stack((ws,hour[:,np.newaxis]),axis=0)
## DEFINE INPUT DATA HERE
data = ws


# In[5]:


# data (num_samples, features)
#data = genfromtxt("Firewheel_FWL1_mast_file_tsec_72-months.csv", delimiter=',')
#data_dir = str(cwd)#+"/data") # full path to data folder with data files
#os.chdir(data_dir) # change directory to data
#files = os.listdir(data_dir) # get filenames from data_dir
#files.sort() # sort them so always in same order
#num_files = len(files)
#for i,file in enumerate(reversed(files)): # delete not .csv files
#    if file[-3:] != "csv":
#        files = np.delete(files,num_files-i-1)
        
#print(files) # check you have the files you want


# In[6]:


batch_size = 512
sequence_length = 20
y_ind = 3 # this is the index you want to predict
chunk_num = 100 # number of chunks to split the data into to divide between train/dev/test
chunk_idx = np.zeros(chunk_num, dtype=int)

raw_data = data
y_raw = data[:,y_ind] # copy y data before norm
data_length, features = data.shape
chunk_length = int(data_length/chunk_num)

for i in range(chunk_num):
    chunk_idx[i] = i*chunk_length
    
#for file in files:
#    raw_data = genfromtxt(file, delimiter=',') # shape (data length, features)

# Break into train, dev, test files
train_size = 0.6
dev_size = 0.2
test_size = 1 - train_size - dev_size

split_1 = int(train_size*chunk_num)
split_2 = int((train_size+dev_size)*chunk_num)

np.random.seed(13)
np.random.shuffle(chunk_idx)

train_chunk = chunk_idx[:split_1]
dev_chunk = chunk_idx[split_1:split_2]
test_chunk = chunk_idx[split_2:]

def combine_idx_data(data, labels, idx_list):
    _,features = data.shape
    data_out = np.zeros((len(idx_list)*chunk_length,features))
    labels_out = np.zeros(len(idx_list)*chunk_length)
    for i in range(len(idx_list)):
        data_out[i*chunk_length:(i+1)*chunk_length,:] = data[idx_list[i]:idx_list[i]+chunk_length,:]
        labels_out[i*chunk_length:(i+1)*chunk_length] = labels[idx_list[i]:idx_list[i]+chunk_length]
    return data_out, labels_out

x_train, y_train = combine_idx_data(raw_data, y_raw, train_chunk)
x_dev, y_dev = combine_idx_data(raw_data, y_raw, dev_chunk)
x_test, y_test = combine_idx_data(raw_data, y_raw, test_chunk)


# In[7]:


def normalize_data(data):
    mu = np.mean(data,axis=0) # compute the mean along axis = 0 (num_samples for raw data)
    cov = np.std(data,axis=0) # using std instead of variance seems to be best
    return mu, cov # returning the normalizations for the data

mu_train, mu_cov = normalize_data(x_train)
#print(mu_train)


# In[8]:


X_train = (x_train-mu_train)/mu_cov
X_dev = (x_dev-mu_train)/mu_cov
X_test = (x_test-mu_train)/mu_cov
#print(np.mean(X_train[:,0],axis=0)) # confirm new mean is 0


# In[32]:


# have (# data examples, inpute_size)
# want (# data examples - num_steps - 1, num_steps, input_size)
# given the index for prediction data (y_ind), creates prediction vector Y
#def reshape_sequences(data, sequence_length, y_ind):
#    data_size, features = data.shape
#    out_data = np.zeros((data_size-sequence_length-1, sequence_length, features))
#    out_y = np.zeros(data_size-sequence_length-1)
#    for i in range(out_data.shape[0]):
#        out_data[i,:,:] = data[i:sequence_length+i,:]
#        out_y[i] = data[sequence_length+i+1,y_ind]
#    return out_data, out_y
def reshape_sequences(data, labels, sequence_length):
    data_size, features = data.shape
    out_data = np.zeros((data_size-sequence_length-1, sequence_length, features))
    out_y = np.zeros(data_size-sequence_length-1)
    for i in range(out_data.shape[0]):
        out_data[i,:,:] = data[i:sequence_length+i,:]
        out_y[i] = labels[sequence_length+i+1]
    return out_data, out_y

def reshape_chunks(data, labels, chunk_length, sequence_length): # reshape each chunk into seq and stack
    reshaped_data = []
    reshaped_labels = []
    data_length, features = data.shape
    num_chunks = int(data_length/chunk_length)
    for i in range(num_chunks):
        new_data, new_labels = reshape_sequences(data[i*chunk_length:(i+1)*chunk_length,:],labels[i*chunk_length:(i+1)*chunk_length],sequence_length)
        if len(reshaped_data) == 0:
            reshaped_data = new_data
            reshaped_labels = new_labels
        else: 
            reshaped_data = np.vstack((reshaped_data,new_data))
            reshaped_labels = np.append(reshaped_labels,new_labels)

            
    return reshaped_data, reshaped_labels
#x_raw, y_raw = reshape_sequences(raw_data, sequence_length, y_ind)
# output is (shortened_data_samples, sequence_length, features), y_raw is (shortened_data_samples)

cx_train, cy_train = reshape_chunks(X_train, y_train, chunk_length, sequence_length)
cx_dev, cy_dev = reshape_chunks(X_dev, y_dev, chunk_length, sequence_length)
cx_test, cy_test = reshape_chunks(X_test, y_test, chunk_length, sequence_length)

#print(bx_train.shape,X_train.shape, by_train.shape)


# In[10]:


def batch(data, labels, batch_size):
    data_length, sequence_length, features = data.shape
    new_data_length = data_length - (data_length % batch_size)
    num_batches = int(data_length/batch_size)
    data_out = np.reshape(data[:new_data_length,:,:],(num_batches, -1, sequence_length, features))
    labels_out = np.reshape(labels[:new_data_length],(num_batches, -1)) # would need to add dim for num_outputs
    return data_out, labels_out
    
bx_train, by_train = batch(cx_train, cy_train, batch_size)


# In[11]:


def initialize_parameters(num_hid_layers, size_hid_layers, n_x, output_size):
    parameters = {}
    total_layers = num_hid_layers+1
    
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


# In[12]:


def forward_prop(X,parameters):
    total_layers = len(parameters)//2
    layer_outputs = {}
    layer_outputs['A0'] = X
    
    for l in range(1,total_layers+1):
        layer_outputs['Z' + str(l)] = tf.matmul(parameters['w' + str(l)],layer_outputs['A' + str(l-1)])+parameters['b' + str(l)]
        layer_outputs['A' + str(l)] = tf.nn.relu(layer_outputs['Z' + str(l)])
        #layer_outputs['A' + str(l)] = tf.nn.dropout(layer_outputs['A' + str(l)],keep_prob)
    
    return layer_outputs['Z' + str(total_layers)]


# In[88]:


def model(X_train, Y_train, X_dev, Y_dev, X_test, Y_test, learning_rate, beta, k_p, reg_type,
          num_epochs, num_hid_layers, size_hid_layers, minibatch_size, log_dir, print_loss = True, 
          plotting = True, print_interval=10, plot_interval=100, parallel=False, queue=0):
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    (num_batches, m, seq_length, n_x) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    losses = []                                        # To keep track of the cost
    train_errs = []                                        # To keep track of the cost
    epochs = [] # keep track of epoch number
    dev_errs = []                                        # To keep track of the cost
    eps = 10**-8.
    O = 1 # output size?
    
    w1_shape = [O, size_hid_layers]
    b1_shape = [O,1]
    
    w1 = tf.get_variable("w1", w1_shape, initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable("b1", b1_shape, initializer = tf.zeros_initializer())
    
    X = tf.placeholder(tf.float32,[None, seq_length, n_x]) # inputs to LSTM are (# data examples, num_steps, input_size)
    Y = tf.placeholder(tf.float32,[None])
    
    # array for the size of the RNN layers
    LSTMsizes = size_hid_layers*np.ones(num_hid_layers).astype(int)
    # create LSTMCells
    rnn_layers = [tf.nn.rnn_cell.LSTMCell(size_hid_layers) for size_hid_layers in LSTMsizes]

    # create a RNN cell composed sequentially of a number of RNNCells
    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

    outputs, state = tf.nn.dynamic_rnn(cell=multi_rnn_cell, inputs=X, dtype=tf.float32)
    #print(outputs.shape)
    #print(outputs[:,1,:].shape)
    # WANT 1 OUTPUT value so outputs is ideally hidden_layer x # data samples, but currently its # data samples, seq, features
    # the 20 outputs are for each sequence step, want the last one I believe
    Z1 = tf.matmul(w1,tf.transpose(outputs[:,-1,:]))+b1 # fully connected layer
    out = Z1
    
    #parameters = initialize_parameters(num_hid_layers, size_hid_layers, n_x, O)
    #out = forward_prop(X, parameters)
    
    #loss = tf.reduce_mean(tf.squared_difference(out, Y)) # L2 loss --> not good for our problem
    loss = tf.reduce_mean(tf.losses.absolute_difference(Y,tf.squeeze(out))) # L1 loss
    
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss) # Optimizer, change the learning rate here

    init = tf.global_variables_initializer() # When init is run later (session.run(init)),
    
    log_name = "Model_"+str(learning_rate)+"_"+str(num_epochs)+"_"+str(num_hid_layers)+"_"+str(size_hid_layers)
    if print_loss:
        utils.set_logger(os.path.join(cwd+"/"+log_dir,log_name+'.log'))
        utils.logging.info("START OF NEW MODEL")
        utils.logging.info("learning rate = %f, hidden layers = %d, hidden units = %d, epochs = %d", learning_rate, num_hid_layers, size_hid_layers, num_epochs) # add other hyperparams
        utils.logging.info("L2 beta = %f, Dropout Keep Prob = %f", beta, k_p)
    
    
    with tf.Session() as sess: # starting tf session --> all computation on tf graph in this with struct
        sess.run(init)
        for epoch in range(2):#num_epochs):
            train_err_batches = []
            for batch in range(num_batches):
                x_batch = np.squeeze(X_train[batch,:,:,:])
                y_batch = np.squeeze(Y_train[batch,:])
                _, loss_val, batch_pred = sess.run([optimizer, loss, out], feed_dict={X: x_batch, Y: y_batch})
                train_err_ind = np.divide(np.mean(abs(batch_pred - y_batch)),np.mean(abs(y_batch))) # absolute error
                train_err_batches.append(train_err_ind)
                if (batch % int(num_batches/print_interval)) == 0:
                    #print("Loss: ",loss_val)
                    # Output the predictions
                    #train_pred = sess.run(out, feed_dict={X: X_train})
                    losses.append(loss_val)
                    dev_pred = sess.run(out, feed_dict={X: X_dev})
                    train_err = np.mean(train_err_batches)#np.divide(np.mean(abs(train_pred - Y_train)),np.mean(abs(Y_train))) # absolute error
                    dev_err = np.divide(np.mean(abs(dev_pred - Y_dev)),np.mean(abs(Y_dev))) # absolute error
                    train_errs.append(train_err)
                    dev_errs.append(dev_err)
                    epochs.append(epoch)
                    if print_loss:
                        utils.logging.info("Batch %d loss: %f", epoch*num_batches+batch, loss_val)    
                        utils.logging.info("Train error: %f", train_err)
                        utils.logging.info("Dev error: %f", dev_err)
        test_pred = sess.run(out, feed_dict={X: X_test})
        test_err = np.divide(np.mean(abs(test_pred - Y_test)),np.mean(abs(Y_test))) # absolute error

        #print('Train error: ', train_err)
        #print('Dev error: ', dev_err)
        if print_loss:
            utils.logging.info("Final test error: %f", test_err)
        if plotting:
            # Plot percent errors during iterations
            fig = plt.figure(num=None, dpi=200, facecolor='w', edgecolor='k'); ax = plt.gca()
            plt.plot(np.squeeze(train_errs))
            plt.plot(np.squeeze(dev_errs))
            ax.set_yscale('log')
            plt.ylabel('Percent error')
            plt.title("Error for learning rate = " + str(learning_rate))
            plt.show()
    ind = np.argmin(dev_errs)
    min_dev = dev_errs[ind]
    min_epoch = epochs[ind] 
    
    results = train_err, Y_train, batch_pred, Y_dev, dev_pred, dev_err, min_dev, min_epoch, test_err
    #results = train_err, dev_err, test_err, min_dev, min_epoch
    return results            


# In[89]:


# Does creating a log file in each call to the model function mess up the main log file?

def hyperparamSearch(X_train, Y_train, X_dev, Y_dev, X_test, Y_test, lr_rng, num_hid_layers_rng, beta_rng, k_p_rng, reg_type,
                     size_hid_layers_rng, num_sims, num_epochs, minibatch_size, log_dir, parallel=False, cores=1):
    # compute random values within the ranges for each param of length num_sims
    num_batches, batch_length, sequence_length, num_features = X_train.shape 
    num_params = 5
    np.random.seed(13) # set seed for rand
    lower_bounds = [lr_rng[0],num_hid_layers_rng[0],size_hid_layers_rng[0],beta_rng[0],k_p_rng[0]]
    upper_bounds = [lr_rng[1],num_hid_layers_rng[1],size_hid_layers_rng[1],beta_rng[1],k_p_rng[1]]
    sample_size = [num_sims, num_params] # num_sims x number of params in search
    samples_params = np.random.uniform(lower_bounds, upper_bounds, sample_size)

    # modifying the initial random parameters
    lr_samples = 10**samples_params[:,0] # log scale
    hl_samples = samples_params[:,1].astype(int) # rounded down to nearest int
    hu_samples = (samples_params[:,2]*num_features).astype(int) # base of 10 neurons used for each level
    beta = samples_params[:,3]
    k_p = samples_params[:,4]
    
    # save the data for the ranges used to the main sim file
    log_name = "Model_"+str(learning_rate)+"_"+str(num_epochs)+"_"+str(num_hid_layers)+"_"+str(size_hid_layers)
    utils.set_logger(os.path.join(cwd+"/"+log_dir,log_name+'.log'))
    utils.logging.info("lr_rng = "+str(lr_rng)+" hidden layers rng = "+str(num_hid_layers_rng)+" hidden units rng = "+str(size_hid_layers_rng)+" num sims = %d", num_sims)
    
    results = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    
    if parallel: # Need cores 
        print("parallelizing the training") # add parallel ability for specified number of cores similar to funct above
        results = multi_sim(X_train, Y_train, X_dev, Y_dev, X_test, Y_test, lr_samples, beta, k_p, reg_type,
                            num_epochs, hl_samples, hu_samples, minibatch_size, log_dir, False, False, 10, 
                            100, True, cores)
        print(results)
    else:
        for i in range(len(lr_samples)):
            
            train_err, Y_train, batch_pred, Y_dev, dev_pred, dev_err, min_dev, min_epoch, test_err = model(X_train, Y_train, X_dev, Y_dev, X_test, Y_test, lr_samples[i], beta[i], k_p[i], reg_type,
                                                                 num_epochs, hl_samples[i], hu_samples[i], minibatch_size, log_dir, False, False) # call model funct

            temp_results = np.array([lr_samples[i], hl_samples[i], hu_samples[i], beta[i], k_p[i], num_epochs, train_err, dev_err, test_err, min_epoch, min_dev])
            #utils.set_logger(os.path.join(cwd+"/"+log_dir,log_dir+'.log')) # reset logger to main log file
            utils.logging.info("START OF NEW MODEL")
            utils.logging.info("learning rate = %f, hidden layers = %d, hidden units = %d, beta = %f, keep_prob = %f, epochs = %d, reg_type = %s", lr_samples[i], hl_samples[i], hu_samples[i], beta[i], k_p[i], num_epochs, reg_type) # add other hyperparams
            utils.logging.info("Train Err = %f, Dev Err = %f, Test Err = %f, Min Dev Err = %f, Min Epoch = %d", train_err, dev_err, test_err, min_dev, min_epoch) # add other hyperparams
            results = np.vstack((results,temp_results))# get all results in a list
        
        # results contain an array of the parameters and then the resulting errors
        results = results[1:,:] # get rid of placeholder row
        results= results[results[:,-1].argsort()] # sort by the lowest dev error
        utils.logging.info("RESULTS")
        utils.logging.info(str(results))
    return results


# In[90]:


learning_rate = 0.008#0.0005
num_epochs = 25 # total number of epochs to iterate through
print_interval = 1 # prints per epoch
minibatch_size = 10
num_hid_layers = 1
size_hid_layers = 64
k_p = 1.
reg_type = 'None'
beta = 0
log_dir = "test_dir"

#results=model(bx_train, by_train, cx_dev, cy_dev, cx_test, cy_test, learning_rate, beta, k_p, reg_type, num_epochs, num_hid_layers, size_hid_layers, minibatch_size, log_dir)

# define the ranges for the hyperparam search
lr_rng = [-4,-2] # range of lr will be done with these values use as 10^r for log scale
num_hid_layers_rng = [2,7] # these will be rounded down to nearest Int so add 1 more to value desired for upper bound
size_hid_layers_rng = [5.0, 20.0] # These values are multiplied by the number of input features
beta_rng = [0.0, 0.0] # regularization coefficient range
k_p_rng = [1.0, 1.0] # dropout "keep probability" range
reg_type = "L2" # set to type of desired regularization "L2", "none" (I don't think L1 is necessary since we don't want feature selection)
parallel = False
cores = 1
num_sims = 10

results = hyperparamSearch(bx_train, by_train, cx_dev, cy_dev, cx_test, cy_test, lr_rng, num_hid_layers_rng, beta_rng, k_p_rng, reg_type,
                     size_hid_layers_rng, num_sims, num_epochs, minibatch_size, log_dir, parallel, cores)

