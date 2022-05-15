# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 10:42:09 2021

@author: Mahdi
"""
import os
import collections
import numpy as np
import scipy.io as sio
import tensorflow as tf
import tensorflow_federated as tff
import matplotlib.pyplot as plt

################################ Functions ################################

def dc_compensation(z):
    """
    Function to estimate and remove DC impairments in the IQ branch
    Parameters:
    z: DC impaired signal sequence (numpy format)
    Returns:
    v: DC removed signal sequence
    """
    iDCest=np.mean(z[:, 0]) # estimated DC on I branch
    qDCest=np.mean(z[:, 1]) # estimated DC on I branch
    v=z-(iDCest+1j*qDCest) # remove estimated DCs
    return v

def blind_iq_compensation(z):
    """
    Function to estimate and compensate IQ impairments for the
    single-branch IQ impairment model
    Parameters:
    z: DC impaired signal sequence (numpy format)
    Returns:
    y: IQ imbalance compensated signal sequence
    """
    I=z[:, 0];Q=z[:, 1]
    theta1=(-1)*np.mean(np.sign(I)*Q)
    theta2=np.mean(abs(I)); theta3=np.mean(abs(Q))
    c1=theta1/theta2
    c2=np.sqrt((theta3**2-theta1**2)/theta2**2)
    return I +1j*(c1*I+Q)/c2


def usergen(x, y, dataset_setup, iid, N0):
    if iid==True:
        sigma = 1 # i.i.d. user fading 
    else:
        sigma = np.random.uniform(low=0.5, high=1.5, size=None) #non-i.i.d. user fading
        
    h = np.random.rayleigh(scale=sigma, size=(x.shape[0],1))
    hIQ=np.concatenate((h,h),axis=1)
    x_u = np.multiply(hIQ,x)+ np.random.normal(0, np.sqrt(N0)/2, x.shape) # Channel-distorted noisy features
    y_u = np.real(y) # Labels
    # print(y_u)
    y_u = np.interp(y_u, (y_u.min(), y_u.max()), (-0.5, 0.5))

    x_u1 = dc_compensation(x_u)  # DC compensation
    # Through Blind IQ compensation after DC compensation
    x_u2 = blind_iq_compensation(x_u1)
    new_x_u2 = np.zeros((len(x_u2), 2))
    new_x_u2[:, 0] = np.real(x_u2)
    new_x_u2[:, 1] = np.imag(x_u2)
    # print(new_x_u2.shape, y_u.shape)
    # print(f'train : {max(y_u)}, {min(y_u)}')

    if dataset_setup=='original':
        D_u = tf.data.Dataset.from_tensor_slices((list(x_u),list(y_u.astype(int))))
    else:
        D_u = tf.data.Dataset.from_tensor_slices((list(new_x_u2),list(y_u.astype(int))))

    return D_u


def testgen(x, y, dataset_setup, iid, N0):
    if iid==True:
        h = np.random.rayleigh(scale=1, size=(x.shape[0],1)) # i.i.d. test fading 
    else:
        h = np.random.rayleigh(scale=np.random.uniform(low=0.5, high=1.5, size=(x.shape[0],1)), size=(x.shape[0],1)) # non-i.i.d. user fading
    hIQ = np.concatenate((h,h),axis=1)
    x_test = np.multiply(hIQ,x) + np.random.normal(0, np.sqrt(N0)/2, x.shape) # Channel-distorted noisy features
    y_test = np.real(y) # Labels
    # print(y_test)
    y_test = np.interp(y_test, (y_test.min(), y_test.max()), (-0.5, 0.5))
    # print(f'test : {max(y_test)}, {min(y_test)}')
    
    x_test1 = dc_compensation(x_test)  # DC compensation
    # Through Blind IQ compensation after DC compensation
    x_test2 = blind_iq_compensation(x_test1)
    new_x_test2 = np.zeros((len(x_test2), 2))
    new_x_test2[:, 0] = np.real(x_test2)
    new_x_test2[:, 1] = np.imag(x_test2)
    # print(new_x_test2.shape, y_test.shape)

    if dataset_setup=='original':
        dataset_test = tf.data.Dataset.from_tensor_slices((list(x_test),list(y_test.astype(int))))
    else:
        dataset_test = tf.data.Dataset.from_tensor_slices((list(new_x_test2),list(y_test.astype(int))))
    
    return dataset_test


def preprocess(dataset):
    U = 5 # Number of wireless users taking part in federated training
    EPOCH = 5 # Number of local epochs for each aggragation round
    AGGREGATION_ROUND = 5 # Number of federated aggregation rounds
    BATCH_SIZE = 1000 
    SHUFFLE_BUFFER = 20
    PREFETCH_BUFFER = 10
    def batch_format_fn(element1,element2):
        return collections.OrderedDict(
            x=tf.reshape(element1, [-1, 2]),
            y=tf.reshape(element2, [-1, 1]))
    return dataset.repeat(EPOCH).shuffle(SHUFFLE_BUFFER).batch(
        BATCH_SIZE).map(batch_format_fn).prefetch(PREFETCH_BUFFER)


def create_keras_model():
    Q = 4 
    M = 2**Q # 16QAM modulation
    Es = 10 # Average symbol energy
    return tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(2,)),
        tf.keras.layers.Dense(M, activation=tf.keras.activations.tanh),])


def model_fn():
  keras_model = create_keras_model()
  return tff.learning.from_keras_model(
      keras_model,
      input_spec=preprocessed_example_dataset.element_spec,
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    
    
################################ Main ################################
def train(snr_g, iid_g, dataset_setup_g):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    MC = 1000 # Number of Monte Carlo trials
    iid = iid_g # set to False to simulate non-iid user fading 

    ### Modulation Parameters
    Q = 4 
    M = 2**Q # 16QAM modulation
    Es = 10 # Average symbol energy

    ### Noise Parameters
    EbN0_dB = snr_g # SNR per bit (dB)
    EsN0_dB = EbN0_dB + 10*np.log10(Q) # SNR per symbol (dB)
    N0 = Es/10**(EsN0_dB/10)

    ### FedRec Training Parameters
    U = 5 # Number of wireless users taking part in federated training
    EPOCH = 5 # Number of local epochs for each aggragation round
    AGGREGATION_ROUND = 5 # Number of federated aggregation rounds
    BATCH_SIZE = 20
    SHUFFLE_BUFFER = 20
    PREFETCH_BUFFER = 10

    ### Load QAM I/Q Symbols
    TRAIN = sio.loadmat('TRAIN_fedrec.mat')
    Train = TRAIN['TRAIN']
    TEST = sio.loadmat('TEST_fedrec.mat')
    Test = TEST['TEST']

    DATASET_SETUP = dataset_setup_g

    ### Generate Real-valued Train/Test Features/Labels
    N_T = int(Train.shape[0]/U) # Size of local user datasets 
    x_train = np.zeros((N_T,2))
    x_train[:,0] = np.real(Train[0:N_T,0])
    x_train[:,1] = np.imag(Train[0:N_T,0])
    y_train = Train[0:N_T,1]-1

    x_test = np.zeros((Test.shape[0],2))
    x_test[:,0] = np.real(Test[:,0])
    x_test[:,1] = np.imag(Test[:,0])
    y_test = Test[:,1]-1


    example_dataset = usergen(x_train, y_train, DATASET_SETUP, iid, N0)
    # print(f"uplink dataset: U_U: {len(example_dataset)}")
    preprocessed_example_dataset=preprocess(example_dataset)
    example_element = next(iter((preprocessed_example_dataset)))

    def model_fn():
        keras_model = create_keras_model()
        return tff.learning.from_keras_model(
            keras_model,
            input_spec=preprocessed_example_dataset.element_spec,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])


    iterative_process = tff.learning.build_federated_averaging_process(
        model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=0.05),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))

    evaluation = tff.learning.build_federated_evaluation(model_fn)
    print(evaluation)


    BER = 0
    for i in range(MC):
        
        print('##############################')
        print('Monte Carlo Trial # ', i+1)
        
        ### Generate Federated User Datasets
        federated_train_data=[]
        for u in range(U):
            D_u = usergen(x_train, y_train, DATASET_SETUP, iid, N0) ### Generate Local Dataset at user u
            # print(f"downlink dataset: D_U: {len(D_u)}")
            federated_train_data.append(preprocess(D_u))
                
        ### Generate Test Dataset
        test_dataset = testgen(x_test, y_test, DATASET_SETUP, iid, N0)
        federated_test_data=[preprocess(test_dataset)]
        
        ### Federated Training
        state = iterative_process.initialize()
        for n in range(AGGREGATION_ROUND):
            state, metrics = iterative_process.next(state, federated_train_data)
            # print(str(metrics))
            
            ### Evaluate the Model
            test_metrics = evaluation(state.model, federated_test_data)
            print(str(test_metrics))
        
        BER = BER + (1-test_metrics['sparse_categorical_accuracy'])/(Q*MC)

        
    print('##############################')
    print('16QAM at Eb/N0=', EbN0_dB, 'dB')
    print('FedRec trained collaboratively by ', U, 'users')
    if iid==True:
            iidstr = 'iid' 
    else:
            iidstr = 'non-iid'
    print(iidstr, ' Rayleigh fading')
    print('BER= ', BER)

    return BER

def main():
    iid = False
    dataset_setup = 'original'
    ber = []
    for i in range(1, 16):
        ber_new = train(i, iid, dataset_setup)
        ber.append(ber_new)
        with open(f'tanh_plot_ber_iid_{iid}_dataset_setup_{dataset_setup}.txt', 'a+') as f:
            f.write(str(i) + ':' + str(ber_new) + '\n')

    with open(f'tanh_plot_ber_iid_{iid}_dataset_setup_{dataset_setup}.txt', 'a+') as f:
            f.write(str(ber) + '\n')

    plt.figure()
    plt.plot(range(1, 16), ber, color = 'green', marker='o', linestyle='dashed')
    plt.xlabel('Eb0/N0')
    plt.ylabel('BER')
    plt.title(f'tanh_plot_ber_iid_{iid}_dataset_setup_{dataset_setup}')
    plt.savefig(f'tanh_plot_ber_iid_{iid}_dataset_setup_{dataset_setup}.png')
    plt.show()

main()