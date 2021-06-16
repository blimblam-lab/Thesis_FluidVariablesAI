import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras as K
from tensorflow.keras.layers import Dense, LSTM

#from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf #toegevoegd voor het laden
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

#Rel RMSE 1: 0.045 | 2: 0.032 | 3: 0.0 (20p)


path = 'C:/Users/joshu/Documents/Joshua School/Masterproef/LSTM Info/LSTMTests/f&dsep/Shifts/NewTries/'
plt.close('all')
Utrain = pd.read_csv(path+'Utrain1.csv', header=None).to_numpy()
Utest = pd.read_csv(path+'Utest1.csv', header=None).to_numpy()
Udev = pd.read_csv(path+'Udev1.csv', header=None).to_numpy()
ytrain = pd.read_csv(path+'ftrain1.csv', header=None).to_numpy()
ytest = pd.read_csv(path+'ftest1.csv', header=None).to_numpy()
ydev = pd.read_csv(path+'fdev1.csv', header=None).to_numpy()
dtrain = pd.read_csv(path+'dtrain1.csv', header=None).to_numpy()
dtest = pd.read_csv(path+'dtest1.csv', header=None).to_numpy()
ddev = pd.read_csv(path+'ddev1.csv', header=None).to_numpy()

UTrain = Utrain[:,0:21]    # of input - take inputs that were shifted for f (second set of the three 21-input sets in the whole)
yTrain = ytrain  #

UTest = Utest[:,0:21]
yTest = ytest

UDev = Udev[:,0:21]
yDev = ydev

time_stepsd = 80 # let state evolve over 10 hours  #MEMORY PROBLEMS NEAR 100
batch_sized = UTrain.shape[0]-time_stepsd+1   #9319
features = 21
output_dim = 70 # use 30 outputs which are recombined using a dense net afterwords

UTrain_LSTM =np.zeros((batch_sized,time_stepsd,features))

for i in range(batch_sized):
    UTrain_LSTM[i,:,:] = UTrain[i:i+time_stepsd,:]       # [0-9],[1-10],[2-11],...

yTrain_LSTM = yTrain[time_stepsd-1:] # the first output is generated after 10 hours

# Validation data
time_steps_devd = time_stepsd # must be equal!
batch_size_devd = UDev[:].shape[0]-time_steps_devd+1
UDev_LSTM = np.zeros((batch_size_devd,time_steps_devd,features))

for i in range(batch_size_devd):
    UDev_LSTM[i,:,:] = UDev[i:i+time_steps_devd,:]

yDev_LSTM = yDev[time_steps_devd-1:]

# Testing data
time_steps_testd = time_stepsd # must be equal!
batch_size_testd = UTest[:].shape[0]-time_steps_testd+1
UTest_LSTM = np.zeros((batch_size_testd,time_steps_testd,features))

for i in range(batch_size_testd):
    UTest_LSTM[i,:,:] = UTest[i:i+time_steps_testd,:]

yTest_LSTM = yTest[time_steps_testd-1:]

n_epoch = 3000
optim = K.optimizers.Adam()
lossf = K.losses.MeanSquaredError()  #   mean_squared_error
metric = [K.metrics.RootMeanSquaredError(), K.metrics.MeanAbsoluteError()]
#regul = K.regularizers.l2(0.001)
batch_size = 128

model = K.Sequential()
model.add(LSTM(output_dim,input_shape=(time_stepsd,features),return_sequences=False,stateful=False,name='LSTM1'))
model.add(Dense(80, activation='tanh',name='layer2'))
model.add(Dense(80, activation='tanh',name='layer3'))
model.add(Dense(60, activation='tanh',name='layer4'))
model.add(Dense(40, activation='tanh',name='layer5'))
model.add(Dense(40, activation='tanh',name='layer6'))
model.add(Dense(30, activation='tanh',name='layer7'))
model.add(Dense(20, activation='tanh',name='layer8'))
model.add(Dense(15, activation='tanh',name='layer9'))
model.add(Dense(10, activation='tanh',name='layer10'))
model.add(Dense(5, activation='tanh',name='layer11'))
model.add(Dense(1, activation='linear',name='outputLayer'))

model.compile(optimizer='Adam', loss='mse', metrics=metric)
earlystop = K.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=1)

'''runfit = model.fit(UTrain_LSTM,yTrain_LSTM,batch_size=batch_size,epochs=n_epoch,shuffle=True, callbacks=earlystop,
                   validation_data=(UDev_LSTM,yDev_LSTM))#
model.save('LSTMf80tsBatch2', save_format='h5')
hist = pd.DataFrame(runfit.history)    #progress printen
hist.to_csv('LSTMf80tsBatch2hist.csv')
'''
model = K.models.load_model('LSTMf80tsBatch2')

yTrain_sim = model.predict(UTrain_LSTM)

yTest_sim = model.predict(UTest_LSTM)
model_eval = model.evaluate(UTest_LSTM,yTest_LSTM)

yTrain_simc = yTrain_sim[:-20]
yTrain_LSTMc = yTrain_LSTM[:-20]
yTest_simcutoff = yTest_sim[-20:]
yTest_simc = yTest_sim[:-20]
yTest_LSTMc = yTest_LSTM[:-20]

def rel_rms_np(y_true,y_sim):
    return np.sqrt(np.mean(np.square(y_true-y_sim)))/np.sqrt(np.mean(np.square(y_true-np.mean(y_true))))

'''def rel_rms_quick(y_tr,y_sim):
    return np.sqrt(np.square(y_tr-y_sim)/np.square(y_tr))'''

rel_RMSEtr = rel_rms_np(yTrain_simc,yTrain_LSTMc)

yDev_sim = model.predict(UDev_LSTM)
yDev_simc = yDev_sim[:-20]
yDevLSTMc = yDev_LSTM[:-20]

rel_RMSEdev = rel_rms_np(yDev_simc,yDevLSTMc)

print('Tr Rel rms: ' + str(rel_rms_np(yTrain_LSTMc,yTrain_simc)))
print('Tst Rel rms: ' + str(rel_rms_np(yTest_LSTMc,yTest_simc)))
#print('Tst Qck Rel rms: ' + str(rel_rms_quick(yTest_LSTMc,yTest_simc)))

#metrics
num_trainsamples = len(yTrain_LSTMc)
num_testsamples = len(yTest_LSTMc)
num_params = model.count_params()
errd = (yTest_simc-yTest_LSTMc)

abs_err = np.abs(errd)           #abs = absolute waarde (alles >=0 en element |R)
avg_err = np.mean(errd)
std_err = np.std(errd)
loss = np.sum(errd**2)/num_testsamples       #MSE
rel_err = loss/(np.sum(np.abs(yTest_LSTMc)**2)/num_testsamples)   #MSE/Mean Squared juiste waarde LSTMc voor _perfo2
rel_RMSE = np.sqrt(rel_err)

LSTMModel_perfo = {'Model activations': 'ReLU-Linear',
                    'Number of training samples': num_trainsamples,
                    'Number of model parameters': num_params,
                    'Ratio training samples-parameters':
                    num_trainsamples/num_params,
                    'Model loss on test data': model_eval[0],
                    'Model RMSE on test data': model_eval[1],
#                    'Model Absolute error on test data': model_eval[2],
                    'Number of test samples': num_testsamples,
                    'Average Error': avg_err,
                    'Standard Deviation of Error': std_err,
                    'Average Absolute Error': np.mean(abs_err),
                    'Maximum Error': np.max(abs_err),
                    'Loss': loss,
                    'Relative Error': rel_err,
                    'Relative RMSE on train set': rel_RMSEtr,
                    'Relative RMSE on validation set': rel_RMSEdev,
                    'Relative RMSE on test set': rel_RMSE}

'''ModelPerfo = pd.DataFrame.from_dict(LSTMModel_perfo, orient='index')       #toch nog is bezien tekst hierboven in een dataframe gieten (csv of tex uiteindelijk)
ModelPerfo.index.name = 'LSTM_f_80tsBatch2_perfo'
ModelPerfo.to_csv('LSTM_f_80tsBatch2_perfo.csv', header=['Value'])
ModelPerfo.to_latex('LSTM_f_80tsBatch2_perfo.tex', header=['Value'])'''

#plotting
hist = pd.read_csv('LSTMf80tsBatch2hist.csv')
plt.plot(hist["root_mean_squared_error"][:], 'r-', label='rmse')
plt.plot(hist["val_root_mean_squared_error"][:], 'b-', label='val-rmse')
plt.legend()

time = np.arange(0.00, len(yTest_LSTM)*0.01, step=0.01)
timeld = np.arange(0.00, len(yTest)*0.01, step=0.01)
timecd = timeld[time_stepsd-1:-20]#np.arange(0.00, len(yTest_LSTMc)*0.02, step=0.02)     #timestep is gekend als 0.01, tijdsrange namaken | hier ineens wel 0 als beginwaarde gebruiken?
timecutoffd = timeld[-20:]
#time = time[:-1]
timecd.shape = (len(timecd), 1)
print(str(timecd.shape) +"|"+ str(yTest_simc.shape))
#assert(timec.shape == yTest_LSTMc.shape)

fig, axs = plt.subplots(2,1,sharex=True)                    #nu wordt displacement geplot
axs[0].plot(timeld, yTest, 'b-', label='y true values')
axs[0].plot(timecd, yTest_simc, 'r:', label='y predicted values')
axs[0].plot(timecutoffd, yTest_simcutoff,'g:', label='cut off samples')
axs[0].grid(b=1,which='both')
axs[1].plot(timecd, errd, 'k', label='Error')
plt.setp(axs[0], ylabel='y true')
plt.setp(axs[1], ylabel='y pred')
plt.setp(axs[-1], xlabel='Time (s)', ylabel='Error')
axs[1].grid(b=1,which='both')

fig.suptitle('Predictions and error of the trained model', fontsize=14)
fig.legend(loc='upper right')

plt.rcParams.update({'font.size': 15})
fig, axs = plt.subplots(2,1,sharex=True)                  #nu wordt displacement geplot
axs[0].plot(timeld[:-10], yTest[:-10], 'b-', label='Target values')
axs[0].plot(timecd, yTest_simc, 'r:', label='Predicted values')
axs[1].plot(timecd, errd, 'k', label='Error')
plt.setp(axs[0], ylabel='y/D')
plt.setp(axs[1], xlabel='Time (s)', ylabel='Error')
axs[0].grid(b=1,which='major')
axs[1].grid(b=1,which='major')
axs[0].legend(loc='upper right')
axs[0].title.set_text('y/D')
axs[1].title.set_text('Error')
fig.suptitle('Displacement model test results', fontsize=20)

#plt.rcParams.update({'font.size': 15})
timel = np.arange(0.00, len(Utest[:,7])*0.01, step=0.01)
fig, axs = plt.subplots(3,1,sharex=True)
axs[0].plot(timel, dtest)
axs[1].plot(timel, ytest)
axs[2].plot(timel, Utest[:,7])
plt.setp(axs[2], xlabel='Time (s)', ylabel='m/s')
#axs[1].grid(b=1,which='both')
axs[0].title.set_text('Target - y/D')
axs[1].title.set_text('Target - $c_f$')
axs[2].title.set_text('Input - $V_y$[8]')
fig.suptitle('Target and input data examples', fontsize=20)

'''fig = plt.figure()
plt.rcParams.update({'font.size': 15})
plt.plot(timel[:-10], yTest[:-10], 'b-', label='y/D true values')
plt.plot(timec, yTest_simc, 'r:', label='y/D predicted values')
plt.plot(timec, errd, 'k', label='Error')
plt.setp(axs[0], xlabel='Time (s)', ylabel='y/D')
plt.grid(b=1,which='major')
plt.suptitle('Displacement: Test predictions and error', fontsize=18)
plt.legend()'''

plt.show()