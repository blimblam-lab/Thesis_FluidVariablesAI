import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras as K
from tensorflow.keras.layers import Dense, LSTM

#from sklearn.preprocessing import MinMaxScaler
'''import tensorflow as tf #toegevoegd voor het laden
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)'''

#Rel RMSE 1:0.039 | 2.1:0.052 | 2.2:0.053 (30p) | 2.3:0.045 (20p) | 3:0.0

path = 'C:/Users/joshu/Documents/Joshua School/Masterproef/LSTM Info/LSTMTests/f&dsep/Shifts/NewTries/'  #/2.AllnodesSeparateShiftFor_f/a.+2backshifts/'

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

time_steps = 50 # let state evolve over 10 hours  #PROBLEMS NEAR 100
batch_sizef = UTrain.shape[0]-time_steps+1 #18746
features = 21
output_dim = 40 # use 30 outputs which are recombined using a dense net afterwords

UTrain_LSTM =np.zeros((batch_sizef,time_steps,features))

for i in range(batch_sizef):
    UTrain_LSTM[i,:,:] = UTrain[i:i+time_steps,:]       # [0-9],[1-10],[2-11],...

yTrain_LSTM = yTrain[time_steps-1:] # the first output is generated after 10 hours

# Validation data
time_steps_dev = time_steps # must be equal!
batch_size_dev = UDev[:].shape[0]-time_steps_dev+1
UDev_LSTM = np.zeros((batch_size_dev,time_steps_dev,features))

for i in range(batch_size_dev):
    UDev_LSTM[i,:,:] = UDev[i:i+time_steps_dev,:]

yDev_LSTM = yDev[time_steps_dev-1:]

# Testing data
time_steps_test = time_steps # must be equal!
batch_size_test = UTest[:].shape[0]-time_steps_test+1
UTest_LSTM = np.zeros((batch_size_test,time_steps_test,features))

for i in range(batch_size_test):
    UTest_LSTM[i,:,:] = UTest[i:i+time_steps_test,:]

yTest_LSTM = yTest[time_steps_test-1:]

n_epoch = 3000
optim = K.optimizers.Adam()
lossf = K.losses.MeanSquaredError()  #   mean_squared_error
metric = [K.metrics.RootMeanSquaredError(), K.metrics.MeanAbsoluteError()]
regul = K.regularizers.l2(0.001)
batch_size = 128

model = K.Sequential()
model.add(LSTM(output_dim,input_shape=(time_steps,features),return_sequences=False,stateful=False,name='LSTM1'))
model.add(Dense(50, activation='relu',name='layer2'))
model.add(Dense(50, activation='relu',name='layer3'))
model.add(Dense(30, activation='relu',name='layer4'))
model.add(Dense(20, activation='relu',name='layer5'))
model.add(Dense(15, activation='relu',name='layer6'))
model.add(Dense(10, activation='relu',name='layer7'))
model.add(Dense(5, activation='relu',name='layer8'))
model.add(Dense(1, activation='linear',name='outputLayer'))

model.compile(optimizer='Adam', loss='mse', metrics=metric)
earlystop = K.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=1)

runfit = model.fit(UTrain_LSTM,yTrain_LSTM,batch_size=batch_size,epochs=n_epoch,shuffle=True, callbacks=earlystop,
                   validation_data=(UDev_LSTM,yDev_LSTM))#,validation_data=(UVal,yVal),validation_steps=5)
model.save('LSTMf50tsBatch3', save_format='h5')
hist = pd.DataFrame(runfit.history)    #progress printen
hist.to_csv('LSTMf50tsBatch3hist.csv')

model = K.models.load_model('LSTMf50tsBatch3')

yTrain_sim = model.predict(UTrain_LSTM)

yTest_sim = model.predict(UTest_LSTM)
model_eval = model.evaluate(UTest_LSTM,yTest_LSTM)

yTrain_simc = yTrain_sim[:-20]
yTrain_LSTMc = yTrain_LSTM[:-20]
yTest_simcutoff = yTest_sim[-20:]
yTest_simc = yTest_sim[:-20]
yTest_LSTMc = yTest_LSTM[:-20]

def rel_rms_np(y_true,y_sim):
    return np.sqrt(np.mean(np.square(y_true-y_sim)))/np.sqrt(np.mean(np.square(y_true-np.mean(y_true)))) #goodness of fit: divided by rms(detrended targets)

rel_RMSEtr = rel_rms_np(yTrain_simc,yTrain_LSTMc)

yDev_sim = model.predict(UDev_LSTM)
yDev_simc = yDev_sim[:-20]
yDevLSTMc = yDev_LSTM[:-20]

rel_RMSEdev = rel_rms_np(yDev_simc,yDevLSTMc)

print('Tr Rel rms: ' + str(rel_rms_np(yTrain_LSTMc,yTrain_simc)))
print('Tst Rel rms: ' + str(rel_rms_np(yTest_LSTMc,yTest_simc)))

#metrics
num_trainsamples = len(yTrain_LSTMc)
num_testsamples = len(yTest_LSTMc)
num_params = model.count_params()
err = (yTest_simc-yTest_LSTMc)

abs_err = np.abs(err)           #abs = absolute waarde (alles >=0 en element |R)
avg_err = np.mean(err)
std_err = np.std(err)
loss = np.sum(err**2)/num_testsamples       #MSE
rel_err = loss/(np.sum(np.abs(yTest_LSTMc)**2)/num_testsamples)   #MSE/Mean Squared juiste waarde
print(str(rel_err))
print(str(np.sum(err**2)/np.sum(yTest_LSTMc**2)))
print(str(np.sqrt(np.mean(np.square(yTest_LSTMc-np.mean(yTest_LSTMc))))))
print(str(np.sqrt(np.mean(np.square(yTest_LSTMc)))))
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

ModelPerfo = pd.DataFrame.from_dict(LSTMModel_perfo, orient='index')       #toch nog is bezien tekst hierboven in een dataframe gieten (csv of tex uiteindelijk)
ModelPerfo.index.name = 'LSTMM_f_50tsBatch3_perfo'
ModelPerfo.to_csv('LSTMM_f_50tsBatch3_perfo.csv', header=['Value'])
ModelPerfo.to_latex('LSTMM_f_50tsBatch3_perfo.tex', header=['Value'])

#plotting
hist = pd.read_csv('LSTMf50tsBatch3hist.csv')
plt.plot(hist["root_mean_squared_error"][:], 'r-', label='rmse')
plt.plot(hist["val_root_mean_squared_error"][:], 'b-', label='val-rmse')
plt.legend()

time = np.arange(0.00, len(yTest_LSTM)*0.01, step=0.01)
timel = np.arange(0.00, len(yTest)*0.01, step=0.01)
timec = timel[time_steps-1:-20]#np.arange(0.00, len(yTest_LSTMc)*0.02, step=0.02)     #timestep is gekend als 0.01, tijdsrange namaken | hier ineens wel 0 als beginwaarde gebruiken?
timecutoff = timel[-20:]
time.shape = (len(time), 1)
print(str(time.shape) +"|"+ str(yTest_LSTM.shape))
assert(time.shape == yTest_LSTM.shape)

f'''ig, axs = plt.subplots(2,1,sharex=True)                    #nu wordt displacement geplot
axs[0].plot(time, yTest_LSTM, 'b--', label='y true values')
axs[0].plot(time, yTest_sim, 'r:', label='y predicted values')
axs[0].grid(b=1,which='both')
axs[1].plot(time, err, 'k', label='Error')
plt.setp(axs[0], ylabel='y true')
plt.setp(axs[1], ylabel='y pred')
plt.setp(axs[-1], xlabel='Time (s)', ylabel='Error')
axs[1].grid(b=1,which='both')
'''
fig, axs = plt.subplots(2,1,sharex=True)                    #nu wordt displacement geplot
axs[0].plot(timel, yTest, 'b-', label='$c_y$ true values')
axs[0].plot(timec, yTest_simc, 'r:', label='$c_y$ predicted values')
axs[0].plot(timecutoff, yTest_simcutoff,'g:', label='cut off samples')
axs[0].grid(b=1,which='both')
axs[1].plot(timec, err, 'k', label='Error')
plt.setp(axs[-1], xlabel='Time (s)', ylabel='Error')
axs[1].grid(b=1,which='both')

fig.suptitle('Predictions and error of the trained model', fontsize=14)
fig.legend(loc='upper right')

plt.rcParams.update({'font.size': 15})
fig, axs = plt.subplots(2,1,sharex=True)                  #nu wordt displacement geplot
axs[0].plot(timel[:-20], yTest[:-20], 'b-', label='Target values')
axs[0].plot(timec, yTest_simc, 'r:', label='Predicted values')
axs[1].plot(timec, err, 'k', label='Error')
plt.setp(axs[0], ylabel='$c_f$')
plt.setp(axs[1], xlabel='Time (s)', ylabel='Error')
axs[0].grid(b=1,which='major')
axs[1].grid(b=1,which='major')
axs[0].legend(loc='upper right')
axs[0].title.set_text('$c_y$')
axs[1].title.set_text('Error')
fig.suptitle('Force model test results', fontsize=20)

plt.show()