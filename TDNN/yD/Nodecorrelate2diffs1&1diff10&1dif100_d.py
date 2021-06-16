import numpy as np
import pandas as pd
from tensorflow import keras as K
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
#changed the patience to 20, had test rel rmse of 0.675 (4d), 0.86, 0.078 and 0.083

Utrain = pd.read_csv('Utrain5.csv', header=None).to_numpy()
Utest = pd.read_csv('Utest5.csv', header=None).to_numpy()
Udev = pd.read_csv('Udev5.csv', header=None).to_numpy()
dtrain = pd.read_csv('dtrain5.csv', header=None).to_numpy()
dtest = pd.read_csv('dtest5.csv', header=None).to_numpy()
ddev = pd.read_csv('ddev5.csv', header=None).to_numpy()

features = (Utrain.shape[1],)
optim = K.optimizers.Adam()
lossf = K.losses.MeanSquaredError()  #   mean_squared_error
metric = [K.metrics.RootMeanSquaredError(), K.metrics.MeanAbsoluteError()]
regul = K.regularizers.l2(0.001)     #loss = l2 * reduce_sum(square(x)) l2=0.001

'''TestModel = K.Sequential([Dense(105, activation='relu', input_shape=[len(Utrain[1]),], name='hidden1'),
                          Dense(84, activation='relu', name='hidden2'),
                          Dense(64, activation='relu', name='Hidden3'),
                          Dense(42, activation='relu', name='Hidden4'),
                          Dense(24, activation='relu', name='Hidden5'),
                          Dense(24, activation='relu', name='Hidden6'),
                          Dense(16, activation='relu', name='Hidden7'),
                          Dense(16, activation='relu', name='Hidden8'),
                          Dense(16, activation='relu', name='Hidden9'),
                          Dense(8, activation='relu', name='Hidden10'),
                          Dense(8, activation='relu', name='Hidden11'),
                          Dense(8, activation='relu', name='Hidden12'),
                          Dense(4, activation='relu', name='Hidden13'),
                          Dense(1, activation='linear', name='output')],name='DifsNodeAugmentData5d')

TestModel.compile(loss=lossf, optimizer=optim, metrics=metric)

earlystop = K.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=1)

Testrun = TestModel.fit(Utrain,dtrain,epochs=1000,batch_size=128, callbacks=earlystop,
                        validation_data=(Udev,ddev),validation_steps=5)         #validationsteps: in hoeveel batches validation data wordt gesplitst (enkel om RAM te sparen)

hist = pd.DataFrame(Testrun.history)    #progress printen
hist.to_csv('DifsNodeAugmDataHistory5d.csv')      #en opslaan           #Deze hist sebiet ook is plotten!

#K.utils.plot_model(TestModel, to_file='./Groot2.png', show_shapes=True, show_layer_names=True)   #structuur opslaan

TestModel.save('DifsNodeAugmentData5d', save_format='h5')'''

TestModel = K.models.load_model('DifsNodeAugmentData4d')
TestModel.summary()
def rel_rms_np(y_true,y_sim):
    return np.sqrt(np.mean(np.square(y_true-y_sim)))/np.sqrt(np.mean(np.square(y_true-np.mean(y_true)))) #goodness of fit: divided by rms(detrended targets)

TestModel_eval = TestModel.evaluate(Udev,ddev)
dpred = TestModel.predict(Udev)
assert (dpred.shape == ddev.shape)

#metrics
num_trainsamples = len(dtrain)
num_testsamples = len(ddev)
num_params = TestModel.count_params()
err = (dpred-ddev)

abs_err = np.abs(err)           #abs = absolute waarde (alles >=0 en element |R)
avg_err = np.mean(err)
std_err = np.std(err)
loss = np.sum(err**2)/num_testsamples       #MSE
rel_err = loss/(np.sum(np.abs(dtest)**2)/num_testsamples)   #MSE/Mean Squared juiste waarde
rel_RMSE = np.sqrt(rel_err)

TestModel_perfo = {'Model activations': 'ReLU-Linear',
                    'Number of training samples': num_trainsamples,
                    'Number of model parameters': num_params,
                    'Ratio training samples-parameters':
                    num_trainsamples/num_params,
                    'Model loss on dev data': TestModel_eval[0],
                    'Model RMSE on dev data': TestModel_eval[1],
                    'Model Absolute error on dev data': TestModel_eval[2],
                    'Number of dev samples': num_testsamples,
                    'Average Error': avg_err,
                    'Standard Deviation of Error': std_err,
                    'Average Absolute Error': np.mean(abs_err),
                    'Maximum Error': np.max(abs_err),
                    'Loss': loss,
                    'Relative Error': rel_err,
                    'Relative RMSE': rel_RMSE}

Model1Perfo = pd.DataFrame.from_dict(TestModel_perfo, orient='index')       #toch nog is bezien tekst hierboven in een dataframe gieten (csv of tex uiteindelijk)
Model1Perfo.index.name = 'TDNNyDdev'
Model1Perfo.to_csv('TDNNyDdev.csv', header=['Value'])
Model1Perfo.to_latex('TDNNyDdev.tex', header=['Value'])

'''# ===== plot of the predictions and error =====
sns.set(context='paper', style='whitegrid')'''

#plotting
"""hist = pd.read_csv('DifsNodeAugmDataHistory5d.csv')
plt.plot(hist["root_mean_squared_error"][:], 'r-', label='rmse')
plt.plot(hist["val_root_mean_squared_error"][:], 'b-', label='val-rmse')
plt.legend()

time = np.arange(0.00, len(dtest)*0.01, step=0.01)     #timestep is gekend als 0.01, tijdsrange namaken | hier ineens wel 0 als beginwaarde gebruiken?
time.shape = (len(time), 1)
print(str(time.shape) +"|"+ str(dtest.shape))
assert(time.shape == dtest.shape)

'''fig, axs = plt.subplots(2,1,sharex=True)                    #nu wordt displacement geplot
axs[0].plot(time[750:2750], dtest[750:2750], 'b--', label='d true values')
axs[0].plot(time[750:2750], dpred[750:2750], 'r:', label='d predicted values')
axs[1].plot(time[750:2750], err[750:2750], 'k', label='Error')
plt.setp(axs[0], ylabel='d true')
plt.setp(axs[1], ylabel='d pred')
plt.setp(axs[-1], xlabel='Time (s)', ylabel='Error')

fig.suptitle('Predictions and error of the trained model', fontsize=14)
fig.legend(loc='upper right')
plt.show()'''

fig, axs = plt.subplots(2,1,sharex=True)                    #nu wordt displacement geplot
axs[0].plot(time, dtest, 'b--', label='d true values')
axs[0].plot(time, dpred, 'r:', label='d predicted values')
axs[1].plot(time, err, 'k', label='Error')
plt.setp(axs[0], ylabel='d true')
plt.setp(axs[1], ylabel='d pred')
plt.setp(axs[-1], xlabel='Time (s)', ylabel='Error')

plt.figure()
plt.plot(time[750:2750], dtest[750:2750], 'b--', linewidth=0.8, label='y/D true values')
plt.plot(time[750:2750], dpred[750:2750], 'r:', linewidth=0.8, label='y/D predicted values')
plt.plot(time[750:2750], dtest[750:2750]-dpred[750:2750], 'k-', linewidth=0.2, label='error')
plt.xlabel('Time (s)')
plt.ylabel('y/D | error')
plt.suptitle('y/D - TDNN: Test predictions and error of the trained model', fontsize=14)
plt.legend(loc='upper right')
plt.grid()

plt.figure()
plt.plot(time[1500:3700], dtest[1500:3700], 'b--', linewidth=0.8, label='y/D target')
plt.plot(time[1500:3700], dpred[1500:3700], 'r:', linewidth=0.8, label='y/D predicted')
plt.plot(time[1500:3700], dpred[1500:3700]-dtest[1500:3700], 'k-', linewidth=0.2, label='error')
plt.xlabel('Time (s)')
plt.ylabel('y/D | error')
plt.suptitle('TDNN - y/D: Test predictions and error of the trained model', fontsize=14)
plt.legend(loc='upper right')
plt.grid()

plt.show()"""