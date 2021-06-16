import numpy as np
import pandas as pd
from tensorflow import keras as K
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
#changed the patience to 20, had test rel rmse of 0.0479724 &    0.0509697

Utrain = pd.read_csv('Utrain4.csv', header=None).to_numpy()
Utest = pd.read_csv('Utest4.csv', header=None).to_numpy()
Udev = pd.read_csv('Udev4.csv', header=None).to_numpy()
ftrain = pd.read_csv('ftrain4.csv', header=None).to_numpy()
ftest = pd.read_csv('ftest4.csv', header=None).to_numpy()
fdev = pd.read_csv('fdev4.csv', header=None).to_numpy()

features = (Utrain.shape[1],)
optim = K.optimizers.Adam()
lossf = K.losses.MeanSquaredError()  #   mean_squared_error
metric = [K.metrics.RootMeanSquaredError(), K.metrics.MeanAbsoluteError()]

TestModel = K.Sequential([Dense(84, activation='relu', input_shape=[len(Utrain[1]),], name='hidden1'),
                          Dense(64, activation='relu', name='Hidden2'),
                          Dense(42, activation='relu', name='Hidden3'),
                          Dense(24, activation='relu', name='Hidden4'),
                          Dense(24, activation='relu', name='Hidden5'),
                          Dense(16, activation='relu', name='Hidden6'),
                          Dense(16, activation='relu', name='Hidden7'),
                          Dense(16, activation='relu', name='Hidden8'),
                          Dense(8, activation='relu', name='Hidden9'),
                          Dense(8, activation='relu', name='Hidden10'),
                          Dense(8, activation='relu', name='Hidden11'),
                          Dense(4, activation='relu', name='Hidden12'),
                          Dense(1, activation='linear', name='output')],name='DifsNodeAugmentData4f')

TestModel.compile(loss=lossf, optimizer=optim, metrics=metric)

earlystop = K.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=1)

'''Testrun = TestModel.fit(Utrain,ftrain,epochs=1000,batch_size=128, callbacks=earlystop,
                        validation_data=(Udev,fdev),validation_steps=5)         #validationsteps: in hoeveel batches validation data wordt gesplitst (enkel om RAM te sparen)

hist = pd.DataFrame(Testrun.history)    #progress printen
hist.to_csv('DifsNodeAugmDataHistory4f.csv')      #en opslaan           #Deze hist sebiet ook is plotten!

#K.utils.plot_model(TestModel, to_file='./Groot2.png', show_shapes=True, show_layer_names=True)   #structuur opslaan

TestModel.save('DifsNodeAugmentData4f', save_format='h5')'''

TestModel = K.models.load_model('DifsNodeAugmentData4f1')
TestModel.summary()
def rel_rms_np(y_true,y_sim):
    return np.sqrt(np.mean(np.square(y_true-y_sim)))/np.sqrt(np.mean(np.square(y_true-np.mean(y_true)))) #goodness of fit: divided by rms(detrended targets)
TestModel_eval = TestModel.evaluate(Udev,fdev)
fpred = TestModel.predict(Udev)
assert (fpred.shape == fdev.shape)

#metrics
num_trainsamples = len(ftrain)
num_testsamples = len(fdev)
num_params = TestModel.count_params()
err = (fpred-fdev)

abs_err = np.abs(err)           #abs = absolute waarde (alles >=0 en element |R)
avg_err = np.mean(err)
std_err = np.std(err)
loss = np.sum(err**2)/num_testsamples       #MSE
rel_err = loss/(np.sum(np.abs(ftest)**2)/num_testsamples)   #MSE/Mean Squared juiste waarde
rel_RMSE = np.sqrt(rel_err)

TestModel_perfo = {'Model activations': 'ReLU-Linear',
                    'Number of training samples': num_trainsamples,
                    'Number of model parameters': num_params,
                    'Ratio training samples-parameters':
                    num_trainsamples/num_params,
                    'Model loss on dev data': TestModel_eval[0],
                    'Model RMSE on dev data': TestModel_eval[1],
                    'Model Absolute error on dev data': TestModel_eval[2],
                    'Number of test samples': num_testsamples,
                    'Average Error': avg_err,
                    'Standard Deviation of Error': std_err,
                    'Average Absolute Error': np.mean(abs_err),
                    'Maximum Error': np.max(abs_err),
                    'Loss': loss,
                    'Relative Error': rel_err,
                    'Relative RMSE': rel_RMSE}

Model1Perfo = pd.DataFrame.from_dict(TestModel_perfo, orient='index')       #toch nog is bezien tekst hierboven in een dataframe gieten (csv of tex uiteindelijk)
Model1Perfo.index.name = 'TDNN Cydev'
Model1Perfo.to_latex('TDNNcydev.tex', header=['Value'])

# ===== plot of the predictions and error =====
'''sns.set(context='paper', style='whitegrid')'''
'''
#plotting
hist = pd.read_csv('DifsNodeAugmDataHistory4f1.csv')

plt.figure()
plt.plot(hist["root_mean_squared_error"][:], 'r-', label='rmse')
plt.plot(hist["val_root_mean_squared_error"][:], 'b-', label='val-rmse')
plt.title('Training progress of the TDNN cy model')
plt.xlabel('epochs')
plt.ylabel('RMSE')
plt.legend()

time = np.arange(0.00, len(ftest)*0.01, step=0.01)     #timestep is gekend als 0.01, tijdsrange namaken | hier ineens wel 0 als beginwaarde gebruiken?
time.shape = (len(time), 1)
print(str(time.shape) +"|"+ str(ftest.shape))
assert(time.shape == ftest.shape)

plt.figure()
plt.plot(time[750:2750], ftest[750:2750], 'b--', linewidth=1, label='cy true values')
plt.plot(time[750:2750], fpredtst[750:2750], 'r:', linewidth=1.2, label='cy predicted values')
plt.plot(time[750:2750], ftest[750:2750]-fpredtst[750:2750], 'k-', linewidth=0.2, label='error')
plt.xlabel('Time (s)')
plt.ylabel('cy | error')
plt.suptitle('Test predictions and error of the trained model', fontsize=14)
plt.legend(loc='upper right')

plt.figure()
plt.plot(time[1300:2750], ftest[1300:2750], 'b--', linewidth=0.8, label='$c_y$ target')
plt.plot(time[1300:2750], fpredtst[1300:2750], 'r:', linewidth=0.8, label='$c_y$ predicted')
plt.plot(time[1300:2750], ftest[1300:2750]-fpredtst[1300:2750], 'k-', linewidth=0.2, label='error')
plt.xlabel('Time (s)')
plt.ylabel('$c_y$ | error')
plt.suptitle('TDNN - $c_y$: Test predictions and error of the trained model', fontsize=14)
plt.legend(loc='upper right')
plt.grid()

plt.show()'''