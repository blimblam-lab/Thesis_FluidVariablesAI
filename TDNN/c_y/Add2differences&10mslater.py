#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#zijn niet echt rico's, eerder verschillen

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

def augmdata(u, f):         #1 en 2 stappen terug in de tijd kijken
    ut3 = u[0:-10, :]
    ut2 = u[8:-2, :]
    ut1 = u[9:-1, :]
    ut0 = u[10:, :]
    ut2 = ut0-ut2         #hier nog mee experimenteren: delen door #tijdstapen voor rico, of tijdstap verder naar achter (5 ipv 2 ofzo)
    ut1 = ut0-ut1         #hier nog mee experimenteren
    ut3 = ut0-ut3

    uout = np.concatenate((ut0, ut1, ut2, ut3), axis=1)  # telkens verschoven datasets naast elkaar zetten
    assert(uout.shape[0] == u.shape[0]-10)           # (axis=0 is naar beneden:tijdlijn)
    assert(uout.shape[1] == u.shape[1]*4)

    fout = np.delete(f, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])    # eerste 10 datapunten verwijderen
    assert(fout.shape[0] == uout.shape[0])

    return uout, fout

afkaparr = [23, 19, 16, 12, 8, 4, 0, 37, 34, 29, 25, 21, -1, -8, 23, 19, 16, 12, 8, 4, 0]
amax = max(afkaparr)
amin = min(afkaparr)

#DATA LOADING:

#path = '/Data_processing/timeshift_correlation/Data_50Hz_100Hz/'
path = 'C:/Users/joshu/Documents/Joshua School/Masterproef/Data_processing/timeshift_correlation/Data_50Hz_100Hz/'

U_amp0p05 = pd.read_csv(path+'U_SWS_A0p05_100.csv',
                        index_col=0).to_numpy()
U_amp0p10 = pd.read_csv(path+'U_SWS_A0p10_100.csv',
                        index_col=0).to_numpy()
U_amp0p15 = pd.read_csv(path+'U_SWS_A0p15_100.csv',
                        index_col=0).to_numpy()
U_amp0p20 = pd.read_csv(path+'U_SWS_A0p20_100.csv',
                        index_col=0).to_numpy()
U_amp0p25 = pd.read_csv(path+'U_SWS_A0p25_100.csv',
                        index_col=0).to_numpy()
U_amp0p30 = pd.read_csv(path+'U_SWS_A0p30_100.csv',
                        index_col=0).to_numpy()
U = [U_amp0p05, U_amp0p10, U_amp0p15, U_amp0p20, U_amp0p25, U_amp0p30]


f_amp0p05 = pd.read_csv(path+'force_SWS_A0p05_100.csv',
                        index_col=0).to_numpy()
f_amp0p10 = pd.read_csv(path+'force_SWS_A0p10_100.csv',
                        index_col=0).to_numpy()
f_amp0p15 = pd.read_csv(path+'force_SWS_A0p15_100.csv',
                        index_col=0).to_numpy()
f_amp0p20 = pd.read_csv(path+'force_SWS_A0p20_100.csv',
                        index_col=0).to_numpy()
f_amp0p25 = pd.read_csv(path+'force_SWS_A0p25_100.csv',
                        index_col=0).to_numpy()
f_amp0p30 = pd.read_csv(path+'force_SWS_A0p30_100.csv',
                        index_col=0).to_numpy()
f = [f_amp0p05, f_amp0p10, f_amp0p15, f_amp0p20, f_amp0p25, f_amp0p30]

#Data shifting and cropping (for each node i (21) and each dataset j (6)), verschuiven voor maximum correlatie
Ush = U.copy()
for j in range(len(f)):
    for i in range(len(afkaparr)):
        if afkaparr[i]>0:               #volgende geeft error als afkaparray[i]=0 bij Ush[j][:-0]
            Ush[j][:-afkaparr[i],i] = U[j][afkaparr[i]:,i]
        if afkaparr[i]<0:
            Ush[j][-afkaparr[i]:,i] = U[j][:afkaparr[i],i] # U iets verder initialiseren
    Ush[j]=Ush[j][-amin:-amax,:]        #De niet overlappende delen afsnijden
    #U[j] = U[j][-amin:-amax,:]          #onverschoven datasets op dezelfde manier inkorten
    f[j] = f[j][-amin:-amax]

#net verschoven dataset samenvoegen tot 1 trainset, 1 devset en 1 testset (shifted)
Utrain = np.concatenate((Ush[0], Ush[1], Ush[2], Ush[4], Ush[5]), axis=0).astype('float32')
Udev = Ush[3][500:3000, :].astype('float32')
Utest = Ush[3].astype('float32')

#de ingekorte f-data:
ftrain = np.concatenate((f[0], f[1], f[2], f[4], f[5]), axis=0).astype('float32')
fdev = f[3][500:3000, :].astype('float32')
ftest = f[3].astype('float32')

#adding two previous shifts and cropping
Utrain2, ftrain2 = augmdata(Utrain, ftrain)
Udev2, fdev2 = augmdata(Udev, fdev)
Utest2, ftest2 = augmdata(Utest, ftest)

np.savetxt('Utrain4.csv', Utrain2, delimiter=',')
np.savetxt('ftrain4.csv', ftrain2, delimiter=',')
np.savetxt('Udev4.csv', Udev2, delimiter=',')
np.savetxt('fdev4.csv', fdev2, delimiter=',')
np.savetxt('Utest4.csv', Utest2, delimiter=',')
np.savetxt('ftest4.csv', ftest2, delimiter=',')

