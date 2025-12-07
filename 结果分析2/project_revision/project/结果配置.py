#######################################接下来是BP神经的结果##############################################
'/gpfs/home/音乐/data/data20/data20_3/'是在以下模型下取得的结果：
后来又做了一次，结果在'/gpfs/home/音乐/data/data20/data20_2/'下
建立简单的全连接BP神经网络
    def build_model(self,optimizer='rmsprop'):
        model = Sequential()
        model.add(Dropout(rate=0.2,input_shape=(self.signal_kind,)))
        model.add(Dense(units=20,activation='relu',input_shape=(self.signal_kind,)))
        model.add(Dropout(rate=0.2))
        model.add(Dense(units=20,activation='relu'))
        model.add(Dropout(rate=0.2))
        model.add(Dense(1,activation='sigmoid'))
        model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
        print(model.summary())
        return model
标签为：
        if self.a[i,2]==1:
            one_shut_train[12,:] = 1/(1+np.exp(-(one_shut_train[13,:]-(disr_time-0.3))*20))
        elif self.a[i,2]==-1:
            one_shut_train[12,:] = 0
回调函数为：
callback_list = [ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=10),\
                EarlyStopping(monitor='val_loss',patience=50),\
                ModelCheckpoint(filepath=filepath+'best.weight.h5',monitor='val_loss',verbose=1,save_best_only=True),\
                TensorBoard(log_dir=filepath+'/my_logs',histogram_freq=1)]
batch_size为:
        self.batch_size=32
        History = model.fit(x=X_train,y=y_train,epochs=self.epochs,\
                            batch_size=self.batch_size,validation_data=(X_val,y_val),verbose=2,\
                            callbacks=callback_list)
        result0 = model.predict(X_test,batch_size=32,verbose=1)
        result2 = model.predict(X,batch_size=32,verbose=1)
归一化为：
        self.min_max[:,0] = train_mat[:-2,:].min(axis=1)
        self.min_max[:,1] = train_mat[:-2,:].max(axis=1)
        #self.min_max = min_max_Ran
        train_mat[:-2,:] = minmax_scale(train_mat[:-2,:],axis=1)

        test_mat[:-2,:] = test_mat[:-2,:]-np.tile(self.min_max[:,0].reshape(-1,1),(1,test_mat.shape[1]))
        Range = (self.min_max[:,1]-self.min_max[:,0]).reshape(-1,1)
        test_mat[:-2,:] = test_mat[:-2,:]/np.tile(Range,(1,test_mat.shape[1]))

寻找最优的thresh是在测试集上找到的

在迭代53次后结束
AUC最大值为0.863139,在阈值为0.100000下取得.
在训练集上的结果:
 train_Res: [214, 13, 35, 6, 89, 34] train_Rate: [0.942736, 0.0572635, 0.213413, 0.0365854, 0.542683, 0.207317073117]

在验证集上的结果:
 val_Res: [210, 16, 25, 2, 98, 39] val_Rate: [0.9292035, 0.0707, 0.1524399, 0.0121953, 0.5975601, 0.237808]

在测试集上的结果:
 test_Res: [215, 12, 25, 4, 98, 36] test_Rate: [0.947, 0.052, 0.1533, 0.0245, 0.601226, 0.2208588]

在所有数据上的结果:
 all_Res: [639, 41, 85, 12, 285, 109] all_Rate: [0.9397, 0.0602941, 0.17311, 0.0244399, 0.58044, 0.22199]


#*************************************************************************************************************************************************
'/gpfs/home/音乐/data/data20/data20_4/'是在以下模型下取得的结果：
    #建立简单的全连接BP神经网络
    def build_model(self,optimizer='rmsprop'):
        model = Sequential()
        model.add(Dropout(rate=0.2,input_shape=(self.signal_kind,)))
        model.add(Dense(units=32,activation='relu',input_shape=(self.signal_kind,)))
        model.add(Dropout(rate=0.2))
        model.add(Dense(units=32,activation='relu'))
        model.add(Dropout(rate=0.2))
        model.add(Dense(1,activation='sigmoid'))
        model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
        print(model.summary())
        return model
标签为:
        if self.a[i,2]==1:
            one_shut_train[12,:] = 1/(1+np.exp(-(one_shut_train[13,:]-(disr_time-0.3))*20))
        elif self.a[i,2]==-1:
            one_shut_train[12,:] = 0

callback_list = [ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=10),\
                EarlyStopping(monitor='val_loss',patience=50),\
                ModelCheckpoint(filepath=filepath+'best.weight.h5',monitor='val_loss',verbose=1,save_best_only=True),\
                TensorBoard(log_dir=filepath+'/my_logs',histogram_freq=1)]
batch_size为:
        self.batch_size=32
        History = model.fit(x=X_train,y=y_train,epochs=self.epochs,\
                            batch_size=self.batch_size,validation_data=(X_val,y_val),verbose=2,\
                            callbacks=callback_list)
        result0 = model.predict(X_test,batch_size=32,verbose=1)
        result2 = model.predict(X,batch_size=32,verbose=1)
归一化为：
        self.min_max[:,0] = train_mat[:-2,:].min(axis=1)
        self.min_max[:,1] = train_mat[:-2,:].max(axis=1)
        #self.min_max = min_max_Ran
        train_mat[:-2,:] = minmax_scale(train_mat[:-2,:],axis=1)

        test_mat[:-2,:] = test_mat[:-2,:]-np.tile(self.min_max[:,0].reshape(-1,1),(1,test_mat.shape[1]))
        Range = (self.min_max[:,1]-self.min_max[:,0]).reshape(-1,1)
        test_mat[:-2,:] = test_mat[:-2,:]/np.tile(Range,(1,test_mat.shape[1]))

寻找最优的thresh是在测试集上找到的

'''与data20_3对比变的是ANN节点数，由20变为32。。。。'''

在迭代51次后结束
AUC最大值为0.815005,在阈值为0.110000下取得.
在训练集上的结果:
 train_Res: [215, 12, 29, 4, 60, 71] train_Rate: [0.9471365, 0.0528634, 0.17682, 0.024390, 0.36585, 0.432926]

在验证集上的结果:
 val_Res: [213, 13, 22, 2, 80, 60] val_Rate: [0.9424, 0.057522, 0.134146, 0.0121, 0.487, 0.36585]

在测试集上的结果:
 test_Res: [221, 6, 23, 3, 81, 56] test_Rate: [0.97356, 0.02643, 0.141, 0.0184, 0.4969, 0.34355]

在所有数据上的结果:
 all_Res: [649, 31, 74, 9, 221, 187] all_Rate: [0.954, 0.0455, 0.15071, 0.018, 0.45010, 0.380]

#*************************************************************************************************************************************************
'/gpfs/home/音乐/data/data20/data20_5/'是在以下模型下取得的结果：
因为结果很好，96%,82%,所以后来又算了一次，保存在'/gpfs/home/音乐/data/data20/data20_1/'下
#建立简单的全连接BP神经网络
    def build_model(self,optimizer='rmsprop'):
        model = Sequential()
        model.add(Dropout(rate=0.2,input_shape=(self.signal_kind,)))
        model.add(Dense(units=20,activation='relu',input_shape=(self.signal_kind,)))
        model.add(Dropout(rate=0.2))
        model.add(Dense(units=20,activation='relu'))
        model.add(Dropout(rate=0.2))
        model.add(Dense(1,activation='sigmoid'))
        model.compile(optimizer='adam',loss='mse',metrics=['mae'])
        print(model.summary())
        return model
标签为:
        if self.a[i,2]==1:
            one_shut_train[12,:] = 1/(1+np.exp(-(one_shut_train[13,:]-(disr_time-0.3))*20))
        elif self.a[i,2]==-1:
            one_shut_train[12,:] = 0

callback_list = [ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=10),\
                EarlyStopping(monitor='val_loss',patience=50),\
                ModelCheckpoint(filepath=filepath+'best.weight.h5',monitor='val_loss',verbose=1,save_best_only=True),\
                TensorBoard(log_dir=filepath+'/my_logs',histogram_freq=1)]
batch_size为:
        self.batch_size=32
        History = model.fit(x=X_train,y=y_train,epochs=self.epochs,\
                            batch_size=self.batch_size,validation_data=(X_val,y_val),verbose=2,\
                            callbacks=callback_list)
        result0 = model.predict(X_test,batch_size=32,verbose=1)
        result2 = model.predict(X,batch_size=32,verbose=1)
归一化为：
        self.min_max[:,0] = train_mat[:-2,:].min(axis=1)
        self.min_max[:,1] = train_mat[:-2,:].max(axis=1)
        #self.min_max = min_max_Ran
        train_mat[:-2,:] = minmax_scale(train_mat[:-2,:],axis=1)

        test_mat[:-2,:] = test_mat[:-2,:]-np.tile(self.min_max[:,0].reshape(-1,1),(1,test_mat.shape[1]))
        Range = (self.min_max[:,1]-self.min_max[:,0]).reshape(-1,1)
        test_mat[:-2,:] = test_mat[:-2,:]/np.tile(Range,(1,test_mat.shape[1]))

寻找最优的thresh是在测试集上找到的

在迭代51次后结束
'''与data20_3相比也就是优化函数由rmsprop变为了adam。。。。'''
AUC最大值为0.884611,在阈值为0.120000下取得.
在训练集上的结果:
 train_Res: [219, 8, 39, 7, 84, 34] train_Rate: [0.9647577, 0.0352422, 0.23780, 0.0426829, 0.5121, 0.2073]

在验证集上的结果:
 val_Res: [214, 12, 23, 5, 103, 33] val_Rate: [0.94690, 0.05309, 0.14024, 0.030488, 0.62804, 0.20121]

在测试集上的结果:
 test_Res: [215, 12, 26, 5, 103, 29] test_Rate: [0.94713, 0.052, 0.1595, 0.03067, 0.63190, 0.17791]

在所有数据上的结果:
 all_Res: [648, 32, 88, 17, 290, 96] all_Rate: [0.95294, 0.04705, 0.17923, 0.0346, 0.59063, 0.1958]



#*************************************************************************************************************************************************
'/gpfs/home/音乐/data/data20/data20_6/'是在以下模型下取得的结果：
建立简单的全连接BP神经网络
    def build_model(self,optimizer='rmsprop'):
        model = Sequential()
        model.add(Dropout(rate=0.2,input_shape=(self.signal_kind,)))
        model.add(Dense(units=20,activation='relu',input_shape=(self.signal_kind,)))
        model.add(Dropout(rate=0.2))
        model.add(Dense(units=20,activation='relu'))
        model.add(Dropout(rate=0.2))
        model.add(Dense(1,activation='sigmoid'))
        model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
        print(model.summary())
        return model
标签为：
        if self.a[i,2]==1:
            one_shut_train[12,:] = minmax_scale(dfsdev[1,:])
        elif self.a[i,2]==-1:
            one_shut_train[12,:] = 0
回调函数为：
callback_list = [ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=10),\
                EarlyStopping(monitor='val_loss',patience=50),\
                ModelCheckpoint(filepath=filepath+'best.weight.h5',monitor='val_loss',verbose=1,save_best_only=True),\
                TensorBoard(log_dir=filepath+'/my_logs',histogram_freq=1)]
batch_size为:
        self.batch_size=32
        History = model.fit(x=X_train,y=y_train,epochs=self.epochs,\
                            batch_size=self.batch_size,validation_data=(X_val,y_val),verbose=2,\
                            callbacks=callback_list)
        result0 = model.predict(X_test,batch_size=32,verbose=1)
        result2 = model.predict(X,batch_size=32,verbose=1)
归一化为：
        self.min_max[:,0] = train_mat[:-2,:].min(axis=1)
        self.min_max[:,1] = train_mat[:-2,:].max(axis=1)
        #self.min_max = min_max_Ran
        train_mat[:-2,:] = minmax_scale(train_mat[:-2,:],axis=1)

        test_mat[:-2,:] = test_mat[:-2,:]-np.tile(self.min_max[:,0].reshape(-1,1),(1,test_mat.shape[1]))
        Range = (self.min_max[:,1]-self.min_max[:,0]).reshape(-1,1)
        test_mat[:-2,:] = test_mat[:-2,:]/np.tile(Range,(1,test_mat.shape[1]))

寻找最优的thresh是在测试集上找到的

'''与data20_3相比标签由sigmoid函数变为了对每炮的密度分别进行最大最小归一化得到的标签。。。。'''

在迭代53次后结束
AUC最大值为0.761912,在阈值为0.220000下取得.
在训练集上的结果:
 train_Res: [218, 9, 26, 5, 67, 66] train_Rate: [0.9603, 0.0396, 0.1585, 0.03048, 0.40853, 0.40243]

在验证集上的结果:
 val_Res: [217, 9, 18, 6, 76, 64] val_Rate: [0.960, 0.03982, 0.109756, 0.03658, 0.46341, 0.39024]

在测试集上的结果:
 test_Res: [215, 12, 17, 5, 72, 69] test_Rate: [0.947136, 0.05286, 0.1042944, 0.03067, 0.4417, 0.42331]

在所有数据上的结果:
 all_Res: [650, 30, 61, 16, 215, 199] all_Rate: [0.95588, 0.04411, 0.12423, 0.0325, 0.43788, 0.40529]

#*************************************************************************************************************************************************
'/gpfs/home/音乐/data/data20/data20_7/'是在以下模型下取得的结果：
    #建立简单的全连接BP神经网络
    def build_model(self,optimizer='rmsprop'):
        model = Sequential()
        model.add(Dropout(rate=0.2,input_shape=(self.signal_kind,)))
        model.add(Dense(units=20,activation='relu',input_shape=(self.signal_kind,)))
        model.add(Dropout(rate=0.2))
        model.add(Dense(units=20,activation='relu'))
        model.add(Dropout(rate=0.2))
        model.add(Dense(1,activation='sigmoid'))
        model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
        print(model.summary())
        return model
标签为:
        if self.a[i,2]==1:
            one_shut_train[12,:] = 1/(1+np.exp(-(one_shut_train[13,:]-(t1+t2)/2)*(10/(t2-t1))))
        elif self.a[i,2]==-1:
            one_shut_train[12,:] = 0
回调函数为：
callback_list = [ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=10),\
                EarlyStopping(monitor='val_loss',patience=50),\
                ModelCheckpoint(filepath=filepath+'best.weight.h5',monitor='val_loss',verbose=1,save_best_only=True),\
                TensorBoard(log_dir=filepath+'/my_logs',histogram_freq=1)]
batch_size为:
        self.batch_size=32
        History = model.fit(x=X_train,y=y_train,epochs=self.epochs,\
                            batch_size=self.batch_size,validation_data=(X_val,y_val),verbose=2,\
                            callbacks=callback_list)
        result0 = model.predict(X_test,batch_size=32,verbose=1)
        result2 = model.predict(X,batch_size=32,verbose=1)
归一化为：
        self.min_max[:,0] = train_mat[:-2,:].min(axis=1)
        self.min_max[:,1] = train_mat[:-2,:].max(axis=1)
        #self.min_max = min_max_Ran
        train_mat[:-2,:] = minmax_scale(train_mat[:-2,:],axis=1)

        test_mat[:-2,:] = test_mat[:-2,:]-np.tile(self.min_max[:,0].reshape(-1,1),(1,test_mat.shape[1]))
        Range = (self.min_max[:,1]-self.min_max[:,0]).reshape(-1,1)
        test_mat[:-2,:] = test_mat[:-2,:]/np.tile(Range,(1,test_mat.shape[1]))

寻找最优的thresh是在测试集上找到的

'''
与data20_3相比标签由1/(1+np.exp(-(one_shut_train[13,:]-(disr_time-0.3))*20))变为了
1/(1+np.exp(-(one_shut_train[13,:]-(t1+t2)/2)*(10/(t2-t1))))
。。。。'''

在迭代52次后结束
AUC最大值为0.895624,在阈值为0.210000下取得.
在训练集上的结果:
 train_Res: [216, 11, 31, 9, 92, 32] train_Rate: [0.95154185, 0.0484581, 0.18902, 0.0548780, 0.56097, 0.195121]

在验证集上的结果:
 val_Res: [218, 8, 22, 7, 101, 34] val_Rate: [0.96460176, 0.03539, 0.13414, 0.0426829, 0.6158, 0.2073170]
在测试集上的结果
 test_Res: [220, 7, 24, 3, 107, 29] test_Rate: [0.969162, 0.03083700, 0.1472392, 0.0184040, 0.6564, 0.1779]

在所有数据上的结果:
 all_Res: [654, 26, 77, 19, 300, 95] all_Rate: [0.9617, 0.0382, 0.1568, 0.038696, 0.61099, 0.193482]
#*************************************************************************************************************************************************

#*************************************************************************************************************************************************
'/gpfs/home/音乐/data/data20/data20_10/'是在以下模型下取得的结果：
    #建立简单的全连接BP神经网络
    def build_model(self,optimizer='rmsprop'):
        model = Sequential()
        model.add(Dropout(rate=0.2,input_shape=(self.signal_kind,)))
        model.add(Dense(units=20,activation='relu',input_shape=(self.signal_kind,)))
        model.add(Dropout(rate=0.2))
        model.add(Dense(units=20,activation='relu'))
        model.add(Dropout(rate=0.2))
        model.add(Dense(1,activation='sigmoid'))
        model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
        print(model.summary())
        return model
标签为:
        if self.a[i,2]==1:
            one_shut_train[12,:] = 1/(1+np.exp(-(one_shut_train[13,:]-(t1+t2)/2)*(10/(t2-t1))))
        elif self.a[i,2]==-1:
            one_shut_train[12,:] = 0
回调函数为：
callback_list = [ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=10),\
                EarlyStopping(monitor='val_loss',patience=50),\
                ModelCheckpoint(filepath=filepath+'best.weight.h5',monitor='val_loss',verbose=1,save_best_only=True),\
                TensorBoard(log_dir=filepath+'/my_logs',histogram_freq=1)]
batch_size为:
        self.batch_size=64
        History = model.fit(x=X_train,y=y_train,epochs=self.epochs,\
                            batch_size=self.batch_size,validation_data=(X_val,y_val),verbose=2,\
                            callbacks=callback_list)
        result0 = model.predict(X_test,batch_size=32,verbose=1)
        result2 = model.predict(X,batch_size=32,verbose=1)
归一化为：
        self.min_max[:,0] = train_mat[:-2,:].min(axis=1)
        self.min_max[:,1] = train_mat[:-2,:].max(axis=1)
        #self.min_max = min_max_Ran
        train_mat[:-2,:] = minmax_scale(train_mat[:-2,:],axis=1)

        test_mat[:-2,:] = test_mat[:-2,:]-np.tile(self.min_max[:,0].reshape(-1,1),(1,test_mat.shape[1]))
        Range = (self.min_max[:,1]-self.min_max[:,0]).reshape(-1,1)
        test_mat[:-2,:] = test_mat[:-2,:]/np.tile(Range,(1,test_mat.shape[1]))
寻找最优的thresh是在测试集上找到的

'''与data20_7不同的是batch_size由32变为了64'''

epoch=51结束
结果为：
AUC最大值为0.882882,在阈值为0.190000下取得.
在训练集上的结果:
 train_Res: [218, 9, 30, 6, 95, 33] train_Rate: [0.960352422, 0.0396475770, 0.18292682, 0.03658536, 0.579268, 0.2012195]

在验证集上的结果:
 val_Res: [214, 12, 26, 7, 94, 37] val_Rate: [0.946902654, 0.053097345, 0.1585365, 0.042682926, 0.5731707, 0.2256097560]

在测试集上的结果:
 test_Res: [217, 10, 22, 5, 105, 31] test_Rate: [0.955947136, 0.044052863, 0.1349693, 0.03067, 0.6441717, 0.190184049079]

在所有数据上的结果:
 all_Res: [649, 31, 78, 18, 294, 101] all_Rate: [0.95441176, 0.0455882352, 0.1588594, 0.036659, 0.5987780, 0.2057026476]


#**********************************************************************************************************************
'/gpfs/home/音乐/data/data20/data20_11/'是在以下模型下取得的结果：
    #建立简单的全连接BP神经网络
    def build_model(self,optimizer='rmsprop'):
        model = Sequential()
        model.add(Dropout(rate=0.2,input_shape=(self.signal_kind,)))
        model.add(Dense(units=20,activation='relu',input_shape=(self.signal_kind,)))
        model.add(Dropout(rate=0.2))
        model.add(Dense(units=20,activation='relu'))
        model.add(Dropout(rate=0.2))
        model.add(Dense(1,activation='sigmoid'))
        model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
        print(model.summary())
        return model
标签为:
        if self.a[i,2]==1:
            one_shut_train[12,:] = 1/(1+np.exp(-(one_shut_train[13,:]-(t1+t2)/2)*(20/(t2-t1))))
        elif self.a[i,2]==-1:
            one_shut_train[12,:] = 0
回调函数为：
callback_list = [ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=10),\
                EarlyStopping(monitor='val_loss',patience=50),\
                ModelCheckpoint(filepath=filepath+'best.weight.h5',monitor='val_loss',verbose=1,save_best_only=True),\
                TensorBoard(log_dir=filepath+'/my_logs',histogram_freq=1)]
batch_size为:
        self.batch_size=128
        History = model.fit(x=X_train,y=y_train,epochs=self.epochs,\
                            batch_size=self.batch_size,validation_data=(X_val,y_val),verbose=2,\
                            callbacks=callback_list)
        result0 = model.predict(X_test,batch_size=32,verbose=1)
        result2 = model.predict(X,batch_size=32,verbose=1)
归一化为：
        self.min_max[:,0] = train_mat[:-2,:].min(axis=1)
        self.min_max[:,1] = train_mat[:-2,:].max(axis=1)
        #self.min_max = min_max_Ran
        train_mat[:-2,:] = minmax_scale(train_mat[:-2,:],axis=1)

        test_mat[:-2,:] = test_mat[:-2,:]-np.tile(self.min_max[:,0].reshape(-1,1),(1,test_mat.shape[1]))
        Range = (self.min_max[:,1]-self.min_max[:,0]).reshape(-1,1)
        test_mat[:-2,:] = test_mat[:-2,:]/np.tile(Range,(1,test_mat.shape[1]))

寻找最优的thresh是在测试集上找到的

'''
与data20_7不同的是batch_size由32变为了128，且(1+np.exp(-(one_shut_train[13,:]-(t1+t2)/2)*(10/(t2-t1))))变为了20/(t2-t1)
'''

在迭代51次后结束
AUC最大值为0.887760,在阈值为0.330000下取得.
在训练集上的结果:
 train_Res: [221, 6, 31, 14, 76, 43] train_Rate: [0.9735682819, 0.02643171, 0.18902439, 0.085365853, 0.463414, 0.2621951219]

在验证集上的结果:
 val_Res: [219, 7, 27, 6, 91, 40] val_Rate: [0.969026548, 0.0309734513, 0.164634146, 0.03658536, 0.55487804, 0.24390243902]

在测试集上的结果:
 test_Res: [222, 5, 38, 6, 86, 33] test_Rate: [0.97797356, 0.02202643, 0.233128, 0.036809815, 0.5276073, 0.2024539877]

在所有数据上的结果:
 all_Res: [662, 18, 96, 26, 253, 116] all_Rate: [0.973529411, 0.026470588, 0.19551934, 0.05295315, 0.51527494, 0.23625254]


#**********************************************************************************************************************
'/gpfs/home/音乐/data/data20/data20_9/'是在以下模型下取得的结果：
    #建立简单的全连接BP神经网络
    def build_model(self,optimizer='rmsprop'):
        model = Sequential()
        model.add(Dropout(rate=0.2,input_shape=(self.signal_kind,)))
        model.add(Dense(units=20,activation='relu',input_shape=(self.signal_kind,)))
        model.add(Dropout(rate=0.2))
        model.add(Dense(units=20,activation='relu'))
        model.add(Dropout(rate=0.2))
        model.add(Dense(1,activation='sigmoid'))
        model.compile(optimizer='adam',loss='mse',metrics=['mae'])
        print(model.summary())
        return model
标签为:
        if self.a[i,2]==1:
            one_shut_train[12,:] = 1/(1+np.exp(-(one_shut_train[13,:]-(t1+t2)/2)*(10/(t2-t1))))
        elif self.a[i,2]==-1:
            one_shut_train[12,:] = 0
回调函数为：
callback_list = [ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=10),\
                EarlyStopping(monitor='val_loss',patience=50),\
                ModelCheckpoint(filepath=filepath+'best.weight.h5',monitor='val_loss',verbose=1,save_best_only=True),\
                TensorBoard(log_dir=filepath+'/my_logs',histogram_freq=1)]
batch_size为:
        self.batch_size=32
        History = model.fit(x=X_train,y=y_train,epochs=self.epochs,\
                            batch_size=self.batch_size,validation_data=(X_val,y_val),verbose=2,\
                            callbacks=callback_list)
        result0 = model.predict(X_test,batch_size=32,verbose=1)
        result2 = model.predict(X,batch_size=32,verbose=1)
归一化为：
        self.min_max[:,0] = train_mat[:-2,:].min(axis=1)
        self.min_max[:,1] = train_mat[:-2,:].max(axis=1)
        #self.min_max = min_max_Ran
        train_mat[:-2,:] = minmax_scale(train_mat[:-2,:],axis=1)

        test_mat[:-2,:] = test_mat[:-2,:]-np.tile(self.min_max[:,0].reshape(-1,1),(1,test_mat.shape[1]))
        Range = (self.min_max[:,1]-self.min_max[:,0]).reshape(-1,1)
        test_mat[:-2,:] = test_mat[:-2,:]/np.tile(Range,(1,test_mat.shape[1]))

寻找最优的thresh是在测试集上找到的
'''与data20_7不同的是优化器由rmsprop变为了adam.....'''

在迭代51次后结束
AUC最大值为0.800451,在阈值为0.240000下取得.
在训练集上的结果:
 train_Res: [216, 11, 26, 6, 69, 63] train_Rate: [0.951, 0.0484581, 0.15853, 0.03658, 0.4207, 0.3841]

在验证集上的结果:
 val_Res: [213, 13, 21, 5, 76, 62] val_Rate: [0.9424, 0.0575, 0.1280, 0.0304, 0.4634, 0.378]

在测试集上的结果:
 test_Res: [213, 14, 25, 5, 78, 55] test_Rate: [0.938, 0.0616, 0.1533, 0.0306, 0.4785, 0.337]

在所有数据上的结果:
 all_Res: [642, 38, 72, 16, 223, 180] all_Rate: [0.9441, 0.0558, 0.1466, 0.0325, 0.4541, 0.3665]

######################################################################################################
'/gpfs/home/音乐/data/data20/data20_12/'是在以下模型下取得的结果：
建立简单的全连接BP神经网络
    def build_model(self,optimizer='rmsprop'):
        model = Sequential()
        model.add(Dropout(rate=0.2,input_shape=(self.signal_kind,)))
        model.add(Dense(units=10,activation='relu',input_shape=(self.signal_kind,)))
        model.add(Dropout(rate=0.2))
        model.add(Dense(units=10,activation='relu'))
        model.add(Dropout(rate=0.2))
        model.add(Dense(1,activation='sigmoid'))
        model.compile(optimizer='adam',loss='mse',metrics=['mae'])
        print(model.summary())
        return model
标签为：
        if self.a[i,2]==1:
            one_shut_train[12,:] = 1/(1+np.exp(-(one_shut_train[13,:]-(disr_time-0.3))*20))
        elif self.a[i,2]==-1:
            one_shut_train[12,:] = 0
回调函数为：
callback_list = [ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=10),\
                EarlyStopping(monitor='val_loss',patience=50),\
                ModelCheckpoint(filepath=filepath+'best.weight.h5',monitor='val_loss',verbose=1,save_best_only=True),\
                TensorBoard(log_dir=filepath+'/my_logs',histogram_freq=1)]
batch_size为:
        self.batch_size=32
        History = model.fit(x=X_train,y=y_train,epochs=self.epochs,\
                            batch_size=self.batch_size,validation_data=(X_val,y_val),verbose=2,\
                            callbacks=callback_list)
        result0 = model.predict(X_test,batch_size=32,verbose=1)
        result2 = model.predict(X,batch_size=32,verbose=1)
归一化为：
        self.min_max[:,0] = train_mat[:-2,:].min(axis=1)
        self.min_max[:,1] = train_mat[:-2,:].max(axis=1)
        #self.min_max = min_max_Ran
        train_mat[:-2,:] = minmax_scale(train_mat[:-2,:],axis=1)

        test_mat[:-2,:] = test_mat[:-2,:]-np.tile(self.min_max[:,0].reshape(-1,1),(1,test_mat.shape[1]))
        Range = (self.min_max[:,1]-self.min_max[:,0]).reshape(-1,1)
        test_mat[:-2,:] = test_mat[:-2,:]/np.tile(Range,(1,test_mat.shape[1]))

寻找最优的thresh是在测试集上找到的

'''与data20-5不同的是模型的每层节点数为10,'''

在迭代53次后结束
AUC最大值为0.791330,在阈值为0.070000下取得.
在训练集上的结果:
 train_Res: [217, 10, 28, 4, 61, 71] train_Rate: [0.95594713, 0.0440528634, 0.17073170, 0.024390243, 0.37195121, 0.4329268292]

在验证集上的结果:
 val_Res: [214, 12, 20, 4, 73, 67] val_Rate: [0.9469026, 0.0530973, 0.1219512, 0.0243902, 0.445121951, 0.408536585]

在测试集上的结果:
 test_Res: [220, 7, 17, 4, 79, 63] test_Rate: [0.9691629, 0.0308370, 0.104294, 0.024539, 0.484662, 0.38650306]

在所有数据上的结果:
 all_Res: [651, 29, 65, 12, 213, 201] all_Rate: [0.9573529, 0.042647, 0.132382, 0.024439918, 0.4338085, 0.40936863]

###########################################################################################################
'/gpfs/home/音乐/data/data20/data20_13/'是在以下模型下取得的结果：
建立简单的全连接BP神经网络
    def build_model(self,optimizer='rmsprop'):
        model = Sequential()
        model.add(Dropout(rate=0.2,input_shape=(self.signal_kind,)))
        model.add(Dense(units=16,activation='relu',input_shape=(self.signal_kind,)))
        model.add(Dropout(rate=0.2))
        model.add(Dense(units=16,activation='relu'))
        model.add(Dropout(rate=0.2))
        model.add(Dense(1,activation='sigmoid'))
        model.compile(optimizer='adam',loss='mse',metrics=['mae'])
        print(model.summary())
        return model
标签为：
        if self.a[i,2]==1:
            one_shut_train[12,:] = 1/(1+np.exp(-(one_shut_train[13,:]-(disr_time-0.3))*20))
        elif self.a[i,2]==-1:
            one_shut_train[12,:] = 0
回调函数为：
callback_list = [ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=10),\
                EarlyStopping(monitor='val_loss',patience=50),\
                ModelCheckpoint(filepath=filepath+'best.weight.h5',monitor='val_loss',verbose=1,save_best_only=True),\
                TensorBoard(log_dir=filepath+'/my_logs',histogram_freq=1)]
batch_size为:
        self.batch_size=32
        History = model.fit(x=X_train,y=y_train,epochs=self.epochs,\
                            batch_size=self.batch_size,validation_data=(X_val,y_val),verbose=2,\
                            callbacks=callback_list)
        result0 = model.predict(X_test,batch_size=32,verbose=1)
        result2 = model.predict(X,batch_size=32,verbose=1)
归一化为：
        self.min_max[:,0] = train_mat[:-2,:].min(axis=1)
        self.min_max[:,1] = train_mat[:-2,:].max(axis=1)
        #self.min_max = min_max_Ran
        train_mat[:-2,:] = minmax_scale(train_mat[:-2,:],axis=1)

        test_mat[:-2,:] = test_mat[:-2,:]-np.tile(self.min_max[:,0].reshape(-1,1),(1,test_mat.shape[1]))
        Range = (self.min_max[:,1]-self.min_max[:,0]).reshape(-1,1)
        test_mat[:-2,:] = test_mat[:-2,:]/np.tile(Range,(1,test_mat.shape[1]))

寻找最优的thresh是在测试集上找到的

'''与data20-5不同的是模型的每层节点数为16,'''
在迭代52次后结束
AUC最大值为0.904435,在阈值为0.170000下取得.
在训练集上的结果:
 train_Res: [220, 7, 32, 11, 79, 42] train_Rate: [0.969162995, 0.0308370044, 0.195121, 0.06707317, 0.4817073170, 0.2560975609756]

在验证集上的结果
 val_Res: [220, 6, 26, 7, 93, 38] val_Rate: [0.973451327, 0.0265486, 0.158536, 0.04268292, 0.567073, 0.231707317073]

在测试集上的结果:
 test_Res: [224, 3, 31, 6, 97, 29] test_Rate: [0.98678414096, 0.0132158, 0.190184, 0.03680981595092, 0.5950920245398, 0.1779141104]

在所有数据上的结果:
 all_Res: [664, 16, 89, 24, 269, 109] all_Rate: [0.976470588, 0.023529411, 0.181262729, 0.04887983, 0.547861507, 0.2219959266]


###########################################################################################################
'/gpfs/home/音乐/data/data20/data20_14/'是在以下模型下取得的结果：
建立简单的全连接BP神经网络
    def build_model(self,optimizer='rmsprop'):
        model = Sequential()
        model.add(Dropout(rate=0.2,input_shape=(self.signal_kind,)))
        model.add(Dense(units=14,activation='relu',input_shape=(self.signal_kind,)))
        model.add(Dropout(rate=0.2))
        model.add(Dense(units=14,activation='relu'))
        model.add(Dropout(rate=0.2))
        model.add(Dense(1,activation='sigmoid'))
        model.compile(optimizer='adam',loss='mse',metrics=['mae'])
        print(model.summary())
        return model
标签为：
        if self.a[i,2]==1:
            one_shut_train[12,:] = 1/(1+np.exp(-(one_shut_train[13,:]-(disr_time-0.3))*20))
        elif self.a[i,2]==-1:
            one_shut_train[12,:] = 0
回调函数为：
callback_list = [ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=10),\
                EarlyStopping(monitor='val_loss',patience=50),\
                ModelCheckpoint(filepath=filepath+'best.weight.h5',monitor='val_loss',verbose=1,save_best_only=True),\
                TensorBoard(log_dir=filepath+'/my_logs',histogram_freq=1)]
batch_size为:
        self.batch_size=32
        History = model.fit(x=X_train,y=y_train,epochs=self.epochs,\
                            batch_size=self.batch_size,validation_data=(X_val,y_val),verbose=2,\
                            callbacks=callback_list)
        result0 = model.predict(X_test,batch_size=32,verbose=1)
        result2 = model.predict(X,batch_size=32,verbose=1)
归一化为：
        self.min_max[:,0] = train_mat[:-2,:].min(axis=1)
        self.min_max[:,1] = train_mat[:-2,:].max(axis=1)
        #self.min_max = min_max_Ran
        train_mat[:-2,:] = minmax_scale(train_mat[:-2,:],axis=1)

        test_mat[:-2,:] = test_mat[:-2,:]-np.tile(self.min_max[:,0].reshape(-1,1),(1,test_mat.shape[1]))
        Range = (self.min_max[:,1]-self.min_max[:,0]).reshape(-1,1)
        test_mat[:-2,:] = test_mat[:-2,:]/np.tile(Range,(1,test_mat.shape[1]))

寻找最优的thresh是在测试集上找到的

'''与data20-5不同的是模型的每层节点数为14,'''

在迭代52次后结束
AUC最大值为0.885476,在阈值为0.110000下取得.
在训练集上的结果:
 train_Res: [216, 11, 38, 7, 92, 27] train_Rate: [0.9515418502, 0.0484581497, 0.2317073170, 0.042682926829, 0.56097560975, 0.16463414634146]

在验证集上的结果:
 val_Res: [215, 11, 26, 5, 106, 27] val_Rate: [0.951327433, 0.04867256637, 0.158536585, 0.0304878048780, 0.64634146341, 0.1646341463414]

在测试集上的结果:
 test_Res: [214, 13, 23, 6, 106, 28] test_Rate: [0.9427312775, 0.057268722466, 0.1411042944, 0.03680981595, 0.6503067484, 0.17177914110]

在所有数据上的结果:
 all_Res: [645, 35, 87, 18, 304, 82] all_Rate: [0.948529411, 0.05147058823, 0.1771894093, 0.03665987, 0.61914460, 0.1670061099]

###########################################################################################################
'/gpfs/home/音乐/data/data20/data20_15/'是在以下模型下取得的结果：
建立简单的全连接BP神经网络
    def build_model(self,optimizer='rmsprop'):
        model = Sequential()
        model.add(Dropout(rate=0.2,input_shape=(self.signal_kind,)))
        model.add(Dense(units=18,activation='relu',input_shape=(self.signal_kind,)))
        model.add(Dropout(rate=0.2))
        model.add(Dense(units=18,activation='relu'))
        model.add(Dropout(rate=0.2))
        model.add(Dense(1,activation='sigmoid'))
        model.compile(optimizer='adam',loss='mse',metrics=['mae'])
        print(model.summary())
        return model
标签为：
        if self.a[i,2]==1:
            one_shut_train[12,:] = 1/(1+np.exp(-(one_shut_train[13,:]-(disr_time-0.3))*20))
        elif self.a[i,2]==-1:
            one_shut_train[12,:] = 0
回调函数为：
callback_list = [ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=10),\
                EarlyStopping(monitor='val_loss',patience=50),\
                ModelCheckpoint(filepath=filepath+'best.weight.h5',monitor='val_loss',verbose=1,save_best_only=True),\
                TensorBoard(log_dir=filepath+'/my_logs',histogram_freq=1)]
batch_size为:
        self.batch_size=32
        History = model.fit(x=X_train,y=y_train,epochs=self.epochs,\
                            batch_size=self.batch_size,validation_data=(X_val,y_val),verbose=2,\
                            callbacks=callback_list)
        result0 = model.predict(X_test,batch_size=32,verbose=1)
        result2 = model.predict(X,batch_size=32,verbose=1)
归一化为：
        self.min_max[:,0] = train_mat[:-2,:].min(axis=1)
        self.min_max[:,1] = train_mat[:-2,:].max(axis=1)
        #self.min_max = min_max_Ran
        train_mat[:-2,:] = minmax_scale(train_mat[:-2,:],axis=1)

        test_mat[:-2,:] = test_mat[:-2,:]-np.tile(self.min_max[:,0].reshape(-1,1),(1,test_mat.shape[1]))
        Range = (self.min_max[:,1]-self.min_max[:,0]).reshape(-1,1)
        test_mat[:-2,:] = test_mat[:-2,:]/np.tile(Range,(1,test_mat.shape[1]))

寻找最优的thresh是在测试集上找到的

'''与data20-5不同的是模型的每层节点数为18,'''
在迭代52次后结束

AUC最大值为0.886895,在阈值为0.130000下取得.
在训练集上的结果:
 train_Res: [221, 6, 33, 13, 79, 39] train_Rate: [0.97356828193, 0.02643171806, 0.201219512, 0.0792682926, 0.4817073170, 0.2378048780487]

在验证集上的结果:
 val_Res: [220, 6, 29, 4, 89, 42] val_Rate: [0.97345132743, 0.0265486725, 0.17682926829, 0.024390243902, 0.54268292682, 0.256097560975]

在测试集上的结果:
 test_Res: [223, 4, 32, 10, 87, 34] test_Rate: [0.982378854, 0.0176211453, 0.19631901, 0.061349693, 0.5337423, 0.2085889570552147]

在所有数据上的结果:
 all_Res: [664, 16, 94, 27, 255, 115] all_Rate: [0.97647058, 0.02352941, 0.1914460, 0.0549898, 0.5193482, 0.23421588594]


###########################################################################################################
'/gpfs/home/音乐/data/data20/data20_16/'是在以下模型下取得的结果：
建立简单的全连接BP神经网络
    def build_model(self,optimizer='rmsprop'):
        model = Sequential()
        model.add(Dropout(rate=0.2,input_shape=(self.signal_kind,)))
        model.add(Dense(units=12,activation='relu',input_shape=(self.signal_kind,)))
        model.add(Dropout(rate=0.2))
        model.add(Dense(units=12,activation='relu'))
        model.add(Dropout(rate=0.2))
        model.add(Dense(1,activation='sigmoid'))
        model.compile(optimizer='adam',loss='mse',metrics=['mae'])
        print(model.summary())
        return model
标签为：
        if self.a[i,2]==1:
            one_shut_train[12,:] = 1/(1+np.exp(-(one_shut_train[13,:]-(disr_time-0.3))*20))
        elif self.a[i,2]==-1:
            one_shut_train[12,:] = 0
回调函数为：
callback_list = [ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=10),\
                EarlyStopping(monitor='val_loss',patience=50),\
                ModelCheckpoint(filepath=filepath+'best.weight.h5',monitor='val_loss',verbose=1,save_best_only=True),\
                TensorBoard(log_dir=filepath+'/my_logs',histogram_freq=1)]
batch_size为:
        self.batch_size=32
        History = model.fit(x=X_train,y=y_train,epochs=self.epochs,\
                            batch_size=self.batch_size,validation_data=(X_val,y_val),verbose=2,\
                            callbacks=callback_list)
        result0 = model.predict(X_test,batch_size=32,verbose=1)
        result2 = model.predict(X,batch_size=32,verbose=1)
归一化为：
        self.min_max[:,0] = train_mat[:-2,:].min(axis=1)
        self.min_max[:,1] = train_mat[:-2,:].max(axis=1)
        #self.min_max = min_max_Ran
        train_mat[:-2,:] = minmax_scale(train_mat[:-2,:],axis=1)

        test_mat[:-2,:] = test_mat[:-2,:]-np.tile(self.min_max[:,0].reshape(-1,1),(1,test_mat.shape[1]))
        Range = (self.min_max[:,1]-self.min_max[:,0]).reshape(-1,1)
        test_mat[:-2,:] = test_mat[:-2,:]/np.tile(Range,(1,test_mat.shape[1]))

寻找最优的thresh是在测试集上找到的

'''与data20-5不同的是模型的每层节点数为12,'''
在迭代52次后结束

AUC最大值为0.891692,在阈值为0.110000下取得.
在训练集上的结果:
 train_Res: [222, 5, 29, 13, 86, 36] train_Rate: [0.977973568281, 0.0220264317180, 0.176829268292, 0.0792682926829, 0.524390243902, 0.2195121951219]

在验证集上的结果:
 val_Res: [218, 8, 27, 7, 94, 36] val_Rate: [0.9646017699115044, 0.035398230088, 0.16463414634, 0.042682926829, 0.5731707317, 0.2195121951219]

在测试集上的结果:
 test_Res: [221, 6, 28, 7, 97, 31] test_Rate: [0.973568281938326, 0.02643171806167401, 0.17177914110, 0.042944785276, 0.595092024, 0.19018404907]

在所有数据上的结果:
 all_Res: [661, 19, 84, 27, 277, 103] all_Rate: [0.9720588235294118, 0.027941176470, 0.17107942973, 0.0549898167, 0.56415478615, 0.209775967]


########################################################################################
'/gpfs/home/音乐/data/data20/data20_17/'是在以下模型下取得的结果：
建立简单的全连接BP神经网络
    def build_model(self,optimizer='rmsprop'):
        model = Sequential()
        model.add(Dropout(rate=0.2,input_shape=(self.signal_kind,)))
        model.add(Dense(units=22,activation='relu',input_shape=(self.signal_kind,)))
        model.add(Dropout(rate=0.2))
        model.add(Dense(units=22,activation='relu'))
        model.add(Dropout(rate=0.2))
        model.add(Dense(1,activation='sigmoid'))
        model.compile(optimizer='adam',loss='mse',metrics=['mae'])
        print(model.summary())
        return model
标签为：
        if self.a[i,2]==1:
            one_shut_train[12,:] = 1/(1+np.exp(-(one_shut_train[13,:]-(disr_time-0.3))*20))
        elif self.a[i,2]==-1:
            one_shut_train[12,:] = 0
回调函数为：
callback_list = [ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=10),\
                EarlyStopping(monitor='val_loss',patience=50),\
                ModelCheckpoint(filepath=filepath+'best.weight.h5',monitor='val_loss',verbose=1,save_best_only=True),\
                TensorBoard(log_dir=filepath+'/my_logs',histogram_freq=1)]
batch_size为:
        self.batch_size=32
        History = model.fit(x=X_train,y=y_train,epochs=self.epochs,\
                            batch_size=self.batch_size,validation_data=(X_val,y_val),verbose=2,\
                            callbacks=callback_list)
        result0 = model.predict(X_test,batch_size=32,verbose=1)
        result2 = model.predict(X,batch_size=32,verbose=1)
归一化为：
        self.min_max[:,0] = train_mat[:-2,:].min(axis=1)
        self.min_max[:,1] = train_mat[:-2,:].max(axis=1)
        #self.min_max = min_max_Ran
        train_mat[:-2,:] = minmax_scale(train_mat[:-2,:],axis=1)

        test_mat[:-2,:] = test_mat[:-2,:]-np.tile(self.min_max[:,0].reshape(-1,1),(1,test_mat.shape[1]))
        Range = (self.min_max[:,1]-self.min_max[:,0]).reshape(-1,1)
        test_mat[:-2,:] = test_mat[:-2,:]/np.tile(Range,(1,test_mat.shape[1]))

寻找最优的thresh是在测试集上找到的

'''与data20-5不同的是模型的每层节点数为22,'''
在迭代52次后结束
AUC最大值为0.905692,在阈值为0.100000下取得.
在训练集上的结果:
 train_Res: [217, 10, 36, 10, 86, 32] train_Rate: [0.955947136563, 0.0440528634361, 0.21951219512, 0.060975609, 0.52439024390, 0.195121951219]

在验证集上的结果:
 val_Res: [216, 10, 26, 6, 101, 31] val_Rate: [0.9557522123, 0.04424778761, 0.15853658536, 0.036585365, 0.61585365853, 0.18902439024390244]

在测试集上的结果:
 test_Res: [219, 8, 29, 4, 105, 25] test_Rate: [0.964757709251, 0.03524229074, 0.177914110, 0.0245398773, 0.6441717, 0.15337423312883436]

在所有数据上的结果:
 all_Res: [652, 28, 91, 20, 292, 88] all_Rate: [0.958823529, 0.0411764705, 0.185336048, 0.040733197, 0.594704684, 0.179226069]

########################################################################################
'/gpfs/home/音乐/data/data20/data20_18/'是在以下模型下取得的结果：
建立简单的全连接BP神经网络
    def build_model(self,optimizer='rmsprop'):
        model = Sequential()
        model.add(Dropout(rate=0.2,input_shape=(self.signal_kind,)))
        model.add(Dense(units=24,activation='relu',input_shape=(self.signal_kind,)))
        model.add(Dropout(rate=0.2))
        model.add(Dense(units=24,activation='relu'))
        model.add(Dropout(rate=0.2))
        model.add(Dense(1,activation='sigmoid'))
        model.compile(optimizer='adam',loss='mse',metrics=['mae'])
        print(model.summary())
        return model
标签为：
        if self.a[i,2]==1:
            one_shut_train[12,:] = 1/(1+np.exp(-(one_shut_train[13,:]-(disr_time-0.3))*20))
        elif self.a[i,2]==-1:
            one_shut_train[12,:] = 0
回调函数为：
callback_list = [ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=10),\
                EarlyStopping(monitor='val_loss',patience=50),\
                ModelCheckpoint(filepath=filepath+'best.weight.h5',monitor='val_loss',verbose=1,save_best_only=True),\
                TensorBoard(log_dir=filepath+'/my_logs',histogram_freq=1)]
batch_size为:
        self.batch_size=32
        History = model.fit(x=X_train,y=y_train,epochs=self.epochs,\
                            batch_size=self.batch_size,validation_data=(X_val,y_val),verbose=2,\
                            callbacks=callback_list)
        result0 = model.predict(X_test,batch_size=32,verbose=1)
        result2 = model.predict(X,batch_size=32,verbose=1)
归一化为：
        self.min_max[:,0] = train_mat[:-2,:].min(axis=1)
        self.min_max[:,1] = train_mat[:-2,:].max(axis=1)
        #self.min_max = min_max_Ran
        train_mat[:-2,:] = minmax_scale(train_mat[:-2,:],axis=1)

        test_mat[:-2,:] = test_mat[:-2,:]-np.tile(self.min_max[:,0].reshape(-1,1),(1,test_mat.shape[1]))
        Range = (self.min_max[:,1]-self.min_max[:,0]).reshape(-1,1)
        test_mat[:-2,:] = test_mat[:-2,:]/np.tile(Range,(1,test_mat.shape[1]))

寻找最优的thresh是在测试集上找到的

'''与data20-5不同的是模型的每层节点数为24,'''
迭代了52轮

AUC最大值为0.901759,在阈值为0.110000下取得.
在训练集上的结果:
 train_Res: [218, 9, 32, 12, 85, 35] train_Rate: [0.96035242290, 0.039647577, 0.1951219, 0.0731707, 0.518292, 0.213414]

在验证集上的结果:
 val_Res: [217, 9, 31, 5, 97, 31] val_Rate: [0.960176991, 0.039823, 0.1890243, 0.03048, 0.5914634, 0.18902]

在测试集上的结果:
 test_Res: [220, 7, 26, 7, 103, 27] test_Rate: [0.969162, 0.030837, 0.15950920, 0.042944, 0.631901, 0.16564]

在所有数据上的结果:
 all_Res: [655, 25, 89, 24, 285, 93] all_Rate: [0.963235, 0.036764, 0.1812627, 0.048879, 0.580448, 0.1894093]


########################################################################################
'/gpfs/home/音乐/data/data20/data20_19/'是在以下模型下取得的结果：
建立简单的全连接BP神经网络
    def build_model(self,optimizer='rmsprop'):
        model = Sequential()
        model.add(Dropout(rate=0.2,input_shape=(self.signal_kind,)))
        model.add(Dense(units=26,activation='relu',input_shape=(self.signal_kind,)))
        model.add(Dropout(rate=0.2))
        model.add(Dense(units=26,activation='relu'))
        model.add(Dropout(rate=0.2))
        model.add(Dense(1,activation='sigmoid'))
        model.compile(optimizer='adam',loss='mse',metrics=['mae'])
        print(model.summary())
        return model
标签为：
        if self.a[i,2]==1:
            one_shut_train[12,:] = 1/(1+np.exp(-(one_shut_train[13,:]-(disr_time-0.3))*20))
        elif self.a[i,2]==-1:
            one_shut_train[12,:] = 0
回调函数为：
callback_list = [ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=10),\
                EarlyStopping(monitor='val_loss',patience=50),\
                ModelCheckpoint(filepath=filepath+'best.weight.h5',monitor='val_loss',verbose=1,save_best_only=True),\
                TensorBoard(log_dir=filepath+'/my_logs',histogram_freq=1)]
batch_size为:
        self.batch_size=32
        History = model.fit(x=X_train,y=y_train,epochs=self.epochs,\
                            batch_size=self.batch_size,validation_data=(X_val,y_val),verbose=2,\
                            callbacks=callback_list)
        result0 = model.predict(X_test,batch_size=32,verbose=1)
        result2 = model.predict(X,batch_size=32,verbose=1)
归一化为：
        self.min_max[:,0] = train_mat[:-2,:].min(axis=1)
        self.min_max[:,1] = train_mat[:-2,:].max(axis=1)
        #self.min_max = min_max_Ran
        train_mat[:-2,:] = minmax_scale(train_mat[:-2,:],axis=1)

        test_mat[:-2,:] = test_mat[:-2,:]-np.tile(self.min_max[:,0].reshape(-1,1),(1,test_mat.shape[1]))
        Range = (self.min_max[:,1]-self.min_max[:,0]).reshape(-1,1)
        test_mat[:-2,:] = test_mat[:-2,:]/np.tile(Range,(1,test_mat.shape[1]))

寻找最优的thresh是在测试集上找到的

'''与data20-5不同的是模型的每层节点数为26,'''
迭代了52轮
在训练集上的结果:
 train_Res: [218, 9, 31, 6, 95, 32] train_Rate: [0.960352422, 0.039647577, 0.1890243, 0.0365853, 0.5792682926829268, 0.1951219512195122] 

在验证集上的结果:
 val_Res: [210, 16, 25, 6, 103, 30] val_Rate: [0.9292035398230089, 0.07079646017699115, 0.1524390, 0.036585365, 0.6280487, 0.18292682926829268] 

在测试集上的结果:
 test_Res: [219, 8, 23, 6, 111, 23] test_Rate: [0.9647577092, 0.03524229074, 0.1411042944785276, 0.036809815, 0.6809815950920245, 0.1411042] 

在所有数据上的结果:
 all_Res: [647, 33, 79, 18, 309, 85] all_Rate: [0.95147058, 0.048529, 0.1608961303, 0.03665987780040733, 0.6293279022403259, 0.1731160896] 


########################################################################################
'/gpfs/home/音乐/data/data20/data20_20/'是在以下模型下取得的结果：
建立简单的全连接BP神经网络
    def build_model(self,optimizer='rmsprop'):
        model = Sequential()
        model.add(Dropout(rate=0.2,input_shape=(self.signal_kind,)))
        model.add(Dense(units=28,activation='relu',input_shape=(self.signal_kind,)))
        model.add(Dropout(rate=0.2))
        model.add(Dense(units=28,activation='relu'))
        model.add(Dropout(rate=0.2))
        model.add(Dense(1,activation='sigmoid'))
        model.compile(optimizer='adam',loss='mse',metrics=['mae'])
        print(model.summary())
        return model
标签为：
        if self.a[i,2]==1:
            one_shut_train[12,:] = 1/(1+np.exp(-(one_shut_train[13,:]-(disr_time-0.3))*20))
        elif self.a[i,2]==-1:
            one_shut_train[12,:] = 0
回调函数为：
callback_list = [ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=10),\
                EarlyStopping(monitor='val_loss',patience=50),\
                ModelCheckpoint(filepath=filepath+'best.weight.h5',monitor='val_loss',verbose=1,save_best_only=True),\
                TensorBoard(log_dir=filepath+'/my_logs',histogram_freq=1)]
batch_size为:
        self.batch_size=32
        History = model.fit(x=X_train,y=y_train,epochs=self.epochs,\
                            batch_size=self.batch_size,validation_data=(X_val,y_val),verbose=2,\
                            callbacks=callback_list)
        result0 = model.predict(X_test,batch_size=32,verbose=1)
        result2 = model.predict(X,batch_size=32,verbose=1)
归一化为：
        self.min_max[:,0] = train_mat[:-2,:].min(axis=1)
        self.min_max[:,1] = train_mat[:-2,:].max(axis=1)
        #self.min_max = min_max_Ran
        train_mat[:-2,:] = minmax_scale(train_mat[:-2,:],axis=1)

        test_mat[:-2,:] = test_mat[:-2,:]-np.tile(self.min_max[:,0].reshape(-1,1),(1,test_mat.shape[1]))
        Range = (self.min_max[:,1]-self.min_max[:,0]).reshape(-1,1)
        test_mat[:-2,:] = test_mat[:-2,:]/np.tile(Range,(1,test_mat.shape[1]))

寻找最优的thresh是在测试集上找到的

'''与data20-5不同的是模型的每层节点数为28,'''
迭代次数为52
AUC最大值为0.894760,在阈值为0.110000下取得.
在训练集上的结果:
 train_Res: [223, 4, 29, 10, 84, 41] train_Rate: [0.982378854, 0.0176211453, 0.17682926829268292, 0.06097560975609756, 0.5121951219512195, 0.25]

在验证集上的结果:
 val_Res: [220, 6, 28, 8, 89, 39] val_Rate: [0.9734513274336283, 0.02654867256637168, 0.17073170731707318, 0.048780487, 0.54268292, 0.2378048]

在测试集上的结果:
 test_Res: [221, 6, 33, 7, 93, 30] test_Rate: [0.97356828, 0.0264317, 0.202453987, 0.0429447, 0.57055214, 0.1840490]

在所有数据上的结果:
 all_Res: [664, 16, 90, 25, 266, 110] all_Rate: [0.97647058, 0.02352941, 0.18329938, 0.0509164, 0.541751527, 0.2240325865580448]

########################################################################################################
根据上面的计划，依次测了以下节点数，32,40,50,60,70,80,90,100,200,300等,结果分如下：
#####################################
当节点数为32时：'/gpfs/home/音乐/data/data20/data20_21/'
在迭代52次后结束，AUC最大值为0.918435,在阈值为0.110000下取得.

在训练集上的结果:
 train_Res: [216, 11, 38, 6, 83, 37] train_Rate: [0.951541850, 0.04845814, 0.23170731, 0.0365853, 0.5060975609756098, 0.22560975609756098]

在验证集上的结果:
 val_Res: [215, 11, 25, 3, 102, 34] val_Rate: [0.9513274336283186, 0.048672566371681415, 0.15243902, 0.018292, 0.621951, 0.2073170731707317]

在测试集上的结果:
 test_Res: [222, 5, 34, 6, 100, 23] test_Rate: [0.9779735682819384, 0.022026431718061675, 0.208588, 0.03680981595092025, 0.61349, 0.141104]

在所有数据上的结果:
 all_Res: [653, 27, 97, 15, 285, 94] all_Rate: [0.960294117, 0.0397058, 0.197556, 0.030549898, 0.58044806, 0.19144602851323828]
####################################
当节点数为50时：'/gpfs/home/音乐/data/data20/data20_22/'
在迭代52次后结束，AUC最大值为0.904827,在阈值为0.150000下取得.

在训练集上的结果:
 train_Res: [219, 8, 31, 11, 84, 38] train_Rate: [0.9647577092, 0.03524229, 0.1890243, 0.0670731, 0.5121951219512195, 0.23170731707317074]

在验证集上的结果:
 val_Res: [217, 9, 29, 3, 96, 36] val_Rate: [0.9601769, 0.03982300, 0.1768292, 0.01829268, 0.5853658, 0.219512]

在测试集上的结果:
 test_Res: [220, 7, 30, 6, 101, 26] test_Rate: [0.9691629, 0.0308370, 0.184049, 0.036809, 0.619631, 0.15950920245398773]

在所有数据上的结果:
 all_Res: [656, 24, 90, 20, 281, 100] all_Rate: [0.964705, 0.035294, 0.18329938, 0.04073319755600815, 0.5723014256619144, 0.203665987]
####################################
当节点数为60时：'/gpfs/home/音乐/data/data20/data20_23/'
在迭代51次后结束，AUC最大值为0.891219,在阈值为0.170000下取得.
在训练集上的结果:
 train_Res: [221, 6, 32, 12, 89, 31] train_Rate: [0.9735682, 0.0264317, 0.1951219, 0.0731707, 0.542682, 0.1890243]

在验证集上的结果:
 val_Res: [216, 10, 27, 7, 99, 31] val_Rate: [0.955752, 0.044247, 0.16463414, 0.0426829, 0.6036585, 0.1890243]

在测试集上的结果:
 test_Res: [218, 9, 27, 7, 100, 29] test_Rate: [0.960352, 0.0396475, 0.1656441, 0.0429447, 0.613496, 0.1779141]

在所有数据上的结果:
 all_Res: [655, 25, 86, 26, 288, 91] all_Rate: [0.9632352, 0.03676, 0.17515274, 0.05295315, 0.5865580, 0.18533604]


####################################
当节点数为70时：'/gpfs/home/音乐/data/data20/data20_24/'
在迭代51次后结束，AUC最大值为0.904354,在阈值为0.140000下取得.
在训练集上的结果:
 train_Res: [213, 14, 32, 10, 94, 28] train_Rate: [0.9383259, 0.061674, 0.1951219, 0.060975, 0.573170, 0.170731]

在验证集上的结果:
 val_Res: [211, 15, 30, 2, 100, 32] val_Rate: [0.933628, 0.0663716, 0.1829268, 0.01219512, 0.6097560, 0.195121]

在测试集上的结果:
 test_Res: [217, 10, 25, 7, 107, 24] test_Rate: [0.95594713, 0.0440528, 0.15337423, 0.0429447, 0.6564417 0.1472392]

在所有数据上的结果:
 all_Res: [641, 39, 87, 19, 301, 84] all_Rate: [0.9426470, 0.057352941, 0.1771894, 0.0386965, 0.61303462, 0.171079]


####################################
当节点数为80时：'/gpfs/home/音乐/data/data20/data20_25/'
在迭代61次后结束，AUC最大值为0.919137,在阈值为0.190000下取得.
在训练集上的结果:
 train_Res: [207, 20, 23, 6, 118, 17] train_Rate: [0.9118942, 0.08810, 0.1402439, 0.0365853, 0.7195121, 0.1036585]

在验证集上的结果:
 val_Res: [201, 25, 23, 2, 125, 14] val_Rate: [0.88938053, 0.11061946, 0.1402439, 0.0121951, 0.7621951, 0.0853658]

在测试集上的结果:
 test_Res: [207, 20, 14, 7, 130, 12] test_Rate: [0.9118942, 0.0881057, 0.085889570, 0.042944782, 0.7975460, 0.0736196]

在所有数据上的结果:
 all_Res: [615, 65, 60, 15, 373, 43] all_Rate: [0.904411, 0.0955882, 0.1221995, 0.0305498, 0.7596741, 0.08757637]


####################################
当节点数为100时：'/gpfs/home/音乐/data/data20/data20_26/'
在87轮迭代后结束，AUC最大值为0.929759,在阈值为0.230000下取得.

在训练集上的结果:
 train_Res: [214, 13, 24, 7, 115, 18] train_Rate: [0.94273127, 0.0572687, 0.1463414, 0.04268292, 0.701219512, 0.1097560975]

在验证集上的结果:
 val_Res: [211, 15, 18, 7, 124, 15] val_Rate: [0.93362831, 0.06637168, 0.109756097, 0.0426829268, 0.7560975, 0.0914634]

在测试集上的结果:
 test_Res: [216, 11, 17, 7, 124, 15] test_Rate: [0.95154185, 0.0484581, 0.10429447, 0.04294478, 0.760736196, 0.092024539]

在所有数据上的结果:
 all_Res: [641, 39, 59, 21, 363, 48] all_Rate: [0.942647058, 0.05735294, 0.12016293, 0.0427698, 0.73930753, 0.09775967]

####################################
当节点数为200时：'/gpfs/home/音乐/data/data20/data20_27/'

迭代59轮后结束，AUC最大值为0.903881,在阈值为0.320000下取得.
在训练集上的结果:
 train_Res: [215, 12, 31, 4, 108, 21] train_Rate: [0.947136563, 0.052863436, 0.18902439, 0.02439024, 0.65853658, 0.1280487804]

在验证集上的结果:
 val_Res: [197, 29, 23, 3, 115, 23] val_Rate: [0.8716814159, 0.12831858, 0.140243, 0.018292682, 0.70121951, 0.140243902]

在测试集上的结果:
 test_Res: [214, 13, 16, 7, 118, 22] test_Rate: [0.94273127, 0.0572687, 0.0981595, 0.0429447, 0.7239263, 0.13496932]

在所有数据上的结果:
 all_Res: [626, 54, 70, 14, 341, 66] all_Rate: [0.9205882, 0.0794117, 0.14256619, 0.0285132, 0.6945010, 0.1344195519]


####################################
当节点数为300时：'/gpfs/home/音乐/data/data20/data20_28/'
在迭代62轮后结束，AUC最大值为0.916151,在阈值为0.320000下取得.
在训练集上的结果:
 train_Res: [207, 20, 31, 2, 110, 21] train_Rate: [0.9118942731, 0.08810572, 0.1890243902, 0.01219512, 0.67073170, 0.1280487]

在验证集上的结果:
 val_Res: [198, 28, 20, 5, 119, 20] val_Rate: [0.87610619, 0.12389380, 0.121951, 0.0304878, 0.725609, 0.1219512]

在测试集上的结果:
 test_Res: [214, 13, 16, 6, 123, 18] test_Rate: [0.942731277, 0.05726872, 0.0981595, 0.03680981, 0.7546012, 0.11042944]

在所有数据上的结果:
 all_Res: [619, 61, 67, 13, 352, 59] all_Rate: [0.91029411, 0.0897058, 0.13645621, 0.0264765, 0.71690427, 0.1201629]

####################################
当节点数为400时：'/gpfs/home/音乐/data/data20/data20_29/'
在迭代71轮后结束，AUC最大值为0.914732,在阈值为0.280000下取得.

在训练集上的结果:
 train_Res: [205, 22, 23, 3, 124, 14] train_Rate: [0.90308370, 0.09691629, 0.1402439, 0.01829268, 0.7560975, 0.08536585]

在验证集上的结果:
 val_Res: [195, 31, 21, 3, 130, 10] val_Rate: [0.8628318, 0.137168141, 0.1280487, 0.018292682, 0.7926829, 0.060975609]

在测试集上的结果:
 test_Res: [205, 22, 16, 3, 132, 12] test_Rate: [0.90308370, 0.0969162, 0.09815950, 0.0184049, 0.8098159, 0.073619]

在所有数据上的结果:
 all_Res: [605, 75, 60, 9, 386, 36] all_Rate: [0.8897058, 0.110294, 0.122199592, 0.01832993, 0.7861507, 0.0733197]


####################################
当节点数为500时：'/gpfs/home/音乐/data/data20/data20_30/'
在迭代60轮后结束，AUC最大值为0.903408,在阈值为0.420000下取得.
在训练集上的结果:
 train_Res: [209, 18, 32, 5, 108, 19] train_Rate: [0.920704, 0.0792951, 0.19512195, 0.0304878, 0.6585365 0.11585365]

在验证集上的结果:
 val_Res: [204, 22, 23, 4, 119, 18] val_Rate: [0.90265486, 0.09734513, 0.1402439, 0.02439024, 0.725609, 0.10975609]

在测试集上的结果:
 test_Res: [211, 16, 16, 5, 122, 20] test_Rate: [0.92951541, 0.07048458, 0.09815950, 0.0306748, 0.748466, 0.1226993]

在所有数据上的结果:
 all_Res: [624, 56, 71, 14, 349, 57] all_Rate: [0.917647054, 0.082352, 0.1446028, 0.0285132, 0.7107942, 0.116089]

####################################
当节点数为600时：'/gpfs/home/音乐/data/data20/data20_31/'
在迭代65轮后结束，AUC最大值为0.896962,在阈值为0.580000下取得.
在训练集上的结果:
 train_Res: [223, 4, 33, 6, 90, 35] train_Rate: [0.982378854, 0.017621145, 0.2012195, 0.0365853658, 0.5487804, 0.21341463]

在验证集上的结果:
 val_Res: [215, 11, 23, 9, 95, 37] val_Rate: [0.951327433, 0.04867256, 0.14024390, 0.05487804, 0.5792682, 0.22560975609756098]

在测试集上的结果:
 test_Res: [222, 5, 19, 11, 103, 30] test_Rate: [0.9779735, 0.022026431718061675, 0.1165644171779141, 0.0674846, 0.63190184, 0.18404]

在所有数据上的结果:
 all_Res: [660, 20, 75, 26, 288, 102] all_Rate: [0.97058823, 0.029411764, 0.1527494908, 0.052953156, 0.58655804, 0.20773930]


####################################
当节点数为700时：'/gpfs/home/音乐/data/data20/data20_32/'
在迭代59后结束，AUC最大值为0.902151,在阈值为0.500000下取得.
在训练集上的结果:
train_Res: [219, 8, 33, 7, 101, 23] train_Rate: [0.9647577, 0.03524229, 0.20121951, 0.042682926, 0.6158536, 0.1402439]

在验证集上的结果:
val_Res: [205, 21, 21, 7, 107, 29] val_Rate: [0.90707964, 0.092920, 0.12804878, 0.042682, 0.652439, 0.176829]

在测试集上的结果:
test_Res: [216, 11, 22, 7, 110, 24] test_Rate: [0.95154185, 0.04845814, 0.134969, 0.0429447, 0.6748466, 0.147239]

在所有数据上的结果:
all_Res: [640, 40, 76, 21, 318, 76] all_Rate: [0.94117647, 0.0588235, 0.15478615, 0.042769857, 0.647657, 0.1547861]


####################################
当节点数为800时：'/gpfs/home/音乐/data/data20/data20_33/'
在迭代63轮后结束，AUC最大值为0.929367,在阈值为0.360000下取得.

在训练集上的结果:
 train_Res: [218, 9, 27, 3, 114, 20] train_Rate: [0.960352422, 0.03964757, 0.1646341, 0.0182926, 0.6951219, 0.1219512]

在验证集上的结果:
 val_Res: [210, 16, 25, 6, 115, 18] val_Rate: [0.92920353, 0.07079646, 0.152439, 0.03658536, 0.70121951, 0.1097560]

在测试集上的结果:
 test_Res: [220, 7, 21, 6, 118, 18] test_Rate: [0.9691629, 0.0308370 0.1288343, 0.0368098, 0.723926, 0.1104294]

在所有数据上的结果:
 all_Res: [648, 32, 73, 15, 347, 56] all_Rate: [0.9529411, 0.04705882, 0.148676, 0.030549, 0.706720, 0.11405295]


####################################
当节点数为900时：'/gpfs/home/音乐/data/data20/data20_34/'
在迭代57轮后结束，AUC最大值为0.882165,在阈值为0.340000下取得.

在训练集上的结果:
 train_Res: [194, 33, 27, 3, 120, 14] train_Rate: [0.85462555, 0.14537444, 0.1646341, 0.01829268, 0.73170737, 0.0853658]

在验证集上的结果:
 val_Res: [184, 42, 22, 2, 127, 13] val_Rate: [0.8141592, 0.1858407, 0.1341463, 0.0121951, 0.774390243, 0.0792682]

在测试集上的结果:
 test_Res: [193, 34, 15, 4, 130, 14] test_Rate: [0.850220, 0.1497797, 0.09202453, 0.024539877, 0.7975460, 0.0858895]

在所有数据上的结果:
 all_Res: [571, 109, 64, 9, 377, 41] all_Rate: [0.8397058, 0.16029411, 0.1303462, 0.01832993, 0.7678207, 0.0835030]


####################################
当节点数为1000时：'/gpfs/home/音乐/data/data20/data20_35/'

在迭代60轮后结束，AUC最大值为0.893097,在阈值为0.250000下取得.
在训练集上的结果:
 train_Res: [185, 42, 20, 3, 128, 13] train_Rate: [0.8149779, 0.18502202, 0.12195121, 0.0182926, 0.7804878, 0.0792682]

在验证集上的结果:
 val_Res: [176, 50, 16, 0, 139, 9] val_Rate: [0.77876106, 0.2212389, 0.0975609, 0.0, 0.847560, 0.05487804]

在测试集上的结果:
 test_Res: [191, 36, 15, 3, 136, 9] test_Rate: [0.8414096, 0.15859030, 0.09202453, 0.0184049, 0.834355, 0.05521472]

在所有数据上的结果:
 all_Res: [552, 128, 51, 6, 403, 31] all_Rate: [0.8117647, 0.188235294, 0.1038696, 0.0122199, 0.82077393, 0.06313645]
####################################
当隐层有三层时，节点数为20,20,20时，：'/gpfs/home/音乐/data/data20/data20_36/'
在迭代51次后结束，AUC最大值为0.899557,在阈值为0.090000下取得.
在训练集上的结果:
 train_Res: [217, 10, 36, 5, 89, 34] train_Rate: [0.955947136, 0.0440528, 0.21951219, 0.0304878, 0.5426829, 0.2073170]

在验证集上的结果:
 val_Res: [214, 12, 20, 7, 102, 35] val_Rate: [0.946902, 0.05309734, 0.1219512, 0.04268296, 0.6219512195121951, 0.21341463]

在测试集上的结果:
 test_Res: [219, 8, 30, 7, 99, 27] test_Rate: [0.9647577, 0.03524229, 0.18404907, 0.04294478, 0.60736196, 0.1656441]

在所有数据上的结果:
 all_Res: [650, 30, 86, 19, 290, 96] all_Rate: [0.955882352, 0.04411764, 0.17515274, 0.038696537, 0.59063136, 0.19551931]

####################################
当隐层有三层时，节点数为100,100,100时，：'/gpfs/home/音乐/data/data20/data20_37/'
在迭代55次后结束，AUC最大值为0.921421,在阈值为0.320000下取得.
在训练集上的结果:
 train_Res: [211, 16, 34, 6, 103, 21] train_Rate: [0.929515418, 0.07048458, 0.2073170, 0.0365853, 0.6280487, 0.12804878]

在验证集上的结果:
 val_Res: [205, 21, 25, 4, 114, 21] val_Rate: [0.9070796, 0.0929203, 0.152439, 0.0243902, 0.69512195, 0.128048]

在测试集上的结果
 test_Res: [215, 12, 15, 7, 124, 17] test_Rate: [0.9471365, 0.0528634, 0.09202453, 0.0429447, 0.76073619, 0.10429447]

在所有数据上的结果:
 all_Res: [631, 49, 74, 17, 341, 59] all_Rate: [0.92794117, 0.07205882, 0.15071283, 0.03462321, 0.6945010, 0.120162]
####################################
当隐层有三层时，节点数为200,200,200时，：'/gpfs/home/音乐/data/data20/data20_38/'
在迭代61次后结束，AUC最大值为0.915205,在阈值为0.320000下取得.
在训练集上的结果:
 train_Res: [199, 28, 26, 3, 119, 16] train_Rate: [0.8766519823788547, 0.123348, 0.158536585, 0.01829268, 0.725609, 0.0975609]

在验证集上的结果:
 val_Res: [192, 34, 22, 1, 127, 14] val_Rate: [0.849557522 0.15044247, 0.13414634146341464, 0.00609756, 0.774390243902439, 0.0853658]

在测试集上的结果:
 test_Res: [208, 19, 15, 5, 129, 14] test_Rate: [0.916299559, 0.0837004, 0.0920245, 0.03067484, 0.791411, 0.0858895]

在所有数据上的结果:
 all_Res: [599, 81, 63, 9, 375, 44] all_Rate: [0.880882352, 0.11911764, 0.1283095, 0.01832993 0.76374745, 0.0896130]

####################################
当隐层有三层时，节点数为300,300,300时，：'/gpfs/home/音乐/data/data20/data20_39/'
在迭代55次后结束，AUC最大值为0.889881,在阈值为0.460000下取得.
在训练集上的结果:
 train_Res: [218, 9, 35, 10, 93, 26] train_Rate: [0.9603524, 0.0396475, 0.21341463, 0.0609756, 0.5670731, 0.158536585]

在验证集上的结果:
 val_Res: [206, 20, 35, 4, 102, 23] val_Rate: [0.9115044, 0.08849557, 0.21341463, 0.0243902, 0.6219512, 0.14024390]

在测试集上的结果:
 test_Res: [216, 11, 21, 5, 109, 28] test_Rate: [0.951541, 0.048458, 0.1288343, 0.030674, 0.6687116, 0.17177]

在所有数据上的结果:
 all_Res: [640, 40, 91, 19, 304, 77] all_Rate: [0.9411764, 0.0588235, 0.18533604, 0.03869653, 0.619144, 0.1568228]
####################################
当隐层有三层时，节点数为400,400,400时，：'/gpfs/home/音乐/data/data20/data20_40/'
在迭代60次后结束，AUC最大值为0.903408,在阈值为0.400000下取得.
在训练集上的结果:
 train_Res: [207, 20, 34, 9, 100, 21] train_Rate: [0.91189427, 0.08810572, 0.20731707, 0.0548780, 0.60975609, 0.12804878]

在验证集上的结果:
 val_Res: [199, 27, 26, 5, 112, 21] val_Rate: [0.880530973, 0.119469026, 0.15853658, 0.03048780, 0.68292682, 0.1280487]

在测试集上的结果:
 test_Res: [211, 16, 15, 7, 121, 20] test_Rate: [0.92951541, 0.0704845, 0.092024539, 0.042944, 0.7423312, 0.12269938]

在所有数据上的结果:
 all_Res: [617, 63, 75, 21, 333, 62] all_Rate: [0.90735294, 0.09264705, 0.152749490, 0.042769857, 0.6782077, 0.1262729]


####################################
当隐层有三层时，节点数为500,500,500时，：'/gpfs/home/音乐/data/data20/data20_41/'
在迭代53次后结束，AUC最大值为0.906948,在阈值为0.390000下取得.
在训练集上的结果:
 train_Res: [212, 15, 29, 4, 109, 22] train_Rate: [0.933920704, 0.0660792951, 0.17682926, 0.0243902439, 0.6646341, 0.1341463]

在验证集上的结果:
 val_Res: [202, 24, 25, 2, 118, 19] val_Rate: [0.8938053097, 0.10619469, 0.15243902, 0.01219512, 0.7195121, 0.1158536]

在测试集上的结果:
 test_Res: [214, 13, 16, 5, 121, 21] test_Rate: [0.94273127, 0.05726872, 0.09815950, 0.030674846, 0.74233128, 0.1288343]

在所有数据上的结果:
 all_Res: [628, 52, 70, 11, 348, 62] all_Rate: [0.923529411, 0.076470588, 0.14256619, 0.022403258, 0.7087576, 0.1262729]


#########################################################################################################
接下来探索便签为的结果：
        if self.a[i,2]==1:
            one_shut_train[12,:] = 1/(1+np.exp(-(one_shut_train[13,:]-(t1+t2)/2)*(10/(t2-t1))))
        elif self.a[i,2]==-1:
            one_shut_train[12,:] = 0
#####################
当有两层隐藏层，节点数为100,100时，计算了多遍
'/gpfs/home/音乐/data/data20/data20_42/'
AUC最大值为0.879098,在阈值为0.340000下取得.
在训练集上的结果:
 train_Res: [195, 32, 31, 2, 114, 17]
 train_Rate: [0.8590308370044053, 0.14096916299559473, 0.18902439024390244, 0.012195121951219513, 0.6951219512195121, 0.10365853658536585]

在验证集上的结果:
 val_Res: [183, 43, 14, 4, 127, 19]
 val_Rate: [0.8097345132743363, 0.1902654867256637, 0.08536585365853659, 0.024390243902439025, 0.774390243902439, 0.11585365853658537]

在测试集上的结果:
 test_Res: [193, 34, 15, 1, 132, 15]
 test_Rate: [0.8502202643171806, 0.14977973568281938, 0.09202453987730061, 0.006134969325153374, 0.8098159509202454, 0.09202453987730061]

在所有数据上的结果:
 all_Res: [571, 109, 60, 7, 373, 51]
 all_Rate: [0.8397058823529412, 0.16029411764705884, 0.12219959266802444, 0.014256619144602852, 0.7596741344195519, 0.10386965376782077]


'/gpfs/home/音乐/data/data20/data20_42_1/'

AUC最大值为0.901759,在阈值为0.250000下取得.
在训练集上的结果:
 train_Res: [222, 5, 30, 5, 102, 27]
 train_Rate: [0.9779735682819384, 0.022026431718061675, 0.18292682926829268, 0.03048780487804878, 0.6219512195121951, 0.16463414634146342]

在验证集上的结果:
 val_Res: [215, 11, 23, 7, 104, 30]
 val_Rate: [0.9513274336283186, 0.048672566371681415, 0.1402439024390244, 0.042682926829268296, 0.6341463414634146, 0.18292682926829268]

在测试集上的结果:
 test_Res: [220, 7, 23, 9, 104, 27]
test_Rate: [0.9691629955947136, 0.030837004405286344, 0.1411042944785276, 0.05521472392638037, 0.6380368098159509, 0.1656441717791411]

在所有数据上的结果:
 all_Res: [657, 23, 76, 21, 310, 84]
 all_Rate: [0.9661764705882353, 0.033823529411764704, 0.15478615071283094, 0.04276985743380855, 0.6313645621181263, 0.1710794297352342]


'/gpfs/home/音乐/data/data20/data20_42_2/'
AUC最大值为0.906557,在阈值为0.210000下取得.
在训练集上的结果:
 train_Res: [220, 7, 30, 9, 97, 28]
 train_Rate: [0.9691629955947136, 0.030837004405286344, 0.18292682926829268, 0.054878048780487805, 0.5914634146341463, 0.17073170731707318]

在验证集上的结果:
 val_Res: [215, 11, 27, 6, 98, 33]
 val_Rate: [0.9513274336283186, 0.048672566371681415, 0.16463414634146342, 0.036585365853658534, 0.5975609756097561, 0.20121951219512196]

在测试集上的结果:
 test_Res: [218, 9, 27, 7, 105, 24]
 test_Rate: [0.960352422907489, 0.039647577092511016, 0.1656441717791411, 0.04294478527607362, 0.6441717791411042, 0.147239263803681]

在所有数据上的结果:
 all_Res: [653, 27, 84, 22, 300, 85]
 all_Rate: [0.9602941176470589, 0.039705882352941174, 0.1710794297352342, 0.04480651731160896, 0.6109979633401222, 0.17311608961303462]


'/gpfs/home/音乐/data/data20/data20_42_3/'
AUC最大值为0.904746,在阈值为0.150000下取得.
在训练集上的结果:
 train_Res: [216, 11, 32, 1, 105, 26]
 train_Rate: [0.9515418502202643, 0.048458149779735685, 0.1951219512195122, 0.006097560975609756, 0.6402439024390244, 0.15853658536585366]

在验证集上的结果:
 val_Res: [202, 24, 23, 2, 114, 25]
 val_Rate: [0.8938053097345132, 0.10619469026548672, 0.1402439024390244, 0.012195121951219513, 0.6951219512195121, 0.1524390243902439]

在测试集上的结果:
 test_Res: [213, 14, 12, 5, 125, 21]
 test_Rate: [0.9383259911894273, 0.06167400881057269, 0.0736196319018405, 0.03067484662576687, 0.7668711656441718, 0.12883435582822086]

在所有数据上的结果:
 all_Res: [631, 49, 67, 8, 344, 72]
 all_Rate: [0.9279411764705883, 0.07205882352941176, 0.1364562118126273, 0.016293279022403257, 0.7006109979633401, 0.14663951120162932]


####################################
当节点数为100时，且把10ms变为了30ms, :'/gpfs/home/音乐/data/data20/data20_43/'
在迭代82次后结束，AUC最大值为0.925353,在阈值为0.150000下取得.
在训练集上的结果:
 train_Res: [206, 21, 15, 13, 121, 15] train_Rate: [0.90748898, 0.092511, 0.09146341, 0.079268292, 0.73780487, 0.091463414]

在验证集上的结果:
 val_Res: [200, 26, 15, 16, 117, 16] val_Rate: [0.88495575221, 0.115044, 0.09146341, 0.097560975, 0.7134146, 0.0975609756097561]

在测试集上的结果:
 test_Res: [214, 13, 9, 10, 129, 15] test_Rate: [0.9427312, 0.0572687, 0.055214723, 0.06134969, 0.79141104, 0.09202453987730061]

在所有数据上的结果:
 all_Res: [620, 60, 39, 39, 367, 46] all_Rate: [0.91176470, 0.08823529, 0.079429735, 0.07942973, 0.747454, 0.09368635437881874]


####################################


#######################################BP神经的结果结束##############################################

##########################################接下来是LSTM中many-to-many模型的结果##################################
'/gpfs/home/音乐/data/data19/data19_1/'是在以下模型下取得的结果：
模型为:
    '''stateful=False的LSTM模型，many-to-many'''
    def build_model(self,optimizer='rmsprop',init='random_normal'):
        model = Sequential()
        model.add(LSTM(units=20,input_shape=(self.time_step,self.signal_kind),\
                       dropout=0.2,recurrent_dropout=0.2,return_sequences=True))
        model.add(LSTM(units=20,dropout=0.2,\
                       recurrent_dropout=0.2,return_sequences=True))
        model.add(TimeDistributed(Dense(units=1,activation='sigmoid')))

        model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
        print(model.summary())
        return model
标签为:
        if self.a[i,2]==1:
            one_shut_train[12,:] = 1/(1+np.exp(-(one_shut_train[13,:]-(disr_time-0.3))*20))
        elif self.a[i,2]==-1:
            one_shut_train[12,:] = 0
回调函数为:
        callback_list = [ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=10),\
                         EarlyStopping(monitor='val_loss',patience=50),\
                         ModelCheckpoint(filepath=filepath+'best.weight.h5',monitor='val_loss',verbose=1,save_best_only=True),\
                         TensorBoard(log_dir=filepath+'/my_logs',histogram_freq=1)]a
batch_size为:
        self.batch_size=128
        History = model.fit(x=X_train,y=y_train,epochs=self.epochs,\
                            batch_size=self.batch_size,validation_data=(X_val,y_val),verbose=2,\
                            callbacks=callback_list)
        result0 = model.predict(X_test,batch_size=32,verbose=1)
        result2 = model.predict(X,batch_size=32,verbose=1)
归一化为：
        self.min_max[:,0] = train_mat[:-2,:].min(axis=1)
        self.min_max[:,1] = train_mat[:-2,:].max(axis=1)
        #self.min_max = min_max_Ran
        train_mat[:-2,:] = minmax_scale(train_mat[:-2,:],axis=1)

        test_mat[:-2,:] = test_mat[:-2,:]-np.tile(self.min_max[:,0].reshape(-1,1),(1,test_mat.shape[1]))
        Range = (self.min_max[:,1]-self.min_max[:,0]).reshape(-1,1)
        test_mat[:-2,:] = test_mat[:-2,:]/np.tile(Range,(1,test_mat.shape[1]))

在迭代142次后结束

结果为:
AUC最大值为0.774182,在阈值为0.140000下取得.
result on train data:
 train_Res: [212, 15, 27, 6, 56, 75]
train_Rate: [0.9339207, 0.0660792, 0.164634146, 0.036585365, 0.3414634, 0.45731707]
result on val data:
 val_Res: [216, 10, 15, 8, 70, 71]
val_Rate: [0.955752, 0.0442477, 0.09146341, 0.0487804, 0.426829, 0.432926]
result on test data
 test_Res: [215, 12, 26, 2, 70, 65]
test_Rate: [0.94713, 0.05286, 0.1595092, 0.0122699, 0.429447, 0.398773]
result on all data:
 all_Res: [643, 37, 68, 16, 196, 211]
all_Rate: [0.945588, 0.054411, 0.13849, 0.0325865, 0.3991, 0.429735]

结果不好

#***************************************************************************************
'/gpfs/home/音乐/data/data19/data19_2/'是在以下模型下取得的结果：
模型为:
    '''stateful=False的LSTM模型，many-to-many'''
    def build_model(self,optimizer='rmsprop',init='random_normal'):
        model = Sequential()
        model.add(LSTM(units=20,input_shape=(self.time_step,self.signal_kind),\
                       dropout=0.2,recurrent_dropout=0.2,return_sequences=True))
        model.add(LSTM(units=20,dropout=0.2,\
                       recurrent_dropout=0.2,return_sequences=True))
        model.add(TimeDistributed(Dense(units=1,activation='sigmoid')))

        model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
        print(model.summary())
        return model
标签为:
        t1 = dfsdev[0,:][np.where(dfsdev[1,:]==dfsdev[1,:].min())][0]
        t2 = dfsdev[0,:][np.where(dfsdev[1,:]==dfsdev[1,:].max())][0]
        if self.a[i,2]==1:
            one_shut_train[12,:] = 1/(1+np.exp(-(one_shut_train[13,:]-(t1+t2)/2)*(10/(t2-t1))))
        elif self.a[i,2]==-1:
            one_shut_train[12,:] = 0

回调函数为:
        callback_list = [ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=10),\
                         EarlyStopping(monitor='val_loss',patience=50),\
                         ModelCheckpoint(filepath=filepath+'best.weight.h5',monitor='val_loss',verbose=1,save_best_only=True),\
                         TensorBoard(log_dir=filepath+'/my_logs',histogram_freq=1)]
batch_size为:
        self.batch_size=128
        History = model.fit(x=X_train,y=y_train,epochs=self.epochs,\
                            batch_size=self.batch_size,validation_data=(X_val,y_val),verbose=2,\
                            callbacks=callback_list)
        result0 = model.predict(X_test,batch_size=32,verbose=1)
        result2 = model.predict(X,batch_size=32,verbose=1)
归一化为：
        self.min_max[:,0] = train_mat[:-2,:].min(axis=1)
        self.min_max[:,1] = train_mat[:-2,:].max(axis=1)
        #self.min_max = min_max_Ran
        train_mat[:-2,:] = minmax_scale(train_mat[:-2,:],axis=1)

        test_mat[:-2,:] = test_mat[:-2,:]-np.tile(self.min_max[:,0].reshape(-1,1),(1,test_mat.shape[1]))
        Range = (self.min_max[:,1]-self.min_max[:,0]).reshape(-1,1)
        test_mat[:-2,:] = test_mat[:-2,:]/np.tile(Range,(1,test_mat.shape[1]))


'''
与data19-1不同的是标签由1/(1+np.exp(-(one_shut_train[13,:]-(disr_time-0.3))*20))变
为了1/(1+np.exp(-(one_shut_train[13,:]-(t1+t2)/2)*(10/(t2-t1))))
'''
在迭代113次后结束
结果为:
AUC最大值为0.763642,在阈值为0.200000下取得.
result on train data:
 train_Res: [215, 12, 31, 3, 60, 70]
train_Rate: [0.947136563876652, 0.05286343612334802, 0.18902439024390244, 0.018292682926829267, 0.36585365853658536, 0.4268292682926829]
result on val data:
 val_Res: [213, 13, 18, 4, 72, 70]
val_Rate: [0.9424778761061947, 0.05752212389380531, 0.10975609756097561, 0.024390243902439025, 0.43902439024390244, 0.4268292682926829]
result on test data:
 test_Res: [213, 14, 19, 2, 75, 67]
test_Rate: [0.9383259911894273, 0.06167400881057269, 0.1165644171779141, 0.012269938650306749, 0.4601226993865031, 0.4110429447852761]
result on all data:
 all_Res: [641, 39, 68, 9, 207, 207]
all_Rate: [0.9426470588235294, 0.057352941176470586, 0.1384928716904277, 0.018329938900203666, 0.4215885947046843, 0.4215885947046843]
结果不好
##########################################LSTM中many-to-many模型的结束##################################

##########################################接下来是LSTM中many-to-one模型的结果##################################
'/gpfs/home/音乐/data/data21/data21_1/'是在以下模型下取得的结果：
模型为:
    def build_model(self,optimizer='rmsprop',init='random_normal'):
        model = Sequential()
        model.add(LSTM(units=20,input_shape=(self.time_step,self.signal_kind),\
                       dropout=0.2,recurrent_dropout=0.2,return_sequences=True))
        model.add(LSTM(units=20,dropout=0.2,recurrent_dropout=0.2))

        model.add(Dense(units=1,activation='sigmoid'))
        model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
        print(model.summary())
        return model
标签为:
    def generate_data(self,data,shut_list,name):
        data1 = np.empty((data.shape[0],data.shape[1],data.shape[2]-1))
        data1[:,:,:-1] = data[:,:,:-2]
        data1[:,:,-1] = data[:,:,-1]
        data_X = []
        data_y = []
        for i,j in enumerate(shut_list):
            if self.a[j,2]==-1:
                y = np.zeros(self.num)
            else:
                a = data1[i][:,-1]
                b = data1[i][:,1]
                t1 = a[np.where(b==b.min())][0]
                t2 = a[np.where(b==b.max())][0]
                y = 1/(1+np.exp(-(data1[i][:,-1][-self.num:]-(t1+t2)/2)*(10/(t2-t1))))
            data_y.append(y)
            for k in range(len(data1[i])-self.time_step):
                x = data1[i][k:k+self.time_step,:-1]
                data_X.append(x)
        data_X = np.array(data_X)
        data_y = np.array(data_y).reshape(-1,1)
        #print('%s data_X shape is %s.'%(name,str(data_X.shape)))
        #print('%s data_y shape is %s.'%(name,str(data_y.shape)))
        return data_X,data_y

回调函数为:
        self.batch_size=128
        callback_list = [ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=5),\
                         EarlyStopping(monitor='val_loss',patience=20),\
                         ModelCheckpoint(filepath=filepath+'best.weight.h5',monitor='val_loss',verbose=1,save_best_only=True),\
                         TensorBoard(log_dir=filepath+'/my_logs',histogram_freq=1)]
归一化为：
        self.min_max[:,0] = train_mat[:-2,:].min(axis=1)
        self.min_max[:,1] = train_mat[:-2,:].max(axis=1)
        #self.min_max = min_max_Ran
        train_mat[:-2,:] = minmax_scale(train_mat[:-2,:],axis=1)

        test_mat[:-2,:] = test_mat[:-2,:]-np.tile(self.min_max[:,0].reshape(-1,1),(1,test_mat.shape[1]))
        Range = (self.min_max[:,1]-self.min_max[:,0]).reshape(-1,1)
        test_mat[:-2,:] = test_mat[:-2,:]/np.tile(Range,(1,test_mat.shape[1]))

结果为:
AUC最大值为0.805803,在阈值为0.210000下取得.
result on train data:
 train_Res: [223, 4, 29, 11, 64, 60]
train_Rate: [0.9823788, 0.01762114, 0.176829, 0.06707, 0.39024, 0.36585]
result on val data:
 val_Res: [224, 2, 27, 3, 76, 58]
val_Rate: [0.9911, 0.008849, 0.16463, 0.01829, 0.4634, 0.35365]
result on test data:
 test_Res: [221, 6, 23, 5, 76, 59]
test_Rate: [0.97356, 0.02643, 0.14110, 0.03067, 0.4662, 0.36196]
result on all data:
 all_Res: [668, 12, 79, 19, 216, 177]
all_Rate: [0.9823, 0.01764, 0.160896, 0.03869, 0.4399, 0.3604]

#**********************************************************************************************************
'/gpfs/home/音乐/data/data21/data21_2/'是在以下模型下取得的结果：
模型为:
    def build_model(self,optimizer='rmsprop',init='random_normal'):
        model = Sequential()
        model.add(LSTM(units=20,input_shape=(self.time_step,self.signal_kind),\
                       dropout=0.2,recurrent_dropout=0.2,return_sequences=True))
        model.add(LSTM(units=20,dropout=0.2,recurrent_dropout=0.2))

        model.add(Dense(units=1,activation='sigmoid'))
        model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
        print(model.summary())
        return model
标签为:
    def generate_data(self,data,shut_list,name):
        data1 = np.empty((data.shape[0],data.shape[1],data.shape[2]-1))
        data1[:,:,:-1] = data[:,:,:-2]
        data1[:,:,-1] = data[:,:,-1]
        data_X = []
        data_y = []
        for i,j in enumerate(shut_list):
            if self.a[j,2]==-1:
                y = np.zeros(self.num)
            else:
                y = 1/(1+np.exp(-(data1[i][:,-1][-self.num:]-(self.b[j,1]-0.3))*20))
            data_y.append(y)
            for k in range(len(data1[i])-self.time_step):
                x = data1[i][k:k+self.time_step,:-1]
                data_X.append(x)
        data_X = np.array(data_X)
        data_y = np.array(data_y).reshape(-1,1)
        #print('%s data_X shape is %s.'%(name,str(data_X.shape)))
        #print('%s data_y shape is %s.'%(name,str(data_y.shape)))
        return data_X,data_y
回调函数为:
        self.batch_size=128
        callback_list = [ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=5),\
                         EarlyStopping(monitor='val_loss',patience=20),\
                         ModelCheckpoint(filepath=filepath+'best.weight.h5',monitor='val_loss',verbose=1,save_best_only=True),\
                         TensorBoard(log_dir=filepath+'/my_logs',histogram_freq=1)]
归一化为：
        self.min_max[:,0] = train_mat[:-2,:].min(axis=1)
        self.min_max[:,1] = train_mat[:-2,:].max(axis=1)
        #self.min_max = min_max_Ran
        train_mat[:-2,:] = minmax_scale(train_mat[:-2,:],axis=1)

        test_mat[:-2,:] = test_mat[:-2,:]-np.tile(self.min_max[:,0].reshape(-1,1),(1,test_mat.shape[1]))
        Range = (self.min_max[:,1]-self.min_max[:,0]).reshape(-1,1)
        test_mat[:-2,:] = test_mat[:-2,:]/np.tile(Range,(1,test_mat.shape[1]))

'''
data21-1不同的是标签变了，由1/(1+np.exp(-(data1[i][:,-1][-self.num:]-(t1+t2)/2)*(10/(t2-t1))))
变为了1/(1+np.exp(-(data1[i][:,-1][-self.num:]-(self.b[j,1]-0.3))*20))

'''

结果为:
AUC最大值为0.859125,在阈值为0.490000下取得.
在29伦后结束
result on train data:
 train_Res: [205, 22, 31, 4, 97, 32]
train_Rate: [0.90308, 0.0969, 0.189024, 0.02439, 0.59146, 0.1951]
result on val data:
 val_Res: [202, 24, 22, 3, 108, 31]
val_Rate: [0.8938, 0.10619, 0.1341, 0.01829, 0.65853, 0.1890]
result on test data:
 test_Res: [209, 18, 13, 4, 113, 33]
test_Rate: [0.9207, 0.07929, 0.07975, 0.0245, 0.69325, 0.20245]
result on all data:
 all_Res: [616, 64, 66, 11, 318, 96]
all_Rate: [0.90588, 0.09411, 0.1344, 0.02240, 0.6476, 0.1955]
训练和测试结束于Sat Nov 10 15:27:07 2018.

#**********************************************************************************************************
'/gpfs/home/音乐/data/data21/data21_3/'是在以下模型下取得的结果：
模型为:
    def build_model(self,optimizer='rmsprop',init='random_normal'):
        model = Sequential()
        model.add(LSTM(units=100,input_shape=(self.time_step,self.signal_kind),\
                       dropout=0.2,recurrent_dropout=0.2,return_sequences=True))
        model.add(LSTM(units=100,dropout=0.2,recurrent_dropout=0.2))

        model.add(Dense(units=1,activation='sigmoid'))
        model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
        print(model.summary())
        return model
标签为:
    def generate_data(self,data,shut_list,name):
        data1 = np.empty((data.shape[0],data.shape[1],data.shape[2]-1))
        data1[:,:,:-1] = data[:,:,:-2]
        data1[:,:,-1] = data[:,:,-1]
        data_X = []
        data_y = []
        for i,j in enumerate(shut_list):
            if self.a[j,2]==-1:
                y = np.zeros(self.num)
            else:
                y = 1/(1+np.exp(-(data1[i][:,-1][-self.num:]-(self.b[j,1]-0.3))*20))
            data_y.append(y)
            for k in range(len(data1[i])-self.time_step):
                x = data1[i][k:k+self.time_step,:-1]
                data_X.append(x)
        data_X = np.array(data_X)
        data_y = np.array(data_y).reshape(-1,1)
        #print('%s data_X shape is %s.'%(name,str(data_X.shape)))
        #print('%s data_y shape is %s.'%(name,str(data_y.shape)))
        return data_X,data_y
回调函数为:
        self.batch_size=128
        callback_list = [ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=5),\
                         EarlyStopping(monitor='val_loss',patience=20),\
                         ModelCheckpoint(filepath=filepath+'best.weight.h5',monitor='val_loss',verbose=1,save_best_only=True),\
                         TensorBoard(log_dir=filepath+'/my_logs',histogram_freq=1)]
归一化为：
        self.min_max[:,0] = train_mat[:-2,:].min(axis=1)
        self.min_max[:,1] = train_mat[:-2,:].max(axis=1)
        #self.min_max = min_max_Ran
        train_mat[:-2,:] = minmax_scale(train_mat[:-2,:],axis=1)

        test_mat[:-2,:] = test_mat[:-2,:]-np.tile(self.min_max[:,0].reshape(-1,1),(1,test_mat.shape[1]))
        Range = (self.min_max[:,1]-self.min_max[:,0]).reshape(-1,1)
        test_mat[:-2,:] = test_mat[:-2,:]/np.tile(Range,(1,test_mat.shape[1]))
结果为:

迭代33轮后结束
AUC最大值为0.902070,在阈值为0.320000下取得.
在训练集上的结果:
train_Res: [219, 8, 21, 3, 122, 18]
train_Rate: [0.9647577092511013, 0.03524229074889868, 0.12804878048780488, 0.018292682926829267, 0.7439024390243902, 0.10975609756097561]
在验证集上的结果:
val_Res: [205, 21, 23, 0, 124, 17]
val_Rate: [0.9070796460176991, 0.09292035398230089, 0.1402439024390244, 0.0, 0.7560975609756098, 0.10365853658536585]
在测试集上的结果:
 test_Res: [209, 18, 17, 4, 123, 19]
 test_Rate: [0.920704845814978, 0.07929515418502203, 0.10429447852760736, 0.024539877300613498, 0.754601226993865, 0.1165644171779141]
在所有数据上的结果:
 all_Res: [633, 47, 61, 7, 369, 54]
 all_Rate: [0.9308823529411765, 0.06911764705882353, 0.12423625254582485, 0.014256619144602852, 0.7515274949083504, 0.109979633401222]

######################################################################
'/gpfs/home/音乐/data/data21/data21_4/'是在以下模型下取得的结果：
模型为:
    def build_model(self,optimizer='rmsprop',init='random_normal'):
        model = Sequential()
        model.add(LSTM(units=100,input_shape=(self.time_step,self.signal_kind),\
                       dropout=0.2,recurrent_dropout=0.2,return_sequences=True))
        model.add(LSTM(units=100,dropout=0.2,recurrent_dropout=0.2))

        model.add(Dense(units=1,activation='sigmoid'))
        model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
        print(model.summary())
        return model
标签为:
    def generate_data(self,data,shut_list,name):
        data1 = np.empty((data.shape[0],data.shape[1],data.shape[2]-1))
        data1[:,:,:-1] = data[:,:,:-2]
        data1[:,:,-1] = data[:,:,-1]
        data_X = []
        data_y = []
        for i,j in enumerate(shut_list):
            if self.a[j,2]==-1:
                y = np.zeros(self.num)
            else:
                a = data1[i][:,-1]
                b = data1[i][:,1]
                t1 = a[np.where(b==b.min())][0]
                t2 = a[np.where(b==b.max())][0]
                y = 1/(1+np.exp(-(data1[i][:,-1][-self.num:]-(t1+t2)/2)*(10/(t2-t1))))
            data_y.append(y)
            for k in range(len(data1[i])-self.time_step):
                x = data1[i][k:k+self.time_step,:-1]
                data_X.append(x)
        data_X = np.array(data_X)
        data_y = np.array(data_y).reshape(-1,1)
        #print('%s data_X shape is %s.'%(name,str(data_X.shape)))
        #print('%s data_y shape is %s.'%(name,str(data_y.shape)))
        return data_X,data_y

回调函数为:
        self.batch_size=128
        callback_list = [ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=5),\
                         EarlyStopping(monitor='val_loss',patience=20),\
                         ModelCheckpoint(filepath=filepath+'best.weight.h5',monitor='val_loss',verbose=1,save_best_only=True),\
                         TensorBoard(log_dir=filepath+'/my_logs',histogram_freq=1)]
归一化为：
        self.min_max[:,0] = train_mat[:-2,:].min(axis=1)
        self.min_max[:,1] = train_mat[:-2,:].max(axis=1)
        #self.min_max = min_max_Ran
        train_mat[:-2,:] = minmax_scale(train_mat[:-2,:],axis=1)

        test_mat[:-2,:] = test_mat[:-2,:]-np.tile(self.min_max[:,0].reshape(-1,1),(1,test_mat.shape[1]))
        Range = (self.min_max[:,1]-self.min_max[:,0]).reshape(-1,1)
        test_mat[:-2,:] = test_mat[:-2,:]/np.tile(Range,(1,test_mat.shape[1]))

结果为:
迭代26轮后结束，AUC最大值为0.823262,在阈值为0.550000下取得.
在训练集上的结果:
 train_Res: [218, 9, 24, 6, 85, 49]  train_Rate: [0.96035242, 0.039647, 0.14634146, 0.03658536, 0.518292682, 0.29878048]
在验证集上的结果:
 val_Res: [213, 13, 14, 5, 91, 54]   val_Rate: [0.942477876, 0.0575221, 0.085365853, 0.0304878, 0.55487804, 0.3292682]
在测试集上的结果:
 test_Res: [215, 12, 15, 4, 95, 49]   test_Rate: [0.9471365, 0.0528634, 0.092024539, 0.0245398, 0.58282208, 0.3006134]
在所有数据上的结果:
 all_Res: [646, 34, 53, 15, 271, 152]  all_Rate: [0.95, 0.05, 0.1079429, 0.03054989, 0.5519348, 0.3095723014256619]

'/gpfs/home/音乐/data/data21/data21_5/'

AUC最大值为0.852599,在阈值为0.370000下取得.
在训练集上的结果:
 train_Res: [208, 19, 21, 8, 104, 31]
train_Rate: [0.9162995594713657, 0.08370044052863436, 0.12804878048780488, 0.04878048780487805, 0.6341463414634146, 0.18902439024390244]
在验证集上的结果:
 val_Res: [207, 19, 21, 4, 104, 35]
val_Rate: [0.915929203539823, 0.084070796460177, 0.12804878048780488, 0.024390243902439025, 0.6341463414634146, 0.21341463414634146]
在测试集上的结果:
 test_Res: [213, 14, 17, 2, 106, 38]
test_Rate: [0.9383259911894273, 0.06167400881057269, 0.10429447852760736, 0.012269938650306749, 0.6503067484662577, 0.2331288343558282]
在所有数据集上的结果:
 all_Res: [628, 52, 59, 14, 314, 104]
all_Rate: [0.9235294117647059, 0.07647058823529412, 0.12016293279022404, 0.028513238289205704, 0.639511201629328, 0.21181262729124237]

'/gpfs/home/音乐/data/data21/data21_6/'

AUC最大值为0.857085,在阈值为0.560000下取得.
在训练集上的结果:
 train_Res: [222, 5, 36, 7, 81, 40]
train_Rate: [0.9779735682819384, 0.022026431718061675, 0.21951219512195122, 0.042682926829268296, 0.49390243902439024, 0.24390243902439024]
在验证集上的结果:
 val_Res: [220, 6, 26, 7, 84, 47]
val_Rate: [0.9734513274336283, 0.02654867256637168, 0.15853658536585366, 0.042682926829268296, 0.5121951219512195, 0.2865853658536585]
在测试集上的结果:
 test_Res: [222, 5, 23, 7, 90, 43]
test_Rate: [0.9779735682819384, 0.022026431718061675, 0.1411042944785276, 0.04294478527607362, 0.5521472392638037, 0.26380368098159507]
在所有数据集上的结果:
 all_Res: [664, 16, 85, 21, 255, 130]
all_Rate: [0.9764705882352941, 0.023529411764705882, 0.17311608961303462, 0.04276985743380855, 0.5193482688391039, 0.26476578411405294]

也是以上模型的结果

'/gpfs/home/音乐/data/data21/data21_7/'
        callback_list = [ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=10),\
                         EarlyStopping(monitor='val_loss',patience=50),\
                         ModelCheckpoint(filepath=filepath+'best.weight.h5',monitor='val_loss',verbose=1,save_best_only=True),\
                         TensorBoard(log_dir=filepath+'/my_logs',histogram_freq=1)]

在迭代55次后结束,AUC最大值为0.831829,在阈值为0.600000下取得.
在训练集上的结果:
 train_Res: [199, 28, 36, 4, 91, 33]
train_Rate: [0.8766519823788547, 0.12334801762114538, 0.21951219512195122, 0.024390243902439025, 0.5548780487804879, 0.20121951219512196]
在验证集上的结果:
 val_Res: [197, 29, 15, 7, 110, 32]
val_Rate: [0.8716814159292036, 0.12831858407079647, 0.09146341463414634, 0.042682926829268296, 0.6707317073170732, 0.1951219512195122]
在测试集上的结果:
 test_Res: [198, 29, 19, 7, 103, 34]
test_Rate: [0.8722466960352423, 0.1277533039647577, 0.1165644171779141, 0.04294478527607362, 0.6319018404907976, 0.2085889570552147]
在所有数据集上的结果:
 all_Res: [594, 86, 70, 18, 304, 99]
all_Rate: [0.8735294117647059, 0.1264705882352941, 0.1425661914460285, 0.03665987780040733, 0.6191446028513238, 0.20162932790224034]

#***************************************LSTM的many-to-one结束******************************************

#***********************************************************************************************
'/gpfs/home/音乐/data/data22/data22_1/'是在以下模型下取得的结果：
模型为:

    #建立简单的全连接BP神经网络
    def build_model(self,optimizer='rmsprop'):
        model = Sequential()
        model.add(Dropout(rate=0.2,input_shape=(self.signal_kind,)))
        model.add(Dense(units=20,activation='relu',input_shape=(self.signal_kind,)))
        model.add(Dropout(rate=0.2))
        model.add(Dense(units=20,activation='relu'))
        model.add(Dropout(rate=0.2))
        model.add(Dense(1,activation='sigmoid'))
        model.compile(optimizer='adam',loss='mse',metrics=['mae'])
        print(model.summary())
        return model
标签为:
        if self.a[i,2]==1:
            one_shut_train[12,:] = 1/(1+np.exp(-(one_shut_train[13,:]-(t1+t2)/2)*(10/(t2-t1))))
        elif self.a[i,2]==-1:
            one_shut_train[12,:] = 0
回掉函数为：
        self.batch_size=128
        callback_list = [ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=10),\
                         EarlyStopping(monitor='val_loss',patience=50),\
                         ModelCheckpoint(filepath=filepath+'best.weight.h5',monitor='val_loss',verbose=1,save_best_only=True),\
                         TensorBoard(log_dir=filepath2+'/my_logs/',histogram_freq=1)]
没有归一化

结果为：
在迭代51次后结束
AUC最大值为0.500000,在阈值为0.000000下取得.
在训练集上的结果:
 train_Res: [0, 227, 0, 0, 164, 0] train_Rate: [0.0, 1.0, 0.0, 0.0, 1.0, 0.0]

在验证集上的结果:
 val_Res: [0, 226, 0, 0, 164, 0] val_Rate: [0.0, 1.0, 0.0, 0.0, 1.0, 0.0]

在测试集上的结果:
 test_Res: [0, 227, 0, 0, 163, 0] test_Rate: [0.0, 1.0, 0.0, 0.0, 1.0, 0.0]

在所有数据上的结果:
 all_Res: [0, 680, 0, 0, 491, 0] all_Rate: [0.0, 1.0, 0.0, 0.0, 1.0, 0.0]

#*************************************************************************************
'/gpfs/home/音乐/data/data22/data22_2/'是在以下模型下取得的结果：
模型为:

    #建立简单的全连接BP神经网络
    def build_model(self,optimizer='rmsprop'):
        model = Sequential()
        model.add(Dropout(rate=0.2,input_shape=(self.signal_kind,)))
        model.add(Dense(units=20,activation='relu',input_shape=(self.signal_kind,)))
        model.add(Dropout(rate=0.2))
        model.add(Dense(units=20,activation='relu'))
        model.add(Dropout(rate=0.2))
        model.add(Dense(1,activation='sigmoid'))
        model.compile(optimizer='adam',loss='mse',metrics=['mae'])
        print(model.summary())
        return model
标签为:
        if self.a[i,2]==1:
            one_shut_train[12,:] = 1/(1+np.exp(-(one_shut_train[13,:]-(t1+t2)/2)*(10/(t2-t1))))
        elif self.a[i,2]==-1:
            one_shut_train[12,:] = 0
回掉函数为：
        self.batch_size=128
        callback_list = [ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=10),\
                         EarlyStopping(monitor='val_loss',patience=50),\
                         ModelCheckpoint(filepath=filepath+'best.weight.h5',monitor='val_loss',verbose=1,save_best_only=True),\
                         TensorBoard(log_dir=filepath2+'/my_logs/',histogram_freq=1)]
每炮内归一化，不是所有测试集和训练集一起归一化，也不是不归一化，。

在迭代52次后结束
AUC最大值为0.782587,在阈值为0.440000下取得.
在训练集上的结果:
 train_Res: [191, 36, 52, 14, 55, 43] train_Rate: [0.8414096, 0.15859, 0.31707, 0.085365, 0.335365, 0.26219]

在验证集上的结果:
 val_Res: [186, 40, 57, 17, 43, 47] val_Rate: [0.82300, 0.17699, 0.34756, 0.103658, 0.26219, 0.28658]

在测试集上的结果:
 test_Res: [184, 43, 44, 20, 59, 40] test_Rate: [0.81057, 0.18942, 0.269938, 0.122699, 0.36196, 0.24539]

在所有数据上的结果:
 all_Res: [561, 119, 153, 51, 157, 130] all_Rate: [0.825, 0.175, 0.31160, 0.10386, 0.31975, 0.26476]

初步看来，不归一化和每炮分别自己内部归一化的效果都不如训练集集体归一化和训练集集体归一化的效果

