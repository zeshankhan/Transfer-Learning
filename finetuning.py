# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 15:39:46 2022

@author: ZESHAN KAHN
"""

def finetune(base_model=None,output_layer='avg_pool',weights='imagenet',weights_new="abc.h5",nb_epoch=1,data_path=None,num_classes=16):
    paths,_,Ys=gather_paths_all(data_path,num_classes=num_classes)
    data_size=len(paths)
    X=gather_images_from_paths(paths,start=0,count=data_size)
    print(X.shape,Ys.shape)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Ys, test_size=0.33, random_state=5)
    
    #model=features_layer(base_model,output_layer,num_classes)
    model=alter_last_layer(base_model,output_layer,num_classes)
    
    for l in model.layers[5:-5]:
        l.trainable = False
    
    model_checkpoint_path = "/kaggle/working/"+data_name+"_"+model_name+"_"+str(nb_epoch)+"_updatingLR_best.h5"
    c1 = ModelCheckpoint(model_checkpoint_path,save_best_only=True,monitor='loss', factor=0.1, patience=5, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=1e-7)
    c2 = ReduceLRBacktrack(best_path=model_checkpoint_path, monitor='loss', factor=0.1, patience=5, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=1e-7)
    model.fit(X_train, Y_train,batch_size=batch_size,epochs=nb_epoch,shuffle=True,verbose=0,validation_data=(X_test, Y_test),callbacks=[c1,c2])
    model.save(weights_new)
    return model

def pred_mid(model,layer,X):
    x=model.layers[layer].output
    model1 = Model(model.input, outputs=x)
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    adag=Adagrad(learning_rate=0.01,initial_accumulator_value=0.1,epsilon=1e-07,name="Adagrad")
    model1.compile(loss='mean_squared_error', optimizer=adag)
    return model1.predict(X)