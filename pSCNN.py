from tensorflow.keras import Input, layers, models, optimizers, callbacks
import matplotlib.pyplot as plt
import pickle, os
from sklearn.metrics import confusion_matrix  


def create_input_layers(xshapes):
    inputs = []
    for xshape in xshapes:
        input_shape_x = (xshape[1], 1)
        input_x = Input(shape = input_shape_x)
    return inputs


def create_convolution_layers(inputs, num_layers = 0):
    convs = []
    for input_x in inputs:
        conv      = layers.Conv1D(32,  5,  activation='relu', input_shape=input_x.get_shape())(input_x)
        conv      = layers.MaxPooling1D(strides=2, padding='valid')(conv)
        #conv      = layers.Dropout(0.25)(conv)
        for i in range(num_layers):
            conv      = layers.Conv1D(32,  5,  activation='relu')(conv)
            conv      = layers.MaxPooling1D(strides=2, padding='valid')(conv)
            #conv      = layers.Dropout(0.25)(conv)
        convs.append(conv)
    return convs



def pSCNN(xshapes, num_conv_layers):
    inputs = create_input_layers(xshapes)
    convs = create_convolution_layers(inputs, num_layers = num_conv_layers)
    if len(convs) >= 2:
        conv_merge = layers.concatenate(convs)
    else:
        conv_merge = convs[0]
    flat      = layers.Flatten()(conv_merge)
    dense     = layers.Dense(100,  activation='relu')(flat)
    dense     = layers.Dropout(0.2)(dense)
    output    = layers.Dense(1, activation='sigmoid')(dense)
    model     = models.Model(inputs= inputs, outputs=output)
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model


def train_pSCNN(model, Xs, y, batch, epochs):
    Xs3d = [X.reshape((X.shape[0], X.shape[1], 1)) for X in Xs]
    model.fit(Xs3d, y, batch_size=batch, epochs=epochs, validation_split = 0.1)
    


def save_pSCNN(model, model_name):
    model_path = f'{model_name}.h5'
    history_path = f'{model_name}.pkl'
    model.save(model_path)
    pickle.dump(model.history.history, open(history_path, "wb" ))


def load_pSCNN(model_name):
    model_path = f'{model_name}.h5'
    history_path = f'{model_name}.pkl'
    model = models.load_model(model_path) 
    history = pickle.load(open(history_path, "rb" ))
    model.history = callbacks.History()
    model.history.history = history
    return model 


def predict_pSCNN(model, Xs):
    Xs3d = [X.reshape((X.shape[0], X.shape[1], 1)) for X in Xs]
    return model.predict(Xs3d)


def evaluate_pSCNN(model, Xs, y):
    Xs3d = [X.reshape((X.shape[0], X.shape[1], 1)) for X in Xs]
    return model.evaluate(Xs3d, y) 


if __name__=="__main__":
    from readBruker import read_bruker_h, read_bruker_hs, plot_h
    from augment import data_augmentation
    import numpy as np
    import pandas as pd
 
    
    model_save_path = 'D:/workspace/DeepNMR_code/models/n10000_epoch100_6layers_0.0001_pc5/'
    model_name = 'test_nmr'
    bTrain = False
    if bTrain:
        stds = read_bruker_hs('D:/workspace/DeepNMR/data/standard', False, True, False)
        aug = data_augmentation(stds, 10000, 5)
        model = pSCNN([aug['R'].shape, aug['S'].shape], 6)
        train_pSCNN(model, [aug['R'], aug['S']], aug['y'], 50, 100)
        save_pSCNN(model, model_save_path + model_name)
    else:
        model = load_pSCNN(model_save_path + model_name)
        plot_loss_accuracy(model)
        
        
        query = read_bruker_h('D:/workspace/DeepNMR/data/known/F1', False, True) #2
 
        stds = read_bruker_hs('D:/workspace/DeepNMR/data/standard', False, True, False)


         p = query['ppm'].shape[0]
         n = len(stds)
         R = np.zeros((n, p), dtype = np.float32)
         Q = np.zeros((n, p), dtype = np.float32)
         for i in range(n):
             R[i, ] = stds[i]['fid']
             Q[i, ] = query['fid']
        yp = predict_MICNN(model, [R, Q])
         

        a = ['Linalyl acetate', 'β-Ionone']
        ids = []
        for i in a:
            for j, s in enumerate(stds):
                if i == s['name']:
                    ids.append(j)
               #print(j)
        print(ids)
         
        y_real = np.zeros((n,1), dtype = np.float32)
        for i in ids:
            y_real[i, 0] = 1
        print(y_real)
         
        ev = evaluate_MICNN(model, [R, Q], y_real)
        yp_real = predict_MICNN(model, [R, Q])
        yp_real_list = [1 if yp_real[i, 0]>=0.5 else 0 for i in range(yp_real.shape[0])]
        yp_real = np.array(yp_real_list).reshape(yp_real.shape)
        cnf_matrix = confusion_matrix(y_real, yp_real)
        
        stds_df = pd.read_csv('D:/workspace/DeepNMR_code/data/standards.csv')
        result_df = pd.DataFrame(columns=['Name','Probability'])
        for i in range(n):
            result_df.loc[len(result_df)] = [stds[i]['name'], yp[i][0]]
        result = pd.merge(stds_df, result_df, on=['Name'])    
        
        