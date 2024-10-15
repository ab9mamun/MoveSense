# -*- coding: utf-8 -*-
import os.path

# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import pickle
import scipy as sp
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.layers import Dense, Conv1D, Flatten, LSTM, concatenate
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import Input

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


def rsquared(X, Y):  # arrays
    try:
        slope, intercept, r_value, p_value, std_err = sp.stats.linregress(X, Y)
        return r_value ** 2
    except:
        return 0


def getmae(Ytrue, Ypred):
    return np.mean(np.abs(Ytrue - Ypred))

def create_frames_for_one_set(mode, df, users, window):
    X = []
    Y = []
    userindices = []
    if mode == 'multimodal':
        feature_start = 3
        feature_end = 68
    elif mode == 'engagement':
        feature_start = 3
        feature_end = 60
    elif mode == 'activity':
        feature_start = 60
        feature_end = 68

    else:
        print('ERROR: Invalid feature set')
        return None, None, None

    hop = max(1, window // 4)
    for user in users:
        fildf = df[df.participant_id == user]
        n = fildf.shape[0]
        start = 0
        while start + window < n:
            x = fildf.values[start:start + window, feature_start:feature_end]
            y = fildf.values[start + window, 60:68]

            X.append(x)
            Y.append(y)
            userindices.append(user)
            start += hop

    return X, Y, userindices


def create_frames(mode, df, all_users, window_size, engagement_percentile, activity_percentile):
    avg_steps = []
    avg_times_opened = []
    for user in all_users:
        fildf = df[df.participant_id == user]
        avg_times_opened.append(fildf.times_opened.to_numpy().mean())
        avg_steps.append(fildf.steps.to_numpy().mean())

    num_users = len(all_users)
    avg_times_opened.sort()
    avg_steps.sort()
    engagement_threshold = avg_times_opened[int(engagement_percentile * num_users)]
    activity_threshold = avg_steps[int(activity_percentile * num_users)]

    surviving_users = []
    for user in all_users:
        fildf = df[df.participant_id == user]
        avg_activity = fildf.steps.to_numpy().mean()
        avg_engagement = fildf.times_opened.to_numpy().mean()
        if avg_activity < activity_threshold or avg_engagement < engagement_threshold:
            continue
        surviving_users.append(user)

    df = df[df.participant_id.isin(surviving_users)]
    surviving_users2 = []
    for user in surviving_users:
        fildf = df[df.participant_id == user]
        if fildf.shape[0] < window_size +1:
            continue
        surviving_users2.append(user)
    df = df[df.participant_id.isin(surviving_users2)]

    num_users = len(surviving_users2)
    superusers = shuffle(surviving_users2, random_state=0)
    #print('Superusers:', superusers)

    # split the data into train, validation and test based on the user
    pivot = int(0.8 * num_users)
    train_users = superusers[:pivot]
    test_users = superusers[pivot:]

    traindf = df[df.participant_id.isin(train_users)]
    testdf = df[df.participant_id.isin(test_users)]

    Xtrain, Ytrain, userindices_train = create_frames_for_one_set(mode, traindf, train_users, window_size)
    Xtest, Ytest, userindices_test = create_frames_for_one_set(mode, testdf, test_users, window_size)


    datadict = {'Xtrain': np.array(Xtrain).astype(float),
                'Ytrain': np.array(Ytrain).astype(float),
                'Xtest': np.array(Xtest).astype(float),
                'Ytest': np.array(Ytest).astype(float),
                'trainuserindices': np.array(userindices_train).astype(int),
                'testuserindices': np.array(userindices_test).astype(int),
                'superusers': np.array(superusers).astype(int)
                }
    return datadict


def getf_score(mat):
    if len(mat) == 1:
        return 1, 1, 1
    tn = mat[0][0]
    tp = mat[1][1]
    fp = mat[0][1]
    fn = mat[1][0]

    if tp == 0:
        return 0, 0, 0
    rec = tp / (tp + fn)
    prec = tp / (tp + fp)
    f = 2 * rec * prec / (rec + prec)

    return rec, prec, f


def test_model(model, arch,  Xtest, Ytest, testuserindices, superusers, label, label_index, phase):

    if arch != 'arima':

        if arch == 'lstm_late':

            loss = model.evaluate(split_input_for_late(Xtest), Ytest[:, label_index], verbose=0)
        else:
            loss = model.evaluate(Xtest, Ytest[:, label_index], verbose=0)

        print(phase, "Loss: ", loss)
        if arch == 'lstm_late':
            Ypred = model.predict(split_input_for_late(Xtest), verbose=0)
        else:
            Ypred = model.predict(Xtest, verbose=0)
        mae = getmae(Ytest[:, label_index], Ypred[:, 0])
        rmse = np.sqrt(np.mean((Ytest[:, label_index] - Ypred[:, 0]) ** 2))
        nrmse = rmse / np.mean(Ytest[:, label_index])
        r2 = rsquared(Ytest[:, label_index], Ypred[:, 0])
    else:
        Ypred = model.forecast(steps=len(Xtest))
        mae = getmae(Ytest[:, label_index], Ypred)
        rmse = np.sqrt(np.mean((Ytest[:, label_index] - Ypred) ** 2))
        nrmse = rmse / np.mean(Ytest[:, label_index])
        r2 = rsquared(Ytest[:, label_index], Ypred)

    mae_peruser = []
    nrmse_peruser = []
    user_peruser = []
    testusers = np.unique(testuserindices)
    for user in testusers:
        test_indices = np.where(testuserindices == user)[0]
        if len(test_indices) < 1:
            continue
        # print(test_indices)
        # print(len(Xtest))
        Xtest_peruser = Xtest[test_indices]
        Ytest_peruser = Ytest[test_indices]


        if arch != 'arima':
            if arch == 'lstm_late':
                Ypred_peruser = model.predict(split_input_for_late(Xtest_peruser), verbose=0)
            else:
                Ypred_peruser = model.predict(Xtest_peruser, verbose=0)

            mae_temp = getmae(Ytest_peruser[:, label_index], Ypred_peruser[:, 0])
            nrmse_temp = np.sqrt(np.mean((Ytest_peruser[:, label_index] - Ypred_peruser[:, 0]) ** 2)) / np.mean(
                Ytest_peruser[:, label_index])
        else:
            Ypred_peruser = model.forecast(steps=len(Xtest_peruser))
            mae_temp = getmae(Ytest_peruser[:, label_index], Ypred_peruser)
            nrmse_temp = np.sqrt(np.mean((Ytest_peruser[:, label_index] - Ypred_peruser) ** 2)) / np.mean(
                Ytest_peruser[:, label_index])


        mae_peruser.append(mae_temp)
        nrmse_peruser.append(nrmse_temp)
        user_peruser.append(user)

    return mae, rmse, nrmse, r2, user_peruser, mae_peruser, nrmse_peruser


def build_model(arch, window_size, featuresize):
    if arch == 'cnn':
        model = Sequential()
        model.add(Conv1D(16, 8, input_shape=(window_size, featuresize), activation='relu'))
        model.add(Conv1D(16, 8, activation='relu'))
        model.add(Flatten())
        model.add(Dense(28, activation='relu'))
        model.add(Dense(1, activation='relu'))

    elif arch == 'lstm':
        model = Sequential()
        model.add(LSTM(64, input_shape=(window_size, featuresize)))  # , return_sequences=True))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='relu'))


    elif arch == 'lin_reg':
        model = Sequential()
        model.add(Dense(1, input_shape=(window_size, featuresize), activation='relu'))

    elif arch == 'lstm_late':
        # Define the input shapes
        input_1 = Input(shape=(window_size, 57))  # First input channel
        input_2 = Input(shape=(window_size, 8))  # Second input channel

        # LSTM and Dense layers for the first input channel
        lstm_1 = LSTM(64, return_sequences=False)(input_1)
        dense_1_1 = Dense(32, activation='relu')(lstm_1)
        dense_1_2 = Dense(16, activation='relu')(dense_1_1)

        # LSTM and Dense layers for the second input channel
        lstm_2 = LSTM(32, return_sequences=False)(input_2)
        dense_2_1 = Dense(16, activation='relu')(lstm_2)
        dense_2_2 = Dense(8, activation='relu')(dense_2_1)

        # Concatenate the two Dense outputs
        concatenated = concatenate([dense_1_2, dense_2_2])

        # Output layer (final dense layer)
        output = Dense(1, activation='relu')(concatenated)

        # Create the model
        model = Model(inputs=[input_1, input_2], outputs=output)


    else:
        print('ERROR: Invalid architecture')
        return None

    return model

def split_input_for_late(X):
    X1 = X[:, :, :57]
    X2 = X[:, :, 57:]
    return [X1, X2]


def train_with_label(arch, Xtrain, Ytrain, Xtest, Ytest, trainuserindices, testuserindices, superusers, label,
                     label_index, window_size, featureset, epochs):
    #print("Now forecasting: " + label)
    if featureset == 'multimodal':
        featuresize = 65
    elif featureset == 'activity':
        featuresize = 8
    elif featureset == 'engagement':
        featuresize = 57
    else:
        featuresize = 0


    if arch == 'arima':
        # Define ARIMA model parameters
        order = (1, 1, 1)  # ARIMA(p, d, q)

        # Fit ARIMA model
        model = ARIMA(Ytrain[:,label_index], order=order)
        print('Now training.')
        arima_result = model.fit()
        print('Now testing.')
        mae, rmse, nrmse, r2, user_peruser, mae_peruser, r2_peruser = test_model(arima_result, arch, Xtest, Ytest, testuserindices, superusers, label, label_index, 'Test')

    if arch != 'arima':
        best_model = None
        best_loss = None
        model = build_model(arch, window_size, featuresize)
        model.compile(loss='mse', optimizer='adam')
        print('Now training.')

        trainusers = np.unique(trainuserindices)
        #print(trainusers)
        if len(trainusers) < 8:
            pivots = [i for i in range(1, len(trainusers))]
        else:
            pivot_step = max(len(trainusers) / 8.0, 1)
            pivots = [round(pivot_step*i) for i in range(1, 8)]

        #print('Pivots:', pivots)
        for i in range(len(pivots)-1):
            val_users = trainusers[int(pivots[i]):int(pivots[i+1])]
            #print('val users:', val_users)
            #print('trainuserindices', trainuserindices)
            val_indices = np.array([]).astype(int)
            train_indices = np.arange(len(trainuserindices)).astype(int)
            for  user in val_users:
                val_temp = np.where(trainuserindices == user)[0]
                val_indices = np.union1d(val_indices, val_temp)
                train_temp = np.where(trainuserindices != user)[0]
                train_indices = np.intersect1d(train_indices, train_temp)

            Xtrain_kfold = Xtrain[train_indices]
            Ytrain_kfold = Ytrain[train_indices]
            Xval_kfold = Xtrain[val_indices]
            Yval_kfold = Ytrain[val_indices]

            if arch == 'lstm_late':
                Xtrain_kfold = split_input_for_late(Xtrain_kfold)
                Xval_kfold = split_input_for_late(Xval_kfold)

            # print('Ytrain_kfold shape:', Ytrain_kfold.shape)
            # print('Yval_kfold shape:', Yval_kfold.shape)
            # print('Xtrain_kfold shape:', Xtrain_kfold.shape)
            # print('Xval_kfold shape:', Xval_kfold.shape)

            history = model.fit(Xtrain_kfold, Ytrain_kfold[:, label_index],
                                validation_data=(Xval_kfold, Yval_kfold[:, label_index]), epochs=epochs, verbose=0)
            loss = history.history['val_loss'][-1]
            if best_loss is None or loss < best_loss:
                best_loss = loss
                best_model = model

        print('Now testing')
        mae, rmse, nrmse, r2, user_peruser, mae_peruser, r2_peruser = test_model(best_model, arch, Xtest, Ytest, testuserindices, superusers, label,
                                                                    label_index, 'Test')

    return mae, rmse, nrmse, r2, user_peruser, mae_peruser, r2_peruser

def plot_learning_curve(history, label, epochs):
    plt.figure()
    x = list(range(1, epochs + 1))
    print(history.history.keys())
    plt.plot(x, history.history['loss'])
    plt.plot(x, history.history['val_loss'])
    plt.legend(['training', 'validation'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs Epochs learning curve for' + label)
    plt.savefig('Loss vs Epochs for '+label+'.png')


def plot_accuracy_curve(history, label, epochs):
    plt.figure()
    x = list(range(1, epochs + 1))
    plt.plot(x, history.history['accuracy'])
    plt.plot(x, history.history['val_accuracy'])
    plt.legend(['training', 'validation'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Accuracy vs Epochs learning curve for' + label)
    plt.savefig('Accuracy vs Epochs for '+label+'.png')


def experiment(featureset, df, all_users, arch, outcome, num_epochs, window_size, engagement_percentile, activity_percentile):
    all_labels = ['Steps', 'Sed minutes', 'LPA mintues', 'MVPA mintues', 'Wear time', 'Sed ratio', 'LPA ratio',
                  'MVPA ratio']
    print(f"============= Outcome: {outcome}, Arch: {arch}, Featureset:", featureset, "======================")
    datadict_combined = create_frames(featureset, df, all_users, window_size, engagement_percentile, activity_percentile)

    Xtrain = datadict_combined['Xtrain']
    Ytrain = datadict_combined['Ytrain']
    Xtest = datadict_combined['Xtest']
    Ytest = datadict_combined['Ytest']
    trainuserindices = datadict_combined['trainuserindices']
    testuserindices = datadict_combined['testuserindices']
    superusers = datadict_combined['superusers']

    outcome_dict = {'steps': 0, 'sed': 1, 'lpa': 2, 'mvpa': 3, 'wear_time': 4, 'sed_ratio': 5, 'lpa_ratio': 6, 'mvpa_ratio': 7}
    label_index = outcome_dict[outcome]

    mae, rmse, nrmse, r2, users, mae_peruser, nrmse_peruser = train_with_label(arch, Xtrain, Ytrain, Xtest,
                                                                                        Ytest, trainuserindices,
                                                                                        testuserindices, superusers,
                                                                                        all_labels[label_index],
                                                                                        label_index, window_size, featureset, num_epochs)
    # plot_learning_curve(histories[-2], 'Steps', epochs) #let's plot the last one
    # plot_accuracy_curve(history, 'Steps', epochs)
    return mae, rmse, nrmse, r2, users, mae_peruser, nrmse_peruser


def run_experiment(featureset, arch, df, all_users, outcome, num_epochs, window_size, engagement_percentile, activity_percentile):
    mae, rmse, nrmse, r2, users, mae_users, nrmse_users = experiment(featureset, df, all_users, arch, outcome, num_epochs, window_size, engagement_percentile, activity_percentile)


    return mae, rmse, nrmse, r2, users, mae_users, nrmse_users


def save(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)


def run_the_things(basepath, arch, dataset, outcome, output_folder, num_epochs, window_size, engagement_percentile, activity_percentile):
    mae_to_return = None
    if dataset == 'sleepwell':
        dataset_token = '21'
    elif dataset == 'bewell':
        dataset_token = '18'
    else:
        print('Invalid dataset name:', dataset)
        return

    df = pd.read_csv(basepath + 'daily_level_more_features_full_' + dataset_token + '.csv')

    print(df.shape)
    all_users = df.participant_id.unique()
    all_cols = df.columns

    #for i in range(len(all_cols)):
    #    print(i, all_cols[i])

    print("Total users:", len(all_users))

    header = "dataset,window_size,engagement_percentile,activity_percentile,arch,epochs,featureset,mae,rmse,nrmse,r2\n"
    body = ""

    if arch == 'arima':
        feature_set_list = ['activity']
    elif arch == 'lstm_late':
        feature_set_list = ['multimodal']
    else:
        feature_set_list = ['multimodal', 'engagement', 'activity']


    for feature_set in feature_set_list:
        mae, rmse, nrmse, r2, users, mae_users, nrmse_users = run_experiment(feature_set, arch, df, all_users, outcome, num_epochs, window_size, engagement_percentile, activity_percentile)
        body+= f"{dataset},{window_size},{engagement_percentile},{activity_percentile},{arch},{num_epochs},{feature_set},{mae},{rmse},{nrmse},{r2}\n"
        create_and_save_peruser(dataset, feature_set, arch, mae_users, nrmse_users, users, num_epochs, output_folder)

        if mae_to_return is None or mae < mae_to_return:
            mae_to_return = mae

    if not os.path.exists(output_folder + f'/regression_results_{dataset}_{outcome}.csv'):
        with open(output_folder + f'/regression_results_{dataset}_{outcome}.csv', 'w') as file:
            file.write(header)

    with open(output_folder + f'/regression_results_{dataset}_{outcome}.csv', 'a') as file:
        file.write(body)

    return mae_to_return

def create_and_save_peruser(dataset, featureset, arch, mae_peruser, nrmse_peruser, users, num_epochs, output_folder):
    result_df = pd.DataFrame(columns=['user', 'mae_peruser', 'nrmse_peruser'])
    result_df['user'] = users
    result_df['mae_peruser'] = mae_peruser
    result_df['nrmse_peruser'] = nrmse_peruser
    result_df.to_csv(output_folder + f'/peruser_results_{dataset}_{featureset}_{arch}_epochs_{num_epochs}.csv')


def create_classification_frames(df, all_users,  mode, threshold, window):
    X = []
    Y = []
    userindices = []
    if mode == 'multimodal':
        feature_start = 3
        feature_end = 68
    elif mode == 'engagement':
        feature_start = 3
        feature_end = 60
    elif mode == 'activity':
        feature_start = 60
        feature_end = 68

    else:
        print('ERROR: Invalid feature set')
        return None, None, None

    hop = max(1, window // 4)
    for user in all_users:
        fildf = df[df.participant_id == user]
        n = fildf.shape[0]
        start = 0
        while start + window < n:
            x = fildf.values[start:start + window, feature_start:feature_end]
            y = int(fildf.values[start + window, 60]>=threshold)

            X.append(x)
            Y.append(y)
            userindices.append(user)
            start += hop

    return X, Y, userindices
def run_classification_with_threshold(basepath, arch, dataset, featureset, num_epochs, threshold, window_size):

    if dataset == 'bewell':
        dataset_token = '18'
    elif dataset == 'sleepwell':
        dataset_token = '21'
    else:
        print('Invalid dataset name:', dataset)
        return
    df = pd.read_csv(basepath + '/daily_level_more_features_full_' + dataset_token + '.csv')
    all_users = df.participant_id.unique()

    all_users = shuffle(all_users)

    pivot = int(0.8 * len(all_users))
    train_users = all_users[:pivot]
    test_users = all_users[pivot:]
    traindf = df[df.participant_id.isin(train_users)]
    testdf = df[df.participant_id.isin(test_users)]



    Xtrain, Ytrain, trainuserindices = create_classification_frames(traindf, train_users, featureset, threshold, window_size)
    Xtest, Ytest, testuserindices = create_classification_frames(testdf, test_users, featureset, threshold, window_size)

    Xtrain = np.array(Xtrain).astype(float)
    Ytrain = np.array(Ytrain).astype(float)
    Xtest = np.array(Xtest).astype(float)
    Ytest = np.array(Ytest).astype(float)

    if featureset == 'multimodal':
        featuresize = 65
    elif featureset == 'activity':
        featuresize = 8
    elif featureset == 'engagement':
        featuresize = 57

    if arch == 'lstm':
        model = Sequential()
        model.add(LSTM(64, input_shape=(window_size, featuresize)))  # , return_sequences=True))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='relu'))

    elif arch == 'lstm_late':
        # Define the input shapes
        input_1 = Input(shape=(window_size, 57))  # First input channel
        input_2 = Input(shape=(window_size, 8))  # Second input channel

        # LSTM and Dense layers for the first input channel
        lstm_1 = LSTM(64, return_sequences=False)(input_1)
        dense_1_1 = Dense(32, activation='relu')(lstm_1)
        dense_1_2 = Dense(16, activation='relu')(dense_1_1)

        # LSTM and Dense layers for the second input channel
        lstm_2 = LSTM(32, return_sequences=False)(input_2)
        dense_2_1 = Dense(16, activation='relu')(lstm_2)
        dense_2_2 = Dense(8, activation='relu')(dense_2_1)

        # Concatenate the two Dense outputs
        concatenated = concatenate([dense_1_2, dense_2_2])

        # Output layer (final dense layer)
        output = Dense(1, activation='sigmoid')(concatenated)

        # Create the model
        model = Model(inputs=[input_1, input_2], outputs=output)

    else:
        print('ERROR: Invalid architecture')
        return None

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    if arch == 'lstm_late':
        model.fit(split_input_for_late(Xtrain), Ytrain, validation_split=0.15, epochs=num_epochs, verbose=0)
        Ypred = model.predict(split_input_for_late(Xtest), verbose=0) > 0.5
        Ypred = Ypred.astype(int)
    else:
        model.fit(Xtrain, Ytrain, validation_split=0.15, epochs=num_epochs, verbose=0)
        Ypred = model.predict(Xtest, verbose=0) > 0.5
        Ypred = Ypred.astype(int)


    acc = accuracy_score(Ytest, Ypred)
    prec = precision_score(Ytest, Ypred)
    rec = recall_score(Ytest, Ypred)
    f1 = f1_score(Ytest, Ypred)
    return acc, prec, rec, f1



def run_classification(basepath, arch, dataset, output_folder, num_epochs, window_size):
    if arch == 'lstm_late' or arch == 'lstm':
        feature_set_list = ['multimodal']
    else:
        feature_set_list = ['multimodal', 'engagement', 'activity']
    header = "dataset,threshold,arch,epochs,featureset,accuracy,precision,recall,f1\n"
    body = ""
    for threshold in [10000]:
        for feature_set in feature_set_list:
            print('Arch:', arch, 'Threshold:', threshold, 'Feature set:', feature_set)
            acc, prec, rec, f1 = run_classification_with_threshold(basepath, arch, dataset, feature_set, num_epochs, threshold, window_size)
            body+= f"{dataset},{threshold},{arch},{num_epochs},{feature_set},{acc},{prec},{rec},{f1}\n"

    outcome = 'steps'

    if not os.path.exists(output_folder + f'/classification_results_{dataset}_{outcome}.csv'):
        with open(output_folder + f'/classification_results_{dataset}_{outcome}.csv', 'w') as file:
            file.write(header)
    with open(output_folder + f'/classification_results_{dataset}_{outcome}.csv', 'a') as file:
        file.write(body)






