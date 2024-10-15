import numpy as np
import pandas as pd


def get_bmis(basepath, dataset, all_users):
    if dataset == 'bewell':
        demdf = pd.read_csv(basepath + 'demographic/Clinical_Dataset_Final_6_22_20.csv')
        users = demdf.ID.to_numpy().astype(int)
        bmis = demdf.BMI_V1.to_numpy().astype(float)
    elif dataset == 'sleepwell':
        demdf = pd.read_csv(basepath + 'demographic/R21_clinical.csv')
        users = demdf.pid.to_numpy().astype(int)
        bmis = demdf.bmi_visit1.to_numpy().astype(float)
    else:
        return None
    mapp = {}
    n = len(users)
    for i in range(n):
        mapp[users[i]] = bmis[i]
    all_bmis = []
    for user in all_users:
        all_bmis.append(mapp[user])
        # print(user, mapp[user])

    return (np.mean(all_bmis), np.std(all_bmis), np.max(all_bmis), np.min(all_bmis))


def get_ages(basepath, dataset, all_users):
    if dataset == 'bewell':
        demdf = pd.read_csv(basepath + 'demographic/Demographic_Dataset_Final_6_22_20.csv')
        users = demdf.ID.to_numpy().astype(int)
        ages = demdf.Age.to_numpy().astype(int)
    elif dataset == 'sleepwell':
        demdf = pd.read_csv(basepath + 'demographic/SleepWell R21_demographics_2020-11-24.csv')
        users = demdf.pid.to_numpy().astype(int)
        ages = demdf.age.to_numpy().astype(int)
    else:
        return None

    mapp = {}
    n = len(users)
    for i in range(n):
        mapp[users[i]] = ages[i]
    all_ages = []
    for user in all_users:
        all_ages.append(mapp[user])
        # print(user, mapp[user])

    return (np.mean(all_ages), np.std(all_ages), np.max(all_ages), np.min(all_ages))


def get_sexes(basepath, dataset, all_users):
    if dataset == 'bewell':
        demdf = pd.read_csv(basepath + 'demographic/Demographic_Dataset_Final_6_22_20.csv')
        users = demdf.ID.to_numpy().astype(int)
        sexes = demdf.Sex.to_numpy().astype(int)
    elif dataset == 'sleepwell':
        demdf = pd.read_csv(basepath + 'demographic/SleepWell R21_demographics_2020-11-24.csv')
        users = demdf.pid.to_numpy().astype(int)
        sexes = demdf.sex.to_numpy().astype(int)
    else:
        return None

    mapp = {}
    n = len(users)
    for i in range(n):
        mapp[users[i]] = sexes[i]
    all_sexes = []
    for user in all_users:
        all_sexes.append(mapp[user])
        # print(user, mapp[user])
    male = np.count_nonzero(all_sexes)
    return male, len(all_users) - male


def get(basepath, dataset):
    if dataset == 'sleepwell':
        dataset_token = '21'
    elif dataset == 'bewell':
        dataset_token = '18'

    df = pd.read_csv(basepath + 'daily_level_more_features_full_' + dataset_token + '.csv')

    # print(df.shape)
    all_users = df.participant_id.unique()
    all_cols = df.columns
    print('==Printing Demographics==')
    print('Dataset:', dataset)
    print('N:', len(all_users))
    stats = get_bmis(basepath, dataset, all_users)
    if stats:
        (mean, std, maxx, minn) = stats
        print('BMI mean:', mean)
        print('BMI std:', std)
        print('BMI max:', maxx)
        print('BMI min:', minn)

    stats = get_ages(basepath, dataset, all_users)
    if stats:
        (mean, std, maxx, minn) = stats
        print('Age mean:', mean)
        print('Age std:', std)
        print('Age max:', maxx)
        print('Age min:', minn)

    stats = get_sexes(basepath, dataset, all_users)
    if stats:
        (male, female) = stats
        print('Male:', male)
        print('Female:', female)
