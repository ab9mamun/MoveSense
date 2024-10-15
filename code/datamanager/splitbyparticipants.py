# -*- coding: utf-8 -*-

import pandas as pd
import os


def split_by_participants(basepath, csvfilename, destination_path):
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    input_file = basepath + csvfilename
    bootstrapdf = pd.read_csv(input_file, nrows=5)
    boot_columns = bootstrapdf.columns
    df = pd.DataFrame(columns=boot_columns)
    cur_par = bootstrapdf.iloc[0, :].participant_id
    # exit_flag = False
    rows_done = 0
    while True:
        rawdf = pd.read_csv(input_file, skiprows=list(range(1, rows_done + 1)), nrows=100000)
        if rawdf.shape[0] == 0:
            df.to_csv(destination_path + csvfilename + '_' + str(cur_par) + '.csv')
            print("Finished processing participant ", cur_par)
            break

        cur_par_df = rawdf[rawdf.participant_id == cur_par]
        other_par_df = rawdf[rawdf.participant_id != cur_par]

        df = pd.concat([df, cur_par_df])
        rows_done += cur_par_df.shape[0]
        while other_par_df.shape[0] > 0:
            df.to_csv(destination_path + csvfilename + '_' + str(cur_par) + '.csv')
            print("Finished processing participant ", cur_par)
            df = other_par_df

            cur_par = df.iloc[0, :].participant_id
            cur_par_df = df[df.participant_id == cur_par]
            other_par_df = df[df.participant_id != cur_par]
            rows_done += cur_par_df.shape[0]
            df = cur_par_df


def split_error_verification_18_level(basepath):
    files = os.listdir(basepath + 'splitbyparticipants')
    files = [file for file in files if '_18' in file]

    print(files)

    for file in files:
        tempdf = pd.read_csv(basepath + 'splitbyparticipants/' + file)
        print('processing: ', file)
        all_participants = tempdf.participant_id.unique()
        if len(all_participants) == 1:
            print(all_participants, 'OK')
        else:
            print(all_participants, 'alert')


def split_error_verification_21_level(basepath):
    files = os.listdir(basepath + 'splitbyparticipants')
    files = [file for file in files if '_21' in file]

    print(files)

    for file in files:
        tempdf = pd.read_csv(basepath + 'splitbyparticipants/' + file)
        print('processing: ', file)
        all_participants = tempdf.participant_id.unique()
        if len(all_participants) == 1:
            print(all_participants, 'OK')
        else:
            print(all_participants, 'alert')


def doit(basepath):
    split_by_participants(basepath, 'fitbit and app usage raw data.csv', basepath + 'splitbyparticipants/')
    # the above function will create some erorrs, now time to fix them.
    split_error_verification_18_level(basepath)
    split_error_verification_21_level(basepath)


