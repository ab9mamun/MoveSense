#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 16:07:41 2022

@author: abdullah
"""
import pandas as pd
import numpy as np
from datetime import datetime

import os


def doit(basepath):
    splitfilespath = basepath + 'splitbyparticipants/'
    destinationpath = basepath + 'daily_level_more_features_splits/'

    if not os.path.exists(destinationpath):
        os.makedirs(destinationpath)

    files = os.listdir(splitfilespath)

    files_18 = [file for file in files if '_18' in file]

    files_18.sort()

    is_weekday_cols = ['is_day_' + str(day) for day in range(7)]
    hourly_duration_cols = ['min_used_hr_' + str(hr) for hr in range(24)]
    hourly_frequency_cols = ['times_opened_hr_' + str(hr) for hr in range(24)]
    columns = ['participant_id', 'date', 'minutes_used', 'times_opened', *is_weekday_cols, *hourly_duration_cols,
               *hourly_frequency_cols,
               'steps', 'is_Sed_minutes', 'is_LPA_minutes', 'is_MVPA_minutes', 'wear_time',
               'sed_ratio', 'lpa_ratio', 'mvpa_ratio']

    fulldf = pd.DataFrame(columns=columns)

    for file in files_18:
        rawdf = pd.read_csv(splitfilespath + file)
        par = rawdf.iloc[0, :].participant_id
        dates = rawdf.date.unique()
        data = []

        for date in dates:
            fildf = rawdf[rawdf.date == date]
            minutes_used = fildf.in_use.to_numpy().sum()

            times_opened = 0
            hourly_duration = [0 for i in range(24)]
            hourly_opened = [0 for i in range(24)]
            weekday_values = [0 for i in range(7)]

            fildf.heart_rate = fildf.heart_rate.replace(' ', 0).astype(int)

            valid_df = fildf[(fildf.heart_rate > 0) | ((fildf.heart_rate == 0) & (fildf.mets >= 1))]

            steps = valid_df.steps.to_numpy().sum()
            is_Sed_minutes = valid_df.is_Sed.to_numpy().sum()
            is_LPA_minutes = valid_df.is_LPA.to_numpy().sum()
            is_MVPA_minutes = valid_df.is_MVPA.to_numpy().sum()
            wear_time = is_Sed_minutes + is_LPA_minutes + is_MVPA_minutes

            if wear_time < 600:
                continue  # not enough data for the day

            sed_ratio = is_Sed_minutes / wear_time
            lpa_ratio = is_LPA_minutes / wear_time
            mvpa_ratio = is_MVPA_minutes / wear_time

            dayOfWeek = datetime.strptime(date, '%m/%d/%Y').weekday()
            weekday_values[dayOfWeek] = 1

            for hr in range(24):
                filhrdf = fildf[fildf.time.str.startswith(str(hr) + ':')]
                hourly_duration[hr] = filhrdf.in_use.to_numpy().sum()

                all_bits = filhrdf.in_use.to_numpy()
                # bit_index = 1 #redundant operation
                bindx_lim = len(all_bits)
                pos_edges = 0
                for bit_index in range(1, bindx_lim):
                    if all_bits[bit_index - 1] == 0 and all_bits[bit_index] == 1:
                        pos_edges += 1

                hourly_opened[hr] = pos_edges

            times_opened = sum(hourly_opened)

            data.append([par, date, minutes_used, times_opened, *weekday_values, *hourly_duration, *hourly_opened,
                         steps, is_Sed_minutes, is_LPA_minutes, is_MVPA_minutes, wear_time,
                         sed_ratio, lpa_ratio, mvpa_ratio])

        if len(data) < 1:
            continue

        data_np = np.asarray(data)
        df = pd.DataFrame(data_np, columns=columns)
        df.to_csv(destinationpath + 'daily_level_' + str(par) + '.csv')

        if df.shape[0] >= 10:
            fulldf = pd.concat([fulldf, df])

    fulldf.to_csv(basepath + 'daily_level_more_features_full_18.csv')


def doit21(basepath):
    splitfilespath = basepath + 'splitbyparticipants/'
    destinationpath = basepath + 'daily_level_more_features_splits_21/'

    if not os.path.exists(destinationpath):
        os.makedirs(destinationpath)

    files = os.listdir(splitfilespath)

    files_21 = [file for file in files if '_21' in file]

    files_21.sort()

    is_weekday_cols = ['is_day_' + str(day) for day in range(7)]
    hourly_duration_cols = ['min_used_hr_' + str(hr) for hr in range(24)]
    hourly_frequency_cols = ['times_opened_hr_' + str(hr) for hr in range(24)]
    columns = ['participant_id', 'date', 'minutes_used', 'times_opened', *is_weekday_cols, *hourly_duration_cols,
               *hourly_frequency_cols,
               'steps', 'is_Sed_minutes', 'is_LPA_minutes', 'is_MVPA_minutes', 'wear_time',
               'sed_ratio', 'lpa_ratio', 'mvpa_ratio']

    fulldf = pd.DataFrame(columns=columns)

    for file in files_21:
        rawdf = pd.read_csv(splitfilespath + file)
        par = rawdf.iloc[0, :].participant_id
        dates = rawdf.date.unique()
        data = []

        for date in dates:
            fildf = rawdf[rawdf.date == date]
            minutes_used = fildf.in_use.to_numpy().sum()

            times_opened = 0
            hourly_duration = [0 for i in range(24)]
            hourly_opened = [0 for i in range(24)]
            weekday_values = [0 for i in range(7)]

            fildf.heart_rate = fildf.heart_rate.replace(' ', 0).astype(int)

            valid_df = fildf[(fildf.heart_rate > 0) | ((fildf.heart_rate == 0) & (fildf.mets >= 1))]

            steps = valid_df.steps.to_numpy().sum()
            is_Sed_minutes = valid_df.is_Sed.to_numpy().sum()
            is_LPA_minutes = valid_df.is_LPA.to_numpy().sum()
            is_MVPA_minutes = valid_df.is_MVPA.to_numpy().sum()
            wear_time = is_Sed_minutes + is_LPA_minutes + is_MVPA_minutes

            if wear_time < 600:
                continue  # not enough data for the day

            sed_ratio = is_Sed_minutes / wear_time
            lpa_ratio = is_LPA_minutes / wear_time
            mvpa_ratio = is_MVPA_minutes / wear_time

            dayOfWeek = datetime.strptime(date, '%m/%d/%Y').weekday()
            weekday_values[dayOfWeek] = 1

            for hr in range(24):
                filhrdf = fildf[fildf.time.str.startswith(str(hr) + ':')]
                hourly_duration[hr] = filhrdf.in_use.to_numpy().sum()

                all_bits = filhrdf.in_use.to_numpy()
                # bit_index = 1 #redundant operation
                bindx_lim = len(all_bits)
                pos_edges = 0
                for bit_index in range(1, bindx_lim):
                    if all_bits[bit_index - 1] == 0 and all_bits[bit_index] == 1:
                        pos_edges += 1

                hourly_opened[hr] = pos_edges

            times_opened = sum(hourly_opened)

            data.append([par, date, minutes_used, times_opened, *weekday_values, *hourly_duration, *hourly_opened,
                         steps, is_Sed_minutes, is_LPA_minutes, is_MVPA_minutes, wear_time,
                         sed_ratio, lpa_ratio, mvpa_ratio])

        if len(data) < 1:
            continue

        data_np = np.asarray(data)
        df = pd.DataFrame(data_np, columns=columns)
        df.to_csv(destinationpath + 'daily_level_' + str(par) + '.csv')

        if df.shape[0] >= 10:
            fulldf = pd.concat([fulldf, df])

    fulldf.to_csv(basepath + 'daily_level_more_features_full_21.csv')