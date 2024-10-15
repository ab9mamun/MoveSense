# -*- coding: utf-8 -*-

from datamanager import datamanager, splitbyparticipants, get_demographics
import sys
import experiments
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

"""
This function can be used to run the experiment with different window size of the featureset. Based on the value of the option parameter,
the function calls the run_the_things function from the corresponding module.
E.g. forecasting_balanced_1_window14 has the window size 14.
"""


def runNum(basepath, dataset, option, epochs):
    # if option == 1:
    #     forecasting_balanced_1_window14.run_the_things(basepath, 'lstm', dataset, epochs)
    # elif option == 2:
    #     forecasting_balanced_1_window21.run_the_things(basepath, 'lstm', dataset, epochs)
    # elif option == 3:
    #     forecasting_balanced_1_window28.run_the_things(basepath, 'lstm', dataset, epochs)
    #
    # elif option == 4:
    #     forecasting_balanced_1_window1.run_the_things(basepath, 'lstm', dataset, epochs)
    #
    # elif option == 5:
    #     forecasting_balanced_1_window3.run_the_things(basepath, 'lstm', dataset, epochs)
    pass


"""
This is the place where the execution starts.
"""

def run(basepath, args):

    print('Welcome to the Activity Forecasting project')
    #forecasting_balanced_1_window14.run_the_things(basepath, 'lstm', 'bewell')

    print('Printing all the parameters for the experiment')
    dataset = args.dataset
    task = args.task
    super_threshold = args.super_threshold
    model = args.model
    outcome = args.outcome
    num_epochs = args.num_epochs
    output_folder = basepath + '/'+ args.output_folder + '/' + args.exp_name + '/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    print('Basepath: ', basepath)
    print('Dataset: ', dataset)

    print('Task: ', task)

    ##runNum(basepath, dataset, 1, num_epochs)
    if task == 'print_demographics':
        get_demographics.get(basepath, dataset)
    #elif task == 'split_by_participants':
    #    splitbyparticipants.split(basepath, dataset)
    elif task == 'run_the_things':
        experiments.run_the_things(basepath, model, dataset, outcome)

    elif task == 'run_everything':
        sub_exps = ['window_chooser', 'model_chooser', 'other_outcomes', 'engagement_chooser', 'activity_chooser']
        sub_output_folders = []

        for sub_exp in sub_exps:
            sub_output_folder = output_folder + sub_exp + '/'
            if not os.path.exists(sub_output_folder):
                os.makedirs(sub_output_folder)
            sub_output_folders.append(sub_output_folder)

        best_window = None
        best_mae = None
        for outcome in ['steps']:  # , 'sed_ratio', 'lpa_ratio', 'mvpa_ratio', 'wear_time']:
            for model in ['lstm']:#['lin_reg', 'arima', 'lstm', 'lstm_late']:
                for window_size in [3, 7, 14, 21]:

                    mae = experiments.run_the_things(basepath, model, dataset, outcome, sub_output_folders[0], num_epochs, window_size, 0, 0)
                    if best_mae is None or mae < best_mae:
                        best_mae = mae
                        best_window = window_size

        print('Best window size: ', best_window, 'MAE:', best_mae)

        best_mae = None
        best_model = None
        for outcome in ['steps']:
            for model in ['lstm', 'lin_reg', 'arima', 'lstm_late']:
                mae = experiments.run_the_things(basepath, model, dataset, outcome, sub_output_folders[1], num_epochs, 7, 0, 0)
                if best_mae is None or mae < best_mae:
                    best_mae = mae
                    best_model = model

        print('Best model: ', best_model, 'MAE:', best_mae)

        best_engagement_percentile = 0
        best_engagement_mae = best_mae

        for engagement_percentile in [0.25, 0.5, 0.75]:
            mae = experiments.run_the_things(basepath, best_model, dataset, 'steps', sub_output_folders[3], num_epochs, best_window, engagement_percentile, 0)
            if best_engagement_mae is None or mae < best_engagement_mae:
                best_engagement_mae = mae
                best_engagement_percentile = engagement_percentile

        best_acitivity_percentile = 0
        best_activity_mae = best_mae
        for activity_percentile in [0.25, 0.5, 0.75]:
            mae = experiments.run_the_things(basepath, best_model, dataset, 'steps', sub_output_folders[4], num_epochs, best_window, 0, activity_percentile)
            if best_activity_mae is None or mae < best_activity_mae:
                best_activity_mae = mae
                best_acitivity_percentile = activity_percentile

        print('Best engagement percentile: ', best_engagement_percentile, 'MAE:', best_engagement_mae)
        print('Best activity percentile: ', best_acitivity_percentile, 'MAE:', best_activity_mae)

        for outcome in ['sed', 'lpa', 'mvpa', 'wear_time']:
           for model in ['lstm']:
               mae = experiments.run_the_things(basepath, model, dataset, outcome, sub_output_folders[2], num_epochs, 7, 0, 0)

        for model in ['lstm']:
            experiments.run_classification(basepath, model, dataset, output_folder, num_epochs, 7)

    elif task == 'plot_peruser':
        plot_peruser(output_folder, basepath+'Per_user_csv.csv')

def plot_peruser(output_folder, filename):
    df = pd.read_csv(filename)
    #print(df)
    participants = df['Participant'].to_numpy()
    early = df['Early'].to_numpy()
    late = df['Late'].to_numpy()
    bewell_participants = participants[:11]
    bewell_early = early[:11]
    bewell_late = late[:11]
    sleepwell_participants = participants[11:]
    sleepwell_early = early[11:]
    sleepwell_late = late[11:]

    fig, (ax1, ax2) = plt.subplots(2, 1)
    bar_width = 0.35
    index = np.arange(len(bewell_participants))
    ax1.bar(index, bewell_early, width=bar_width, label='Early fusion')
    ax1.bar(index + bar_width, bewell_late, width=bar_width, alpha=0.7, label='Late fusion')
    # Adding labels and formatting
    ax1.set_xlabel('Prediabetes study participants')
    ax1.set_ylabel('MAE')
    ax1.set_ylim(0, 6000)
    ax1.set_xticks(index + bar_width / 2)
    ax1.set_xticklabels(bewell_participants)
    ax1.legend(loc='upper left', fontsize='small')

    index = np.arange(len(sleepwell_participants))
    ax2.bar(index, sleepwell_early, width=bar_width, label='Early fusion')
    ax2.bar(index + bar_width, sleepwell_late, width=bar_width, alpha=0.7, label='Late fusion')
    # Adding labels and formatting
    ax2.set_xlabel('Sleep study participants')
    ax2.set_ylabel('MAE')
    ax2.set_ylim(0, 6000)
    ax2.set_xticks(index + bar_width / 2)
    ax2.set_xticklabels(sleepwell_participants)
    ax2.legend(loc='upper left', fontsize='small')

    plt.tight_layout()  # Optional: Adjust layout for better spacing
    plt.savefig(output_folder + 'mae_per_user.png')


def main(args):
    basepath = '../data/app_usage/'
    run(basepath, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the model on the FRI dataset.')

    parser.add_argument('--dataset',
                        default='bewell',
                        help='Choose from bewell, sleepwell',
                        type=str)
    parser.add_argument('--output_folder',
                        default='output',
                        help='The folder where the output will be stored',
                        type=str)
    parser.add_argument('--exp_name',
                        default='default_exp_name',
                        help='A unique name for the experiment. If not unique, the existing experiment will be overwritten.',
                        type=str)
    parser.add_argument('--task',
                        default='print_demographics',
                        help='Choose from regression, classification, print_demographics',
                        type=str)
    parser.add_argument('--super_threshold',
                        default='1.00',
                        help='Choose from None, 0.10, 0.25, 0.50, 0.75, 0.90, 1.00',
                        type=float)

    parser.add_argument('--outcome',
                        default='steps',
                        help='Choose from steps, sed_ratio, lpa_ratio, mvpa_ratio, wear_time',
                        type=str)

    parser.add_argument('--model',
                        default='lstm',
                        help='Options: lstm, cnn',
                        type=str)

    parser.add_argument('--learning_rate',
                        default='0.01',
                        help='Choose learning rate from a valid numeric value greater than 0 and less than 1',
                        type=str)

    parser.add_argument('--num_epochs',
                        default=2,
                        help='Number of epochs to train the model',
                        type=int)

    parser.add_argument('--gpu_ids',
                        default='0',
                        type=str)
    parser.add_argument('--batch_size',
                        default=32,
                        type=int)

    main(args=parser.parse_args())