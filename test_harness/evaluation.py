#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Model evaluation functions module of thesis testing harness.

Author:  Christopher Good
Version: 1.0.0

Usage: evaluation.py

"""
# See following link for proper docstring documentation
# https://pandas.pydata.org/docs/development/contributing_docstring.html 

### Futures ###
#TODO

### Built-in Imports ###
import datetime
from operator import truediv
import os
import time


### Other Library Imports ###
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import seaborn as sns

### Local Imports ###
#TODO

### Constants ###
#TODO

### Function Definitions ###

def test_model(model, test_dataset, **hyperparams):
    """
    """

    print(f'Testing {model.name} with test dataset...')

    # Record start time for model evaluation
    model_test_start = time.process_time()

    # Get prediction values for test dataset
    pred_test = model.predict(test_dataset,
                              verbose=1,
                              ).argmax(axis=1)

    # Record end time for model evaluation
    model_test_end = time.process_time()

    # Get time elapsed for testing model
    model_test_time = datetime.timedelta(seconds=(model_test_end - model_test_start))

    print('Testing completed!')
   

    return pred_test, model_test_time


def calculate_model_statistics(pred_test, target_test, labels,
                               **hyperparams):
    """
    """
    labels = hyperparams['all_class_labels']

    overall_acc = metrics.accuracy_score(target_test, pred_test)
    precision = metrics.precision_score(target_test, pred_test, average='micro')
    recall = metrics.recall_score(target_test, pred_test, average='micro')
    kappa = metrics.cohen_kappa_score(target_test, pred_test)
    confusion_matrix = metrics.confusion_matrix(target_test, pred_test, labels=range(len(labels)))

    # Supress/hide invalid value warning
    # np.seterr(invalid='ignore')

    # Calculate average accuracy and per-class accuracies
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)

    # Get classification report
    classification_report = metrics.classification_report(target_test, 
                                                          pred_test, 
                                                          labels=range(len(labels)),
                                                          target_names=labels, 
                                                          digits=3)

    results = {
        'overall_accuracy': overall_acc,
        'average_accuracy': average_acc,
        'precision_score': precision,
        'recall_score': recall,
        'cohen_kappa_score': kappa,
        'confusion_matrix': confusion_matrix,
        'per_class_accuracies': each_acc,
        'labels': labels,
        'classification_report': classification_report,
    }

    return results

def create_confusion_matrix_plot(confusion_matrix, labels, model_name, 
                                 output_path='./', iteration=None):
    """
    """

    # Create filename for confusion matrix image file
    if iteration is not None:
        filename = f'experiment_{iteration+1}_{model_name}_confusion_matrix.png'
    else:
        filename = f'experiment_{model_name}_confusion_matrix.png'

    # Create full file name for confusion matrix image file
    cm_filename = os.path.join(output_path, filename)

    # Create annotations for confusion matrix
    print('Creating confusion matrix annotations...')
    cm_sum = np.sum(confusion_matrix, axis=1, keepdims=True)
    cm_perc = confusion_matrix / cm_sum.astype(float) * 100
    annot = np.empty_like(confusion_matrix).astype(str)
    nrows, ncols = confusion_matrix.shape
    for i in range(nrows):
        for j in range(ncols):
            c = confusion_matrix[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    

    # Create confusion matrix dataframe
    print('Creating confusion matrix plot...')
    cm = pd.DataFrame(confusion_matrix, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=(30,30))
    sns.heatmap(cm, annot=annot, fmt='', ax=ax)

    if iteration is not None:
        plt.title(f'Experiment #{iteration+1} {model_name} Confusion Matrix')
    else:
        plt.title(f'Experiment w/ {model_name} Confusion Matrix')

    print('Saving confusion matrix plot...')
    plt.savefig(cm_filename)

    # Clear plot data for next plot
    plt.clf()

def output_experiment_results(experiment_info):
    """
    """

    # Get variables from dictionary
    experiment_name = experiment_info['experiment_name']
    model_name = experiment_info['model_name']
    model_train_time = experiment_info['model_train_time']
    model_test_time = experiment_info['model_test_time']
    overall_acc = experiment_info['overall_accuracy']
    average_acc = experiment_info['average_accuracy']
    per_class_accuracies = experiment_info['per_class_accuracies']
    precision = experiment_info['precision_score']
    recall = experiment_info['recall_score']
    kappa = experiment_info['cohen_kappa_score']
    labels = experiment_info['labels']
    classification_report = experiment_info['classification_report']
    
    # Print results
    print('---------------------------------------------------')
    if experiment_name is None:
        print('             MODEL EXPERIMENT RESULTS              ')
    else:
        print(f'          "{experiment_name}" RESULTS              ')
    print('---------------------------------------------------')
    print(f' MODEL NAME: {model_name}')
    print('---------------------------------------------------')
    print(f'{model_name} train time: {model_train_time}')
    print(f'{model_name} test time:  {model_test_time}')
    print('...................................................')
    print(f'{model_name} overall accuracy:  {overall_acc}')
    print(f'{model_name} average accuracy:  {average_acc}')
    print(f'{model_name} precision score:   {precision}')
    print(f'{model_name} recall score:      {recall}')
    print(f'{model_name} cohen kappa score: {kappa}')
    print('...................................................')
    print(f'{model_name} Per-class accuracies:')
    for i, label in enumerate(labels):
        print(f'{label}: {per_class_accuracies[i]}')
    print('---------------------------------------------------')
    print('              CLASSIFICATION REPORT                ')
    print('...................................................')
    print(classification_report)
    print('---------------------------------------------------')
    print()