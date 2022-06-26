#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Model training functions module of thesis testing harness.

Author:  Christopher Good
Version: 1.0.0

Usage: training.py

"""
# See following link for proper docstring documentation
# https://pandas.pydata.org/docs/development/contributing_docstring.html 

### Futures ###
#TODO

### Built-in Imports ###
import datetime
import os
import time

### Other Library Imports ###
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import (
    EarlyStopping,
    LearningRateScheduler,
    ModelCheckpoint,
)

### Local Imports ###
from models.models import get_optimizer


### Constants ###
#TODO

### Function Definitions ###

def create_training_summary_plot(history, experiment_name, output_path):
    """
    """
    print('Creating training summary plots...')

    # Plot loss
    plt.subplot(211)
    plt.title('Cross Entropy Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], color='orange', label='test')

    # Plot accuracy
    plt.subplot(212)
    plt.title('Sparse Categorical Accuracy')
    plt.plot(history.history['sparse_categorical_accuracy'], color='blue', label='train')
    if 'val_sparse_categorical_accuracy' in history.history:
        plt.plot(history.history['val_sparse_categorical_accuracy'], color='orange', label='test')

    # save plot to file
    print('Saving training summary plot to file...')
    filename = os.path.join(output_path, f'{experiment_name}_training_summary_plot.png')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    plt.clf()

def train_model(model, train_dataset, val_dataset,
                iteration = None, **hyperparams):
    """
    """
    # Initialize variables from the hyperparameters
    experiment_name = hyperparams['experiment_name']
    epochs = hyperparams['epochs']
    epochs_before_decay = hyperparams['epochs_before_decay']
    lr_decay_rate = hyperparams['lr_decay_rate']
    patience = hyperparams['patience']
    loss = hyperparams['loss']
    model_metrics = hyperparams['metrics']
    output_path = hyperparams['output_path']
    model_save_period = hyperparams['model_save_period']
    optimizer = get_optimizer(**hyperparams)
    callbacks = []

    # Determine ID string for experiment
    if experiment_name is not None:
        experiment_id = experiment_name
    elif iteration is not None:
        experiment_id = f'experiment_{iteration+1}'
    else:
        experiment_id = 'experiment'

    # Create best weights path filename
    best_weights_path = os.path.join(output_path, 
        f'{model.name}_best_weights_{experiment_id}.hdf5')
    
    checkpoint_path_prefix = os.path.join(output_path, f'{experiment_id}_checkpoint_')

    if model_save_period is not None:
        # Create callback to save model weights every 'period' number
        # of epochs
        cb_periodic_model_checkpoint = ModelCheckpoint(checkpoint_path_prefix + '{epoch:08d}.hdf5', period=model_save_period)
        callbacks.append(cb_periodic_model_checkpoint)

    if patience is not None:
        # Create callback to stop training early if metrics don't improve
        cb_early_stopping = EarlyStopping(monitor='val_loss', 
                                        patience=patience, 
                                        verbose=1, 
                                        mode='auto',
                                        restore_best_weights=True)
        callbacks.append(cb_early_stopping)

    if val_dataset is not None:
        # Create callback to save model weights if the model performs
        # better than the previously trained models
        cb_save_best_model = ModelCheckpoint(best_weights_path, 
                                            monitor='val_loss', 
                                            verbose=1, 
                                            save_best_only=True, 
                                            mode='auto')
        callbacks.append(cb_save_best_model)
    
    if lr_decay_rate is not None and epochs_before_decay is not None:
        # This function keeps the initial learning rate for a set number of epochs
        # and reduces it at decay rate after that
        def scheduler(epoch, lr):
            if epoch < epochs_before_decay:
                return lr
            else:
                print(f'Learning rate reduced from {lr} to {lr*lr_decay_rate}...')
                return lr * lr_decay_rate
                # return lr * tf.math.exp(-0.1)

        # Create learning rate scheduler callback for learning rate decay
        cb_lr_decay = LearningRateScheduler(scheduler)

        callbacks.append(cb_lr_decay)

    # Compile the model with the appropriate loss function, optimizer,
    # and metrics
    model.compile(loss=loss, 
                  optimizer=optimizer, 
                  metrics=model_metrics,
                  loss_weights=None,
                  weighted_metrics=None,
                  run_eagerly=None,
                  )
    
    # Display a summary of the model being trained
    model.summary()

    # Record start time for model training
    model_train_start = time.process_time()

    # Train the model
    model_history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs, 
            verbose=1,
            shuffle=True, 
            callbacks=callbacks
        )

    # Record end time for model training
    model_train_end = time.process_time()

    # Calculate training and testing times
    model_train_time = datetime.timedelta(seconds=(model_train_end - model_train_start))
    
    # Write model history to file
    with open(os.path.join(output_path,
         f'{experiment_id}_training_history.txt'), 'w') as hf:

        hf.write(f'EXPERIMENT #{iteration+1} MODEL HISTORY:\n')
        hf.write('-----------------------------------------------\n')
        hf.write(f'MODEL: {model.name}\n')
        hf.write('-----------------------------------------------\n')

        # Save model summary to file as well
        model.summary(print_fn=lambda x: hf.write(x + '\n'))

        # Show epoch with best validation value
        if patience is not None:
            hf.write(f'Best Epoch: {cb_early_stopping.best_epoch}\n')

        # Get number of epochs model actually ran for
        ran_epochs = model_history.params['epochs']
        if patience is not None:
            if cb_early_stopping.stopped_epoch > 0:
                ran_epochs = cb_early_stopping.stopped_epoch

        # Save info from each epoch to file
        for epoch in range(ran_epochs):
            hf.write(f'EPOCH: {epoch+1}\n')
            for key in model_history.history.keys():
                hf.write(f'  {key}: {model_history.history[key][epoch]}\n')

    create_training_summary_plot(model_history, experiment_id, output_path)

    # Save final weights
    # final_weights_path = os.path.join(output_path, 
    #     f'{model.name}_final_weights_{experiment_id}.hdf5')
    # model.save_weights(final_weights_path)

    return model, model_train_time