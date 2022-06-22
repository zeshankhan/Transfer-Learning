# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 15:39:31 2022

@author: ZESHAN KAHN
"""

def alter_last_layer(base_model=None,output_layer='avg_pool',num_classes=16):
    if(output_layer=="global_average_pooling2d"):
        x=base_model.layers[-2].output
    else:
        x = base_model.get_layer(output_layer).output
    x = Dense(num_classes, name="output")(x)
    model = Model(base_model.input, outputs=x)
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    adag=Adagrad(learning_rate=0.01,initial_accumulator_value=0.1,epsilon=1e-07,name="Adagrad")
    model.compile(loss='mean_squared_error', optimizer=adag)
    return model

def features_layer(base_model=None,output_layer='avg_pool',num_classes=16):
    if(output_layer=="global_average_pooling2d"):
        x=base_model.layers[-2].output
    else:
        x = base_model.get_layer(output_layer).output
    model = Model(base_model.input, outputs=x)
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    adag=Adagrad(learning_rate=0.01,initial_accumulator_value=0.1,epsilon=1e-07,name="Adagrad")
    model.compile(loss='mean_squared_error', optimizer=adag)
    return model


from tensorflow.python.platform import tf_logging as logging

class ReduceLRBacktrack(ReduceLROnPlateau):
    def __init__(self, best_path, *args, **kwargs):
        super(ReduceLRBacktrack, self).__init__(*args, **kwargs)
        self.best_path = best_path

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            logging.warning('Reduce LR on plateau conditioned on metric `%s` '
                            'which is not available. Available metrics are: %s',
                             self.monitor, ','.join(list(logs.keys())))
        if not self.monitor_op(current, self.best): # not new best
            if not self.in_cooldown(): # and we're not in cooldown
                if self.wait+1 >= self.patience: # going to reduce lr
                    # load best model so far
                    print("Backtracking to best model before reducting LR")
                    self.model.load_weights(self.best_path)

        super().on_epoch_end(epoch, logs) # actually reduce LR
