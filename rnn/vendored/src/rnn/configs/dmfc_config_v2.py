"""DMFC model v2 configuration: rank-1 integrator + comp RNN."""

config = {
    # Task
    'dt': 1/30,
    'tau': 0.1,
    'vector_lengths': [1, 2, 3, 4, 5, 6],
    'training_set': 'circular',
    'delays': [0.2, 0.3, 0.4],
    'scale': 1,

    # Component 1: Rank-1 Integrator RNN
    'integrator': {
        'input_size': 6,       # [startx, starty, dir_onehot(4)]
        'hidden_size': 32,
        'tau': 0.1,
    },

    # Component 2: Computation RNN
    'comp_rnn': {
        'hidden_size': 32,
        'tau': 0.1,
    },

    # Readout
    'readout': {
        'output_size': 2,      # target x, y
    },

    # Training
    'training': {
        'optimizer': 'AdamW',
        'lr': 1e-3,
        'weight_decay': 0.0,
        'batch_size': 300,
        'max_epochs': 20000,
        'early_stopping_patience': 1000,
        'grad_clip': 1.0,
        'ramp_lambda': 1.0,    # weight of aux ramp loss
    },
}
