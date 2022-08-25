def train():
    # Default values for hyper-parameters we're going to sweep over
    defaults = dict(
        optimizer = 'Adam',
        epochs = 25,
        regularization = True,    
        dropout = True
    )

    
    # Initialize a new wandb run
    wandb.init(project="sign_language_recognition", config= defaults)

    # Config is a variable that holds and saves hyperparameters and inputs
    config = wandb.config


    # neural network layers    
    lenet5 = Sequential()
    if config.regularization:
        lenet5.add(Conv2D(filters=6, kernel_size=(5,5), strides=1, kernel_initializer='he_uniform', kernel_regularizer=keras.regularizers.l2(0.01),  activation='relu', input_shape=(80,80,3), padding='same'))
    else:
        lenet5.add(Conv2D(filters=6, kernel_size=(5,5), strides=1, kernel_initializer='he_uniform',  activation='relu', input_shape=(80,80,3), padding='same'))
    lenet5.add(BatchNormalization())
    lenet5.add(MaxPooling2D(pool_size=(3,3))) 
    if config.dropout:
        lenet5.add(Dropout(0.5))
    lenet5.add(Conv2D(filters=16, kernel_size=(5,5), strides=1,  activation='relu', padding='valid'))
    lenet5.add(BatchNormalization())
    lenet5.add(MaxPooling2D(pool_size=(3,3))) 
    if config.dropout:
        lenet5.add(Dropout(0.5))
    lenet5.add(Flatten()) 
    lenet5.add(Dense(units=120, activation='relu'))
    if config.dropout:
        lenet5.add(Dropout(0.5))
    lenet5.add(Dense(units=84, activation='relu'))
    if config.dropout:
        lenet5.add(Dropout(0.25))
    lenet5.add(Dense(units=26, activation='softmax'))


    # testing different loss functions
    loss = 'sparse_categorical_crossentropy'

    # Instantiate an accuracy metric.
    accuracy = Accuracy()

    if config.optimizer == 'Adam':
        optimizer = Adam()

    if config.optimizer == 'RMSprop':
        optimizer = RMSprop()

    lenet5.compile(optimizer=optimizer, loss=loss, metrics=['accuracy']) 

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5) 
    lenet5.fit(x_train, y_train, 
               batch_size=32,
               epochs=config.epochs,
               validation_data=(x_val, y_val),
               callbacks=[es, WandbCallback()]
               )
    
# Configure the sweep – specify the parameters to search through, the search strategy, the optimization metric et all.
sweep_config = {
    'method': 'random', #grid, random
    'metric': {
      'name': 'accuracy',
      'goal': 'maximize'   
    },
    'parameters': {
        # testing epochs variance effect
        'epochs': {
            'values': [25,50]
        },
        # testing optimizer variance effect
        'optimizer': {
            'values': ['Adam', 'RMSprop']
        },
        # testing optimizer variance effect
        'regularization': {
            'values': [True, False]
        },
        # testing the addition of dropout layers
        'dropout': {
            'values': [True, False]
        }
    }
}







def train():
    # Default values for hyper-parameters we're going to sweep over
    defaults = dict(
        optimizer = 'Adam',
        dropout_1 = True,
        dropout_2 = True,
        dropout_3 = True,
        dropout_4 = True,
        dropout_value = 0.25
    )

    
    # Initialize a new wandb run
    wandb.init(project="sign_language_recognition", config= defaults)

    # Config is a variable that holds and saves hyperparameters and inputs
    config = wandb.config


    # neural network layers    
    lenet5 = Sequential()
    lenet5.add(Conv2D(filters=6, kernel_size=(5,5), strides=1, kernel_initializer='he_uniform', kernel_regularizer=keras.regularizers.l2(0.01),  activation='relu', input_shape=(80,80,3), padding='same'))
    lenet5.add(BatchNormalization())
    lenet5.add(MaxPooling2D(pool_size=(3,3))) 
    if config.dropout_1:
        if config.dropout_value==0.25:
            lenet5.add(Dropout(0.25))
        else:
            lenet5.add(Dropout(0.5))
    lenet5.add(Conv2D(filters=16, kernel_size=(5,5), strides=1,  activation='relu', padding='valid'))
    lenet5.add(BatchNormalization())
    lenet5.add(MaxPooling2D(pool_size=(3,3))) 
    if config.dropout_2:
        if config.dropout_value==0.25:
            lenet5.add(Dropout(0.25))
        else:
            lenet5.add(Dropout(0.5))
    lenet5.add(Flatten()) 
    lenet5.add(Dense(units=120, activation='relu'))
    if config.dropout_3:
        if config.dropout_value==0.25:
            lenet5.add(Dropout(0.25))
        else:
            lenet5.add(Dropout(0.5))
    lenet5.add(Dense(units=84, activation='relu'))
    if config.dropout_4:
        if config.dropout_value==0.25:
            lenet5.add(Dropout(0.25))
        else:
            lenet5.add(Dropout(0.5))
    lenet5.add(Dense(units=26, activation='softmax'))


    # testing different loss functions
    loss = 'sparse_categorical_crossentropy'

    # Instantiate an accuracy metric.
    accuracy = Accuracy()

    if config.optimizer == 'Adam':
        optimizer = Adam()

    if config.optimizer == 'RMSprop':
        optimizer = RMSprop()

    lenet5.compile(optimizer=optimizer, loss=loss, metrics=['accuracy']) 

    # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5) 
    lenet5.fit(x_train, y_train, 
               batch_size=32,
               epochs=50,
               validation_data=(x_val, y_val),
               # callbacks=[es, WandbCallback()]
               callbacks=[WandbCallback()]
               )
    
# Configure the sweep – specify the parameters to search through, the search strategy, the optimization metric et all.
sweep_config = {
    'method': 'bayes', #grid, random
    'metric': {
      'name': 'val_loss',
      'goal': 'minimize'   
    },
    # 'method': 'random', #grid, random
    # 'metric': {
    #   'name': 'accuracy',
    #   'goal': 'maximize'   
    # },
    'parameters': {
        # testing epochs variance effect
        'optimizer': {
            'values': ['Adam', 'RMSprop']
        },
        # testing optimizer variance effect
        'dropout_1': {
            'values': [True, False]
        },
        'dropout_2': {
            'values': [True, False]
        },
        'dropout_3': {
            'values': [True, False]
        },
        'dropout_4': {
            'values': [True, False]
        },
        # testing the addition of dropout layers
        'dropout_value': {
            'values': [0.25, 0.5]
        }
    }
}






def train():
    # Default values for hyper-parameters we're going to sweep over
    defaults = dict(
        optimizer = 'Adam',
        dropout_2 = 0.25,
        dropout_3 = 0.25
    )

    
    # Initialize a new wandb run
    wandb.init(project="sign_language_recognition", config= defaults)

    # Config is a variable that holds and saves hyperparameters and inputs
    config = wandb.config


    # neural network layers    
    lenet5 = Sequential()
    lenet5.add(Conv2D(filters=6, kernel_size=(5,5), strides=1, kernel_initializer='he_uniform', kernel_regularizer=keras.regularizers.l2(0.01),  activation='relu', input_shape=(80,80,3), padding='same'))
    lenet5.add(BatchNormalization())
    lenet5.add(MaxPooling2D(pool_size=(3,3))) 
    lenet5.add(Conv2D(filters=16, kernel_size=(5,5), strides=1,  activation='relu', padding='valid'))
    lenet5.add(BatchNormalization())
    lenet5.add(MaxPooling2D(pool_size=(3,3))) 
    if config.dropout_2==0.25:
        lenet5.add(Dropout(0.25))
    else:
        lenet5.add(Dropout(0.5))
    lenet5.add(Flatten()) 
    lenet5.add(Dense(units=120, activation='relu'))
    if config.dropout_3==0.25:
        lenet5.add(Dropout(0.25))
    else:
        lenet5.add(Dropout(0.5))
    lenet5.add(Dense(units=84, activation='relu'))
    lenet5.add(Dense(units=26, activation='softmax'))


    # testing different loss functions
    loss = 'sparse_categorical_crossentropy'

    # Instantiate an accuracy metric.
    accuracy = Accuracy()

    if config.optimizer == 'Adam':
        optimizer = Adam()

    if config.optimizer == 'RMSprop':
        optimizer = RMSprop()

    lenet5.compile(optimizer=optimizer, loss=loss, metrics=['accuracy']) 

    # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5) 
    lenet5.fit(x_train, y_train, 
               batch_size=32,
               epochs=100,
               validation_data=(x_val, y_val),
               # callbacks=[es, WandbCallback()]
               callbacks=[WandbCallback()]
               )
    
# Configure the sweep – specify the parameters to search through, the search strategy, the optimization metric et all.
sweep_config = {
    'method': 'grid', #grid, random
    'metric': {
      'name': 'val_loss',
      'goal': 'minimize'   
    },
    'parameters': {
        'optimizer': {
            'values': ['Adam', 'RMSprop']
        },
        'dropout_2': {
            'values': [0.25, 0.5]
        },
        'dropout_3': {
            'values': [0.25, 0.5]
        }
    }
}





def train():
    # Default values for hyper-parameters we're going to sweep over
    defaults = dict(
        epochs = 10
    )

    
    # Initialize a new wandb run
    wandb.init(project="sign_language_recognition", config=defaults)

    # Config is a variable that holds and saves hyperparameters and inputs
    config = wandb.config


    # neural network layers    
    lenet5 = Sequential()
    lenet5.add(Conv2D(filters=6, kernel_size=(5,5), strides=1, kernel_initializer='he_uniform', kernel_regularizer=keras.regularizers.l2(0.01),  activation='relu', input_shape=(80,80,3), padding='same'))
    lenet5.add(BatchNormalization())
    lenet5.add(MaxPooling2D(pool_size=(3,3))) 
    lenet5.add(Conv2D(filters=16, kernel_size=(5,5), strides=1,  activation='relu', padding='valid'))
    lenet5.add(BatchNormalization())
    lenet5.add(MaxPooling2D(pool_size=(3,3))) 
    lenet5.add(Dropout(0.5))
    lenet5.add(Conv2D(filters=32, kernel_size=(5,5), strides=1,  activation='relu', padding='valid'))
    lenet5.add(BatchNormalization())
    lenet5.add(MaxPooling2D(pool_size=(3,3))) 
    lenet5.add(Dropout(0.25))
    lenet5.add(Flatten()) 
    lenet5.add(Dense(units=120, activation='relu'))
    lenet5.add(Dropout(0.25))
    lenet5.add(Dense(units=84, activation='relu'))
    lenet5.add(Dense(units=26, activation='softmax'))


    # testing different loss functions
    loss = 'sparse_categorical_crossentropy'

    # Instantiate an accuracy metric.
    accuracy = Accuracy()

    optimizer = Adam()

    lenet5.compile(optimizer=optimizer, loss=loss, metrics=['accuracy']) 

    # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5) 
    lenet5.fit(x_train, y_train, 
               batch_size=32,
               epochs=config.epochs,
               validation_data=(x_val, y_val),
               # callbacks=[es, WandbCallback()]
               callbacks=[WandbCallback()]
               )
    
# Configure the sweep – specify the parameters to search through, the search strategy, the optimization metric et all.
sweep_config = {
    'method': 'grid', #grid, random
    'metric': {
      'name': 'val_loss',
      'goal': 'minimize'   
    },
    'parameters': {
        'epochs': {
            'values': [10, 25, 50]
        }
    }
}
