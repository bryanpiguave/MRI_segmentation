BATCH_SIZE = 64
BUFFER_SIZE = 1000
im_width = 256
im_height = 256
learning_rate = 1e-4
EPOCHS = 100
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
import argparse
from src.utils import train_generator
from src.models import unet
from src.loss import dice_coef_loss,iou,Tversky,dice_coef
import pandas as pd

parser= argparse.ArgumentParser()
parser.add_argument("--train-csv",default="data/df_train.csv")
parser.add_argument("--val_csv",default="data/df_val.csv")
parser.add_argument("--checkpoint", type=str, help="path where")
args = parser.parse_args()

df_train=pd.read_csv(args.train_csv)
df_val =pd.read_csv(args.val_csv)

def main():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    train_generator_args = dict(rotation_range=0.2,
                            width_shift_range=0.05,
                            height_shift_range=0.05,
                            shear_range=0.05,
                            zoom_range=0.05,
                            horizontal_flip=True,
                            fill_mode='nearest')


    train_gen = train_generator(df_train, BATCH_SIZE,
                                    train_generator_args,
                                    target_size=(im_height, im_width))
        
    test_gener = train_generator(df_val, BATCH_SIZE,
                                    dict(),
                                    target_size=(im_height, im_width))
        
    #callbacks
    earlystopping = EarlyStopping(monitor='val_loss',
                                mode='min', 
                                verbose=1, 
                                patience=20
                                )
    # save the best model with lower validation loss

    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                mode='min',
                                verbose=1,
                                patience=10,
                                min_delta=0.0001,
                                factor=0.2
                                )


    decay_rate = learning_rate / EPOCHS
    opt = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=decay_rate, amsgrad=False)
    model = unet()

    model.compile(optimizer=opt, loss=dice_coef_loss, metrics=[iou, dice_coef, Tversky])

    callbacks = [ModelCheckpoint('Unet.hdf5', verbose=1, save_best_only=True), earlystopping,reduce_lr]

    history = model.fit(train_gen,
                        steps_per_epoch=len(df_train) / BATCH_SIZE, 
                        epochs=EPOCHS, 
                        callbacks=callbacks,
                        validation_data = test_gener,
                        validation_steps=len(df_val) / BATCH_SIZE)


    model.save("Unet.h5")

if __name__=="__main__":
    main()