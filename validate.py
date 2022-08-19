import tensorflow as tf
from tensorflow.keras.models import Model, load_model, save_model
from src.utils import train_generator
from src.loss import dice_coef_loss,iou,Tversky,dice_coef
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt 
BATCH_SIZE = 60
BUFFER_SIZE = 1000
im_width = 256
im_height = 256
learning_rate = 1e-4
EPOCHS=1

model = load_model("Unet.h5", compile=False)

def main():
	return 0 	

if __name__ == "__main__":
    df_test = pd.read_csv("data/df_val.csv")
    decay_rate = learning_rate / EPOCHS

    opt = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=decay_rate, amsgrad=False)
    model.compile(optimizer=opt, loss=dice_coef_loss, metrics=[iou, dice_coef, Tversky])

    test_gen = train_generator(df_test, BATCH_SIZE,
                                dict(),
                                target_size=(im_height, im_width))
    results = model.evaluate(test_gen, steps=len(df_test) / BATCH_SIZE)
    for i in range(104,105):
        img = cv2.imread(df_test['filename'].iloc[i])
        img = cv2.resize(img ,(im_height, im_width))
        img = img / 255
        img = img[np.newaxis, :, :, :]
        pred=model.predict(img)

        plt.figure(figsize=(12,12))
        plt.subplot(1,3,1)
        plt.imshow(np.squeeze(img))
        plt.title('Original Image')
        plt.subplot(1,3,2)
        plt.imshow(np.squeeze(cv2.imread(df_test['mask'].iloc[i])))
        plt.title('Original Mask')
        plt.subplot(1,3,3)
        plt.imshow(np.squeeze(pred) > .5)
        plt.title('Prediction')
        plt.savefig("validation.jpg")

