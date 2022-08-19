import tensorflow as tf 
from glob import glob
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow_io as tfio


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def image_example(image_string, mask):
  image_shape = tfio.experimental.image.decode_tiff(image_string,index=0,name=None).shape

  feature = {
      'height': _int64_feature(image_shape[0]),
      'width': _int64_feature(image_shape[1]),
      'depth': _int64_feature(image_shape[2]),
      'image_raw': _bytes_feature(image_string),
      'segmentation':_bytes_feature(mask)
  }

  return tf.train.Example(features=tf.train.Features(feature=feature))

# Create a dictionary describing the features.
image_feature_description = {
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'depth': tf.io.FixedLenFeature([], tf.int64),
    'image_raw': tf.io.FixedLenFeature([], tf.string),
    'segmentation': tf.io.FixedLenFeature([], tf.string)
}

def _parse_image_function(example_proto):
  # Parse the input tf.train.Example proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, image_feature_description)




def main():


    train_files = []
    mask_files = glob('/data/MRI/data/lgg-mri-segmentation/kaggle_3m/*/*_mask*')

    for i in mask_files:
        train_files.append(i.replace('_mask',''))

    df = pd.DataFrame(data={"filename": train_files, 'mask' : mask_files})
    df_train, df_test = train_test_split(df,test_size = 0.20)


    df_train.to_csv("data/df_train.csv")
    df_test.to_csv("data/df_val.csv")


    

    return 0
if __name__ == "__main__":
    main()