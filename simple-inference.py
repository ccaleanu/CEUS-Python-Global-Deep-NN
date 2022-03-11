import numpy as np
import tensorflow as tf
import pathlib
import config
import os
from tensorflow.keras.preprocessing import image
import PIL

# model path
BEST_MODEL_PATH_FILE = "./Output/Xception/from-scratch/all/best_model-20-Jul-2021_1615.h5"

# data path
data_path = "E:/MY/My Databases/MEDICAL/CEUS/UMF/DBV50LEZIUNImini/FNH/"
data_dir = pathlib.Path(data_path)
rois = list(data_dir.glob('*.jpg'))
image_count = len(rois)
print(image_count)

# # display some/all data (optional)
# im = PIL.Image.open(str(rois[10]), 'r')
# im.show()

# # load the saved model
model = tf.keras.models.load_model(BEST_MODEL_PATH_FILE)


# load all images into a list
img_height = 224
img_width = 224
images = []
for img in os.listdir(data_path):
    img = os.path.join(data_path, img)
    img = image.load_img(img, target_size=(img_width, img_height))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    images.append(img)

# stack up images list to pass for prediction
images = np.vstack(images)

#hard vote implementation
predictions = model.predict(images)
pred = tf.argmax(predictions, axis=-1)
hist = tf.histogram_fixed_width(pred, [0,4], nbins=5)
lesion = tf.argmax(hist)
labels = ['FNH', 'HCC', 'HMG', 'MHIPER', 'MHIPO']
print(labels[lesion])


