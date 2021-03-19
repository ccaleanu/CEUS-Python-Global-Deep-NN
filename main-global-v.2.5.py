import numpy as np
import tensorflow as tf
import pathlib
import time
import pickle
import random
import matplotlib.pyplot as plt
import gc
import config

myModel = __import__(config.myModelType + '.' + config.myModelName, fromlist=[config.myModelName])
myClassifier = getattr(myModel, config.myModelName)

# ceusimage contains custom generator for leave-one-patient implementation
from ceusutils.ceusimage import CeusImagesGenerator 
# needed for saving the best model (not the last one) 
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

start_time = time.time()

all_experiments={}
all_experimentsx={}

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

AUTOTUNE = tf.data.experimental.AUTOTUNE

t = time.localtime()
timestamp = time.strftime('%d-%b-%Y_%H%M', t)
BACKUP_NAME = "../Output/output" + "-" + timestamp
LOG_DIR = "../LOGS/fit/" + time.strftime('%d-%b-%Y_%H%M', t)

data_dir_p = pathlib.Path(config.data_dir)
image_count = len(list(data_dir_p.glob('*/*.jpg')))
print("Total number of DB images:", image_count)
p_dict = CeusImagesGenerator.patients_sets(config.data_dir)
print("Lesions and patients:")
for item in p_dict.items():
    print(item)
print("=============================================")

for nrexp in range(config.EXPERIMENTS):
    # due to possibly shuffle, p_dict shoud not be removed from here  
    p_dict = CeusImagesGenerator.patients_sets(config.data_dir)
    x_dict = p_dict.copy()
    num_classes = len(p_dict)
    print("Starting experiment", nrexp+1)
    for lesion in p_dict:
        if config.SHUFFLE:
            random.shuffle(p_dict[lesion])
        best_val = []
        x_val = []
        if config.PATIENS_TAKEN:
            p_dict[lesion]=p_dict[lesion][0:config.PATIENS_TAKEN]
            x_dict[lesion]=x_dict[lesion][0:config.PATIENS_TAKEN]
        for one_out in p_dict[lesion]:
            print("Lesion:", lesion, ", Total patients:", len(p_dict[lesion]))
            current_index = (p_dict[lesion].index(one_out))+1
            print("Index One_out:", one_out, ", Current Index:", current_index, "from:", len(p_dict[lesion]))
            
            datagen = CeusImagesGenerator(config.data_dir, one_out, lesion)
            
            train_ds = datagen.image_dataset_from_files(
                subset="training",
                seed=123,
                image_size=(config.img_height, config.img_width),
                color_mode=config.color_mode,
                batch_size=config.batch_size)

            val_ds = datagen.image_dataset_from_files(
                subset="validation",
                seed=123,
                image_size=(config.img_height, config.img_width),
                color_mode=config.color_mode,
                batch_size=config.batch_size)
            
            train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
            val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
            
            # if pathlib.Path.exists(pathlib.Path("trained_model")):
            #     model = tf.keras.models.load_model('trained_model')
            # else:
            
            model = myClassifier.build(num_classes)

            model.compile(optimizer='adam',
                            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                            metrics=['accuracy'])
            
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
            mc = ModelCheckpoint('../Output/best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
            tb = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1)
            
            # tensorboard just for the first patient of each lesion
            if (current_index == 1 and nrexp+1 ==1):
                history = model.fit(
                    train_ds,
                    validation_data=val_ds,
                    epochs=config.EPOCHS,
                    verbose=1,
                    callbacks=[es, mc, tb]
                )
            else:
                history = model.fit(
                    train_ds,
                    validation_data=val_ds,
                    epochs=config.EPOCHS,
                    verbose=1,
                    callbacks=[es, mc]
                ) 
            
            # simple vote - used, if hard vote fails - do not comment
            ## best_val.append(mc.best)
            bst = float(np.round(max(history.history['val_accuracy']), 2))
            best_val.append(bst)

            # load the saved model
            model = tf.keras.models.load_model('../Output/best_model.h5')
            #hard vote implementation
            predictions = model.predict((val_ds))
            pred = tf.argmax(predictions, axis=-1)
            hist = tf.histogram_fixed_width(pred, [0,4], nbins=5)
            # test the current prediction vs right answe
            if tf.argmax(hist) == list(p_dict.keys()).index(lesion):
                x_val.append(1)
            else:
                x_val.append(bst)          

            if config.DISPLAY_TRAINING:
                acc = history.history['accuracy']
                val_acc = history.history['val_accuracy']
                loss = history.history['loss']
                val_loss = history.history['val_loss']
                epochs_range = range(config.EPOCHS)
                plt.figure(figsize=(8, 8))
                plt.subplot(1, 2, 1)
                plt.plot(epochs_range, acc, label='Training Accuracy')
                plt.plot(epochs_range, val_acc, label='Validation Accuracy')
                plt.legend(loc='lower right')
                plt.title('Training and Validation Accuracy')
                plt.subplot(1, 2, 2)
                plt.plot(epochs_range, loss, label='Training Loss')
                plt.plot(epochs_range, val_loss, label='Validation Loss')
                plt.legend(loc='upper right')
                plt.title('Training and Validation Loss')
                """             
                plt.draw()
                plt.pause(0.001)
                plt.ion()
                """
                plt.show()

            # prevent memory leak
            del train_ds
            del val_ds
            del model
            del history
            print("Garbage collected: ", gc.collect())
            tf.keras.backend.clear_session()          
                        
            print("=========End one out==========")

        p_dict[lesion]=[p_dict[lesion], best_val]
        x_dict[lesion]=[x_dict[lesion], x_val]
        print("=============End one lesion=======================")
    all_experiments[str(nrexp)] = p_dict
    all_experimentsx[str(nrexp)] = x_dict
    print("===============End one experiment=============================")
print("======================End all experimets=================================")


f = open(BACKUP_NAME + "simple-vote.pkl","wb")
pickle.dump(all_experiments, f)
f.close()

r = open("config.py", "r") 
f = open(BACKUP_NAME + "simple-vote.txt","w")
f.write( str(all_experiments))
f.write("\n")
for line in r:
    f.write(line)
f.close()
r.close()

f = open(BACKUP_NAME + "hard-vote.pkl","wb")
pickle.dump(all_experimentsx, f)
f.close()

r = open("config.py", "r") 
f = open(BACKUP_NAME + "hard-vote.txt","w")
f.write( str(all_experimentsx))
f.write("\n")
for line in r:
    f.write(line)
f.close()
r.close()

print("Total elapse time: ", (time.time() - start_time)//60, "minutes")