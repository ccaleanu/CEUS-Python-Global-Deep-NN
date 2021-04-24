import numpy as np
import tensorflow as tf
import pathlib
import time
import pickle
import random
import matplotlib.pyplot as plt
import gc
import config
import os
import sys

if config.myModelType == 'model.classic':
    myModel = __import__(config.myModelType + '.' + 'AllClassic', fromlist=['AllClassic'])
    myClassifier = getattr(myModel, 'AllClassic')
else:
    myModel = __import__(config.myModelType + '.' + config.myModelName, fromlist=[config.myModelName])
    myClassifier = getattr(myModel, config.myModelName)

# ceusimage contains custom generator for leave-one-patient implementation
from ceusutils.ceusimage import CeusImagesGenerator 
# needed for saving the best model (not the last one) 
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from ceusutils.ceusstatistics import ceusstatistics

start_time = time.time()

if config.PATIENS_TAKEN:
    total_patients = config.PATIENS_TAKEN*5
else:
    total_patients = 91

all_experiments={}
all_experimentsx={}
all_cm={}

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
BACKUP_NAME = "./Output/output" + "-" + timestamp
LOG_DIR = "./LOGS/fit/" + time.strftime('%d-%b-%Y_%H%M', t)
BEST_MODEL_PATH_FILE = './Output/best_model' + '-' + timestamp + '.h5'

cwd = os.getcwd()
if os.path.basename(cwd) != 'Global classif':
    sys.exit('Change the current dir to "Global classif"!')

data_dir_p = pathlib.Path(config.data_dir)
image_count = len(list(data_dir_p.glob('*/*.jpg')))
if image_count == 0:
    sys.exit('No data set loaded. Check the path "data_dir" from config.py!')
print("Total number of DB images:", image_count)
p_dict = CeusImagesGenerator.patients_sets(config.data_dir)
print("Lesions and patients:")
for item in p_dict.items():
    print(item)
print("=============================================")
count_processed_patients = 0
for nrexp in range(config.EXPERIMENTS):
    # due to possibly shuffle, p_dict shoud not be removed from here  
    p_dict = CeusImagesGenerator.patients_sets(config.data_dir)
    x_dict = p_dict.copy()
    y_true = []
    y_pred = []
    cm = []
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
            start_time_patient = time.time()
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
            
            #es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=config.patience)
            es = EarlyStopping(monitor='val_accuracy', mode='auto', verbose=1, patience = config.patience)
            mc = ModelCheckpoint(BEST_MODEL_PATH_FILE, monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
            
            # tensorboard just for the first patient of each lesion
            if (config.TF_Board and current_index == 1 and nrexp+1 ==1):
                tb = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1)
                history = model.fit(
                    train_ds,
                    validation_data=val_ds,
                    epochs=config.EPOCHS,
                    verbose=1,
                    callbacks=[tb, mc, es]
                )
            else:
                history = model.fit(
                    train_ds,
                    validation_data=val_ds,
                    epochs=config.EPOCHS,
                    verbose=1,
                    callbacks=[mc, es]
                ) 
            
            # simple vote - used, if hard vote fails - do not comment
            ## best_val.append(mc.best)
            #bst = float(np.round(max(history.history['val_accuracy']), 2))
            bst = model.evaluate(val_ds)[1]
            best_val.append(bst)

            # load the saved model
            model = tf.keras.models.load_model(BEST_MODEL_PATH_FILE)
            #hard vote implementation
            predictions = model.predict((val_ds))
            pred = tf.argmax(predictions, axis=-1)
            hist = tf.histogram_fixed_width(pred, [0,4], nbins=5)
            # test the current prediction vs right answe
            if tf.argmax(hist) == list(p_dict.keys()).index(lesion):
                x_val.append(1)
            else:
                # accuracy over presented pictures, doesn't match global confusion matrix (gcm)
                x_val.append(bst)
                # accuracy over patients, does match gcm
                # x_val.append(0)          

            y_true.append(list(p_dict.keys()).index(lesion))
            y_pred.append(tf.argmax(hist))    
            cm = (tf.math.confusion_matrix(y_true, y_pred, num_classes=5)).numpy()

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
            count_processed_patients = count_processed_patients + 1
            ETA = round((time.time() - start_time_patient)/60,2)
            print('Time per patient [min]:', ETA)
            percent = round((100*count_processed_patients)//(total_patients*config.EXPERIMENTS), 2)
            remaining = round(ETA*(config.EXPERIMENTS*total_patients-count_processed_patients),2)
            print(percent, '% completed,', remaining, ' mins left')                  
            print("=========End one out==========")
        
        p_dict[lesion]=[p_dict[lesion], best_val]
        x_dict[lesion]=[x_dict[lesion], x_val]
        print("=============End one lesion=======================")
    all_experiments[str(nrexp)] = p_dict
    all_experimentsx[str(nrexp)] = x_dict
    all_cm[str(nrexp)] = cm
    print("===============End one experiment=============================")
#compute global confusion matrix
gcm = sum(x for x in all_cm.values())
print("======================End all experimets=================================")

f = open(BACKUP_NAME + "simple-vote.pkl","wb")
pickle.dump(all_experiments, f)
f.close()

r = open("./Current/config.py", "r") 
f = open(BACKUP_NAME + "simple-vote.txt","w")
f.write( str(all_experiments))
f.write("\n")
for line in r:
    f.write(line)
f.close()
r.close()

f = open(BACKUP_NAME + "hard-vote.pkl","wb")
pickle.dump([all_experimentsx, all_cm], f)
f.close()

r = open("./Current/config.py", "r") 
f = open(BACKUP_NAME + "hard-vote.txt","w")
f.write( str(all_experimentsx))
f.write("\n")
f.write( str(all_cm))
f.write("\n")
f.write( str(gcm))
f.write("\n")
for line in r:
    f.write(line)
f.write("\n")
f.write('Time per patient [min]: ' + str(ETA))
f.write("\n")
f.write("Total elapse time [min]: " + str((time.time() - start_time)//60))
f.write("\n")
f.write(sys.executable)
f.close()
r.close()

print("Total elapse time: ", (time.time() - start_time)//60, "minutes")

ceusstatistics(BACKUP_NAME + "hard-vote", False)