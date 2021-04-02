# disable GPU
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#data_dir = "d:/MY/DBV50LEZIUNI"
data_dir = 'e:/MY/My Databases/MEDICAL/CEUS/UMF/DBV50LEZIUNI'

#PATIENS_TAKEN = {False = all patients, number}
#PATIENS_TAKEN = False
PATIENS_TAKEN = 11

EPOCHS = 12
#Early Stopping patience
#patience = EPOCHS/2
patience = 10

SHUFFLE = True
#SHUFFLE = False

EXPERIMENTS = 5

DISPLAY_TRAINING = False
TF_Board = False

# myModelType = {model.classic, model.custom}
# classic myModelName: any from https://keras.io/api/applications/, e.g., ResNet50, MobileNetV2, etc.
# custom myModelName: {SequentialS, Sequential12, Sequential17}

myModelType = 'model.classic'
myModelName = 'MobileNetV2'
#myModelName = 'NASNetMobile'
#myModelName ='EfficientNetB0'
#myModelName ='DenseNet121'
#myModelName = 'ResNet50'
#myModelName = 'EfficientNetB3'

#myModelType = 'model.custom'
#myModelName = 'SequentialS'
#myModelName = 'Sequential12'
#myModelName = 'Sequential17'

batch_size = 16

#color_mode must be one of {"rbg", "rgba", "grayscale"}
#color_mode = 'grayscale'
color_mode = 'rgb'
#depth must be one of {1 - for grayscale, 3 - for rgb and rgba}
#depth = 1
depth = 3

#settings for all custom Sequentials
#img_height = 180
#img_width = 180

#settings for all Classics
img_height = 224
img_width = 224
#weights={'imagenet', None}
weights=None
#trainable = {True, False}
trainable = True
AUG = True
PREPROC = True
