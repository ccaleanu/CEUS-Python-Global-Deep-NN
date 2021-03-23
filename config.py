# disable GPU
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

data_dir = "E:/MY/My Databases/MEDICAL/CEUS/UMF/DBV50LEZIUNI"

#PATIENS_TAKEN = False = all patients
#PATIENS_TAKEN = False
PATIENS_TAKEN = 11

EPOCHS = 40
#Early Stopping patience
patience = 20

#SHUFFLE = True
SHUFFLE = False

EXPERIMENTS = 1

DISPLAY_TRAINING = False

# myModelType = {model.classic, model.custom}
# classic myModelName: {ResNet50, MobileNetV2}
# custom myModelName: {Sequential12, Sequential17}

myModelType = 'model.classic'
#myModelName = 'MobileNetV2'
#myModelName = 'NASNetMobile'
#myModelName ='EfficientNetB0'
#myModelName ='DenseNet121'
myModelName = 'ResNet50'

#myModelType = 'model.custom'
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
weights='imagenet'
#trainable = {True, False}
trainable = False
