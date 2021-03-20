# disable GPU
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# myModelType = {model.classic, model.custom}
# classic myModelName: {ResNet50, MobileNetV2}
# custom myModelName: {Sequential12, Sequential17}

myModelType = 'model.classic'
#myModelName = 'MobileNetV2'
#myModelName = 'ResNet50'
myModelName = 'NASNetMobile'

#myModelType = 'model.custom'
#myModelName = 'Sequential12'
#myModelName = 'Sequential17'

batch_size = 16

#settings for all custom Sequentials
#img_height = 180
#img_width = 180
#depth must be one of {1 - for grayscale, 3 - for rgb and rgba}
#depth = 1

#settings for NASNetMobile
img_height = 224
img_width = 224
depth = 3
color_mode = 'rgb'

#color_mode must be one of {"rbg", "rgba", "grayscale"}
#color_mode = 'grayscale'

data_dir = "E:/MY/My Databases/MEDICAL/CEUS/UMF/DBV50LEZIUNI"

#PATIENS_TAKEN = False = all patients
#PATIENS_TAKEN = False
PATIENS_TAKEN = 11

EPOCHS = 40
#Early Stopping patience
patience = 20

#SHUFFLE = False
SHUFFLE = True

EXPERIMENTS = 1

DISPLAY_TRAINING = False