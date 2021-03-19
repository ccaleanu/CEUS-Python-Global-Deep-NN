# disable GPU
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# myModelType = {model.classic, model.custom}
# classic myModelName: {ResNet50, MobileNetV2}
# custom myModelName: {Sequential12, Sequential17}

#myModelType = 'model.classic'
#myModelName = 'MobileNetV2'
#myModelName = 'ResNet50'

myModelType = 'model.custom'
myModelName = 'Sequential12'
#myModelName = 'Sequential17'

batch_size = 16

img_height = 180
img_width = 180
#depth must be one of {1 - for grayscale, 3 - for rgb and rgba}
depth = 1
#color_mode must be one of {"rbg", "rgba", "grayscale"}
color_mode = 'grayscale'

data_dir = "E:/MY/My Databases/MEDICAL/CEUS/UMF/DBV50LEZIUNI"

#PATIENS_TAKEN = False = all patients
#PATIENS_TAKEN = False
#PATIENS_TAKEN = 11
PATIENS_TAKEN = 1

EPOCHS = 3

#SHUFFLE = False
SHUFFLE = True

EXPERIMENTS = 1

DISPLAY_TRAINING = True