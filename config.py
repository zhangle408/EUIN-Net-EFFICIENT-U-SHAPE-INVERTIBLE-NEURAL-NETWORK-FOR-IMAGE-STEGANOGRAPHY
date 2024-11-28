# Super parameters
nf = 3
gc = 32

clamp = 2.0
channels_in = 3
log10_lr = -5
lr = 10 ** log10_lr
epochs = 1000
weight_decay = 1e-5
init_scale = 0.01

lamda_reconstruction = 2
lamda_guide = 1
lamda_low_frequency = 1
device_ids = [0,1,2,3]

imageSize = 128
iters_per_epoch = 5500
# Train:
batch_size = 48
cropsize = 128
betas = (0.5, 0.999)
weight_step = 30
gamma = 0.5

# Val:
cropsize_val = 128
batchsize_val = 48
shuffle_val = False
val_freq = 1
val_total = 550

# Dataset
TRAIN_PATH = '/data/data/Imagenet2012/ILSVRC2012_img_train'
# TRAIN_PATH = '/data/data/paris/paris_train/'
# VAL_PATH = '/data/data/paris/paris_val/'
VAL_PATH = '/data/data/Imagenet2012/ILSVRC2012_img_val'
coverdir = '/data/EUIN-Net/num_1/cover/'
secretdir = '/data/EUIN-Net/num_1/secret/'
format_train = 'png'
format_val = 'png'

# Display and logging:
loss_display_cutoff = 2.0
loss_names = ['L', 'lr']
silent = False
live_visualization = False
progress_bar = False


# Saving checkpoints:

MODEL_PATH = '/data/EUIN-Net/model/'
checkpoint_on_error = False
SAVE_freq = 1

IMAGE_PATH = '/data/EUIN-Net/image/'
IMAGE_PATH_cover = IMAGE_PATH + 'cover/'
IMAGE_PATH_secret_1 = IMAGE_PATH + 'secret_1/'
IMAGE_PATH_secret_2 = IMAGE_PATH + 'secret_2/'
IMAGE_PATH_steg = IMAGE_PATH + 'steg/'
IMAGE_PATH_secret_rev_1 = IMAGE_PATH + 'secret-rev_1/'
IMAGE_PATH_secret_rev_2 = IMAGE_PATH + 'secret-rev_2/'



# Load:
suffix = 'model_checkpoint_00100.pt'
tain_next = False
trained_epoch = 0

pretrain = True
PRETRAIN_PATH = '/data/EUIN-Net/model/'
suffix_pretrain = 'model_checkpoint_00190'
