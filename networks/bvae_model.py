from keras import backend as K
from keras.models import Model
from keras.layers import Input, Conv2D
from keras.callbacks import Callback
from keras.optimizers import Adam

save_dir = './saved_models/'
