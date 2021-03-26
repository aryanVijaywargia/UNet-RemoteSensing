import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, BatchNormalization, Activation, Dense, Dropout,Lambda, MaxPooling2D, GlobalMaxPool2D, Permute, UpSampling2D, 
from tensorflow.keras.layers import Lambda, RepeatVector, Reshape, concatenate, add, Conv2D, Conv2DTranspose, LeakyReLU, Flatten, Convolution2D, GlobalMaxPooling2D
from keras.layers import merge, core
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.activations import softmax
from tensorflow.keras.layers import Add,Multiply
from tensorflow.keras import backend as K


def conv2d_block(input_tensor, n_filters, kernel_size=3,strides=1,batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),strides=(strides, strides),padding="same", kernel_initializer="he_normal")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x

def encoder_decoder(x1,ni, kernel_size=3, batchnorm=True,times=None):
    x=GlobalMaxPooling2D()(x1)
    x=Reshape(target_shape=(1,1,times))(x)
    x=Conv2D(filters=ni//2, kernel_size=(kernel_size, kernel_size),padding="same", kernel_initializer="he_normal")(x)
    x=LeakyReLU(alpha=0.1)(x)
    x=Conv2D(filters=ni, kernel_size=(kernel_size, kernel_size),padding="same", kernel_initializer="he_normal")(x)
    x=Activation('sigmoid')(x)
    
    x2=Conv2D(filters=ni, kernel_size=(kernel_size, kernel_size),padding="same", kernel_initializer="he_normal")(x1)
    x2=Activation('sigmoid')(x2)
    
    
    x11=Multiply()([x1,x2])
    x12=Multiply()([x1,x])
    x13=Add()([x11,x12])
    
    return x13

def DownBlock(x,ni,nf, kernel_size=3, batchnorm=True,down=None):
    inp=x
    x=conv2d_block(x,nf,3,2)
    x=conv2d_block(x,nf,3)
    x=Add()([x,conv2d_block(inp,nf,3,2)])
    if down is not None:
        return encoder_decoder(x,nf, kernel_size=3, batchnorm=True,times=128)
    else:
        return x

def UpBlock(down,cross,ni,nf, kernel_size=3, batchnorm=True,down1=None):
    x=Conv2DTranspose(filters=nf, kernel_size=(3, 3),strides=(2,2),padding="same", kernel_initializer="he_normal")(down)
    print(x)
    print(cross)
    x=concatenate([x,cross])
    x=conv2d_block(x,nf,3)
    if down1 is not None:
        return encoder_decoder(x,nf, kernel_size=3, batchnorm=True,times=256)
    else:
    return x

def get_unet(input_img, n_filters=128, dropout=0.15, batchnorm=True):
    
    d1=DownBlock(input_img,7,128,3, True,12)
    print("D1")
    d2=DownBlock(d1,128,256)
    print("D2")
    d3=DownBlock(d2,256,512)
    print("D3")
    d4=DownBlock(d3,512,1024)
    print("D4")
    print(d4)
    u1=UpBlock(d4,d3,1024,512)
    print("U1")
    u2=UpBlock(u1,d2,512,256,3,True,12)
    print("U2")
    u3=UpBlock(u2,d1,256,128)
    print("U3")
    
    print("X")
    print(u3)
    
    outputs = Conv2DTranspose(filters=255, kernel_size=(3,3),strides=(2,2),padding="same", kernel_initializer="he_normal")(u3)
    print("Yes")
    outputs = core.Reshape((128*128,255))(outputs)
    outputs = core.Activation('softmax')(outputs)        
    model = Model(inputs=[input_img], outputs=[outputs])
    print("Yes1")
    
    return model




