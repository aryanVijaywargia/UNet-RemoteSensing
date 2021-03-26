from make_dataset import get_data, get_data1
from utils import get_CDLRGB, view_mapped, callbacks
from train_model import get_unet
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.layers import Input
import matplotlib.pyplot as plt
%matplotlib inline


def main():

    input_img = Input((128, 128, 7), name='img')
    model = get_unet(input_img, n_filters=7, dropout=0.15, batchnorm=True)
    #model = get_unet()

    model.compile(optimizer=Adam(lr=0.1), loss="categorical_crossentropy", metrics=["categorical_accuracy"])
    model.summary()

    train_generator = get_data(train_ids[0:1], batch_size=1)
    valid_generator = get_data(valid_ids[0:1],batch_size=1)
    
    x_tr,y_tr=get_data1(train_ids[0:2], batch_size=2)

    #Load the pretrained model
    #model = models.load_model('saved_models/model3.h5')
    
    results =model.fit(train_generator, steps_per_epoch=10 , epochs=500, verbose=1, validation_data=valid_generator, validation_steps=1, callbacks=callbacks)
    
    model.evaluate(x=x_tr,y=y_tr,batch_size=1)
    print(y_tr.shape)
    
    y_test = model.predict(x_tr)
    print(y_test.shape)
    
    print(y_test[1])
    print(y_tr[1])
    
    true_img = y_tr[1].reshape(128,128,255)
    print(true_img.shape)
    print(true_img)
    
    true_img=np.argmax(true_img,axis=-1)
    
    print(true_img)
    true_img=true_img.reshape(128,128)
    true_img=true_img/255
    print(true_img)
    
    plt.imshow(true_img, cmap='gray')
    plt.show()
    
    pred_img = y_test[1].reshape(128,128,255)
    
    print(pred_img.shape)
    print(pred_img)
    
    pred_img=np.argmax(pred_img,axis=-1)

    print(pred_img)
    pred_img=pred_img.reshape(128,128)
    pred_img=pred_img/255
    print(pred_img)
    
    plt.imshow(pred_img, cmap='gray')
    plt.show()
    
    print("The color of class 5 is: ", getCDLRGB(5))
    
    view_mapped(true_img*255)
    view_mapped(pred_img*255)
    
if __name__ == '__main__':
    main();    
    
    
    
    
    
    
    
    
    
    









