from config import PATH_TRAIN_OUTPUT, PATH_TRAIN_OUTPUT
from random import shuffle


ids = next(os.walk(PATH_TRAIN_OUTPUT))[2]
print(len(ids))

shuffle(ids)

train_ids = ids[0:1500]
valid_ids = ids[1500:]

print(len(train_ids))

def get_data(ids,batch_size):
    while True:
    ids_batches = [ids[i:min(i+batch_size,len(ids))] for i in range(0, len(ids), batch_size)] 
    
    for b in range(len(ids_batches)):

        k=-1
        X = np.zeros((len(ids_batches[b]), im_height, im_width, 7), dtype=np.float32)
        y = np.zeros((len(ids_batches[b]), im_height*im_width, 255), dtype=np.float32)
        for c in range(len(ids_batches[b])):
            k=k+1
            for r in range(1,7):
                img = load_img(path_train_input + 'lc8' + ids_batches[b][c][3:-4] + '_' + str(r) + '.tif', color_mode="grayscale")
                x_img = img_to_array(img)
                x_img = resize(x_img, (128, 128), mode='constant', preserve_range=True)
                for p in range(128):
                    for q in range(128):
                    #print(x_img[p][q]/255)
                    X[k][p][q][r-1]=x_img[p][q]/255

        #k=k+1
        # Save images
        #X[k, ..., 0] = temp1 / 255  

        # Load masks

        mask = img_to_array(load_img(path_train_output+ids_batches[b][c], color_mode="grayscale"))
        mask = resize(mask, (128, 128), mode='constant', preserve_range=True)
        
        inc=-1
        for p in range(128):
            for q in range(128):
                num=int(mask[p][q])
                temp=np.zeros((255), dtype=np.float32)
                temp[num]=1
                inc=inc+1
                y[k][inc]=temp
                print

    yield X,y          


def get_data1(ids,batch_size):
    ids_batches = [ids[i:min(i+batch_size,len(ids))] for i in range(0, len(ids), batch_size)] 
    #ids_out_batches = [ids_out[j:min(j+batch_size,len(ids_out))] for j in range(0, len(ids_out), batch_size)]
    # Load images
    for b in range(len(ids_batches)):
      #print(b)
      #print(":")
      #print(ids_batches[b])
      k=-1
      X = np.zeros((len(ids_batches[b]), im_height, im_width, 7), dtype=np.float32)
      y = np.zeros((len(ids_batches[b]), im_height*im_width, 255), dtype=np.float32)
      for c in range(len(ids_batches[b])):
        k=k+1
        #temp1=np.zeros((1,7),dtype=np.float32)
        for r in range(1,7):
          img = load_img(path_train_input + 'lc8' + ids_batches[b][c][3:-4] + '_' + str(r) + '.tif', color_mode="grayscale")
          x_img = img_to_array(img)
          x_img = resize(x_img, (128, 128), mode='constant', preserve_range=True)
          for p in range(128):
            for q in range(128):
              #print(x_img[p][q]/255)
              X[k][p][q][r-1]=x_img[p][q]/255
          
        #k=k+1
        # Save images
        #X[k, ..., 0] = temp1 / 255  

        # Load masks

        mask = img_to_array(load_img(path_train_output+ids_batches[b][c], color_mode="grayscale"))
        mask = resize(mask, (128, 128), mode='constant', preserve_range=True)
        
        inc=-1
        for p in range(128):
          for q in range(128):
            num=int(mask[p][q])
            temp=np.zeros((255), dtype=np.float32)
            temp[num]=1
            inc=inc+1
            y[k][inc]=temp
            print

      return X,y





















