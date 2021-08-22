from PIL import Image
from tensorflow.keras.preprocessing import image
import numpy as np 

def read_image( image_path : str, size : tuple([int, int])):
    '''
    Read the  image from path
    --------------------------------------------
    image loaded to PIL (Python Imaging Library)
    --------------------------------------------
    Return : Image with resized  by size
    '''
    return(Image.open(image_path).resize(size))

def get_image_embeddings(model, object_image : image):
    '''
    load ml trained model and image from PIL
    model = trained machine learning/ deep learning model
    -----------------------------------------------------
    convert image into 3d array and added additional
    dimension for model input
    -----------------------------------------------------
    return embeddings of give image
    '''
    image_array = np.expand_dims(image.img_to_array(object_image), axis = 0)
    image_embedding = model.predict(image_array)

    return image_embedding


    

