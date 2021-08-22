from src import utils
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16

def get_similarity_score(image_1 : str, image_2 : str):
    '''
    Read the image from path
    --------------------------------------------------------------------------------
    image loaded to PIL (Python Imaging Library) generate embeddings of image using 
    multiple pretrain model 
    --------------------------------------------------------------------------------
    Return the similarity score of 2 different images
    '''
    similarity_score = {}
    models = {
        'VGG16' : VGG16(weights='imagenet'),
        'ResNet50' : ResNet50(weights='imagenet')
    }
    image_1 = utils.read_image(image_1, size=((224, 224)))
    image_2 = utils.read_image(image_2, size=((224, 224)))
    for model_name, model in models.items():
        image1_embedding_vector = utils.get_image_embeddings(model, image_1)
        image2_embedding_vector = utils.get_image_embeddings(model, image_2)
        similarity_score[model_name] = cosine_similarity(image1_embedding_vector, image2_embedding_vector).reshape(1,)

    return(similarity_score)