
from src import calc_similarity as cs
image_1 = './data/image.jpg'
image_2 = './data/cat.jpg'

if __name__ == '__main__':
    similarity_score = cs.get_similarity_score(image_1, image_2)

print(similarity_score)