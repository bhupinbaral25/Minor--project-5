
from src import calc_similarity as cs
image_1 = './data/image.jpg'
image_2 = './data/cat.jpg'
image_3 = './data/dog.jpg'

if __name__ == '__main__':
    similarity_score = cs.get_similarity_score(image_1, image_2)
    similarity_score2 = cs.get_similarity_score(image_1, image_3)

print('Similarity_Score of disimilar image =', similarity_score)
print('Similarity_Score of similar image =', similarity_score2)
