from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, models
from numpy import dot
from numpy.linalg import norm
from annoy import AnnoyIndex

class Model():
    
    def __init__():
        bert = models.Transformer('12epoch_multilingual_model')
        pooling = models.Pooling(bert.get_word_embedding_dimension(), 'cls')
        self.model =  SentenceTransformer(modules=[bert, pooling])
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/stsb-xlm-r-multilingual')
        
        length_of_vector = 768
        self.annoy_index = AnnoyIndex(length_of_vector, 'angular')
        self.annoy_index.load('angular_sentence_transformer.annoy')

    def cos_sim(A, B):
        return dot(A, B)/(norm(A)*norm(B))

    def get_indices(question,n=5):
        output = self.model.encode(question)
        return self.annoy_index.get_nns_by_vector(output, n)





