from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, models
from numpy import dot
from numpy.linalg import norm
from annoy import AnnoyIndex

#model_path : pre-trained model directory path
#annoy_path : annoy file path
class Model(model_directory_path,annoy_path):
    def __init__(self):
        bert = models.Transformer(model_directory_path)
        pooling = models.Pooling(bert.get_word_embedding_dimension(), 'cls')
        self.model =  SentenceTransformer(modules=[bert, pooling])
        #sentence-transformers/stsb-xlm-r-multilingual
        self.tokenizer = AutoTokenizer.from_pretrained(model_directory_path)
        
        length_of_vector = 768
        self.annoy_index = AnnoyIndex(length_of_vector, 'angular')
        self.annoy_index.load(annoy_path)

    def cos_sim(A, B):
        return dot(A, B)/(norm(A)*norm(B))

    def get_indices(self,question,n=5):
        output = self.model.encode(question)
        return self.annoy_index.get_nns_by_vector(output, n)





