from sentence_transformers import SentenceTransformer
from tensorflow.keras.models import load_model

text_embeding_model = SentenceTransformer('Fine_tunned_embeding_model')
classification_model = load_model('classification_model.keras')

class analyser : 
    def __init__( self ):
        self.embeder = text_embeding_model
        self.classifier = classification_model
        self.sentiments = [ 'negative' ,'neutral' ,'positive' ]
    
    def get_response( self ,text ):
        emb = self.embeder.encode( text )
        prob_dist = self.classifier.predict(emb.reshape(1,-1))[ 0 ]
        sentiment = self.sentiments[prob_dist.argmax()]
        sent_prob_dist = {'negative': prob_dist[0] , 'neutral' : prob_dist[1] ,'positive' : prob_dist[2] }
        return sentiment ,sent_prob_dist