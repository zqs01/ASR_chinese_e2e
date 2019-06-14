from Predictor.Bases import BaseModel


class CnnTransformer(BaseModel):
    def __init__(self, config, vocab):
        super(CnnTransformer, self).__init__()
        self.config = config
        self.vocab = vocab

        self.cnns