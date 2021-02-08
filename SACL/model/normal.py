"""
    see `tutorial - pytorch <https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html#define-the-model>`__
"""
import torch.nn as nn

class TextSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, num_class=2):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim, 10)
        # self.bn = nn.BatchNorm1d(10, eps=1e-05, momentum=0.1, affine=True)
        self.fc2 = nn.Linear(10, num_class)
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()


    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.zero_()
        self.fc2.weight.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.zero_()

    def forward(self, text):
        self.result = []
        # embedded = self.embedding(text, None)
        # embedded = self.fc2(self.fc1(self.embedding(text, None)))
        sentence_embedding=self.embedding(text, None)
        fc1=self.fc1(sentence_embedding)
        # fc1_bn=self.bn(fc1)
        fc2=self.fc2(fc1)
        self.result.append(fc1)
        return self.softmax(fc2)

    def get_fc_result(self):
        return self.result

