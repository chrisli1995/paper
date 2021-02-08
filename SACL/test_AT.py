import OpenAttack
import torch
import pickle
import codecs

# Design a feedforward neural network as the the victim sentiment analysis model
def make_model(vocab_size):
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

    return TextSentiment(vocab_size)

def make_batch(data, vocab):
    batch_x = [
        [
            vocab[token] if token in vocab else vocab["<UNK>"]
                for token in inst.tokens
        ] for inst in data
    ]
    max_len = max( [len(inst.tokens) for inst in data] )
    batch_x = [
        sentence + [vocab["<PAD>"]] * (max_len - len(sentence))
            for sentence in batch_x
    ]
    batch_y = [
        inst.y for inst in data
    ]
    return torch.LongTensor(batch_x), torch.LongTensor(batch_y)

adversarial_samples = pickle.load(codecs.open('./my_data/ad_data/adversarial_samples_test.pkl', 'rb'))
vocab = pickle.load(codecs.open('./my_data/vocab/vocab_SST.pkl','rb'))
batch_x,batch_y=make_batch(adversarial_samples,vocab)
# print(len(batch_x))
# print(len(batch_y))
model=make_model(16200)
model.load_state_dict(torch.load('./my_data/model/model_test.pkl'))
model.eval()
print(torch.argmax(model(batch_x),1))
print(batch_y)
sum=0
for k,i in enumerate(torch.argmax(model(batch_x),1)):
    if i == batch_y[k]:
        sum+=1

    else:
        print(adversarial_samples[k])

print(sum/len(batch_y))
# clsf = OpenAttack.PytorchClassifier(model, word2id=vocab)
# accuracy = len(adversarial_samples.eval(clsf).correct()) / len(adversarial_samples)
# print("accuracy %lf" % (accuracy))