'''
This example code shows how to conduct adversarial training to improve the robustness of a sentiment analysis model.
The most important part is the "attack()" function, in which adversarial examples are easily generated with an API "attack_eval.generate_adv()" 
'''
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

# Choose SST-2 as the dataset
def prepare_data():
    vocab = {
        "<UNK>": 0,
        "<PAD>": 1
    }
    train, valid, test = OpenAttack.loadDataset("SST")
    tp = OpenAttack.text_processors.DefaultTextProcessor()
    for dataset in [train, valid, test]:
        for inst in dataset:
            inst.tokens = list(map(lambda x:x[0], tp.get_tokens(inst.x)))
            for token in inst.tokens:
                if token not in vocab:
                    vocab[token] = len(vocab)
    return train, valid, test, vocab

def prepare_data_self(dataset_train=None, dataset_valid=None, dataset_test=None):
    vocab = {
        "<UNK>": 0,
        "<PAD>": 1
    }

    f1 = codecs.open(dataset_train, 'rb')
    train = pickle.load(f1)
    f1.close()

    f2 = codecs.open(dataset_valid, 'rb')
    valid = pickle.load(f2)
    f2.close()

    f3 = codecs.open(dataset_test, 'rb')
    test = pickle.load(f3)
    f3.close()
    # print(train)
    # print(valid)
    # print(test)
    tp = OpenAttack.text_processors.DefaultTextProcessor()

    for dataset in [train, valid, test]:
        for inst in dataset:
            inst.tokens = list(map(lambda x: x[0], tp.get_tokens(inst.x)))
            for token in inst.tokens:
                if token not in vocab:
                    vocab[token] = len(vocab)
    return train, valid, test, vocab

# Batch data
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

# Train the victim model for one epoch 
def train_epoch(model, dataset, vocab, batch_size=128, learning_rate=5e-3):
    dataset = dataset.shuffle().reset_index()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    avg_loss = 0
    for start in range(0, len(dataset), batch_size):
        train_x, train_y = make_batch(dataset[start: start + batch_size], vocab)
        train_x = train_x.to(device)
        train_y = train_y.to(device)
        pred = model(train_x)
        loss = criterion(pred.log(), train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
    return avg_loss / len(dataset)

# Train the victim model and conduct evaluation
def train_model(model, data_train, data_valid, vocab, num_epoch=10, save=False):
    mx_acc = None
    mx_model = None
    for i in range(num_epoch):
        loss = train_epoch(model, data_train, vocab)
        clsf = OpenAttack.PytorchClassifier(model, word2id=vocab)
        accuracy = len(data_valid.eval(clsf).correct()) / len(data_valid)
        print("Epoch %d: loss: %lf, accuracy %lf" % (i, loss, accuracy))
        if mx_acc is None or mx_acc < accuracy:
            mx_model = model.state_dict()
    model.load_state_dict(mx_model)
    if save:
        torch.save(model.state_dict(),'./my_data/model/model_test.pkl')
    return model

# Launch adversarial attacks and generate adversarial examples 
def attack(classifier, dataset, dataset_valid,attacker = OpenAttack.attackers.PWWSAttacker()):
    attack_eval = OpenAttack.attack_evals.DefaultAttackEval(
        attacker = attacker,
        classifier = classifier,
        success_rate = True
    )
    correct_samples = dataset.eval(classifier).correct()
    # accuracy = len(correct_samples) / len(dataset)
    accuracy = len(dataset_valid.eval(classifier).correct()) / len(dataset_valid)

    adversarial_samples = attack_eval.generate_adv(correct_samples)
    attack_success_rate = attack_eval.get_result()["Attack Success Rate"]

    print("Accuracy: %lf%%\nAttack success rate: %lf%%" % (accuracy * 100, attack_success_rate * 100))

    tp = OpenAttack.text_processors.DefaultTextProcessor()
    for inst in adversarial_samples:
        inst.tokens = list(map(lambda x:x[0], tp.get_tokens(inst.x)))

    return adversarial_samples

def main(attacker,adversarial_samples=None):
    print("Loading data")
    train, valid, test, vocab = prepare_data() # Load dataset
    # pickle_file = codecs.open('./my_data/vocab/vocab_SST.pkl', 'wb')
    # pickle.dump(vocab, pickle_file)
    # pickle_file.close()
    model = make_model(len(vocab)) # Design a victim model

    print("Training")
    trained_model = train_model(model, train, valid, vocab) # Train the victim model

    print("Generating adversarial samples (this step will take dozens of minutes)")
    clsf = OpenAttack.PytorchClassifier(trained_model, word2id=vocab) # Wrap the victim model


    if adversarial_samples==None:
        adversarial_samples = attack(clsf, train, test, attacker=attacker)  # Conduct adversarial attacks and generate adversarial examples
        print('Saveing adversarial_samples')
        pickle_file = codecs.open('./my_data/ad_data/adversarial_samples_test.pkl', 'wb')
        pickle.dump(adversarial_samples, pickle_file)
        pickle_file.close()
    else:
        print('Loading adversarial_samples')
        adversarial_samples = pickle.load(codecs.open(adversarial_samples, 'rb'))

    print("Adversarially training classifier")
    finetune_model = train_model(trained_model, train + adversarial_samples, valid, vocab, save=True) # Retrain the classifier with additional adversarial examples

    print("Testing enhanced model (this step will take dozens of minutes)")
    attack(clsf, train, test, attacker=attacker) # Re-attack the victim model to measure the effect of adversarial training

if __name__ == '__main__':
    main(OpenAttack.attackers.PWWSAttacker())