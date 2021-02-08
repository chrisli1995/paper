'''
This example code shows how to conduct adversarial training to improve the robustness of a sentiment analysis model.
The most important part is the "attack()" function, in which adversarial examples are easily generated with an API "attack_eval.generate_adv()"
'''
import OpenAttack
import torch
from utils import SupervisedAdversarialContrastiveLossNew
from utils import ATFLLoss
import pickle
import codecs
import os



# Design a feedforward neural network as the the victim sentiment analysis model
def make_model(vocab_size,num_class=2):
    """
    see `tutorial - pytorch <https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html#define-the-model>`__
    """
    from model import TextSentimentLSTM


    return TextSentimentLSTM(vocab_size, num_class=num_class)

# Choose SST-2 as the dataset
def prepare_data():
    vocab = {
        "<UNK>": 0,
        "<PAD>": 1
    }
    train, valid, test = OpenAttack.loadDataset("SST")
    # print(train)
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
# def train_epoch(model, dataset, vocab, batch_size=128, learning_rate=5e-3):
#     dataset = dataset.shuffle().reset_index()
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     model.train()
#     # torch.nn.CrossEntropyLoss()=torch.nn.logSoftmax()+torch.nn.NLLLoss()
#     criterion = torch.nn.NLLLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     avg_loss = 0
#     for start in range(0, len(dataset), batch_size):
#         if len(dataset[start: start + batch_size])>1:
#             # print(len(dataset[start: start + batch_size]))
#             # print(dataset[0].data(),type(dataset[0].data()))
#             train_x, train_y = make_batch(dataset[start: start + batch_size], vocab)
#             train_x=train_x.to(device)
#             train_y=train_y.to(device)
#             # print(train_x.shape)
#             # print(train_y)
#             # pred has been softmax
#             pred = model(train_x)
#             # print(pred.log())
#             # print('result:',model.get_fc_result(),'   ',len(model.get_fc_result()))
#             loss = criterion(pred.log(), train_y)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             avg_loss += loss.item()
#     return avg_loss / len(dataset)
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
        # print(train_x.shape)
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
def train_model(model, data_train, data_valid, vocab, num_epoch=10,learning_rate=5e-3):
    mx_acc = None
    mx_model = None
    for i in range(num_epoch):
        loss = train_epoch(model, data_train, vocab,learning_rate=learning_rate)
        clsf = OpenAttack.PytorchClassifier(model, word2id=vocab)
        accuracy = len(data_valid.eval(clsf).correct()) / len(data_valid)
        print("Epoch %d: loss: %lf, accuracy %lf" % (i, loss, accuracy))
        if mx_acc is None or mx_acc < accuracy:
            mx_model = model.state_dict()
    model.load_state_dict(mx_model)
    return model

# Batch data contained adversarial data
def make_batch_adversarial(data, vocab):
    batch_x = [
        [
            vocab[token] if token in vocab else vocab["<UNK>"]
                for token in inst.data()['meta']['tokens']
        ] for inst in data
    ]
    # print(data[0].data()['meta']['original'])

    batch_original = [
        [
            vocab[token] if token in vocab else vocab["<UNK>"]
            for token,_ in OpenAttack.DefaultTextProcessor().get_tokens(sentence=inst.data()['meta']['original'])
        ]if 'original' in inst.data()['meta'].keys() else [1] for inst in data
    ]

    max_len=max([max(len(x) for x in batch_x),max(len(xa) for xa in batch_original)])

    batch_x = [
        sentence + [vocab["<PAD>"]] * (max_len - len(sentence))
            for sentence in batch_x
    ]

    batch_y = [
        inst.data()['y_orig'] for inst in data
    ]

    batch_original_index = [
        1 if sentence[0]!=1 else 0
        for sentence in batch_original
    ]
    # print(batch_original_index)

    batch_original = [
        sentence + [vocab["<PAD>"]] * (max_len - len(sentence))
        for sentence in batch_original
    ]

    # print(torch.LongTensor(batch_x).shape)
    # print(torch.LongTensor(batch_y).shape)
    # print(torch.LongTensor(batch_original).shape)
    # print('-------------------------------------------------------')
    return torch.LongTensor(batch_x), torch.LongTensor(batch_y), torch.LongTensor(batch_original), batch_original_index

# Train the victim model by adversarial data for one epoch
def train_epoch_adversarial(model, dataset_adversarial, vocab, alpha,batch_size=128, learning_rate=5e-3, dataset_origin=None):
    dataset = (dataset_origin+dataset_adversarial).shuffle().reset_index()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    # torch.nn.CrossEntropyLoss()=torch.nn.logSoftmax()+torch.nn.NLLLoss()
    # criterion = torch.nn.NLLLoss()
    AdversarialContrastiveLoss=SupervisedAdversarialContrastiveLossNew(alpha)
    # AdversarialContrastiveLoss=ATFLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    avg_loss = 0
    for start in range(0, len(dataset), batch_size):
        # print(dataset_adversarial[0].data(),type(dataset_adversarial[0].data()))
        train_x, train_y, train_x_original, batch_original_index = make_batch_adversarial(dataset[start: start + batch_size], vocab)
        # print('train_x_original:',train_x_original)
        # print('batch_original_index',batch_original_index)
        train_x=train_x.to(device)
        train_y = train_y.to(device)
        train_x_original = train_x_original.to(device)
        # pred has been softmax
        # print(len(train_x_original+train_x_adversarial),type(train_x_original),type(train_x_adversarial))
        # pred = model(torch.cat((train_x,train_x_original),0))
        # print(pred,type(pred)) # <class 'torch.Tensor'>
        # x_embeding=model.get_fc_result()
        pred = model(train_x)
        x_embeding1 = model.get_fc_result()
        pred1 = model(train_x_original)
        x_embeding2 = model.get_fc_result()
        x_embeding=x_embeding1
        x_embeding.append(x_embeding2[0])
        # print(x_embeding[0],type(x_embeding[0])) # <class 'torch.Tensor'>
        # print(x_embeding[0].shape)
        loss = AdversarialContrastiveLoss(pred, train_y, x_embeding, adversarial_index=batch_original_index, pred1=pred1)
        # print(loss) # <class 'torch.Tensor'>
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
    return avg_loss / len(dataset_adversarial)

# Train the victim model and conduct evaluation
def train_model_adversarial(model, dataset_adversarial, data_valid, vocab, alpha,num_epoch=10, dataset_origin=None,learning_rate=5e-3):
    mx_acc = None
    mx_model = None
    for i in range(num_epoch):
        loss = train_epoch_adversarial(model,dataset_adversarial, vocab, dataset_origin=dataset_origin,alpha=alpha,learning_rate=learning_rate)
        clsf = OpenAttack.PytorchClassifier(model, word2id=vocab)
        accuracy = len(data_valid.eval(clsf).correct()) / len(data_valid)
        print("Epoch %d: loss: %lf, accuracy %lf" % (i, loss, accuracy))
        if mx_acc is None or mx_acc < accuracy:
            mx_model = model.state_dict()
    model.load_state_dict(mx_model)
    return model


# Launch adversarial attacks and generate adversarial examples
def attack(classifier, dataset, dataset_valid, attacker = OpenAttack.attackers.PWWSAttacker()):
    attack_eval = OpenAttack.attack_evals.DefaultAttackEval(
        attacker = attacker,
        classifier = classifier,
        success_rate = True
    )
    correct_samples = dataset.eval(classifier).correct()
    accuracy = len(dataset_valid.eval(classifier).correct()) / len(dataset_valid)

    adversarial_samples = attack_eval.generate_adv(correct_samples)
    print(len(adversarial_samples))
    attack_success_rate = attack_eval.get_result()["Attack Success Rate"]

    print("Accuracy: %lf%%\nAttack success rate: %lf%%" % (accuracy * 100, attack_success_rate * 100))

    tp = OpenAttack.text_processors.DefaultTextProcessor()
    for inst in adversarial_samples:
        inst.tokens = list(map(lambda x:x[0], tp.get_tokens(inst.x)))

    return adversarial_samples

def main(attacker,adversarial_samples=None, alpha=1, cnn=False):
    print("Loading data")
    # train, valid, test, vocab = prepare_data() # Load dataset
    # train, valid, test, vocab = prepare_data_self('./my_data/data/AGNews/AGNews_train.pkl',
    #                                               './my_data/data/AGNews/AGNews_valid.pkl',
    #                                               './my_data/data/AGNews/AGNews_test.pkl')
    train, valid, test, vocab = prepare_data_self('./my_data/data/semeval/semeval_train.pkl',
                                                  './my_data/data/semeval/semeval_valid.pkl',
                                                  './my_data/data/semeval/semeval_test.pkl')
    # train, valid, test, vocab = prepare_data_self('./my_data/data/QNLI/QNLI_train.pkl', './my_data/data/QNLI/QNLI_valid.pkl', './my_data/data/QNLI/QNLI_test.pkl')
    # print(vocab)
    model = make_model(len(vocab),num_class=19) # Design a victim model
    # print('train:',train[0].data())train: {'x_orig': 'a stirring , funny and finally transporting re-imagining of beauty and the beast and 1930s horror films', 'y_orig': 1, 'idx': 0, 'meta': {'tokens': ['a', 'stirring', ',', 'funny', 'and', 'finally', 'transporting', 're', '-', 'imagining', 'of', 'beauty', 'and', 'the', 'beast', 'and', '1930s', 'horror', 'films']}}

    print("Training")
    trained_model = train_model(model, train, valid, vocab,num_epoch=10,learning_rate=5e-3) # Train the victim model

    print("Generating adversarial samples (this step will take dozens of minutes)")
    clsf = OpenAttack.PytorchClassifier(trained_model, word2id=vocab) # Wrap the victim model
    # adversarial_samples = attack(clsf, train, valid) # Conduct adversarial attacks and generate adversarial examples
    # print(adversarial_samples[0].data())# {'x_orig': ' A inspiration , amusing a eventually transport r - imagining o smasher a t animal and 1930s horror films', 'y_orig': 1, 'prd': 0, 'idx': 0, 'meta': {'original': 'a stirring , funny and finally transporting re-imagining of beauty and the beast and 1930s horror films', 'info': {'Succeed': True}, 'tokens': ['A', 'inspiration', ',', 'amusing', 'a', 'eventually', 'transport', 'r', '-', 'imagining', 'o', 'smasher', 'a', 't', 'animal', 'and', '1930s', 'horror', 'films']}}

    # print('Saveing adversarial_samples')
    # pickle_file = codecs.open('my_data/ad_data/adversarial_samples_TextFoolerAttacker.pkl', 'wb')
    # pickle.dump(adversarial_samples, pickle_file)
    # pickle_file.close()
    # print('Loading adversarial_samples')
    # adversarial_samples=pickle.load(codecs.open('./my_data/ad_data/adversarial_samples_TextFoolerAttacker.pkl','rb'))
    if not os.path.exists(adversarial_samples):
        adversarial_samples_path = adversarial_samples
        adversarial_samples = attack(clsf, train, valid, attacker=attacker)
        print('Saveing adversarial_samples')
        pickle_file = codecs.open(adversarial_samples_path, 'wb')
        pickle.dump(adversarial_samples, pickle_file)
        pickle_file.close()
    else:
        print('Loading adversarial_samples')
        adversarial_samples = pickle.load(codecs.open(adversarial_samples, 'rb'))

    print("Adversarially training classifier")
    # finetune_model = train_model(trained_model, train+adversarial_samples, valid, vocab)
    finetune_model = train_model_adversarial(learning_rate=5e-4,num_epoch=10,model=trained_model, dataset_adversarial=adversarial_samples, data_valid=valid, vocab=vocab, dataset_origin=train,alpha=alpha) # Retrain the classifier with additional adversarial examples

    print("Testing enhanced model (this step will take dozens of minutes)")
    # clsf = OpenAttack.PytorchClassifier(finetune_model, word2id=vocab)
    attack(clsf, train, valid, attacker=attacker) # Re-attack the victim model to measure the effect of adversarial training
if __name__ == '__main__':
    main(OpenAttack.attackers.PWWSAttacker(),'./my_data/ad_data/adversarial_samples_lstm_semeval.pkl', 1)