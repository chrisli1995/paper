"""
    see `tutorial - pytorch <https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html#define-the-model>`__
"""

import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, embedding_size=200, hidden_size=256):
        """
        :param input_dim: 词数量
        :param emb_dim: 词嵌入维度
        :param hid_dim: 隐藏层维度
        :param dropout: 随机失活概率
        :param device: 在什么环境下运行
        :param n_layer: lstm的层数
        :param rnn_type: rnn类型
        :param bidirectional: 是否是双向rnn
        """
        super(RNN, self).__init__()
        self.n_layer = 1
        self.hid_dim = hidden_size
        self.bidirectional = True
        self.hid_num = 1 + int(self.bidirectional)
        self.embedding_size = embedding_size
        self.dropout = nn.Dropout(0.5)
        self.rnn_type = "LSTM"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.rnn_type == "GRU":
            self.rnn = nn.GRU(self.embedding_size, self.hid_dim, bidirectional=self.bidirectional,
                              num_layers=self.n_layer, batch_first=True)
        if self.rnn_type == "LSTM":
            self.rnn = nn.LSTM(self.embedding_size, self.hid_dim, bidirectional=self.bidirectional,
                               num_layers=self.n_layer, batch_first=True)

    def forward(self, inputs):
        # src = [input_len, batch_size]
        #         embedded = self.dropout(self.word_embedding(inputs.long()))
        # embedded = [input_len, batch_size, emb_dim]
        hidden = self.init_hidden(inputs.size(0))

        if self.rnn_type == "LSTM" and self.bidirectional:
            output, (hidden, cell_state) = self.rnn(inputs, hidden)
            # outputs = [input_len, batch_size, hid dim * n_directions]
            # hidden_state = [n_layers * n_directions, batch_size, hid_dim]
            # cell_state = [n_layers * n_directions,batch_size,hid_dim]
            # outputs are always from the top hidden layer
        else:
            output = ""
            hidden_state = ""

        hidden = hidden.permute(1, 0, 2)
        # hidden = [batch_size,n_layers * n_directions, hid_dim]

        hidden_forward = hidden[:, 0:1, :]
        hidden_back = hidden[:, 1:2, :]

        sentence__representation = torch.cat((hidden_forward, hidden_back), dim=2).squeeze(dim=1)
        #         [batch_size, hid_dim*2]
        return sentence__representation

    def init_hidden(self, batch_size):
        if self.bidirectional:
            return (torch.rand(self.n_layer * 2, batch_size, self.hid_dim).to(self.device),
                    torch.rand(self.n_layer * 2, batch_size, self.hid_dim).to(self.device))
        else:
            return (torch.rand(self.n_layer, batch_size, self.hid_dim).to(self.device),
                    torch.rand(self.n_layer, batch_size, self.hid_dim).to(self.device))


class TextSentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, num_class=2, hidden_size=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # self.fc1 = nn.Linear(embed_dim, 10)
        self.fc1 = RNN(embedding_size=embed_dim, hidden_size=hidden_size)
        # self.bn = nn.BatchNorm1d(10, eps=1e-05, momentum=0.1, affine=True)
        self.fc2 = nn.Linear(hidden_size * 2, num_class)
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        # self.fc1.weight.data.uniform_(-initrange, initrange)
        # self.fc1.bias.data.zero_()
        self.fc2.weight.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.zero_()

    def forward(self, text):
        self.result = []
        # embedded = self.embedding(text, None)
        # embedded = self.fc2(self.fc1(self.embedding(text, None)))
        sentence_embedding = self.embedding(text)

        fc1 = self.fc1(sentence_embedding)
        # fc1_bn=self.bn(fc1)
        fc2 = self.fc2(fc1)
        self.result.append(fc1)
        return self.softmax(fc2)

    def get_fc_result(self):
        return self.result
