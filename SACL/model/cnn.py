"""
    see `tutorial - pytorch <https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html#define-the-model>`__
"""
import torch.nn as nn
import torch


class TextCNN(nn.Module):
    def __init__(self, embed_dim=32, output_size_of_cnn=256):
        """
        :param vec_dim: 词向量的维度
        :param filter_num: 每种卷积核的个数
        :param sentence_max_size:一篇文章的包含的最大的词数量
        :param label_size:标签个数，全连接层输出的神经元数量=标签个数
        :param kernel_list:卷积核列表
        """
        super(TextCNN, self).__init__()
        chanel_num = 1
        # nn.ModuleList相当于一个卷积的列表，相当于一个list
        # nn.Conv1d()是一维卷积。in_channels：词向量的维度， out_channels：输出通道数
        # nn.MaxPool1d()是最大池化，此处对每一个向量取最大值，所有kernel_size为卷积操作之后的向量维度
        self.filter_num = 100
        self.embedding_size = embed_dim
        self.kernel_list = [2, 2, 2]
        self.output_size_of_cnn = output_size_of_cnn
        self.dropout = nn.Dropout(0.5)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(chanel_num, self.filter_num, (kernel, self.embedding_size)),
            nn.ReLU(),
            # 经过卷积之后，得到一个维度为sentence_max_size - kernel + 1的一维向量
            nn.AdaptiveMaxPool2d((1, 1))
        )
            for kernel in self.kernel_list])
        # 全连接层，因为有2个标签
        self.fc = nn.Linear(self.filter_num * len(self.kernel_list), self.output_size_of_cnn)
        # dropout操作，防止过拟合

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        # print('x:',x.shape)
        in_size = x.size(0)  # x.size(0)，表示的是输入x的batch_size
        out = [con(x) for con in self.convs]
        out = torch.cat(out, dim=1)
        out = out.view(in_size, -1)  # 设经过max pooling之后，有output_num个数，将out变成(batch_size,output_num)，-1表示自适应
        out = self.dropout(out)
        out = self.fc(out)  # nn.Linear接收的参数类型是二维的tensor(batch_size,output_num),一批有多少数据，就有多少行
        return out


class TextSentimentCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, num_class=2, output_size_of_cnn=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # self.fc1 = nn.Linear(embed_dim, 10)
        self.fc1 = TextCNN(embed_dim=embed_dim, output_size_of_cnn=output_size_of_cnn)
        # self.bn = nn.BatchNorm1d(10, eps=1e-05, momentum=0.1, affine=True)
        self.fc2 = nn.Linear(output_size_of_cnn, num_class)
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

        # [batch_size,1,sentence_len,emd_dim]
        fc1 = self.fc1(sentence_embedding)
        # fc1_bn=self.bn(fc1)
        fc2 = self.fc2(fc1)
        self.result.append(fc1)
        return self.softmax(fc2)

    def get_fc_result(self):
        return self.result
