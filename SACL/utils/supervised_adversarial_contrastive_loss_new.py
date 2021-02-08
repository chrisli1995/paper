import torch
import numpy as np

class SupervisedAdversarialContrastiveLossNew():
    '''
    :param float alpha: weight for loss
    '''
    def __init__(self,alpha=0.5):
        self.alpha=alpha
    '''
    :param tensor pred: 
    :param list label: 
    :param int temperature: 
    :param tensor train_embedding: 
    **Default:** None
    '''
    def __call__(self, pred, labels, train_embeddings, temperature=10, adversarial_index=None, *args, **kwargs):
        if self.alpha!=1:
            # print(len(train_embeddings))
            embeddings_x=train_embeddings[0]
            embeddings_original=train_embeddings[1]
            # print(embeddings_x[0],type(embeddings_x[0]),len(embeddings_x))
            # print(embeddings_original[0],type(embeddings_original[0]),len(embeddings_original))


            criterion = torch.nn.NLLLoss()
            # Cross Entropy
            loss_ce = criterion(pred.log(), labels)

            sum_label = self.sum_label(labels)
            # print(sum_label)
            # print(sum_label.keys)
            # Adversarial contrastive
            loss_ac = 0
            if adversarial_index is not None:
                for k,i in enumerate(adversarial_index):
                    if i==1:
                        label_index = self.same_label_index(labels,k)
                        # print('label_index:',label_index)
                        # print(embeddings_x[k],'\n',embeddings_original[k])
                        relation_original_adversarial=torch.exp(torch.dot(self.norm_vector(embeddings_x[k],p=2), self.norm_vector(embeddings_original[k],p=2))/temperature)
                        # print(relation_original_adversarial)
                        # relation_data_same_label=self.relation_data_same_label(embeddings_x[k],embeddings_x,label_index,temperature)
                        relation_data_not_same_label=self.relation_data_not_same_label(embeddings_x[k],embeddings_x,label_index,temperature)
                        # print(relation_data_same_label)
                        # print(relation_original_adversarial,type(relation_original_adversarial))# tensor(1.0101, device='cuda:0', grad_fn=<ExpBackward>) <class 'torch.Tensor'>
                        # print(relation_data_same_label,type(relation_data_same_label))# tensor(58.9858, device='cuda:0', grad_fn=<AddBackward0>) <class 'torch.Tensor'>
                        # print((relation_original_adversarial/relation_data_same_label).log())# tensor(-4.0673, device='cuda:0', grad_fn=<LogBackward>)
                        # print(sum_label[int(labels[i])])
                        # print('roa:',relation_original_adversarial,'rdsl:',relation_data_same_label/(sum_label[int(

                        # add same label average
                        # loss_ac+=-((relation_original_adversarial/(relation_data_same_label/(sum_label[int(labels[i])]-1))).log()/(sum_label[int(labels[i])]-1))
                        # delete same label average
                        loss_ac += -((relation_original_adversarial / (relation_data_not_same_label )).log() / (sum_label[int(labels[i])] - 1))
                        #

            # print('loss_ce:',loss_ce,'loss_ac:',loss_ac)
            return self.alpha*loss_ce+(1-self.alpha)*loss_ac
        else:
            criterion = torch.nn.NLLLoss()
            # Cross Entropy
            loss_ce = criterion(pred.log(), labels)
            return loss_ce


    # divide dataset to two sub_dataset
    def div_data(self,embeddings, rate=0.5):
        len_embeddings=int(len(embeddings)*rate)

        embeddings_1 = embeddings[:len_embeddings]
        embeddings_2 = embeddings[len_embeddings:]

        return embeddings_1,embeddings_2


    def relation_data_same_label(self,target_embedding,train_embeddings,label_index,temperature):
        relation_data_same_label=0
        for i in label_index:
            # print(torch.exp(torch.dot(target_embedding,train_embeddings[i])/temperature))
            relation_data_same_label+=torch.exp(torch.dot(self.norm_vector(target_embedding,p=2),self.norm_vector(train_embeddings[i],p=2))/temperature)

        return relation_data_same_label

    def relation_data_not_same_label(self,target_embedding,train_embeddings,label_index,temperature):
        relation_data_same_label=0
        for k,i in enumerate(train_embeddings):
            if k not in label_index:
                # print(torch.exp(torch.dot(target_embedding,train_embeddings[i])/temperature))
                relation_data_same_label+=torch.exp(torch.dot(self.norm_vector(target_embedding,p=2),self.norm_vector(i,p=2))/temperature)

        return relation_data_same_label

    def same_label_index(self,labels, k):
        label_index = []
        for i,l in enumerate(labels):
            if labels[k] == l:
                label_index.append(i)

        label_index.remove(k)
        return label_index

    def sum_label(self,labels):
        sum_label={}
        for i in labels:
            sum_label.setdefault(int(i), 1)
            sum_label[int(i)]+=1

        return sum_label

    def norm_vector(self, embedding, p=2):
        norm=torch.norm(embedding,p=p)
        return embedding/norm