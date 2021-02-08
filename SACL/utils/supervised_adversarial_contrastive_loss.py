import torch
import numpy as np

class SupervisedAdversarialContrastiveLoss():
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
    def __call__(self, pred, labels, train_embeddings, temperature=20,*args, **kwargs):
        if self.alpha!=1:
            # print(len(train_embeddings))
            original_embeddings=train_embeddings[:int(len((train_embeddings))/2)]
            # print(original_embeddings[0],type(original_embeddings[0]),len(original_embeddings))
            # print(len(original_embeddings))
            adversarial_embeddings=train_embeddings[int(len((train_embeddings))/2):]
            # print(adversarial_embeddings[0],type(adversarial_embeddings[0]),len(adversarial_embeddings))
            # print(len(adversarial_embeddings))

            criterion = torch.nn.NLLLoss()
            # Cross Entropy
            # labels_fb=labels
            # labels_fb=torch.cat((labels_fb,labels_fb))
            # loss_ce = criterion(pred.log(), labels_fb)
            loss_ce = criterion(torch.split(pred,len(original_embeddings))[0].log(), labels)

            sum_label = self.sum_label(labels)
            # print(sum_label)
            # print(sum_label.keys)
            # Adversarial contrastive
            loss_ac = 0
            for i in range(len(original_embeddings)):
                label_index = self.same_label_index(labels,i)
                # print(original_embeddings[i].shape)
                # print(original_embeddings[i].t(),original_embeddings[i].t().shape)
                relation_original_adversarial=torch.exp(torch.dot(self.norm_vector(original_embeddings[i],p=2), self.norm_vector(adversarial_embeddings[i],p=2))/temperature)
                # relation_data_same_label=self.relation_data_same_label(original_embeddings[i],original_embeddings,label_index,temperature)
                relation_data_not_same_label=self.relation_data_not_same_label(original_embeddings[i],original_embeddings,label_index,temperature)
                # print(relation_original_adversarial,type(relation_original_adversarial))# tensor(1.0101, device='cuda:0', grad_fn=<ExpBackward>) <class 'torch.Tensor'>
                # print(relation_data_same_label,type(relation_data_same_label))# tensor(58.9858, device='cuda:0', grad_fn=<AddBackward0>) <class 'torch.Tensor'>
                # print((relation_original_adversarial/relation_data_same_label).log())# tensor(-4.0673, device='cuda:0', grad_fn=<LogBackward>)
                # print(sum_label[int(labels[i])])
                # print('roa:',relation_original_adversarial,'rdsl:',relation_data_same_label/(sum_label[int(labels[i])]-1),'all:',(relation_original_adversarial/(relation_data_same_label/(sum_label[int(labels[i])]-1))).log())
                # loss_ac+=-((relation_original_adversarial/(relation_data_same_label/(sum_label[int(labels[i])]-1))).log()/(sum_label[int(labels[i])]-1))
                loss_ac += -((relation_original_adversarial / (
                            relation_data_not_same_label )).log() / (
                                         sum_label[int(labels[i])] - 1))

            return self.alpha*loss_ce+(1-self.alpha)*loss_ac
        else:
            original_embeddings = train_embeddings[:int(len((train_embeddings)) / 2)]
            adversarial_embeddings = train_embeddings[int(len((train_embeddings)) / 2):]
            criterion = torch.nn.NLLLoss()
            labels_fb=labels
            labels_fb=torch.cat((labels_fb,labels_fb))
            loss_ce = criterion(pred.log(), labels_fb)
            return loss_ce


    #
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