import torch
import numpy as np

class ATFLLoss():

    def __init__(self):
        pass

    def __call__(self, pred, labels, train_embeddings, pred1=None, temperature=10, adversarial_index=None, *args, **kwargs):
        if adversarial_index is not None:
            embeddings_x=train_embeddings[0]
            embeddings_original=train_embeddings[1]
            criterion = torch.nn.NLLLoss()
            # Cross Entropy
            loss_ce = criterion(pred.log(), labels)
            loss_atfl = 0
            if adversarial_index is not None:
                for k,i in enumerate(adversarial_index):
                    if i==1:
                        loss_atfl+=torch.norm((pred1[k]-pred[k]),p=2)
            # print('loss_ce:',loss_ce,'loss_atfl:',loss_atfl)
            return loss_ce+0.1*loss_atfl
            # return loss_ce
        else:
            original_embeddings = train_embeddings[:int(len((train_embeddings)) / 2)]
            adversarial_embeddings = train_embeddings[int(len((train_embeddings)) / 2):]
            len_dataset=len(original_embeddings)
            criterion = torch.nn.NLLLoss()
            # loss_ce = criterion(torch.split(pred, len(original_embeddings))[0].log(), labels)
            labels_fb = labels
            labels_fb = torch.cat((labels_fb, labels_fb))
            loss_ce = criterion(pred.log(), labels_fb)
            loss_atfl = 0
            for i in range(len_dataset):
                loss_atfl+=torch.norm((pred[i]-pred[i+len_dataset]),p=2)

            return loss_ce+0.5*loss_atfl


    def norm_vector(self, embedding, p=2):
        norm=torch.norm(embedding,p=p)
        return embedding/norm
