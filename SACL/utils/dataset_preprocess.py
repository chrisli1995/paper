import pickle
import codecs
import OpenAttack

def add_original_token(dataset):
    if isinstance(dataset,str):
        f=codecs.open(dataset,'rb')
        dataset=pickle.load(f)

    dataset=dataset.data()
    print(dataset)
    for data in dataset:
        if 'original' in data['meta'].keys():
            data['meta']['original_tokens']=OpenAttack.DefaultTextProcessor().get_tokens(sentence=data['meta']['original'])

        else:
            data.data()['meta']['original_tokens']=[-1]

    print(dataset)
    return OpenAttack.utils.Dataset(dataset)
            
if __name__ == '__main__':
    dataset=add_original_token('/home/lwd/cstools/project/TextAattck/my_data/ad_data/adversarial_samples.pkl')
    #
    # with codecs.open('/home/lwd/cstools/project/TextAattck/my_data/ad_data/adversarial_samples_new.pkl','wb') as f:
    #     pickle.dump(dataset,f)
    # f = codecs.open('/home/lwd/cstools/project/TextAattck/my_data/ad_data/adversarial_samples_new.pkl', 'rb')
    # dataset = pickle.load(f)
    # for data in dataset.data():
    #     print(data)

    # dataset1 = OpenAttack.utils.Dataset(dataset.data())
    # print(dataset1)



