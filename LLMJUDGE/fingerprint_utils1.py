import json
import jieba
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import BertTokenizer, BertModel
import torch
import nltk
from nltk.corpus import stopwords
import numpy as np
from tqdm import tqdm
from scipy.linalg import logm, expm, norm
import random


class ModelCopyright():
    def __init__(self, mode=1):
        self.mode = 1

    def get_template(self, eval_source):
        with open(eval_source, 'r', encoding='utf-8') as f:
            # json template:
            return json.load(f)

    def get_same_label(self, reference_template, candidate_template, label=1):
        same_dict = {}
        if label >= 0:
            for category, content in reference_template.items():
                generation = []
                if content[0]['label'] == candidate_template[category][0]['label'] and content[0]['label'] == label:
                    generation.append(reference_template[category][0]['generation'])
                    generation.append(candidate_template[category][0]['generation'])
                    same_dict[category] = generation
            return same_dict

        # 所有prompt
        elif label == -1:
            for category, content in reference_template.items():
                generation = []
                generation.append(reference_template[category][0]['generation'])
                generation.append(candidate_template[category][0]['generation'])
                same_dict[category] = generation
            return same_dict

        # 所有在reference_template攻击成功的prompt
        elif label == -2:
            for category, content in reference_template.items():
                generation = []
                if content[0]['label'] == 0:
                    generation.append(reference_template[category][0]['generation'])
                    generation.append(candidate_template[category][0]['generation'])
                    same_dict[category] = generation
            return same_dict

        # r1s0
        elif label == -3:
            for category, content in reference_template.items():
                generation = []
                if content[0]['label'] != candidate_template[category][0]['label'] and content[0]['label'] == 1:
                    generation.append(reference_template[category][0]['generation'])
                    generation.append(candidate_template[category][0]['generation'])
                    same_dict[category] = generation
            return same_dict

        # r0s1
        elif label == -4:
            for category, content in reference_template.items():
                generation = []
                if content[0]['label'] != candidate_template[category][0]['label'] and content[0]['label'] == 0:
                    generation.append(reference_template[category][0]['generation'])
                    generation.append(candidate_template[category][0]['generation'])
                    same_dict[category] = generation
            return same_dict

    def get_compare_examples(self, reference_template, candidate_template):
        compare_examples = {}
        compare_examples['r1s1'] = self.get_same_label(reference_template, candidate_template, label=1)
        compare_examples['r0s0'] = self.get_same_label(reference_template, candidate_template, label=0)
        compare_examples['r1s0'] = self.get_same_label(reference_template, candidate_template, label=-3)
        compare_examples['r0s1'] = self.get_same_label(reference_template, candidate_template, label=-4)
        return compare_examples


    def get_bert_embedding(self, text, method='cls', device='cpu'):
        if device == 'gpu' and torch.cuda.is_available():
            device = 'cuda'
        elif device == 'mps' and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
        device = torch.device(device)

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        model.to(device)

        # Tokenize the input text
        tokens = tokenizer.encode(text, add_special_tokens=True)
        num_chunks = (len(tokens) + 511) // 512  # Calculate the number of chunks (each chunk up to 512 tokens)

        embeddings = []

        for i in range(num_chunks):
            chunk_tokens = tokens[i * 512:(i + 1) * 512]
            chunk_tokens = tokenizer.convert_ids_to_tokens(chunk_tokens)
            chunk_inputs = tokenizer(' '.join(chunk_tokens), return_tensors='pt', truncation=True, max_length=512).to(
                device)

            with torch.no_grad():
                outputs = model(**chunk_inputs)
            hidden_states = outputs.last_hidden_state

            if method == 'cls':
                chunk_embedding = hidden_states[0][0].cpu()
            elif method == 'mean':
                chunk_embedding = hidden_states.mean(dim=1)[0].cpu()
            elif method == 'max':
                chunk_embedding = hidden_states.max(dim=1)[0].cpu()
            else:
                raise ValueError("Unknown method")

            embeddings.append(chunk_embedding)

        # Aggregate embeddings
        if method in ['mean', 'max']:
            sentence_embedding = torch.stack(embeddings).mean(dim=0)
        else:
            sentence_embedding = embeddings[0]  # For cls method, take the embedding from the first chunk

        return sentence_embedding

    def remove_stopwords(self, text):
        # 确保下载了停用词列表
        stop_words = set(stopwords.words('english'))
        # 分词
        words = text.split()
        # 去除停用词
        filtered_words = [word for word in words if word.lower() not in stop_words]
        # 将过滤后的词汇重新组合成字符串
        filtered_text = ' '.join(filtered_words)
        return filtered_text

    # def similarity_tools(self, vec1, vec2, method = 'cosine'):
    #
    #     vec1 = np.array(vec1) if not isinstance(vec1, np.ndarray) else vec1
    #     vec2 = np.array(vec2) if not isinstance(vec2, np.ndarray) else vec2
    #
    #     # 检查向量是否为空或者全零
    #     if np.any(np.isnan(vec1)) or np.any(np.isnan(vec2)):
    #         return 0  # 如果向量中含有nan，返回相似度为0
    #
    #     if np.all(vec1 == 0) or np.all(vec2 == 0):
    #         return 0  # 如果向量是全零，返回相似度为0
    #
    #     # 余弦相似度
    #     if method == 'cosine':
    #         norm_vec1 = np.linalg.norm(vec1)
    #         norm_vec2 = np.linalg.norm(vec2)
    #
    #         dot_product = np.dot(vec1, vec2)
    #         return dot_product / (norm_vec1 * norm_vec2)
    #     # 欧式距离
    #     elif method == 'euclidean':
    #         return np.linalg.norm(vec1 - vec2)
    #     # 曼哈顿距离
    #     elif method == 'manhattan':
    #         return np.sum(np.abs(vec1 - vec2))
    #     # 杰卡德相似系数
    #     elif method == 'jaccard':
    #         intersection = np.sum(np.minimum(vec1, vec2))
    #         union = np.sum(np.maximum(vec1, vec2))
    #         return intersection / union
    #
    #     # Geodesic
    #     elif method == 'geodesic':
    #         # 先归一化向量，确保它们在球面上
    #         vec1 = vec1 / np.linalg.norm(vec1)
    #         vec2 = vec2 / np.linalg.norm(vec2)
    #
    #         # 计算地质距离
    #         dot_product = np.dot(vec1, vec2)
    #         return np.arccos(np.clip(dot_product, -1.0, 1.0))  # 使用np.clip确保输入在有效范围内

    def similarity_tools(self, vec1, vec2, method='cosine'):

        vec1 = np.array(vec1) if not isinstance(vec1, np.ndarray) else vec1
        vec2 = np.array(vec2) if not isinstance(vec2, np.ndarray) else vec2

        # 检查向量是否为空或者全零
        if np.any(np.isnan(vec1)) or np.any(np.isnan(vec2)):
            return 0  # 如果向量中含有nan，返回相似度为0

        if np.all(vec1 == 0) or np.all(vec2 == 0):
            return 0  # 如果向量是全零，返回相似度为0

        # 余弦相似度
        if method == 'cosine':
            norm_vec1 = np.linalg.norm(vec1)
            norm_vec2 = np.linalg.norm(vec2)
            if norm_vec1 == 0 or norm_vec2 == 0:
                return 0

            dot_product = np.dot(vec1, vec2)
            cosine_similarity = dot_product / (norm_vec1 * norm_vec2)

            # 将相似度从 [-1, 1] 归一化到 [0, 1]
            return (cosine_similarity + 1) / 2

        # 欧几里得相似度（还可以）
        elif method == 'euclidean':
            euclidean_distance = np.linalg.norm(vec1 - vec2)

            # 归一化欧几里得距离：将其映射到 [0, 1]，这里假设最大距离为向量维度的平方根
            max_distance = np.sqrt(len(vec1))
            return 1 - (euclidean_distance / max_distance)

        # 曼哈顿相似度
        elif method == 'manhattan':
            manhattan_distance = np.sum(np.abs(vec1 - vec2))

            # 归一化曼哈顿距离：最大曼哈顿距离为所有维度上最大绝对差值之和
            max_distance = len(vec1) * np.max(np.abs(vec1) + np.abs(vec2))
            return 1 - (manhattan_distance / max_distance)

        # 杰卡德相似系数（适用于二进制数据或非负向量）也还行
        elif method == 'jaccard':
            vec1 = np.maximum(0, vec1)  # 确保非负
            vec2 = np.maximum(0, vec2)  # 确保非负
            intersection = np.sum(np.minimum(vec1, vec2))
            union = np.sum(np.maximum(vec1, vec2))

            if union == 0:
                return 1  # 两个向量都为0时，相似度为1
            return intersection / union

        # Geodesic 相似度
        elif method == 'geodesic':
            # 先归一化向量，确保它们在球面上
            vec1 = vec1 / np.linalg.norm(vec1)
            vec2 = vec2 / np.linalg.norm(vec2)

            # 计算点积
            dot_product = np.dot(vec1, vec2)

            # 计算测地距离 (反余弦函数) 并转换为相似性度量
            geodesic_distance = np.arccos(np.clip(dot_product, -1.0, 1.0))

            # 将测地距离归一化为相似性（0到1之间）
            return 1 - geodesic_distance / np.pi

    def get_bleu_score(self, reference, candidate):
        # bleu_score = modelcopyright.get_bleu_score(content[0]['generation'], candidate_template[category][0]['generation'])
        reference = list(' '.join(jieba.cut(reference)).split())
        candidate = ' '.join(jieba.cut(candidate)).split()
        smoothing_function = SmoothingFunction().method1
        return sentence_bleu(reference, candidate, smoothing_function=smoothing_function)

    def get_similarity(self, examples, device='mps', method='cosine'):
        # examples format: {key: [content1, content2]}
        similarity = np.array([])
        for category, content in tqdm(examples.items()):
            reference = self.get_bert_embedding(self.remove_stopwords(content[0]), device=device)
            candidate = self.get_bert_embedding(self.remove_stopwords(content[1]), device=device)
            # print(mc.get_similarity(reference, candidate, method = 'cosine'))
            similarity = np.append(similarity, self.similarity_tools(reference, candidate, method=method))
            # print('similarity:', similarity, 'reference:', self.remove_stopwords(content[0]), 'candidate:', self.remove_stopwords(content[1]))
            # print('similarity:', similarity) # 检查节点
        return similarity

    def get_IPscore(self, compare_examples, device='mps', method='cosine', weight=[1, 0.1, 0.2, 1]):
        r1s1_score = self.get_similarity(compare_examples['r1s1'], device=device, method=method)
        r0s0_score = self.get_similarity(compare_examples['r0s0'], device=device, method=method)
        r1s0_score = self.get_similarity(compare_examples['r1s0'], device=device, method=method)
        r0s1_score = self.get_similarity(compare_examples['r0s1'], device=device, method=method)
        # print('r0s0_score:', r0s0_score)

        # 计算每个数据集的总和
        sum_r1s1 = np.sum(r1s1_score)
        # print('sum_r1s1:', r1s1_score.shape)
        sum_r0s0 = np.sum(r0s0_score)
        # print('sum_r0s0:', r0s0_score.shape)
        sum_r1s0 = np.sum(r1s0_score)
        # print('sum_r1s0:', r1s0_score.shape)
        sum_r0s1 = np.sum(r0s1_score)
        # print('sum_r0s1:', r0s1_score.shape)
        print('sum_r1s1:', r1s1_score.shape[0], ' sum_r0s0:', r0s0_score.shape[0], ' sum_r1s0:', r1s0_score.shape[0], ' sum_r0s1:', r0s1_score.shape[0])

        print('r1s1_ave:', sum_r1s1 / r1s1_score.shape[0],
              'r0s0_ave:', sum_r0s0 / r0s0_score.shape[0],
              'r1s0_ave:', sum_r1s0 / r1s0_score.shape[0],
              'r0s1_ave:', sum_r0s1 / r0s1_score.shape[0])

        # 将所有总和相加
        total_sum = (sum_r1s1 * weight[0] +
                     sum_r0s0 * weight[1] +
                     sum_r1s0 * weight[2] +
                     sum_r0s1 * weight[3])

        # 计算所有数据集的总得分数
        total_scores_count = len(r1s1_score) + len(r0s0_score) + len(r1s0_score) +len(r0s1_score)
        print('total_scores_count:', total_scores_count)

        # 计算最终的平均得分
        final_score = total_sum / total_scores_count

        # 归一化得分
        min_score = 0  # 假设最小值为0，实际应根据数据调整
        max_score = 1  # 假设最大值为1，实际应根据数据调整
        normalized_score = (final_score - min_score) / (max_score - min_score)

        # 确保归一化后的得分在0-1之间
        normalized_score = np.clip(normalized_score, 0, 1)

        return {
            'r1s1_score': r1s1_score.tolist(),
            'r0s0_score': r0s0_score.tolist(),
            'r1s0_score': r1s0_score.tolist(),
            'r0s1_score': r0s1_score.tolist(),
            'final_score': final_score
        }


class ResultStatistic():

    def __init__(self, mode=1):
        self.mode = mode
        self.mc = ModelCopyright()

    def get_template(self, eval_source):
        with open(eval_source, 'r', encoding='utf-8') as f:
            # json template:
            return json.load(f)

    def get_example_sum(self, files):
        examples = self.get_template(files)
        sum = 0
        for category, content in examples.items():
            sum += 1
        return sum

    def get_label_sum(self, files, label=1):
        examples = self.get_template(files)
        sum = 0
        for category, content in examples.items():
            if content[0]['label'] == label:
                sum += 1
        return sum

    def get_category_sum(self, files):
        examples = self.get_template(files)
        sum_list = []
        for category, content in examples.items():
            sum_list.append(category)
        return len(set(sum_list))

    def sample_examples(self, examples, sample_size=20):
        sampled = {}
        examples_key = list(examples.keys())
        random_key = random.sample(examples_key, sample_size)
        for key in random_key:
            sampled[key] = examples[key]
        return sampled

    def compare_metric(self, reference_template, candidate_template, device='gpu', method='cosine'):
        compare_examples = self.mc.get_compare_examples(reference_template, candidate_template)

        # 对每个数据集随机挑选 20 个示例
        r1s1_sample = self.sample_examples(compare_examples['r1s1'])
        r0s0_sample = self.sample_examples(compare_examples['r0s0'])
        r1s0_sample = self.sample_examples(compare_examples['r1s0'])
        r0s1_sample = self.sample_examples(compare_examples['r0s1'])

        # 打印每个数据集随机选择的样本
        # print("r1s1 (Random 20 examples):", r1s1_sample)
        # print("r0s0 (Random 20 examples):", r0s0_sample)
        # print("r1s0 (Random 20 examples):", r1s0_sample)
        # print("r0s1 (Random 20 examples):", r0s1_sample)

        r1s1_score = self.mc.get_similarity(r1s1_sample, device=device, method=method)
        # r0s0_score = self.mc.get_similarity(r0s0_sample, device=device, method=method)
        # r1s0_score = self.mc.get_similarity(r1s0_sample, device=device, method=method)
        # r0s1_score = self.mc.get_similarity(r0s1_sample, device=device, method=method)

        print(r1s1_score)


def run_all_metrics_and_save(reference_template, candidate_template, output_file, device='mps'):
    mc = ModelCopyright()
    compare_examples = mc.get_compare_examples(reference_template, candidate_template)

    methods = ['cosine', 'euclidean', 'manhattan', 'jaccard', 'geodesic']
    all_scores = {}

    for method in methods:
        print(f"Running for method: {method}")
        scores = mc.get_IPscore(compare_examples, device=device, method=method)
        all_scores[method] = scores

    # 将结果保存为JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_scores, f, ensure_ascii=False, indent=4)

    print(f"Scores saved to {output_file}")

if __name__ == '__main__':
    eval_file_source = './test_cases_new/llama2_7b_GCG/eval_llama2_7b_chat.json'
    eval_file_Suspect_1 = 'test_cases_new/llama2_7b_GCG/eval_llama2_7b_chinese.json'
    eval_file_Suspect_2 = 'test_cases_new/llama2_7b_GCG/eval_mistral_7b_v3.json'
    mc = ModelCopyright()
    rs = ResultStatistic()
    reference_template = mc.get_template(eval_file_source)
    candidate_template_1 = mc.get_template(eval_file_Suspect_1)
    candidate_template_2 = mc.get_template(eval_file_Suspect_2)
    compare_examples_1 = mc.get_compare_examples(reference_template, candidate_template_1)
    compare_examples_2 = mc.get_compare_examples(reference_template, candidate_template_2)

    # rs.compare_metric(reference_template, candidate_template_1

    # print(len(r1s0))
    # print(rs.get_label_sum(eval_file_Suspect_1))
    # print('vicuna_7b_v1_5:', mc.get_IPscore(compare_examples_1, method='geodesic'), 'mistral_7b_v3:', mc.get_IPscore(compare_examples_2, method='geodesic'))
    print('eval_llama2_7b_chinese.json:', mc.get_IPscore(compare_examples_1, method='geodesic'))
    # labels = ['cosine', 'euclidean', 'manhattan', 'jaccard', 'geodesic']

    # run_all_metrics_and_save(reference_template, candidate_template_1, 'output_scores.json')



