import os
import subprocess
from tqdm import tqdm

path = './test_cases/llama2b_gcg_10/test_cases_individual_behaviors'
model = '/home/ubuntu/tools/hfd/HarmBench-Mistral-7b-val-cls'
data_path = 'data/behavior_datasets/harmbench_behaviors_text_all.csv'

def file_name_listdir(file_dir):
    files_list = []
    for file in os.listdir(file_dir):
        files_list.append(file)
    return files_list

files_list = file_name_listdir(path)

for sub_path in tqdm(files_list):
    cmd = ('sh ./scripts/evaluate_completions.sh' + ' '
           + model + ' '
           + data_path + ' '
           + path + '/' + sub_path + '/results.json '
           + path + '/' + sub_path + '/eval.json')
    print(cmd)
    subprocess.run(cmd, shell=True, check=True)
