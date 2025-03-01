import os
import subprocess
from tqdm import tqdm

path = './test_cases/llama2b_gcg_10/test_cases_individual_behaviors'
model = 'llama2_7b'
data_path = 'data/behavior_datasets/harmbench_behaviors_text_all.csv'

def file_name_listdir(file_dir):
    files_list = []
    for file in os.listdir(file_dir):
        files_list.append(file)
    return files_list

files_list = file_name_listdir(path)

for sub_path in tqdm(files_list):
    cmd = ('sh ./scripts/generate_test_cases.sh' + ' '
           + model + ' '
           + data_path + ' '
           + path + '/' + sub_path + '/test_cases.json '
           + path + '/' + sub_path + '/results.json 512 False')
    print(cmd)
    # subprocess.run(cmd, shell=True, check=True)
