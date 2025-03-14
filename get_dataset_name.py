import sys, yaml, os

config_path = os.path.join(os.path.dirname(__file__), './utils/dataset_config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

dataset_config = sys.argv[1]  
task_id = int(sys.argv[2])
datasets = config[dataset_config]['DATASETS']

# print(datasets[task_id])
dataset_name = datasets[task_id]
# n24news의 경우 task_id 7 이상은 텍스트 모달리티로 간주
if dataset_config == 'n24news' and task_id >= 7:
    dataset_name = dataset_name + "_text"

print(dataset_name)