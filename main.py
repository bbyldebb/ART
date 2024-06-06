import warnings
warnings.filterwarnings('ignore')
import os
import yaml
import pandas as pd
import sys
sys.path.extend(['models', 'models/diagnosis_tasks', 'models/unified_representation'])
from models.diagnosis_tasks.diag_workflow import diag_workflow
from models.unified_representation.train import train
from utils.public_functions import load_samples, hash_init, save_json, save_pkl, load_pkl

dataset = 'D1'
workflow = ['AD', 'FT', 'RCL']
# config
config = yaml.load(open(f'config/{dataset}.yaml', 'r'), Loader=yaml.FullLoader)
cases = pd.read_csv(config['path']['case_path'])
ad_cases_label = load_pkl(config['path']['ad_case_path'])
node_hash, node_dict, type_hash, type_dict, channel_dict = hash_init(config['path']['hash_dir'])
print('load config.')
res_dir = f'res/{dataset}'
tmp_dir = f'{res_dir}/tmp'
model_path = f'{res_dir}/model.pkl'
res_path = f'{res_dir}/res.json'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)
# pre-training
train_samples, test_samples = load_samples(config['path']['sample_dir'])
print('load samples.')
input_samples = train_samples if config['train_samples_num'] == 'whole' else train_samples[: config['train_samples_num']]
if os.path.exists(model_path):
    model = load_pkl(model_path)
    print('model loaded.')
else:
    model = train(input_samples, config['model_param'])
    save_pkl(model_path, model)
    print('model trained.')
# diagnosis tasks
tmp_res, eval_res = diag_workflow(config['downstream_param'], model, train_samples, test_samples,
                                cases, ad_cases_label, 
                                node_dict, type_hash, type_dict, channel_dict=None,
                                workflow=workflow)
# save output_results
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)
if 'AD' in tmp_res:
    save_json(f'{tmp_dir}/pre_interval.json', tmp_res['AD']['pre_interval'])
if 'FT' in tmp_res:
    save_json(f'{tmp_dir}/pre_types.json', tmp_res['FT']['pre_types'])
if 'RCL' in tmp_res:
    tmp_res['RCL']['rank_df'].to_csv(f'{tmp_dir}/rank_df.csv', index=False)
# save eval_results
save_json(res_path, eval_res)
