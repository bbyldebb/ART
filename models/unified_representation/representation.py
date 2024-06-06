'''
system-level deviation
instance-level deviation
'''
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from layers import create_dataloader_AR

# method: 'num', 'prob'
def SLD(model, test_samples, method='num', t_value=3):
    mse = nn.MSELoss(reduction='none')
    system_level_deviation_df = pd.DataFrame()
    dataloader = create_dataloader_AR(test_samples, batch_size=128, shuffle=False)
    model.eval()
    with torch.no_grad():
        for batch_ts, batched_graphs, batched_feats, batched_targets in dataloader:
            z, h = model(batched_graphs, batched_feats)
            loss = mse(h, batched_targets) # 128,46,130
            if method == 'prob':
                max = torch.max(torch.sum(loss, dim=-1), dim=-1).values.unsqueeze(dim=-1)
                min = torch.min(torch.sum(loss, dim=-1), dim=-1).values.unsqueeze(dim=-1)
                root_prob = torch.softmax((torch.sum(loss, dim=-1) - min) / (max - min), dim=-1)
                sorted_indices = torch.argsort(root_prob, dim=1, descending=True)
                root_prob = torch.gather(root_prob, 1, sorted_indices)
                loss = torch.gather(loss, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, loss.size(-1)))
                cumulative_sum = torch.cumsum(root_prob, dim=1)

                root_prob = torch.softmax(torch.sum(loss, dim=-1), dim=-1)
                sorted_indices = torch.argsort(root_prob, dim=1, descending=True)
                root_prob = torch.gather(root_prob, 1, sorted_indices)
                loss = torch.gather(loss, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, loss.size(-1)))
                cumulative_sum = torch.cumsum(root_prob, dim=1)

                t_value_indices = torch.argmax((cumulative_sum > t_value).to(torch.int), dim=1)
                selected_indices = torch.zeros_like(loss)
                for i in range(root_prob.shape[0]):
                    selected_indices[i, :t_value_indices[i]+1, ] = 1
                system_level_deviation = torch.sum(selected_indices * loss, dim=1)
            elif method == 'num':
                instance_deviation = torch.sum(loss, dim=-1)
                topk_values, topk_indices = torch.topk(instance_deviation, k=t_value, dim=-1)
                mask = torch.zeros_like(instance_deviation)
                mask = mask.scatter_(1, topk_indices, 1).unsqueeze(-1)
                system_level_deviation = torch.sum(loss * mask, dim=1)

            tmp_df = pd.DataFrame(system_level_deviation.detach().numpy())
            tmp_df['timestamp'] = batch_ts
            system_level_deviation_df = pd.concat([system_level_deviation_df, tmp_df])
    return system_level_deviation_df.reset_index(drop=True)

def ILD(model, test_samples):
    mse = nn.MSELoss(reduction='none')
    instance_level_deviation_df = pd.DataFrame()
    dataloader = create_dataloader_AR(test_samples, batch_size=128, shuffle=False)
    model.eval()
    with torch.no_grad():
        for batch_ts, batched_graphs, batched_feats, batched_targets in dataloader:
            z, h = model(batched_graphs, batched_feats)
            loss = mse(h, batched_targets)
            batch_size, instance_size, channel_size = loss.shape
            string_tensor = np.array([str(row.tolist()) for row in loss.reshape(-1, channel_size)])
            tmp_df = pd.DataFrame(string_tensor.reshape(batch_size, instance_size))
            tmp_df['timestamp'] = batch_ts
            instance_level_deviation_df = pd.concat([instance_level_deviation_df, tmp_df])
    return instance_level_deviation_df.reset_index(drop=True)      

def aggregate_instance_representations(cases, instance_level_deviation_df, before=60, after=300):
    instance_representations = []
    for _, case in cases.iterrows():
        instance_representation = []
        agg_df = instance_level_deviation_df[(instance_level_deviation_df['timestamp']>=(case['timestamp']-before)) & (instance_level_deviation_df['timestamp']<(case['timestamp']+after))]
        for col_name, col_data in agg_df.items():
            if col_name == 'timestamp':
                continue
            # mean
            # instance_representation.append(torch.mean(torch.tensor([eval(item) for item in col_data]), dim=0))
            # max
            # tmp = torch.tensor([eval(item) for item in col_data])
            # max_row_index = np.argmax(tmp.sum(axis=1))
            # instance_representation.append(tmp[max_row_index])
            # no-aggregation
            instance_representation.extend([(col_name, eval(item)) for item in col_data])
        # instance_representations.append(torch.stack(instance_representation))
    # return torch.stack(instance_representations)
        instance_representations.append(instance_representation)
    return instance_representations

def aggregate_failure_representations(cases, system_level_deviation_df, type_hash=None, before=60, after=300):
    failure_representations, type_labels = [], []
    for _, case in cases.iterrows():
        agg_df = system_level_deviation_df[(system_level_deviation_df['timestamp']>=(case['timestamp']-before)) & (system_level_deviation_df['timestamp']<(case['timestamp']+after))]
        failure_representations.append(list(agg_df.mean()[:-1])) # mean
        if type_hash:
            type_labels.append(type_hash[case['failure_type']])
        else:
            type_labels.append(case['failure_type'])
    return failure_representations, type_labels

# failure_representations, type_labels = aggregate_failure_representations(cases, SLD(model, test_samples, 'num', 3), type_hash, before, after)
# failure_representations, type_labels = aggregate_failure_representations(cases, SLD(model, test_samples, 'prob', 0.9), type_hash, before, after)
# instance_representations = aggregate_instance_representations(cases, ILD(model, test_samples), before, after)
