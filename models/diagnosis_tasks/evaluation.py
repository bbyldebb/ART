import numpy as np
from collections import Counter
from sklearn.metrics import f1_score, precision_score, recall_score

def eval_AD(pre_interval, ad_cases_label, impact_window=5*60, verbose=False):
    # match
    pre_dict = {key: set() for key in pre_interval}
    ad_dict = {key: set() for key in ad_cases_label}
    for s, e in pre_interval:
        for case_ts in ad_cases_label:
            case_s, case_e = case_ts-impact_window, case_ts+impact_window
            if not (case_s > e or case_e < s):
                pre_dict[(s, e)].add(case_ts)
                ad_dict[case_ts].add((s, e))
    # calculate
    TP = len([key for key,value in ad_dict.items() if len(value) > 0])
    FP = len([key for key,value in pre_dict.items() if len(value) == 0])
    FN = len([key for key,value in ad_dict.items() if len(value) == 0])
    precision = np.round(TP / (TP + FP), 4)
    recall = np.round(TP / (TP + FN), 4)
    f1 = np.round(2 * precision * recall / (precision + recall), 4)
    density = np.round(np.mean([len(value) for key,value in pre_dict.items() if len(value) > 0]), 2)
    if verbose:
        print(f'precision: {precision}, recall: {recall}, f1: {f1}, density: {density}')
    return precision, recall, f1

def eval_FT(root, labels, pre, num_leaf_nodes, channel_dict=None, verbose=False):
    avg_type='weighted'
    precision = np.round(precision_score(labels, pre, average=avg_type), 4)
    recall = np.round(recall_score(labels, pre, average=avg_type), 4)
    f1 = np.round(f1_score(labels, pre, average=avg_type), 4)
    label_rate = np.round(num_leaf_nodes/len(root.failure_infos), 4)
    if verbose:
        print(f'precision, recall, {avg_type}-f1, label_rate')
        print(precision, recall, f1, label_rate, f'({num_leaf_nodes} / {len(root.failure_infos)})')
    if channel_dict:
        print_channel_detials(root, channel_dict)
    return precision, recall, f1

def print_channel_detials(node, channel_dict):
    if node is None:
        return
    indent = '     ' * node.depth
    if node.left is None and node.right is None:
        print(indent + f' | Split Dimension: {node.split_dim}, Split Criteria: {node.criteria}, Split Value: {node.split_value}, Num Vectors: {len(node.failure_infos)}, In distance: {node.in_distance}')
        print(indent + ' * ' + f'[{node.label_id}] ' + str(Counter([failure_info.label for failure_info in node.failure_infos])))
    else:
        print(indent + f'Split Dimension: {node.split_dim}, Split Criteria: {node.criteria}, Split Value: {node.split_value}, Num Vectors: {len(node.failure_infos)}, In distance: {node.in_distance}, {channel_dict[node.split_dim]}')
    print_channel_detials(node.left, channel_dict)
    print_channel_detials(node.right, channel_dict)


def eval_RCL(rank_df, k=5, verbose=False):
    topks = np.zeros(k)
    for _, case in rank_df.iterrows():
        for i in range(k):
            if case['cmdb_id'] in case[f'Top{i+1}']:
                topks[i: ] += 1
                break
    topK = list(np.round(topks / len(rank_df), 4))
    avgK = np.round(np.mean(topks / len(rank_df)), 4)
    if verbose:
        print(f'topK: {topK}, avgK: {avgK}')
    return topK, avgK
