'''
a workflow for multiple diagnosis tasks
'''
from anomaly_detection import AD
from diagnosis_tasks.failure_triage import FT
from diagnosis_tasks.root_cause_localization import RCL

def diag_workflow(config, model, train_samples, test_samples, 
                  cases, ad_cases_label, 
                  node_dict, type_hash, type_dict, channel_dict=None,
                  workflow=['AD', 'FT', 'RCL']):
    tmp_res, eval_res = {}, {}
    # anomaly detection
    if 'AD' in workflow:
        print('AD start. |', '*' *100)
        tmp_param = config['AD']
        split_ratio = tmp_param['split_ratio']
        method = tmp_param['method']
        t_value = tmp_param['t_value']
        q = tmp_param['q']
        level = tmp_param['level']
        delay = tmp_param['delay']
        impact_window = tmp_param['impact_window']
        verbose = tmp_param['verbose']

        pre_interval, precision, recall, f1 = AD(model, train_samples, test_samples, ad_cases_label, 
                                                split_ratio, method, t_value, q, level, delay, impact_window, verbose)
        tmp_res['AD'] = {'pre_interval': pre_interval}
        eval_res['AD'] = {'precision': precision, 'recall': recall, 'f1': f1}
        print('AD done. | ', eval_res['AD'])
    # failure triage & interpretable channel details
    if 'FT' in workflow:
        print('FT start. |', '*' *100)
        tmp_param = config['FT']
        split_ratio = tmp_param['split_ratio']
        method = tmp_param['method']
        t_value = tmp_param['t_value']
        before = tmp_param['before']
        after = tmp_param['after']
        max_clusters = tmp_param['max_clusters']
        verbose = tmp_param['verbose']

        pre_types, precision, recall, f1 = FT(model, test_samples, cases, type_hash, type_dict, 
                                            split_ratio, method, t_value, before, after, max_clusters, channel_dict=channel_dict, verbose=verbose)
        tmp_res['FT'] = {'pre_types': pre_types}
        eval_res['FT'] = {'precision': precision, 'recall': recall, 'f1': f1}
        print('FT done. |', eval_res['FT'])
    # root cause localization
    if 'RCL' in workflow:
        print('RCL start. |', '*' *100)
        tmp_param = config['RCL']
        split_ratio = tmp_param['split_ratio']
        method = tmp_param['method']
        t_value = tmp_param['t_value']
        before = tmp_param['before']
        after = tmp_param['after']
        verbose = tmp_param['verbose']

        rank_df, topK, avgK = RCL(model, test_samples, cases, node_dict, 
                                split_ratio, method, t_value, before, after, verbose)
        tmp_res['RCL'] = {'rank_df': rank_df}
        eval_res['RCL'] = {'topK': topK, 'avgK': avgK}
        print('RCL done. |', eval_res['RCL'])
    return tmp_res, eval_res