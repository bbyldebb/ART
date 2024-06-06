'''
anomaly detection
'''
import numpy as np
from utils.spot import SPOT
from models.unified_representation.representation import SLD
from evaluation import eval_AD
def run_spot(his_val, q=1e-2, level=0.98, verbose=False):
    model = SPOT(q)
    model.fit(his_val, [])
    model.initialize(level=level, verbose=verbose)
    return model.extreme_quantile

def get_threshold(his_val, q=1e-2, level=0.98, verbose=False):
    try:
        if len(set(his_val))==1:
            threshold = his_val[0]
        else:
            threshold = run_spot(his_val, q, level, verbose)
    except:
        threshold = sorted(his_val)[int(len(his_val)*level)]
    return threshold

def get_pre_interval(eval_slds_sum, delay=600):
    pre_ts_df = eval_slds_sum[eval_slds_sum['outlier']==1]
    pre_ts_df['diff'] = [0] + np.diff(pre_ts_df['timestamp']).tolist()
    pre_interval = []
    start_ts, end_ts = None, None
    for _, span in pre_ts_df.iterrows():
        if start_ts is None:
            start_ts = int(span['timestamp'])
        if span['diff'] >= delay:
            pre_interval.append((start_ts, end_ts))
            start_ts = int(span['timestamp'])
        end_ts = int(span['timestamp'])
    pre_interval.append((start_ts, end_ts))
    # filter
    pre_interval = [(item[0], item[1]) for item in pre_interval if item[0]!=item[1]]
    return pre_interval

def AD(model, train_samples, test_samples, ad_cases_label, split_ratio=0.6, method='num', t_value=3, q=0.1, level=0.95, delay=600, impact_window=300, verbose=False):
    total_samples = train_samples + test_samples
    total_samples.sort(key=lambda x: x[0])
    ts_list = sorted([item[0] for item in total_samples])
    split_ts = ts_list[int(len(ts_list)*split_ratio)]

    ad_cases_label = [item for item in ad_cases_label if item > split_ts]
    init_samples = [item for item in train_samples if item[0] <= split_ts]
    eval_samples = [item for item in total_samples if item[0] > split_ts]
    init_slds = SLD(model, init_samples, method, t_value)
    eval_slds = SLD(model, eval_samples, method, t_value)
    init_slds_sum = init_slds.set_index('timestamp').sum(axis=1).reset_index()
    eval_slds_sum = eval_slds.set_index('timestamp').sum(axis=1).reset_index()

    threshold = get_threshold(init_slds_sum[0].values, q, level)
    if threshold > 0:
        eval_slds_sum['outlier'] = eval_slds_sum[0].apply(lambda x: 1 if x > threshold else 0)
        if verbose:
            print(f'threshold is {threshold}.')
            print(f"outlier ratio is {np.round(sum(eval_slds_sum['outlier']==1)/len(eval_slds_sum), 4)}.")
        pre_interval = get_pre_interval(eval_slds_sum, delay)
        precision, recall, f1 = eval_AD(pre_interval, ad_cases_label, impact_window, verbose)
        return pre_interval, precision, recall, f1
    else:
        print('threshold is invalid.')
