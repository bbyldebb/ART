'''
failure triage & an interpretable channel analysis
'''
import copy
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from models.unified_representation.representation import SLD, aggregate_failure_representations
from evaluation import eval_FT

# construct a split tree
class FailureInfo:
    def __init__(self, vector, label):
        self.vector = vector
        self.label = label

class Node:
    cluster_id = -1
    common_split_dims = set()

    def __init__(self, failure_infos, depth):
        self.failure_infos = failure_infos
        self.depth = depth
        self.left = None
        self.right = None
        self.flag = 1
        self.split_value = None
        self.cluster_id = -1
        self.update_criteria()
        self.update_label_id()

    def update_criteria(self):
        if self.flag:
            vectors = np.array([failure_info.vector for failure_info in self.failure_infos])
            variances = np.var(vectors, axis=0)
            for dim in Node.common_split_dims:
                variances[dim] = 0
            split_dim = np.argmax(variances)
            criteria = variances[split_dim]
            self.split_dim, self.criteria = split_dim, criteria
    
    def update_label_id(self):
        label_counts = Counter([failure_info.label for failure_info in self.failure_infos])
        most_common_label = max(label_counts, key=label_counts.get)
        self.label_id = most_common_label
    
    def update_in_distance(self):
        vectors = [info.vector for info in self.failure_infos]
        self.in_distance = np.mean(1 - cosine_similarity(vectors))

def split_cluster(root, max_depth=50, min_cluster_size=1):
    leaf_nodes = [root]
    while leaf_nodes:
        max_criteria_node = max(leaf_nodes, key=lambda x: x.criteria)
        split_dim = max_criteria_node.split_dim
        vectors = np.array([failure_info.vector for failure_info in max_criteria_node.failure_infos])
        max_cosine_distance = -1
        best_percentile = None
        for percentile in vectors[:, split_dim]:
            left_failure_infos = [failure_info for failure_info in max_criteria_node.failure_infos if failure_info.vector[split_dim] <= percentile]
            right_failure_infos = [failure_info for failure_info in max_criteria_node.failure_infos if failure_info.vector[split_dim] > percentile]
            if len(left_failure_infos) >= min_cluster_size and len(right_failure_infos) >= min_cluster_size:
                left_vectors = np.array([failure_info.vector for failure_info in left_failure_infos])
                right_vectors = np.array([failure_info.vector for failure_info in right_failure_infos])
                cosine_distance = np.mean(1- cosine_similarity(left_vectors, right_vectors))
                if cosine_distance > max_cosine_distance:
                    max_cosine_distance = cosine_distance
                    best_percentile = percentile
        if best_percentile is not None:
            max_criteria_node.split_value = best_percentile
            left_failure_infos = [failure_info for failure_info in max_criteria_node.failure_infos if failure_info.vector[split_dim] <= best_percentile]
            right_failure_infos = [failure_info for failure_info in max_criteria_node.failure_infos if failure_info.vector[split_dim] > best_percentile]
            max_criteria_node.left = Node(left_failure_infos, max_criteria_node.depth + 1)
            max_criteria_node.right = Node(right_failure_infos, max_criteria_node.depth + 1)
        max_criteria_node.flag = 0
        max_criteria_node.update_in_distance()
        Node.common_split_dims.add(split_dim)
        leaf_nodes = [node for node in get_leaf_nodes(root) if ((node.depth<max_depth) & node.flag)]
    for leaf_node in get_leaf_nodes(root):
        Node.cluster_id += 1
        leaf_node.cluster_id = Node.cluster_id

def merge_nodes(root, max_clusters):
    while len(get_leaf_nodes(root)) >= (max_clusters+1):
        min_avg_cosine_distance = float('inf')
        node_to_merge = None
        for node in get_parent_nodes_of_leaves(root):
            if node.in_distance < min_avg_cosine_distance:
                min_avg_cosine_distance = node.in_distance
                node_to_merge = node
        if node_to_merge is not None:
            node_to_merge.left = None
            node_to_merge.right = None
        else:
            break

def get_leaf_nodes(node):
    if node.left is None and node.right is None:
        node.update_criteria()
        return [node]
    leaf_nodes = []
    if node.left is not None:
        leaf_nodes.extend(get_leaf_nodes(node.left))
    if node.right is not None:
        leaf_nodes.extend(get_leaf_nodes(node.right))
    return leaf_nodes

def get_parent_nodes_of_leaves(root):
    parent_nodes = set()
    leaf_nodes = get_leaf_nodes(root)
    for node in leaf_nodes:
        parent_node = find_parent(root, node)
        if parent_node is not None:
            parent_nodes.add(parent_node)
    return list(parent_nodes)

def find_parent(root, node):
    if root is None:
        return None
    if root.left == node or root.right == node:
        return root
    left_parent = find_parent(root.left, node)
    right_parent = find_parent(root.right, node)
    return left_parent if left_parent is not None else right_parent

def init_prediction(root, type_hash):
    init_labels, pre, clusters = [], [], []
    leaf_nodes = get_leaf_nodes(root)
    for leaf_node in leaf_nodes:
        for info in leaf_node.failure_infos:
            init_labels.append(type_hash[info.label])
            pre.append(type_hash[leaf_node.label_id])
            clusters.append(leaf_node.cluster_id)
    return init_labels, pre, clusters

def test_prediction(root, test_failure_infos, type_dict):
    pre, clusters = [], []
    for test_failure_info in test_failure_infos:
        current_node = root
        while current_node.left is not None or current_node.right is not None:
            if test_failure_info.vector[current_node.split_dim] <= current_node.split_value:
                current_node = current_node.left
            else:
                current_node = current_node.right
        pre_id, cluster_id = current_node.label_id, current_node.cluster_id
        pre.append({v: k for k, v in type_dict.items()}[pre_id])
        clusters.append(cluster_id)
    return pre, clusters

def FT(model, test_samples, cases, type_hash, type_dict, split_ratio=0.7, method='num', t_value=3, before=60, after=300, max_clusters=15, channel_dict=None, verbose=False):
    failure_representations, type_labels = aggregate_failure_representations(cases, SLD(model, test_samples, method, t_value), type_hash, before, after)
    spilit_index = int(len(cases)*split_ratio)
    init_failure_infos = [FailureInfo(failure_representations[_], type_dict[type_labels[_]]) for _ in range(spilit_index)]
    test_failure_infos = [FailureInfo(failure_representations[_], type_dict[type_labels[_]]) for _ in range(spilit_index, len(failure_representations))]
    test_labels = [type_labels[_] for _ in range(spilit_index, len(failure_representations))]
    # Cutting Divisions
    Node.cluster_id = -1
    Node.common_split_dims = set()
    splitting_root = Node(init_failure_infos, depth=0)
    split_cluster(splitting_root, max_depth=50, min_cluster_size=1)
    # Backtracking Merge
    merged_root = copy.deepcopy(splitting_root)
    merge_nodes(merged_root, max_clusters)
    num_leaf_nodes = len(get_leaf_nodes(merged_root))
    if verbose:
        # evaluate the initialization sets.
        init_labels, init_pre, init_clusters = init_prediction(merged_root, {v: k for k, v in type_dict.items()})
        print('init_prediction: ', end='')
        eval_FT(merged_root, init_labels, init_pre, num_leaf_nodes, verbose=True)
    # evaluate the test sets.
    test_pre, test_clusters = test_prediction(merged_root, test_failure_infos, type_dict)
    if verbose:
        print('test_prediction: ', end='')
    precision, recall, f1 = eval_FT(merged_root, test_labels, test_pre, num_leaf_nodes, channel_dict, verbose)
    pre_types = [type_dict[item] for item in test_pre]
    return pre_types, precision, recall, f1
