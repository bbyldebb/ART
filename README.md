# Getting Started

## Environment
Python 3.9.13, PyTorch 1.12.1, scikit-learn 1.1.2, and DGL 0.9.0 are suggested.

## Dataset
Dataset D1 is collected from a simulated e-commerce microservice system, which is deployed in a real cloud environment with traffic consistent with real business flow. The system comprises 46 instances, including 40 microservice instances and 6 virtual machines. Failure records were collected by replaying the failures over several days in May 2022. The
failure scenarios are derived from actual failures (Container Hardware, Container Network, Node CPU, Node Disk, and
Node Memory-related failures). The collected records were labeled with their respective root cause instances and failure types. The raw data and labels of D1 are publicly available at https://anonymous.4open.science/r/Aiops-Dataset-1E0E.

Dataset D2 is collected from the management system of a top-tier commercial bank, which comprises 18 instances, including microservices, servers, databases, and dockers. Two experienced operators examined the failure records from January 2021 to June 2021 and labeled the root cause instances and failure types (Memory, CPU, Netowork, Disk, JVM-Memory, and JVM-CPU-related failures). Each operator conducted the labeling process separately and cross-checked the labels to ensure consensus. D2 has been used in the International AIOps Challenge 2021 https://aiops-challenge.com. Due to the non-disclosure agreement, we cannot make it publicly available. 

We have preprocessed two raw datasets and placed them in the following folder. 

D1: ART/data/D1

D2: ART/data/D2

In the folder:
- `cases`: There are two files in this directory. 

  ad_cases.pkl: The list of failure timestamps. 
  
  cases.csv: The four items in the table header indicate the failure injection time, failure level, root cause of the failure, and failure type respectively. 

- `hash_info`: There are four files in this directory. They all hold a dictionary that records the correspondence between names and indexes.

- `samples`: There are three files in this directory, all samples (samples.pkl), samples for pre-training (train_samples.pkl), and samples for evaluation (test_samples.pkl). 

  Each sample is a tuple: (timestamp, graphs, features of each node). Graphs indicate the topology of the microservice system generated from call relationships and deployment information; Features of each node are composed of pod metric feats, pod trace feats, pod log feats and node metric feats. 

## Demo
We provide a demo. 
Before running the following commands, please unzip D1.zip and D2.zip.

```
python main.py
```

## Parameter Description in the Demo

Take dataset D1 as an example.

### Pre-training in SSL

* `instance_dim`: The number of microservice instances. (default: 46)
* `num_heads`: The number of attention heads in a Transformer Encoder. (default: 2)
* `tf_layers`: The number of layers of Transformer Encoder. (default: 1)
* `channel_dim`: The number of data channels. (default: 130)
* `gnn_hidden_dim`: The hidden dimension of GraphSAGE. (default: 64)
* `gnn_out_dim`: The output dimension of GraphSAGE. (default: 32)
* `noise_rate`: Dropout ratio. (default: 0.3)
* `gnn_layers`: The number of layers of GraphSAGE. (default: 2)
* `gru_hidden_dim`: The hidden dimension of GRU. (default: 32)
* `gru_layers`: The number of layers of GRU. (default: 1)
* `epochs`: The training epochs. (default: 1000)
* `batch_size`: The batch size. (default: 4)
* `learning_rate`: The learning rate. (default: 0.001)

### Downstream tasks.

* `split_ratio`: The ratio for splitting the initialization set and the test set. (default: 0.6)
* `method & t_value`: The methods and parameters for aggregating ILD to obtain SLD. The method can be either 'num' or 'prob', representing the specified number and probability methods, respectively. 'num' can take 't_value' in the range [1, instance_dim], while 'prob' can take 't_value' in the range [0,1). (default: 'num', 3)
* `q & level`: The initialize parameters for SPOT. (default: 0.1, 0.95)
* `delay`: Consider anomalies with intervals smaller than the delay as the same failure. (default: 600)
* `impact_window`: The impact range of a single failure. Functions similarly to 'before' and 'after'. (default: 300)
* `before & after`: The failure injection time is inject_ts. Assuming the failure impact range is [inject_ts-before, inject_ts+after], take the data within this time window for analysis. (default: 59, 300)
* `max_clusters`: The maximum number of clusters obtained from the cut tree in failure triage. (default: 25)
* `verbose`: Control the output to be either concise or verbose. (default: False)

More details can be found in the configuration file: ART/config/D1.yaml.

