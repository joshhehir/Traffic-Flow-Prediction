## Traffic Flow Prediction

Traffic Flow Prediction with Neural Networks(SAEs、LSTM、GRU) 

COS30018 - Intelligent Systems

## Getting Started

These instructions will explain the process of getting the system up and running on your local machine.

### Prerequisites

Graphviz - Graph Visualization Software  
Python 3.6.x
```
keras
matplotlib
pydot
pandas
scikit-learn
tensorflow
```

## Train the model

**Run command below to train the model:**

```
python train.py --model model_name
```

You can choose "lstm", "gru" or "saes" as arguments. The ```.h5``` weight file was saved at model folder.


## Experiment

Data are obtained from the Caltrans Performance Measurement System (PeMS). Data are collected in real-time from individual detectors spanning the freeway system across all major metropolitan areas of the State of California.
	
	device: Tesla K80
	dataset: PeMS 5min-interval traffic flow data
	optimizer: RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
	batch_szie: 256 


**Run command below to run the program:**

```
python main.py
```

These are the details for the traffic flow prediction experiment.


| Metrics | MAE | MSE | RMSE | MAPE |  R2  | Explained variance score |
| ------- |:---:| :--:| :--: | :--: | :--: | :----------------------: |
| LSTM | 7.21 | 98.05 | 9.90 | 16.56% | 0.9396 | 0.9419 |
| GRU | 7.20 | 99.32 | 9.97| 16.78% | 0.9389 | 0.9389|
| SAEs | 7.06 | 92.08 | 9.60 | 17.80% | 0.9433 | 0.9442 |

![evaluate](/TFPS/images/eva.png)

## Authors

* **Josh Hehir** [102932561@student.swin.edu.au](mailto:102932561@student.swin.edu.au)  
* **David Nguyen** [102927185@student.swin.edu.au](mailto:102927185@student.swin.edu.au)  
* **Brian Burns** [102313108@student.swin.edu.au](mailto:102313108@student.swin.edu.au) 

## Acknowledgments

* **xiaochus** - *Base code* - [Traffic Flow Prediction](https://github.com/xiaochus/TrafficFlowPrediction)
