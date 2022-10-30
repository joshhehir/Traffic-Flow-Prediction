## Traffic Flow Prediction

Traffic Flow Prediction with Neural Networks(SAEs、LSTM、GRU, SRNN) 

COS30018 - Intelligent Systems

## Getting Started

These instructions will explain the process of getting the system working on your local machine.

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

**Run command below to open the training gui to train your models:**

```
python tfps_train_gui.py
```

![evaluate](/TFPS/images/train_gui.png)

## Experiment

**Run command below to run the program:**

```
python main.py
```

These are the details for the traffic flow prediction experiment.


| Metrics | MAE | MSE | RMSE | MAPE |  R2  | Explained variance score |
| ------- |:---:| :--:| :--: | :--: | :--: | :----------------------: |
| LSTM | 14.70 | 416.07 | 20.4 | 13.77% | 0.975 | 0.975 |
| GRU  | 15.21 | 457.73 | 21.39 | 16.47% | 0.972 | 0.973 |
| SRNN | 17.83 | 588.97 | 24.27 | 21.87% | 0.965 | 0.965 |
| SAEs | 55.26 | 6241.41| 79.00 | 66.74 | 0.625 | 0.664 |

![evaluate](/TFPS/images/eva.png)

## Routing Prediction

**Run command below to open the routing gui:**

```
python routing_gui.py
```

Use the routing gui to predict a path from an origin scats point to a destination scats point.

![evaluate](/TFPS/images/routing_gui.png)

## Authors

* **Josh Hehir** [102932561@student.swin.edu.au](mailto:102932561@student.swin.edu.au)  
* **David Nguyen** [102927185@student.swin.edu.au](mailto:102927185@student.swin.edu.au)  
* **Brian Burns** [102313108@student.swin.edu.au](mailto:102313108@student.swin.edu.au) 

## Acknowledgments

* **xiaochus** - *Base code* - [Traffic Flow Prediction](https://github.com/xiaochus/TrafficFlowPrediction)
