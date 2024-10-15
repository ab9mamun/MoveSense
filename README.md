# MoveSense

### Multimodal Physical Activity Forecasting in Free-Living Clinical Settings: Hunting Opportunities for Just-in-Time Interventions
_Abdullah Mamun, Krista S. Leonard, Megan E. Petrov, Matthew P. Buman, Hassan Ghasemzadeh_


MoveSense is an LSTM-based multimodal time series forecasting system for estimating the next-day physical activity, such as the number of steps or duration of light physical activity of a person.

Read the full preprint here: https://arxiv.org/abs/2410.09643

Bibtex for citing the work:
```
@misc{mamun2024multimodalphysicalactivityforecasting,
      title={Multimodal Physical Activity Forecasting in Free-Living Clinical Settings: Hunting Opportunities for Just-in-Time Interventions}, 
      author={Abdullah Mamun and Krista S. Leonard and Megan E. Petrov and Matthew P. Buman and Hassan Ghasemzadeh},
      year={2024},
      eprint={2410.09643},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.09643}, 
}
```

### Method
The MoveSense system uses multimodal forecasting models with early fusion or late fusion of past physical activities and engagement with smartphone applications with lifestyle monitoring. The LSTM forecasting models outperform the baselines (ARIMA and Linear Regression) with a significant margin.

The paper investigates the effect of various parameters, including window size, different architectures, different modalities, early fusion vs late fusion, choice of participants based on engagement thresholds based on certain percentiles, etc.

Sample commands to run all the experiments:
For the prediabetes dataset
``` 
python main.py --task run_everything --dataset bewell --num_epochs 20 --exp_name e1
```
For the sleep dataset
```
python main.py --task run_everything --dataset sleepwell --num_epochs 20 --exp_name e1
```

### Dataset
The original dataset is not available for public use at this point. We are considering publicly sharing some synthetic data of a similar format in the near future.

### Contact Information
For questions or concerns, please contact Abdullah Mamun (a.mamun@asu.edu)
