# Smart Nudging Causal Analysis
This repository contains code for training a causal machine learning model based on simulated data for smart green nudging.

# Installation
To install the required packages, run the following:
```
pip install -r requirements.txt
```

# Usage
To create synthetic data of a nudging experiment, run:
```
python src/simulate_data.py
```
This script generates data with the following columns:
- T: Indicates a random treatment with the nudge.
- X: Features of the individuals subject to the nudge.
- Y: The target variable supposedly influenced by the nudge (e.g., the return of products in e-commerce).
The generated data is separated into training and testing datasets.

To tune, train, and save a CausalForestDML estimator (a causal forest embedded in a double machine learning framework) to predict the effect of T on Y depending on features X, run:
```
python src/cml_training.py
```

To create evaluation plots based on the separate test data, showing the group average treatment effects and the inverse propensity score estimator for different shares of treated individuals, run:
```
python src/cml_evaluation.py
```

# Citation
Please consider citing us if you find this helpful for your work:
```
@article{vonZahn.2024,
  title={Smart Green Nudging: Reducing Product Returns through Digital Footprints and Causal Machine Learning},
  author={von Zahn, Moritz and Bauer, Kevin and Mihale-Wilson, Cristina and Jagow, Johanna and Speicher, Maximilian and Hinz, Oliver},
  journal={Marketing Science},
  pages={forthcoming},
  year={2024}
}
 ```
