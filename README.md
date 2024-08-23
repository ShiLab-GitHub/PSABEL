# PSABEL
Source code and data for "Predict peptide self-assemble behaviors via high-throughput screening method in tandem with machine learning"
## Data
* original_dataset contains raw data
*   assemble.csv - sequences capable of self-assembly
*   unassemble.csv - sequences that can't self-assemble
*   data_all_3descriptor.csv - Multi-class raw data
* preprocess_dataset contains preprocessed data
*   dataset_2class_1descriptor.csv - only calculate mordred descriptor of 2 class
*   dataset_2class_1descriptor_onlypio.csv - only calculate piomed descriptor of 2 class
*   dataset_2class_2descriptor.csv - mordred and piomed descriptors of 2 class
*   dataset_2class_3descriptor.csv - mordred, piomed and one-hot descriptors of 2 class
*   dataset_multi-class_3descriptor.csv - mordred, piomed and one-hot descriptors of multi-class
## Source codes
* preprocess-2class.ipynb: preprocess 2-class data to generate each descriptor 
* preprocess-multi-class_3descriptor.ipynb: preprocess multi-class data to generate 3 descriptors
* train-2class-only_mordred.ipynb: Only the mordred descriptor of 2 class was used for training and prediction
* train-2class-only_pio.ipynb: Only the piomed descriptor of 2 class was used for training and prediction
* train-2class-2descriptor.ipynb: Mordred and piomed descriptors of 2 class were used for training and prediction
* train-2class-3descriptor.ipynb: Mordred, piomed and one-hot adescriptors of 2 class were used for training and prediction
* train-multi-class-3descriptor.ipynb: Mordred, piomed and one-hot adescriptors of multi-class were used for training and prediction
* net_2class contains the network codes
*   main.py: train the network model and make predictions
*   model.py: details of 2class network model
## Requirements
* Python == 3.9.18
* PyTorch == 1.12.0
* sklearn == 1.3.1
* Numpy == 1.26.0
* Pandas == 2.1.1
* Mordred == 1.2.0
* matplotlib == 3.8.0
* pybiomed == 1.0
* rdkit == 2023.3.3
* seaborn == 0.13.0
## Operation steps
1. Install dependencies, including torch1.12, sklearn, numpy and pandas ...
2. run preprocess-2class.ipynb and train-2class-XXX.ipynb to get 2-class result
3. run preprocess-multi-class.ipynb and train-multi-class-XXX.ipynb to get multi-class result
4. run main.py to get 2-class network result
## Installation
git clone xxx
