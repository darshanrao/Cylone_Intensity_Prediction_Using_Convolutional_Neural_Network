# Predicting Tropical Cyclone Intensity using Convolutional Neural Networks (CNN)

This repository contains the code and resources for a research project aimed at predicting the intensity of tropical cyclones (TCs) using an innovative approach based on image processing and deep learning techniques. The research paper associated with this project presents a Convolutional Neural Network (CNN) architecture modified to estimate TC intensity using meteorological geostationary satellite imagery.

## Abstract
Tropical cyclones, also known as typhoons or hurricanes, are among the most affecting types of weather phenomena. They can form in five oceans: Atlantic Ocean, North Pacific Ocean, Western Pacific Ocean, South Pacific Ocean, and Indian Ocean. Accurate monitoring and assessment of TC intensity before their formation are crucial for forecasting and mitigating human loss and suffering. Previous techniques for monitoring cyclones have shown limited effectiveness. However, this research proposes a novel image processing-based approach combined with deep learning for predicting TC intensity.

### Key Contributions
- Utilization of meteorological geostationary satellite imagery as the input data for TC intensity prediction.
- Modification of the CNN architecture to suit the cyclone intensity estimation task.
- Experimental results on a publicly available benchmark dataset, demonstrating the accuracy and stability of the proposed method under various scenarios.
- Highlighting the potential of data science in advancing meteorological research.

## Dataset
The research employs the Tropical Cyclone Image Dataset provided by Prof. Hsuan-Tien Lin from National Taiwan University. The dataset can be accessed and downloaded from the following link: [Tropical Cyclone Image Dataset](https://www.csie.ntu.edu.tw/~htlin/program/TCIR/).

## Code Organization
The code is organized as follows:
- `preprocess_and_train.ipynb`: Jupyter notebook containing the code for data preprocessing, including image augmentation and normalization, as well as the code for training the model using the preprocessed dataset.
- `evaluate.ipynb`: Jupyter notebook for evaluating the trained model on the test set and generating performance metrics.
- `ATLN_Model_Results.ipynb`: Jupyter notebook providing results for the model when used on Atlantic 2015-15 Dataset.
- `EPAC_Model_Results.ipynb`: Jupyter notebook providing results for the model when used on East Pacific 2015-15 Dataset.
- `WPAC_Model_Results.ipynb`: Jupyter notebook providing results for the model when used on West Pacific 2015-15 Dataset.
- `ATLN_Model_Results.ipynb`: Jupyter notebook providing results for the model when used on Overall(All three Regions)  2015-15 Dataset.

## Requirements
- Python 3.7 or higher
- TensorFlow 2.x
- NumPy
- Matplotlib
- PIL (Python Imaging Library)

## Usage
1. Clone this repository to your local machine: `git clone https://github.com/darshanrao/Cylone_Intensity_Prediction_Using_Neural_Network.git`
2. Download the Tropical Cyclone Image Dataset from the link provided above.
3. Place the downloaded dataset in the appropriate directory.
4. Open and run the Jupyter notebook `Model_Training.ipynb` to perform data preprocessing and training.
5. Open and run the Jupyter notebook `evaluate.ipynb` to evaluate the trained model on the test set and generate performance metrics.

## Acknowledgments
We acknowledge Prof. Hsuan-Tien Lin from National Taiwan University for providing the Tropical Cyclone Image Dataset used in this research. We also thank the open-source community for providing valuable tools and libraries that enabled this project.
