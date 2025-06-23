#Skeleton-Based Action Recognition using ST-GCN#

This repository contains the source code for a graduation project on real-time human action recognition. The system uses skeleton data extracted from standard 2D videos and classifies actions using a Spatio-Temporal Graph Convolutional Network (ST-GCN).


#Abstract#

This project presents a robust system for human action recognition that operates on real-time video streams from a standard webcam. By leveraging a skeleton-based approach, the system models the spatio-temporal dynamics of human movement, making it inherently invariant to superficial visual details like clothing, background, and lighting conditions. We utilize the YOLOv8-Pose model for efficient skeleton extraction and a lightweight ST-GCN model for classification. The model was trained on a 5-class subset of the KTH Human Actions dataset and achieved an overall test accuracy of 93.6%.

#Features#

Real-Time Performance: Classifies actions in real-time from a standard webcam feed.

High Accuracy: Achieves 93.6% accuracy on the unseen test set.

Robustness: Focuses on skeletal movement, making it insensitive to changes in background, lighting, or apparel.

5 Action Classes: Successfully recognizes the following actions:

Boxing 

Hand Clapping 

Hand Waving 

Running 

Walking 


#Dataset#

This project utilizes the KTH Human Actions Dataset. The dataset contains videos of 25 subjects performing six different actions in various scenarios. For this project, 5 of the 6 action classes were used.
Note: Due to its large size, the dataset is not included in this repository. You need to download it from the official website and process it using the generate_dataset.py script included in this project.


#Installation#

1.Clone the repository:

Bash:

git clone https://github.com/KULLANICI_ADINIZ/PROJE_ADINIZ.git
cd PROJE_ADINIZ

2. Install dependencies:
 
Bash:

pip install -r requirements.txt

4. Download Pre-Tranied Model:

Download the trained model weights (kth_best_model_5_class.pth) from the link below and place the file inside a trained_models/ directory in the project's root folder.

https://drive.google.com/drive/folders/12ZEUeiXZXcVXIpJhlBW4QeikuXk9D17-?usp=drive_link


#USAGE#

The project includes scripts for data preparation, training, evaluation, and a live demonstration.

1. Data Preparation:
   
To process the raw KTH dataset videos into skeleton data in .npy format, run:

Bash:

python src/generate_dataset.py

2. Training:

To train the model from scratch on the prepared dataset, run:

Bash:

python src/train.py

The script will train the model for 50 epochs and save the best performing model based on validation accuracy as kth_best_model_5_class.pth.

3. Evaluation:
   
To evaluate the performance of the trained model on the test set, run:

Bash

python src/test.py

This will print a classification report and save a confusion matrix image.

4. Live Demo
   
To run the live action recognition demo using your webcam, make sure the pre-trained model is in the correct directory and run:

Bash:

python src/live_demo.py


#Results#

The model achieved high performance on the test set, which was completely held out from the training and validation processes.

Overall Test Accuracy: 93.60% 

Classification Report:

Class                    	Precision	Recall	F1-Score

boxing               	     0.9459	0.9790	0.9622

handclapping               0.9026	0.9653	0.9329

handwaving          	      0.9924	0.9097	0.9493
 
running	                   0.9241	0.9306	0.9273

walking	                   0.9214	0.8958	0.9085


#Model Architecture#

The system uses a lightweight ST-GCNv2 model implemented in PyTorch. The architecture is designed for fast training and inference, consisting of 4 main convolution layers (2 spatial and 2 temporal).
This structure allows the model to learn both the spatial relationships between body joints and their temporal patterns over time simultaneously, which is key to distinguishing different actions.

