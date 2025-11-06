# **Identifying Prognostic Microenvironment Niches in Ovarian Cancer from Multiplex Immunofluorescence Images Using NaroNet**
Eduardo Amo González

This repository contains the implementation and adaptation of **NaroNet**, a graph-based neural network designed to analyze high-dimensional **multiplex immunofluorescence (mIF)** images and uncover the **hierarchical spatial organization** of the tumor microenvironment (TME).  
In this project, NaroNet is applied to a cohort of mIF images from **142 patients diagnosed with High-Grade Serous Ovarian Cancer (HGSOC)** to identify **prognostic microenvironmental niches** or spatial tissue regions whose cellular composition and organization are predictive of patient outcomes.

All the code can be found in the following github repo: 

---

## Overview

**NaroNet** (Neighborhood-Aware Representation of the Microenvironment through Neural Networks) models the **Tumor Microenvironment (TME)** as a hierarchy of three biologically interpretable layers:

- **Phenotypes** : clusters of cellular patches defined by biomarker expression.  
- **Neighborhoods** : spatial interactions between phenotypes within local regions.  
- **Areas** : higher-order structures integrating multiple neighborhoods across the tissue.  

By learning this multiscale representation, NaroNet captures how **spatial organization of immune and stromal cells** correlates with clinical outcomes such as **survival** or **relapse**.  
This approach bridges histopathology, spatial omics, and machine learning to support **explainable and biologically grounded insights** into cancer architecture.

---

## NaroNet environment

This project was developed using a **Conda environment** closely mirroring the original NaroNet software setup.  
Maintaining consistent versions is essential to ensure compatibility with legacy dependencies.

1. The provided file **`NaroNet_env.yml`** defines the full software environment used in this project, including all Conda and pip dependencies.

2. To create and activate the environment, run:

   ```bash
   conda env create -f NaroNet_env.yml
   conda activate NaroNet
   ```
3. Key environment features:

  Uses Python 3.8, PyTorch 1.8.0, and CUDA 11.1 to ensure GPU compatibility and alignment with the original implementation.
  
- Includes all major NaroNet dependencies and packages for Data analysis (pandas, numpy, scikit-learn), visualization (matplotlib, seaborn)

- Also includes all pip-installed dependencies required by the project (e.g., opencv-python, tqdm, scikit-image, albumentations).

- Using newer versions of Python or PyTorch may lead to dependency conflicts or incompatibility with legacy code.

## Patient Data Analysis
The file data_analysis_edu.ipynb performs the exploratory and preprocessing steps required for clinical data analysis of the 142 HGSOC patients.
It includes:

- Loading and harmonizing clinical metadata (survival, relapse, age, treatment).

- Linking image filenames with patient records and cleaning missing or inconsistent entries.

- Clustering patients into risk groups using both percentile-based and K-Means approaches on survival and relapse times.

- Generating visualizations of variable distributions with custom plotting utilities (barplotdf, boxplotdf, violinplotdf).

This notebook serves as both a descriptive overview of the patient cohort and a preprocessing pipeline preparing categorical variables for model training.

---

## Using NaroNet
The version of NaroNet including all my minor modifications is included as a downloadable zip file
### main.py
The central script controlling NaroNet execution.
It guides the flow through the following modules:

- Image preprocessing: normalization, patch extraction, and contrastive embedding.

- Patch Contrastive Learning (PCL): learns compact feature representations for tissue patches.

- Architecture Search: explores multiple model configurations to identify the optimal one.

- Training and Prediction: fits the NaroNet model to classify or predict patient outcomes.

- Bio-Insight Extraction: identifies tissue regions and microenvironmental niches driving predictions.

All experiment-specific configurations are loaded from DatasetParameters.py.

### DatasetParameters.py
The full parameter set needs to be specified for each experiment. In the case of the project, the defined architecture was for example:
```python
elif 'eduamgo' in path: # Explore version
        # Patch contrastive learning parameters
        args['PCL_embedding_dimensions'] = 256
        args['PCL_batch_size']=128
        args['PCL_epochs']=1000
        args['PCL_patch_size']=45
        args['PCL_alpha_L']=1.2 # The value of alpha_L in the manuscript
        args['PCL_ZscoreNormalization']=True        
        args['PCL_width_CNN']=2 # [1, 2, 4]           
        args['PCL_depth_CNN']=101#4 # [1, 2, 4] 

        # Label you want to infer with respect the images.
        args['experiment_Label'] = ['Survival_risk_quart_3', 'Survival_risk_quart_4', 'Survival_risk_km_3','Survival_risk_km_4', 'Relapse_risk_quart_3', 'Relapse_risk_quart_4', 'Relapse_risk_km_3', 'Relapse_risk_km_4']
        
        # Optimization Parameters
        args['num_samples_architecture_search'] = 100 # [40, 200, 400, 500] # Markus reccomended 500
        args['epochs'] = 40 #if debug=='Index' else hp.quniform('epochs', 20, 100, 1) if debug=='Object' else 30
        args['epoch'] = 0
        args['lr_decay_factor'] = 0.5 #if debug=='Index' else hp.uniform('lr_decay_factor', 0, 0.75) if debug=='Object' else 0.5
        args['lr_decay_step_size'] = 12 #if debug=='Index' else hp.quniform('lr_decay_step_size', 2, 20, 1) if debug=='Object' else 12
        args['weight_decay'] = 0.01 #if debug=='Index' else hp.choice('weight_decay',[0.1,0.01,0.001,0.0001,0.00001]) if debug=='Object' else 0.01
        args['batch_size'] = 6 #if debug=='Index' else hp.choice('batch_size', [6,12,16,20]) if debug=='Object' else 6
        args['lr'] = 0.0001 #if debug=='Index' else hp.choice('lr', [0.0005, 0.0001, 0.00005,0.00001]) if debug=='Object' else 0.0001
        args['useOptimizer'] = 'ADAM' #if debug=='Index' else hp.choice("useOptimizer", ['ADAM', 'ADAMW', 'ADABound']) if debug=='Object' else 'ADAM'

        # General
        args['context_size'] = 25 #if debug=='Index' else hp.choice("context_size", [10, 15, 25, 50]) if debug=='Object' else 15  #A larger value = more context captured, but also more noise and higher computational cost.
        args['num_classes'] = 3
        args['MultiClass_Classification']=1
        args['showHowNetworkIsTraining'] = False # Creates a GIF of the learning clusters!
        args['visualizeClusters'] = True
        args['learnSupvsdClust'] = True
        args['recalculate'] = False
        args['folds'] = 10 # 10 #He always does a 10 fold validation
        args['device'] = 'cuda' # This allowed me to do it with GPU, working fine
        args['normalizeFeats'] = True #if debug=='Index' else hp.choice("normalizeFeats", [True,False]) if debug=='Object' else False        
        args['normalizeCells'] = False #if debug=='Index' else hp.choice("normalizeCells", [True,False]) if debug=='Object' else False        
        args['Batch_Normalization'] = True #if debug=='Index' else hp.choice("Batch_Normalization", [True,False]) if debug=='Object' else True
        args['normalizePercentile'] = False #if debug=='Index' else hp.choice("normalizePercentile", [True,False]) if debug=='Object' else False
        args['dataAugmentationPerc'] = 0.0001 #if debug=='Index' else hp.choice("dataAugmentationPerc", [0,0.0001,0.001,0.01,0.1]) if debug=='Object' else 0.0001

        # Neural Network
        args['hiddens'] = 44 #if debug=='Index' else hp.choice('hiddens', [20,32,44,64,96]) if debug=='Object' else 32 # sometimes he tests a shortened version for this, but i wanna be careful              
        args['clusters1'] = 6 #if debug=='Index' else hp.choice('clusters1',[6,7,8,9,10,11,12]) if debug=='Object' else 10      #Phenotypes 
        args['clusters2'] = 21 #if debug=='Index' else hp.choice('clusters2',[6,9,12,15,18,21]) if debug=='Object' else 9        #Neighborhoods
        args['clusters3'] = 17 #if debug=='Index' else hp.choice('clusters3',[5,8,11,14,17,20]) if debug=='Object' else 17         # Areas
        args['LSTM'] = False # I leave it to False as we don't have temporal sequences
        args['GLORE'] = False #if debug=='Index' else hp.choice('GLORE',[True,False]) if debug=='Object' else False # Might leave it as True in the future as it improves interpretability, although he usually leaves it as false
        args['Phenotypes'] = True  
        args['DeepSimple'] = False
        args['isAttentionLayer'] = False #if debug else hp.choice("isAttentionLayer", [True,False]) # Although he usually leaves it as false
        args['ClusteringOrAttention'] = True #if debug=='Index' else hp.choice("ClusteringOrAttention", [True,False]) if debug=='Object' else True        
        args['1cell1cluster'] = False #if debug=='Index' else hp.choice("1cell1cluster", [True,False]) if debug=='Object' else False                
        args['dropoutRate'] = 0.05 #if debug else hp.uniform('dropoutRate', 0.01, 0.1) # 7 if debug=='Index' else hp.choice('dropoutRate', [0.05, 0.1, 0.15, 0.2,0.25,0.3,0.35,0.40]) if debug=='Object' else 0.4       
        args['AttntnSparsenss'] = False #if debug else hp.choice("AttntnSparsenss", [True,False])
        args['attntnThreshold'] = 0.6 #if debug=='Index' else hp.choice('attntnThreshold', [0,.2,.4,.6,.8]) if debug=='Object' else 0  
        args['GraphConvolution'] = 'IncepNet' #if debug=='Index' else hp.choice('GraphConvolution', ['ResNet', 'IncepNet', 'JKNet']) if debug=='Object' else 'ResNet'                
        args['n-hops'] = 1 #if debug=='Index' else hp.choice('n-hops', [1, 2, 3]) if debug=='Object' else 3                        
        args['modeltype'] = 'SAGE'# I keep SAGE as it seems like the tendency
        args['ObjectiveCluster'] = True #if debug else hp.choice('ObjectiveCluster', [True, False])
        args['ReadoutFunction'] = False #if debug else hp.choice('ReadoutFunction', ['MAX', 'SUM', 'DeepSets']) # Choose the readout function    
        args['NearestNeighborClassification'] = False #if debug == 'Index' else hp.choice('NearestNeighborClassification', [True, False]) if debug == 'Object' else False
        args['NearestNeighborClassification_Lambda0'] = 0.01 #if debug == 'Index' else hp.choice("NearestNeighborClassification_Lambda0", [0.1, 0.01, 0.001, 0.0001]) if debug == 'Object' else 0.01
        args['NearestNeighborClassification_Lambda1'] = 0.01 #if debug == 'Index' else hp.choice("NearestNeighborClassification_Lambda1", [0.1, 0.01, 0.001, 0.0001]) if debug == 'Object' else 0.01
        args['NearestNeighborClassification_Lambda2'] = 0.01 #if debug == 'Index' else hp.choice("NearestNeighborClassification_Lambda2", [0.1, 0.01, 0.001, 0.0001]) if debug == 'Object' else 0.01
        args['KinNearestNeighbors'] = 5 #if debug == 'Index' else hp.choice('KinNearestNeighbors', [5, 10]) if debug == 'Object' else 5

        # Losses
        args['pearsonCoeffSUP'] = False #if debug == 'Index' else hp.choice("pearsonCoeffSUP", [True, False]) if debug == 'Object' else False
        args['pearsonCoeffUNSUP'] = False #if debug == 'Index' else hp.choice("pearsonCoeffUNSUP", [True, False]) if debug == 'Object' else False
        args['orthoColor'] = True #if debug=='Index' else hp.choice("orthoColor", [True,False]) if debug=='Object' else True
        args['orthoColor_Lambda0'] = 0.1 #if debug=='Index' else hp.choice("orthoColor_Lambda0", [0.1,0.01,0.001,0.0001,0.00001]) if debug=='Object' else 0.1                                
        args['orthoColor_Lambda1'] = 0.00001 #if debug=='Index' else hp.choice("orthoColor_Lambda1", [0.1,0.01,0.001,0.0001,0.00001]) if debug=='Object' else 0.00001                              
        args['ortho'] = False #if debug=='Index' else hp.choice("ortho", [True,False]) if debug=='Object' else False                        
        args['ortho_Lambda0'] = 0.1 #if debug=='Index' else hp.choice("ortho_Lambda0", [0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.1                                
        args['ortho_Lambda1'] = 0 #if debug=='Index' else hp.choice("ortho_Lambda1", [0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0                                
        args['ortho_Lambda2'] = 0 #if debug=='Index' else hp.choice("ortho_Lambda2", [0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0                                
        args['min_Cell_entropy'] = True #if debug=='Index' else hp.choice("min_Cell_entropy", [True,False]) if debug=='Object' else True                                        
        args['min_Cell_entropy_Lambda0'] = 1 #if debug=='Index' else hp.choice("min_Cell_entropy_Lambda0", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 1   
        args['min_Cell_entropy_Lambda1'] = 0.0001 #if debug=='Index' else hp.choice("min_Cell_entropy_Lambda1", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.0001
        args['min_Cell_entropy_Lambda2'] = 0.01 #if debug=='Index' else hp.choice("min_Cell_entropy_Lambda2", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.01
        args['MinCut'] = True #if debug=='Index' else hp.choice("MinCut", [True,False]) if debug=='Object' else True        
        args['MinCut_Lambda0'] = 0 #if debug=='Index' else hp.choice("MinCut_Lambda0", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0        
        args['MinCut_Lambda1'] = 0.1 #if debug=='Index' else hp.choice("MinCut_Lambda1", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.1        
        args['MinCut_Lambda2'] = 0.1 #if debug=='Index' else hp.choice("MinCut_Lambda2", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.1        
        args['F-test'] = False #if debug == 'Index' else hp.choice("F-test", [True, False])
        args['Max_Pat_Entropy'] = False #if debug=='Index' else hp.choice('Max_Pat_Entropy', [True, False]) if debug=='Object' else False                
        args['Max_Pat_Entropy_Lambda0'] = 0.0001 #if debug=='Index' else hp.choice("Max_Pat_Entropy_Lambda0", [1,0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.0001        
        args['Max_Pat_Entropy_Lambda1'] = 0.1 #if debug=='Index' else hp.choice("Max_Pat_Entropy_Lambda1", [1,0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.1        
        args['Max_Pat_Entropy_Lambda2'] = 0.1 #if debug=='Index' else hp.choice("Max_Pat_Entropy_Lambda2", [1,0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.1        
        args['UnsupContrast'] = False #1 if debug else hp.choice("UnsupContrast", [True,False])
        args['UnsupContrast_Lambda0'] = 0 #if debug == 'Index' else hp.choice("UnsupContrast_Lambda0", [1, 0.1, 0.01, 0.001, 0.0001]) if debug == 'Object' else 0
        args['UnsupContrast_Lambda1'] = 0 #if debug == 'Index' else hp.choice("UnsupContrast_Lambda1", [1, 0.1, 0.01, 0.001, 0.0001]) if debug == 'Object' else 0
        args['UnsupContrast_Lambda2'] = 0 #if debug == 'Index' else hp.choice("UnsupContrast_Lambda2", [1, 0.1, 0.01, 0.001, 0.0001]) if debug == 'Object' else 2    
        args['Lasso_Feat_Selection'] = False #if debug=='Index' else hp.choice("Lasso_Feat_Selection", [True,False]) if debug=='Object' else False         
        args['Lasso_Feat_Selection_Lambda0'] = 0.1 # if debug=='Index' else hp.choice("Lasso_Feat_Selection_Lambda0", [1,0.1, 0.01,0.001,0]) if debug=='Object' else 0.1         
        args['SupervisedLearning_Lambda0'] = 1 #if debug=='Index' else hp.choice("SupervisedLearning_Lambda0", [1,0.1, 0.01,0.001,0.0001]) if debug=='Object' else 1         
        args['SupervisedLearning_Lambda1'] = 1 #if debug=='Index' else hp.choice("SupervisedLearning_Lambda1", [1,0.1, 0.01,0.001,0.0001]) if debug=='Object' else 1         
        args['SupervisedLearning_Lambda2'] = 1 #if debug=='Index' else hp.choice("SupervisedLearning_Lambda2", [1,0.1, 0.01,0.001,0.0001]) if debug=='Object' else 1         
        args['SupervisedLearning_Lambda3'] = 1 #if debug=='Index' else hp.choice("SupervisedLearning_Lambda3", [1,0.1, 0.01,0.001,0.0001]) if debug=='Object' else 1         
        args['SupervisedLearning'] = True #if debug == 'Index' else hp.choice("SupervisedLearning", [True,False]) if debug=='Object' else True

```

### architecture_search.py
Implements the automated architecture search routine.
It balances computational cost and model interpretability, identifying the best-performing configuration for the ovarian cancer dataset.

## Auxiliary Scripts
A series of utility scripts support:

- **Architecture search processor** : processes the result of the architecture search experiment in order to visualise the parameters more clearly
- **Image cleaner** : ensures that the image files are named properly and allows to move some batches to a sepparate folder if needed
- **Tiff plotting** : framework for visualising any tiff image as a png image for easier interpretation


## Sbatch Scripts
The sbatch_scripts/ directory includes SLURM-compatible job submission scripts for high-performance computing (HPC) clusters.
These scripts have automated:

- Environment activation and dependency setup.
- GPU resource allocation.
