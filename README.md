# README

**Release Tag -** [v1.0.0-TML25_A4_12](https://github.com/highgroundmaster/TML25_A4_12/releases/tag/v1.0.0-TML25_A4_12)

[https://github.com/highgroundmaster/TML25_A4_12/releases/tag/v1.0.0-TML25_A4_12](https://github.com/highgroundmaster/TML25_A4_12/releases/tag/v1.0.0-TML25_A4_12)

| Name | ID |
| --- | --- |
| Rithvika Pervala | 7072443 |
| Bushra Ashfaque | 7070555 |

---

# Task 1: Network Dissection - `/Task-1`

This task involved using the `CLIP-dissect` library to identify and compare the concepts learned by the neurons of two `ResNet18` models: one trained on ImageNet and another on Places365.

### Key Files for the Solution

- `describe_neurons.py`: The primary script from the `CLIP-dissect` library used to perform the neuron dissection.
- `data_utils.py`: **Modified file**. This script was edited to teach the program how to load the custom `resnet18_places365.pth` model. The `get_target_model` function was updated to handle this specific case.
- `dlbroden.sh`: **Modified file**. This shell script was edited to use `curl` instead of `wget` for downloading the Broden dataset on macOS.
- `TML25_A4_12_Task_1.ipynb`: **Primary analysis file**. A Jupyter Notebook created to load the final `.csv` results, perform the comparative analysis, and generate the plots used in the final report.
- `requirements.txt` - Python Packages required for running the code
- `report.pdf` - Report on Network Dissection Analysis with statistics.
- `results` - Folder containing results for each ResNet Model.

### How to Reproduce the Results

1. **Install Dependencies:** 
    - A `requirements.txt` file is present in the folder. Install the dependencies accordingly
        
        ```bash
        pip install -r requirements.txt
        ```
        
    - Alternatively, the below command can be used to install necessary libraries can be installed via pip.
        
        ```bash
        pip install torch torchvision matplotlib pandas similarity utils
        
        ```
        
2. **Set Up the Dataset:**
First, download and unzip the Broden probing dataset by running the modified download script. This only needs to be done once.
    
    ```bash
    ./dlbroden.sh
    
    ```
    
3. **Run Network Dissection:**
Execute the dissection script for both models. Note: This is a computationally intensive process that can take a significant amount of time, especially on a CPU.
    - **For the ResNet18 Places365 model:**
    
    ```bash
    python3 describe_neurons.py --target_model resnet18_places365 --target_layers layer2,layer3,layer4 --device cpu
    
    ```
    
    - **For the standard ResNet18 ImageNet model:**
    
    ```bash
    python3 describe_neurons.py --target_model resnet18 --target_layers layer2,layer3,layer4 --device cpu
    
    ```
    
4. **Analyze the Output:**
The scripts will generate two results folders (e.g., `results/resnet18_places365_25_07_21_01_25` and `results/resnet18_25_07_21_02_50`), each containing a `descriptions.csv` file. The final analysis and visualization can be performed by running the 
`TML25_A4_12_Task_1.ipynb` notebook.

---

# Task 2: Grad-CAM, AblationCAM, and ScoreCAM - `/Task-2`

This task involved using the `pytorch-grad-cam` library to visualize the model's focus for the 10 specified ImageNet images using three different Class Activation Map (CAM) methods.

### Key Files

- `TML25_A4_12_Task_2.ipynb`: **Primary solution file**. This single Jupyter Notebook contains the complete solution for Task 2. The notebook:
    - Loads the pre-trained `ResNet50` model.
    - Downloads the 10 specified images from their URLs.
    - Initializes `GradCAM`, `AblationCAM`, and `ScoreCAM`.
    - Loops through each image to generate and plot the CAM visualizations.
- `requirements.txt` - Python Packages required for running the code.
- `report.pdf` - Report on comparison on Explainability of Grad-CAM, AblationCAM, and ScoreCAM.
- `CAM_results.png` - Plot comparing Grad-CAM, AblationCAM, and ScoreCAM heatmaps.

### How to Reproduce the Results

1. **Install Dependencies:** 
    - A `requirements.txt` file is present in the folder. Install the dependencies accordingly
        
        ```bash
        pip install -r requirements.txt
        ```
        
    - Alternatively, the below command can be used to install necessary libraries can be installed via pip.
        
        ```bash
        pip install pytorch-gradcam torch torchvision matplotlib Pillow
        
        ```
        
2. **Run the Notebook:**
The entire process is contained within the notebook. Open `TML25_A4_12_Task_2.ipynb` in a Jupyter environment and run all cells sequentially from top to bottom. The notebook will automatically download the images, run the models, and display the final plots.

---

# Task 3: LIME - `/Task-3`

- This task involves generating LIME explanations for 10 ImageNet images and visualizing and analyzing the LIME Masks, confidence.

### Key Files

- `TML25_A4_12_Task_3.ipynb`: **Primary solution file**. This single Jupyter Notebook contains the complete solution for Task 2. The notebook:
    - Loads the pre-trained `ResNet50` model.
    - Downloads the 10 specified images from their URLs.
    - Generate LIME Explanations in accordance with the custom complexity analysis functions.
    - Visualizes LIME Masks and segments of each image.
    - Submits the custom parameters per image to the evaluation server.
- `requirements.txt` - Python Packages required for running the code
- `lime_results.png` - PNG File containing the Plotted LIME Masks and Segments
- `params.pkl` - Pickle File containing the Parameters used to run LIME
- `report.pdf` - Report containing methodology, analysis and observations on LIME.

### How to Reproduce the Results

1. **Install Dependencies:** 
    - A `requirements.txt` file is present in the folder. Install the dependencies accordingly
        
        ```bash
        pip install -r requirements.txt
        ```
        
    - Alternatively, the below command can be used to install necessary libraries can be installed via pip.
        
        ```bash
        pip install lime torch torchvision matplotlib Pillow scikit-learn skimage numpy Requests
        
        ```
        
2. **Run the Notebook:**
    - The entire process is contained within the notebook - `TML25_A4_12_Task_3.ipynb`
    - Open `TML25_A4_12_Task_3.ipynb` in a Jupyter environment and run all cells sequentially from top to bottom.

---

# Task 4: GradCAM vs LIME - `/Task-4`

- This task involves making a comparison between the LIME and GradCAM results obtained from Task 2 and 3 respectively.

### Key Files

- `TML25_A4_12_Task_4.ipynb` - **Primary Analysis** Notebook containing snippets from `TML25_A4_12_Task_2.ipynb` and `TML25_A4_12_Task_3.ipynb` to compute combined IoU.
- `report.pdf` - Report Comparing LIME and GradCAM - IoU and other analysis.
- `IoU_Lime_GradCAM.png` - Plot comparing LIME and GradCAM with IoU annotations.
- `requirements.txt` - Python Packages required for running the code.

### How to Reproduce the Results

1. **Install Dependencies:** 
    - A `requirements.txt` file is present in the folder. Install the dependencies accordingly
        
        ```bash
        pip install -r requirements.txt
        ```
        
    - Alternatively, the below command can be used to install necessary libraries can be installed via pip.
        
        ```bash
        pip install pytorch-gradcam lime torch torchvision matplotlib Pillow scikit-learn skimage numpy
        ```
        
2. **Run the Notebook:**
    - The entire process is contained within the notebook - `TML25_A4_12_Task_4.ipynb`
    - Open `TML25_A4_12_Task_4.ipynb` in a Jupyter environment and run all cells sequentially from top to bottom.

---
