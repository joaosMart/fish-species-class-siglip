# Fish Species Classification with SigLIP Vision-Language Models

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joaosMart/fish-species-class-siglip/)

This repository contains the implementation of **"Temporal Aggregation of Vision-Language Features for High-Accuracy Fish Classification in Automated Monitoring"**, a state-of-the-art approach for detecting and classifying salmonid fish species using SigLIP vision-language models.

## 🎯 Key Results

- **99.1% accuracy** for fish detection (presence/absence)
- **98.2% accuracy** for multi-fish detection (single vs. multiple fish)
- **97.5% accuracy** for 3-way species classification (Trout, Salmon, Arctic Char)
- Robust zero-shot detection without fine-tuning

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Pipeline Components](#pipeline-components)
  - [1. Fish Detection](#1-fish-detection)
  - [2. Multi-Fish Detection](#2-multi-fish-detection)
  - [3. Feature Extraction](#3-feature-extraction)
  - [4. Species Classification](#4-species-classification)
- [Usage Guide](#usage-guide)
- [Model Details](#model-details)
- [Results](#results)
- [Citation](#citation)

## 🔬 Project Overview

This project implements a complete pipeline for automated fish monitoring in aquaculture environments:

1. **Detection**: Identify frames containing fish using zero-shot vision-language models
2. **Multiple Fish Detection**: To find which videos contain multiple fish instances and which have single fish instances.
3. **Classification**: Classify fish species using temporal feature aggregation.
  

### Project Structure

```
fish-species-class-siglip/
├── Code/
│   ├── fish-detection/
│   │   ├── Fish_Detection_and_Evaluation.ipynb      # Main detection experiments, validation and evaluation
│   │   └── SigLIP_fish_detection_prediction_savings.ipynb
│   ├── multi-fish-detection/
│   │   ├── Multi_Fish_Detection.ipynb               # Multi-fish detection experiments, validation and evaluation
│   │   ├── Frame_Level_Multi_Fish_Detection.ipynb   # Frame-level inference and saving of scores
│   │   └── Video_Level_MultiFish_Detection.ipynb    # Video-level detection and saving of videos
│   └── species-classification/
│       ├── Feature_Extraction.ipynb              # SigLIP and ResNet feature extraction
│       ├── Classification_Central_Frame_C.ipynb  # Central frame training
│       ├── Evaluation.ipynb                      # Models evaluation (SigLIP and ResNet features)
│       └── resnet_transfer.ipynb                 # fine-tuned ResNet baseline
├── LICENSE
└── README.md
```

## 🚀 Installation

### Option 1: Google Colab (Recommended)

All notebooks are designed to run in Google Colab. Simply click the "Open in Colab" badges in each notebook. 

### Option 2: Local Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/fish-species-class-siglip.git
cd fish-species-class-siglip
```


### System Requirements
- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM
- 50GB+ storage for datasets

## 📊 Dataset Setup

### Required Datasets

1. **Fish Detection Dataset**
   - [Validation Set](https://drive.google.com/drive/folders/1BAfJ8vT-DW_dbzK-oM0bzZDSMX-r-5nA?usp=drive_link)
   - Contains: `fish/` and `no_fish/` folders and a test_set_final folder. The test_set_final folder includes independent fish and no_fish folders  from the previous ones that was used for evaluation. 

2. **Multi-Fish Detection Dataset**
   - [Multiple vs. Single Fish Dataset](https://drive.google.com/drive/folders/13Q2YsdajGGiR3UzDW8_zdcwAunMZRGzb?usp=sharing)
   - Contains: `one_fish/` and `more_than_one_fish/` folders

3. **Species Classification Videos**
   - Available for sharing on a reasonable request to the Icelandic Marine and Freshwater Research Institute.
   - Three species: Brown/Sea Trout  (Urriði), Atlantic Salmon (Lax), Arctic Char (Bleikja)

### Dataset Structure

```
datasets/
├── detection_data/
│   └── validation_set/
│       ├── fish/          # 168 images with fish
│       ├── no_fish/       # 163 images without fish
│       └── test_set_final/
└── multi_fish_detection_data/
    └── validation_fish_counting/
        ├── one_fish/      # 100 single fish images
        └── more_than_one_fish/  # 100 multiple fish images

```

## 🔧 Pipeline Components

### 1. Fish Detection

**Notebook**: `Code/fish-detection/Fish_Detection_and_Evaluation.ipynb`

Shows the implementation and experiments performed that are connected to the zero-shot fish detection using SigLIP vision-language models with prompt engineering. This includes model performance, threshold selection, validation and evaluation. 

```python
# Key configuration
MODEL_NAME = 'ViT-SO400M-14-SigLIP'
CHECKPOINT = 'webli'
DETECTION_THRESHOLD = 0.977989  # Optimized threshold

# Ensemble prompts for fish detection
positive_prompts = [
    "Salmon-like fish swimming",
    "An underwater photo of a salmon-like fish seen clearly swimming.",
    "Image of salmon-like fish in a contained environment.",
    "A photo of a salmon-like fish in a controlled river environment.",
    "Image of at least one salmon-like fish in a contained environment."
]

negative_prompts = [
    "An image of an empty white water container.",
    "A contained environment with nothing in it.",
    "An image of a empty container with nothing in it."
]
```

**Key Features:**
- Model architecture comparison (9 different CLIP/SigLIP/EVA variants)
- Prompt engineering with ensemble averaging
- Threshold optimization via F1-score maximization
- Comprehensive evaluation metrics

**Notebook**: `Code/fish-detection/SigLIP_fish_detection_prediction_savings.ipynb`

Using the set of best hyper-parameters (model, prompts, and threshold) defined in the last notebook it performs the detection, by filtering and saving the data for the next phase.

### 2. Multi-Fish Detection

**Notebook**: `Code/multi-fish-detection/Multi_Fish_Detection.ipynb`

Distinguishes between single and multiple fish in frames. It includes all the exeriments connected with the frame-level and video-level multiple fish detection. For the frame-level detection it includes the experiments with prompt engineering, cross-validation, threshold selection and evaluation.

```python
# Optimized prompts for multi-fish detection
MULTIPLE_FISH_PROMPTS = [
    "Salmon-like fishes swimming",
    "Image of two or more salmon-like fish in a contained environment."
]

SINGLE_FISH_PROMPTS = [
    "Clear image of a single fish swimming in a river."
]
```
**Key Features:**
- Zero-shot classification for fish counting
- Cross-validation with 30 repetitions
- ROC analysis and threshold optimization

**Notebooks**:
- `Code/multi-fish-detection/Frame_Level_Multi_Fish_Detection.ipynb`
- `Code/multi-fish-detection/Video_Level_MultiFish_Detection.ipynb`

Destinguish between video instances with one fish and more fish. Videos with more than one fish are not passed to the next phase.

### 3. Feature Extraction

**Notebook**: `Code/species-classification/Feature_Extraction.ipynb`

Extracts SigLIP and ResNet-50 features from video frames for species classification. First selects the central 11 frames and then extracts features for all the frames using the SigLIP model or only for the central frame using the central frame.

```python
# Feature extraction configuration
WINDOW_SIZE = 11  # frames for temporal aggregation
MODEL_NAME = 'ViT-SO400M-14-SigLIP' or 'ResNet-50' 

# Center expansion strategy for frame selection
def select_frames_by_center_expansion(video_data, window_size=11):
    """Select frames with highest mean probability"""
    # Implementation details in notebook
    

class FeatureExtractor:
    """
    SigLIP-based feature extractor for fish species classification.

    Uses ViT-SO400M-14-SigLIP model as feature extractor.
    """
    # Implementation details in notebook

class ResNetFeatureExtractor:
    """
    ResNet-50 feature extractor for baseline comparison.

    Implements ResNet-50 feature extraction as described in the paper
    for comparison with SigLIP-based approach.
    """
    # Implementation details in notebook


```

**Key Features:**
- Temporal frame selection (11-frame windows)
- SigLIP feature extraction (1152-dimensional)
- ResNet-50 baseline feature extraction (central frame only
- Batch processing for efficiency

### 4. Species Classification

**Notebook**: `Code/species-classification/Classification_Central_Frame_C.ipynb`

Classifies fish species using extracted features.

```python
# Classification models
models = {
    'SVM': LinearSVC(C=optimized_C, class_weight='balanced'),
    'LogisticRegression': LogisticRegression(C=optimized_C, class_weight='balanced')
}

# Three species
class_names = ['Bleikja', 'Lax', 'Urridi']  # Arctic Char, Salmon, Trout
```

**Key Features:**
- Multi-seed evaluation (10 random seeds)
- Hyperparameter optimization via random search
- Balanced class weights for imbalanced data
- Comprehensive metrics reporting

## 📖 Usage Guide

### Complete Pipeline Workflow

#### Step 1: Fish Detection on Videos

```python
# In SigLIP_fish_detection_prediction_savings.ipynb

# Define your video list
fish_path_list = [
    "/path/to/video1.mp4",
    "/path/to/video2.mp4",
    "/path/to/video3.mp4"
]

# Run detection (automatic checkpointing every 10 videos)
# Output: Scores-ViT-SO400M-14-SigLIP.pkl
```

#### Step 2: Filter Detected Frames

```python
# Apply threshold to get frames with fish
DETECTION_THRESHOLD = 0.977989

# Process and filter results
# Output: fish_detection_results.json
```

#### Step 3: Extract Features

```python
# In Feature_Extraction.ipynb

# Load filtered frames
filtered_data = load_filtered_data('filtered_data_class.json')

# Extract SigLIP features
# Output: features/ViT-SO400M-14-SigLIP/*.npz
```

#### Step 4: Classify Species

```python
# In Classification_Central_Frame_C.ipynb

# Load features and train classifier
data_dir = "features/ViT-SO400M-14-SigLIP"
results = run_multi_seed_optimization(
    data_dir=data_dir,
    class_names=['Bleikja', 'Lax', 'Urridi'],
    n_seeds=10
)
```

### Quick Start Examples

#### Example 1: Single Image Fish Detection

```python
import open_clip
from PIL import Image
import torch

# Load model
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-SO400M-14-SigLIP', pretrained='webli'
)
tokenizer = open_clip.get_tokenizer('ViT-SO400M-14-SigLIP')

# Process image
image = preprocess(Image.open("test_image.jpg")).unsqueeze(0)
text = tokenizer(["A fish", "No fish"])

# Get predictions
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    print(f"Fish probability: {similarity[0,0]:.2%}")
```

#### Example 2: Batch Video Processing

```python
# Process multiple videos efficiently
from tqdm import tqdm

results = {}
for video_path in tqdm(video_list):
    video_results = process_video_batch(
        video_path, 
        text_features, 
        batch_size=128
    )
    results[video_path] = video_results
```

## 🤖 Model Details

### SigLIP Architecture

- **Model**: ViT-SO400M-14-SigLIP
- **Pretraining**: WebLI dataset
- **Features**: 1152-dimensional normalized vectors
- **Context**: Vision-language alignment without fine-tuning

### Performance Comparison

| Model | Fish Detection F1 | Species Classification F1 |
|-------|------------------|--------------------------|
| ViT-SO400M-14-SigLIP | **99.1%** | **96.8%** |
| ViT-B-16-SigLIP | 98.5% | 95.2% |
| ResNet-50 (baseline) | - | 89.3% |
| ViT-L-14 CLIP | 94.2% | 91.5% |

### Optimal Hyperparameters

- **Fish Detection Threshold**: 0.977989
- **Multiple Fish Detection Threshold**: 0.962
- **SVM C parameter**: ~10-50 (varies by seed)
- **Temporal Window**: 11 frames


## 📊 Results

### Fish Detection Performance
- **Accuracy**: 99.1%
- **Precision**: 100%
- **Recall**: 98.2%
- **F1-Score**: 99.1%

### Species Classification Performance (3-way)
- **Balanced Accuracy**: 96.5%
- **Macro F1-Score**: 96.8%
- **Per-species F1**: 
  - Arctic Char: 97.2%
  - Salmon: 96.1%
  - Trout: 97.1%

### Multi-Fish Detection
- **Accuracy**: 99.4%
- **Single vs Multiple F1**: 99.3%

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{YourPaper2024,
  title={Temporal Aggregation of Vision-Language Features for High-Accuracy Fish Classification in Automated Monitoring},
  author={Your Name and Collaborators},
  journal={Journal Name},
  year={2024}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or collaborations, please contact:
- Email: joao.da.silva.martins@hafogvatn.is
- GitHub Issues: [Create an issue](https://github.com/yourusername/fish-species-class-siglip/issues)

## 🙏 Acknowledgments

- Thanks to the [Marine & Freshwater Research Institute of Iceland](https://www.hafogvatn.is/en) for providing the video data. 
- OpenCLIP team for the SigLIP implementation
- Google Colab for computational resources
