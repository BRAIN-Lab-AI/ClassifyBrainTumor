# Brain Tumor Detection and Classification using Deep Learning

# Enhancing Brain Tumor Detection in Low-Quality MRI Scans Using Transfer Learning

## Introduction
This project develops a deep learning model aimed at accurately classifying brain tumors from MRI scans, focusing especially on cases with low-quality images. By improving existing transfer learning methods through changes in the model architecture and advanced data manipulation techniques—including both image augmentation and degradation—the approach achieves better diagnostic results. The performance of the enhanced model is tested across several tumor categories: normal, pituitary, meningioma, and glioma. This work aims to address common diagnostic challenges in under-resourced medical settings, making reliable brain tumor detection more accessible and effective.
## Project Metadata
### Authors
- Oulaya Elargab
- **Supervisor Name:** Dr. Muzammil Behzad
- **Affiliations:** KFUPM

### Project Documents
- **Presentation:** [Project Presentation](/presentation.pptx)
- **Report:** [Project Report](/report.pdf)

### Reference Paper
- [Brain Tumor Classification using Deep Learning Algorithms][https://arxiv.org/abs/2112.10752](https://www.ijraset.com/best-journal/brain-tumor-classification-using-deep-learning-algorithms)

### Reference Dataset
- [MRI images Dataset]([https://laion.ai/blog/laion-5b/](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)


## Project Technicalities

### Terminologies
- **MRI (Magnetic Resonance Imaging):** A non-invasive imaging technique used to visualize detailed internal structures, especially soft tissues like the brain.
- **Brain Tumors (Glioma, Meningioma, Pituitary):** The project focuses on classifying MRI scans into three tumor types (glioma, meningioma, pituitary) and a "no tumor" category.
- **Transfer Learning:** A technique where models pretrained on large datasets (e.g., ImageNet) are adapted to a new, domain-specific task by modifying and fine-tuning their architecture.
- **Degraded Images:** Gaussian Noise: Random pixel noise mimicking sensor errors.
Blurring: Simulating motion or focus loss. Downsampling: Reducing and re-scaling image resolution.
- **Dropout:** A regularization technique that randomly disables a fraction of neurons during training to prevent overfitting.

### Problem Statements
- **Problem 1:** Dealing with poor-quality MRI scans commonly encountered in real clinical settings, where imaging artifacts such as noise, blur, and low resolution affect diagnosis accuracy.
- **Problem 2:** Existing models struggle to maintain diagnostic reliability under practical clinical conditions, especially in resource-limited settings where imaging quality cannot always be assured.


### Loopholes or Research Areas
- **Evaluation Under Degraded Conditions:** Existing studies rarely evaluate model performance under realistic low-quality MRI conditions, limiting their clinical applicability.
- **Generalization and Robustness:** Many transfer learning models struggle to maintain consistent performance across diverse and imperfect imaging data, leading to potential diagnostic errors.
- **Resource Limitations in Deployment:** High-performing models often require computational resources that are impractical for use in under-resourced medical environments.

### Problem vs. Ideation: Proposed 3 Ideas to Solve the Problems
1. **Simulating Realistic Image Degradation:** Apply controlled degradation techniques (Gaussian noise, blurring, downsampling) to training data to simulate real-world MRI variability and train models to be resilient.
2. **Enhancing Transfer Learning Architectures:** Deepen the classification head with additional dense layers and apply dropout regularization to improve the model’s learning capacity and generalization across challenging conditions.
3. **Adopting Lightweight Fine-Tuning Strategies** Freeze pretrained base layers and fine-tune only the added classification layers with minimal computational overhead, enabling practical deployment even in environments with limited hardware resources.

### Proposed Solution: Code-Based Implementation
This repository provides an implementation of enhanced transfer learning models for brain tumor classification using TensorFlow/Keras.
The solution addresses real-world challenges by simulating degraded MRI images and optimizing model robustness.

- **Enhanced Transfer Learning Architectures:** Pretrained models (VGG16, VGG19, ResNet50V2) are adapted with deeper classification heads and dropout regularization to improve feature learning and generalization.
- **Degradation Simulation Pipeline:** Controlled degradation techniques (Gaussian noise, blurring, downsampling) are applied to MRI scans to simulate real-world imaging imperfections during training.
- **Optimized Training Strategy:** Fine-tuning strategies minimize computational cost by freezing base layers and training only newly added dense layers, using early stopping and learning rate scheduling.


### Key Components
- **`brain_tumor_classification.ipynb`**: The main Google Colab notebook containing all code for the project, including:
   - Loading and degrading MRI images
   - Building and training baseline and enhanced transfer learning models
   - Evaluating classification performance with accuracy, F1-score, and confusion matrices
- **Dataset Access**: To download the dataset from Kaggle in Colab:
   1. **Create a Kaggle API token**: 
         - Go to your Kaggle account:https://www.kaggle.com/account
         - Scroll down to **API** section and click **"Create New API Token"**
         - This will download a file named `kaggle.json`
   2. **Upload** the `kaggle.json` file in the notebook

## Model Workflow
The workflow of the enhanced brain tumor classification system is designed to improve model resilience under degraded imaging conditions through careful data processing, model fine-tuning, and evaluation:

1. **Input:**
   - **MRI Images:** Grayscale MRI scans resized to 224×224 pixels.
   - **Data Augmentation:** Randomized image degradation is applied to simulate real-world variability.
   - **Label Encoding:** Tumor classes (glioma, meningioma, pituitary tumor, no tumor) are encoded for multi-class classification.

2. **Training Process:**
   - **Feature Extraction:** Pretrained convolutional layers (frozen) extract key features from input images.
   - **Classification Head:** Enhanced dense layers with dropout refine the extracted features for final classification.
   - **Optimization:** Categorical cross-entropy loss and Adam optimizer guide the model during training, with real-time augmentation to enhance robustness.
   - **Validation Monitoring:** Validation loss and accuracy are tracked to apply early stopping and prevent overfitting.

3. **Evaluation:**
   - **Performance Metrics:** Model performance is assessed using accuracy, class-wise F1-scores, and confusion matrices.
   - **Robustness Testing:** Models are evaluated on both clean and degraded MRI datasets to verify real-world applicability.


## How to Run the Code

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/oulaya-arg/brain-tumor-mri-classification.git
   cd brain-tumor-mri-classification

    ```

2. **Open the Colab Notebook**
    Create a virtual environment and install the required dependencies.
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3. **Train the Model:**
    Configure the training parameters in the provided configuration file and run:
    ```bash
    python train.py --config configs/train_config.yaml
    ```

4. **Generate Images:**
    Once training is complete, use the inference script to generate images.
    ```bash
    python inference.py --checkpoint path/to/checkpoint.pt --input "A surreal landscape with mountains and rivers"
    ```

## Acknowledgments
- **Open-Source Communities:** Thanks to the contributors of PyTorch, Hugging Face, and other libraries for their amazing work.
- **Individuals:** Special thanks to bla, bla, bla for the amazing team effort, invaluable guidance and support throughout this project.
- **Resource Providers:** Gratitude to ABC-organization for providing the computational resources necessary for this project.



