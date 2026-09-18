**TITLE**
Explainable ResNet-Based Deep Learning Framework for Diabetic Retinopathy Classification

**Description**
This project implements a comparative analysis of ResNet architectures (ResNet-34, ResNet-50, ResNet-101, and ResNet-152) using transfer learning for binary classification of Diabetic Retinopathy (DR) from retinal fundus images. The framework integrates Grad-CAM (Gradient-weighted Class Activation Mapping) to provide visual explanations highlighting clinically relevant retinal regions influencing model predictions. The study systematically benchmarks these architectures under identical experimental conditions, emphasizing performance, generalization, and explainability, enabling trustworthy AI deployment in clinical ophthalmology diagnostics.

**Dataset Information**
The model was trained on the Kaggle DR Detection dataset comprising 2,838 high-resolution retinal fundus images, evenly split between DR and No DR classes (1,408 DR and 1,430 No DR). An 80:20 split was applied, with 2,270 images for training (1,126 DR, 1,144 No DR) and 568 for testing (282 DR, 286 No DR).

**Code Overview**
The pipeline integrates:
1. Data Preprocessing & Augmentation
   - Resizing to 224×224 pixels, normalization using ImageNet mean and standard deviation, contrast enhancement, and Gaussian filtering for noise reduction.
2. Model Training (Transfer Learning)
   - Fine-tunes pre-trained ResNet-34, ResNet-50, ResNet-101, and ResNet-152 independently.
   - Uses Adam optimizer, learning rate = 0.0001, batch size = 16, 15 epochs, and binary cross-entropy loss.
   - Tracks training/testing loss and accuracy across epochs.
3. Performance Evaluation
   - Confusion matrix generation for training and testing sets.
   - Computation of accuracy, misclassification rate, specificity, recall, precision, negative predictive value, false positive rate, false negative rate, and F1-score.
4. Robustness Analysis
   - Five-fold cross-validation on the best-performing ResNet-34 model.
   - McNemar's test for statistical significance of performance differences between architectures.
5. Visualization & Interpretability
   - Confusion matrices
   - Training/testing loss and accuracy curves
   - Grad-CAM heatmaps highlighting retinal regions (hemorrhages, exudates) influencing predictions
   - Comparative analysis of Grad-CAM outputs across all four ResNet architectures

All analyses are implemented in PyTorch and TensorFlow, with plotting via Matplotlib and Seaborn.

**Usage Instructions**
1. Dataset Preparation
   - Organize the dataset directory as:
     ```
     /path/to/dataset/
        ├── DR/
        └── No_DR/
     
   - Update the path in the code:
     python
     data_dir = "/path/to/your/DR_dataset"
     ```
2. Run Training and Evaluation
   - Execute the Python script or Jupyter notebook:
     bash
     python ResNet_DR_Classification.py
     
     or open and run `ResNet_DR_Classification.ipynb` in Jupyter/Colab.
3. Outputs Generated
   - Training/validation accuracy and loss plots
   - Classification reports per model
   - Confusion matrices for training and testing sets
   - Five-fold cross-validation performance plots
   - McNemar's test statistical results
   - Grad-CAM interpretability heatmaps

**Requirements**
| Library | Version (Recommended) |
|----------|-----------------------|
| Python | ≥ 3.9 |
| PyTorch | ≥ 2.1 |
| torchvision | ≥ 0.16 |
| TensorFlow | ≥ 2.12 |
| numpy | ≥ 1.25 |
| scikit-learn | ≥ 1.3 |
| matplotlib | ≥ 3.8 |
| seaborn | ≥ 0.12 |
| OpenCV | ≥ 4.8 |

Install dependencies:
bash
pip install torch torchvision tensorflow numpy scikit-learn matplotlib seaborn opencv-python

**Methodology Summary**
| Component | Description |
|------------|-------------|
| Base Models | ResNet-34, ResNet-50, ResNet-101, ResNet-152 |
| Training Strategy | Transfer learning with fine-tuning of pre-trained models |
| Preprocessing | Resizing (224×224), normalization, contrast enhancement, Gaussian filtering |
| Dataset Split | 80% training / 20% testing with preserved class distribution |
| Evaluation Metrics | Accuracy, Misclassification Rate, Specificity, Recall, Precision, NPV, FPR, FNR, F1-Score |
| Robustness Analysis | Five-fold cross-validation, McNemar's test |
| Interpretability | Grad-CAM heatmaps highlighting retinal regions influencing predictions |

**Performance Summary**
| Model | Accuracy | F1-Score | Notes |
|--------|-----------|----------|-------|
| ResNet-34 | 98.9% | 0.989 | Best overall performance, lowest FPR (1.0%) and FNR (1.1%) |
| ResNet-50 | 98.1% | 0.981 | Stable performance, slightly lower than ResNet-34 |
| ResNet-101 | 98.4% | 0.984 | Highest specificity (99.0%), good balance |
| ResNet-152 | 97.89% | 0.979 | Highest misclassification rate (2.11%), overfitting tendency |

- McNemar's test: test statistic = 3.6818, p-value = 0.0550 (no statistically significant difference at 95% confidence level)
- Five-fold cross-validation (ResNet-34): average training accuracy = 98.84%, validation accuracy = 97.05%
- Average training loss = 0.0368, validation loss = 0.1078

**Visualization Examples**
- Confusion Matrices – class-wise accuracy comparison for training and testing sets
- Training/Testing Loss & Accuracy Curves – convergence analysis across epochs
- Five-Fold Cross-Validation Plots – robustness and generalizability assessment
- Grad-CAM Heatmaps – highlight retinal regions (hemorrhages, exudates, microaneurysms) influencing predictions

**Evaluation Environment**
Developed and tested on:
- OS: Windows 10 / Linux
- Platform: Google Colab (GPU/TPU acceleration)
- Framework: PyTorch and TensorFlow
- Libraries: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, OpenCV

**Results and Discussion**
- ResNet-34 achieved the best performance with 98.9% accuracy and 0.989 F1-score, outperforming deeper architectures.
- Classification performance generally improved with deeper architectures up to a point, but extremely deep networks (ResNet-152) showed declining performance, suggesting overfitting.
- McNemar's test confirmed no statistically significant difference between architectures at the 95% confidence level (p = 0.055).
- Five-fold cross-validation demonstrated consistent and generalizable learning with minimal variation across folds.
- Grad-CAM visualizations showed that deeper models produced more localized attention maps, but better focus did not always translate to superior classification results.
- Moderate-depth architectures represent the best trade-off among feature learning capability, optimization robustness, and generalization with limited data.
- The framework is designed as a preliminary screening and referral support tool for clinical practice, where avoiding false negatives is especially important.

**Limitations**
- Current study is limited to a single dataset and binary classification, restricting generalizability to diverse clinical environments and multi-class DR severity grading.
- Patient-level stratification was not possible due to unavailable patient identifiers.
- Grad-CAM outputs remained qualitative only, as ophthalmologist annotations were unavailable for quantitative validation.
- The impact of individual preprocessing steps was not evaluated through ablation studies.
- Ensemble and advanced architectures were not explored; only ResNet variants were benchmarked.
- Future work should address these gaps through multi-center datasets, clinical metadata integration (blood glucose levels, medical history), advanced architectures (Vision Transformers, EfficientNet), real-time mobile/edge deployment, and expert-validated explainability analysis.

**Conclusion**
The proposed Explainable ResNet-Based Deep Learning Framework delivers:
- State-of-the-art accuracy (98.9%) with ResNet-34 for binary DR classification
- Comprehensive benchmarking of four ResNet architectures under identical experimental conditions
- Robustness validation through five-fold cross-validation and McNemar's statistical test
- Clinically interpretable Grad-CAM explanations highlighting disease-related retinal regions

This framework represents a practical, trustworthy AI pipeline for ophthalmology diagnostics, emphasizing accuracy, reliability, and transparency, with strong potential for clinical deployment as a screening and referral support tool.
