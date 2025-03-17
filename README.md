# AI-Resume-Screener: Resume Classifier System

## Overview

The Resume Classifier System is an intelligent application that automatically categorizes resumes into specific job categories using advanced natural language processing and machine learning techniques. This system helps streamline the recruitment process by reducing the manual effort required for resume screening and improving the quality of candidate matching.

## Features

- **Multi-category Classification**: Categorizes resumes into 25+ job categories including Data Science, Java Developer, Testing, and more
- **BERT-based Model**: Leverages the power of BERT (Bidirectional Encoder Representations from Transformers) for superior text understanding
- **Preprocessing Pipeline**: Includes comprehensive text preprocessing to handle common resume formatting issues
- **Confidence Scores**: Provides confidence scores for classification results to aid decision-making
- **Model Flexibility**: Supports both BERT and spaCy models depending on requirements
- **Training and Evaluation**: Includes complete training pipeline with detailed evaluation metrics

## Technical Architecture

The Resume Classifier System consists of the following components:

1. **ResumeScreener Class**: The core component that handles all aspects of resume classification
2. **Text Preprocessing Module**: Cleans and normalizes resume text for better model performance
3. **Model Training Pipeline**: Configures and trains the classification model
4. **Inference Engine**: Provides predictions for new resume inputs
5. **Model Persistence**: Allows saving and loading of trained models

## How It Works

1. **Resume Preprocessing**: The system first cleans and normalizes the resume text by removing URLs, email addresses, phone numbers, and extra whitespace, and converting all text to lowercase.

2. **Feature Extraction**: For BERT-based models, the system tokenizes the text and converts it to model-specific inputs. For spaCy models, it creates document vectors using pre-trained word embeddings.

3. **Classification**: The trained model predicts the most likely job category for each resume.

4. **Evaluation**: The system provides detailed performance metrics including precision, recall, and F1-score for each job category.

## Getting Started

### Prerequisites

- Python 3.7 or higher
- PyTorch
- Transformers library (for BERT models)
- spaCy (for spaCy models)
- scikit-learn
- pandas

### Installation

```python
# Install required packages
pip install torch transformers spacy scikit-learn pandas

# Download spaCy models if needed
python -m spacy download en_core_web_lg
```

### Usage Example

```python
import pandas as pd
from resumescreener import ResumeScreener

# Load your resume dataset
df = pd.read_csv('resume_dataset.csv')

# Extract features and labels
resumes = df['Resume'].tolist()
categories = df['Category'].tolist()

# Create category mapping
unique_categories = list(set(categories))
category_mapping = {category: idx for idx, category in enumerate(unique_categories)}
label_mapping = {idx: category for idx, category in enumerate(unique_categories)}
labels = [category_mapping[category] for category in categories]

# Initialize and train the model
screener = ResumeScreener(model_type="bert")
screener.train(resumes, labels, label_mapping, test_size=0.2)

# Save the trained model
screener.save_model('resume_classifier_model')

# Use the model to classify a new resume
new_resume = "Experienced data scientist with expertise in Python, machine learning..."
predicted_category = screener.predict(new_resume)
print(f"Predicted category: {predicted_category}")
```

## Performance Metrics

Based on the latest evaluation, the model achieves:
- Overall accuracy: 74%
- Strong performance on categories like Network Security Engineer, Arts, and Electrical Engineering
- Room for improvement on categories like Business Analyst and DotNet Developer

## Future Improvements

1. Data augmentation techniques to improve performance on underrepresented categories
2. Class weighting to address class imbalance
3. Hyperparameter tuning for better model performance
4. Domain-specific pre-training for improved accuracy
5. Integration with cloud services for scalable deployment

## Contributing

Contributions to improve the Resume Classifier System are welcome. Please feel free to submit a pull request or open an issue to discuss potential improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
