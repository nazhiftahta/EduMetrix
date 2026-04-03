# Student Performance Analyzer App

An interactive Streamlit app that predicts student exam scores and provides AI-powered insights using Ollama.

## Features

- **Overview**: Project summary and feature descriptions
- **Dataset Statistics**: Visualizations of the Student Performance dataset
- **Analysis & Insights**: Key findings and conclusions from the data
- **Prediction**: Input student data to predict exam scores
- **AI Chatbot**: Ask questions about student performance (powered by Ollama)

## Prerequisites

1. **Python 3.8+** installed
2. **Ollama** installed and running locally

### Install Ollama

Visit https://ollama.ai/ to download and install Ollama for your OS.

After installation, run:
```
bash
# Pull the Llama 3.2 model (or any model you prefer)
ollama pull llama3.2

# Start Ollama service
ollama serve
```

## Installation

1. Clone or download this repository

2. Install the required Python packages:
```
bash
pip install -r requirements.txt
```

## Running the App

1. Make sure Ollama is running:
```
bash
ollama serve
```

2. Run the Streamlit app:
```
bash
streamlit run app.py
```

3. Open your browser and navigate to:
```
http://localhost:8501
```

## Project Structure

```
├── app.py                      # Main Streamlit application
├── model_1_tubesAIlab.pkl    # Trained Linear Regression model
├── dataset/
│   └── StudentPerformanceFactors.csv
├── requirements.txt            # Python dependencies
├── Tubes_AI_LAB_LAHH.ipynb   # Original notebook
└── README.md                  # This file
```

## Using the App

### 1. Overview
Read about the project and understand the features being analyzed.

### 2. Dataset Statistics
Explore the dataset with interactive visualizations:
- Numerical feature distributions
- Categorical feature distributions
- Correlation analysis

### 3. Analysis & Insights
View detailed analysis and conclusions:
- Key findings from correlation analysis
- Performance by gender
- Study hours vs exam score relationships

### 4. Prediction
Enter student information to predict exam scores:
- Fill in all the student details in the form
- Click "Predict Exam Score" to get the prediction
- Receive AI-powered feedback on the prediction

### 5. AI Chatbot
Ask questions about student performance:
- The chatbot is limited to student performance topics only
- It can provide insights based on the prediction context
- Ask about study tips, factors affecting scores, etc.

## Model Information

- **Primary Model**: Linear Regression
- **Target**: Exam Score (0-100)
- **Features**: 19 features including study hours, attendance, parental involvement, etc.

## Troubleshooting

### Ollama Connection Error
If you get "Cannot connect to Ollama":
1. Make sure Ollama is installed
2. Run `ollama serve` in a terminal
3. The app uses `http://localhost:11434` to connect

### Model Not Found
Ensure `model_1_tubesAIlab.pkl` is in the same directory as `app.py`.

### Dataset Not Found
Ensure the `dataset/StudentPerformanceFactors.csv` file exists.

## Notes

- The AI chatbot uses Ollama's Llama 3.2 model by default. You can change this in the `chat_with_ollama` function in `app.py`.
- The chatbot is specifically configured to only answer questions related to student performance.
- The prediction model was trained on the Student Performance Factors dataset.
