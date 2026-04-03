"""
Student Performance Prediction & Analysis App
============================================
This app uses a Linear Regression model to predict exam scores and provides
AI-powered insights using Ollama.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Student Performance Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .sidebar-content {
        background-color: #ffffff;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .insight-box {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
        margin: 10px 0;
    }
    .chat-message {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .chat-user {
        background-color: #e8f5e9;
        border-left: 4px solid #4CAF50;
    }
    .chat-ai {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
</style>
""", unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    try:
        model = joblib.load("model_1_tubesAIlab.pkl")
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure model_1_tubesAIlab.pkl is in the same directory.")
        return None

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('dataset/StudentPerformanceFactors.csv')
    return df

# Preprocessing function (matching the notebook)
def preprocess_data(df):
    df = df.copy()
    df["Teacher_Quality"] = df["Teacher_Quality"].fillna(df["Teacher_Quality"].mode()[0])
    df["Parental_Education_Level"] = df["Parental_Education_Level"].fillna(df["Parental_Education_Level"].mode()[0])
    df["Distance_from_Home"] = df["Distance_from_Home"].fillna(df["Distance_from_Home"].mode()[0])
    return df

# Feature columns
num_cols = ["Hours_Studied", "Attendance", "Previous_Scores", "Sleep_Hours", 
            "Tutoring_Sessions", "Physical_Activity"]

cat_cols = ["Parental_Involvement", "Access_to_Resources", "Extracurricular_Activities",
            "Motivation_Level", "Internet_Access", "Family_Income", "Teacher_Quality",
            "School_Type", "Peer_Influence", "Learning_Disabilities",
            "Parental_Education_Level", "Distance_from_Home", "Gender"]

# Chat with Ollama
def chat_with_ollama(prompt, context=""):
    """Send a message to Ollama and get response"""
    try:
        import requests
        
        # Prepare the system prompt to limit the chatbot to student performance
        system_prompt = """You are a student performance analysis assistant. 
You are specialized in analyzing factors that affect student academic performance.
You should ONLY answer questions related to student performance, education, study habits, 
and factors affecting exam scores. If asked about other topics, politely redirect 
to student performance topics.

Context for this conversation: {}
""".format(context)
        
        url = "http://localhost:11434/api/chat"
        payload = {
            "model": "llama3.2",  # or other available model
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "stream": False
        }
        
        response = requests.post(url, json=payload, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            return result["message"]["content"]
        else:
            return f"Error: Unable to connect to Ollama. Please ensure Ollama is running. (Status: {response.status_code})"
            
    except ImportError:
        return "Error: requests library not found. Please install it: pip install requests"
    except ConnectionError:
        return "Error: Cannot connect to Ollama. Please ensure Ollama is running on your local machine.\n\nTo start Ollama, run: ollama serve"
    except Exception as e:
        return f"Error: {str(e)}"

# Sidebar navigation
st.sidebar.title("🎓 Student Performance Analyzer")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select Page:",
    ["🏠 Overview", "📊 Dataset Statistics", "🔍 Analysis & Insights", "🎯 Prediction", "💬 AI Chatbot"]
)

# Main content based on selected page
df = load_data()
df = preprocess_data(df)
model = load_model()

if page == "🏠 Overview":
    st.title("🎓 Student Performance Prediction & Analysis")
    st.markdown("---")
    
    # Hero section
    st.markdown("""
    <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 15px; color: white; margin-bottom: 30px;">
        <h1 style="font-size: 3em; margin-bottom: 10px;">Student Performance Analyzer</h1>
        <p style="font-size: 1.3em;">Predict exam scores and get AI-powered insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Students", f"{len(df):,}")
    with col2:
        st.metric("Average Exam Score", f"{df['Exam_Score'].mean():.1f}")
    with col3:
        st.metric("Max Exam Score", f"{df['Exam_Score'].max()}")
    with col4:
        st.metric("Min Exam Score", f"{df['Exam_Score'].min()}")
    
    st.markdown("---")
    
    # Features overview
    st.subheader("📋 Features Analyzed")
    
    features_info = {
        "Study Hours": "Hours_Studied - Time spent studying per week",
        "Attendance": "Attendance - Class attendance percentage",
        "Previous Scores": "Previous_Scores - Previous academic scores",
        "Sleep Hours": "Sleep_Hours - Daily sleep duration",
        "Tutoring": "Tutoring_Sessions - Number of tutoring sessions",
        "Physical Activity": "Physical_Activity - Weekly physical activity hours",
        "Parental Involvement": "Parental_Involvement - Level of parent engagement",
        "Resources": "Access_to_Resources - Access to learning materials",
        "Motivation": "Motivation_Level - Student's motivation level",
        "Internet": "Internet_Access - Internet availability",
        "Family Income": "Family_Income - Family economic status",
        "Teacher Quality": "Teacher_Quality - Quality of teachers",
        "School Type": "School_Type - Public or Private school",
        "Peer Influence": "Peer_Influence - Impact of peers",
        "Learning Disabilities": "Learning_Disabilities - Any learning challenges",
        "Parental Education": "Parental_Education_Level - Parents' education level",
        "Distance": "Distance_from_Home - Distance to school",
        "Gender": "Gender - Student's gender"
    }
    
    for feature, desc in features_info.items():
        st.markdown(f"- **{feature}**: {desc}")
    
    st.markdown("---")
    
    # Model info
    st.subheader("🤖 Model Information")
    st.markdown("""
    - **Primary Model**: Linear Regression
    - **Secondary Model**: Ollama (LLM) for AI insights
    - **Target Variable**: Exam Score
    
    The model predicts exam scores based on various student factors and provides
    AI-powered recommendations to improve academic performance.
    """)
    
    # How to use
    st.subheader("📖 How to Use")
    st.markdown("""
    1. **Overview**: See project summary and features
    2. **Dataset Statistics**: Explore the dataset with visualizations
    3. **Analysis & Insights**: Understand key findings and conclusions
    4. **Prediction**: Enter student data to predict exam scores
    5. **AI Chatbot**: Ask questions about student performance
    """)

elif page == "📊 Dataset Statistics":
    st.title("📊 Dataset Statistics")
    st.markdown("---")
    
    # Basic info
    st.subheader("📋 Basic Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Total Records**: {len(df)}")
        st.write(f"**Total Features**: {len(df.columns)}")
        st.write(f"**Numerical Features**: {len(num_cols)}")
        st.write(f"**Categorical Features**: {len(cat_cols)}")
    
    with col2:
        st.write(f"**Missing Values**: {df.isnull().sum().sum()}")
        st.write(f"**Duplicate Rows**: {df.duplicated().sum()}")
        st.write(f"**Average Exam Score**: {df['Exam_Score'].mean():.2f}")
        st.write(f"**Std Dev**: {df['Exam_Score'].std():.2f}")
    
    st.markdown("---")
    
    # Numerical distributions
    st.subheader("📈 Numerical Features Distribution")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(num_cols):
        sns.histplot(df[col], bins=20, kde=True, ax=axes[i], color=sns.color_palette("pastel")[i])
        axes[i].set_title(f"{col} Distribution")
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("---")
    
    # Categorical distributions
    st.subheader("📊 Categorical Features Distribution")
    
    selected_cat = st.selectbox("Select categorical feature:", cat_cols)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(data=df, x=selected_cat, ax=ax, palette="pastel")
    ax.set_title(f"{selected_cat} Distribution")
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    ax.tick_params(axis='x', rotation=25)
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("---")
    
    # Correlation heatmap
    st.subheader("🔥 Correlation with Exam Score")
    
    corr = df.select_dtypes("number").corr()["Exam_Score"].drop("Exam_Score").sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=corr.values, y=corr.index, palette="rocket", ax=ax)
    ax.axvline(0, color="gray", linewidth=1)
    ax.set_title("Correlation with Exam Score")
    ax.set_xlabel("Correlation Coefficient")
    ax.set_ylabel("")
    plt.tight_layout()
    st.pyplot(fig)

elif page == "🔍 Analysis & Insights":
    st.title("🔍 Analysis & Insights")
    st.markdown("---")
    
    # Key findings
    st.subheader("📌 Key Findings")
    
    # Calculate correlations
    corr = df.select_dtypes("number").corr()["Exam_Score"].drop("Exam_Score").sort_values(ascending=False)
    
    insights = []
    
    # Positive correlations
    positive_corr = corr[corr > 0].head(3)
    for feat, val in positive_corr.items():
        insights.append(f"• **{feat}** has a positive correlation ({val:.3f}) with exam scores")
    
    # Negative correlations
    negative_corr = corr[corr < 0].head(3)
    for feat, val in negative_corr.items():
        insights.append(f"• **{feat}** has a negative correlation ({val:.3f}) with exam scores")
    
    for insight in insights:
        st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Detailed analysis
    st.subheader("📊 Detailed Analysis")
    
    # Hours studied vs Exam Score
    st.markdown("### Study Hours Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x="Hours_Studied", y="Exam_Score", hue="Gender", 
                       palette=["#90CAF9", "#FFD54F"], alpha=0.7, data=df, ax=ax)
        ax.set_title("Hours Studied vs Exam Score")
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(x="Gender", y="Exam_Score", palette=["#90CAF9", "#FFD54F"], data=df, ax=ax)
        ax.set_title("Exam Score by Gender")
        plt.tight_layout()
        st.pyplot(fig)
    
    # Attendance analysis
    st.markdown("### Attendance Analysis")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x="Attendance", y="Exam_Score", hue="Gender", 
                   palette=["#90CAF9", "#FFD54F"], alpha=0.7, data=df, ax=ax)
    ax.set_title("Attendance vs Exam Score")
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("---")
    
    # Conclusions
    st.subheader("🎯 Conclusions")
    
    conclusions = """
    Based on the analysis of the Student Performance Factors dataset:
    
    1. **Hours Studied** is the most important factor affecting exam scores.
       Students who study more tend to perform better in exams.
    
    2. **Attendance** shows a strong positive correlation with exam performance.
       Regular attendance is crucial for academic success.
    
    3. **Previous Scores** indicate that past academic performance is a good
       predictor of future exam scores.
    
    4. **Sleep Hours** affect exam performance - both too little and too much
       sleep can negatively impact scores.
    
    5. **Tutoring Sessions** can help improve exam scores when needed.
    
    6. **Socioeconomic Factors** like Family Income and Parental Education Level
       also influence student performance.
    
    7. **School Type** (Public vs Private) shows some influence on scores.
    
    8. **Gender** differences exist but are relatively small compared to other factors.
    
    **Recommendations:**
    - Encourage regular study habits
    - Ensure consistent class attendance
    - Maintain healthy sleep patterns
    - Provide access to tutoring when needed
    - Address learning disabilities early
    """
    
    st.markdown(conclusions)

elif page == "🎯 Prediction":
    st.title("🎯 Exam Score Prediction")
    st.markdown("---")
    
    if model is None:
        st.error("Model not loaded. Please check the model file.")
    else:
        st.subheader("Enter Student Information")
        
        # Create input form
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Study & Academic Info")
            hours_studied = st.slider("Hours Studied per Week", 0, 50, 20)
            attendance = st.slider("Attendance (%)", 0, 100, 80)
            previous_scores = st.slider("Previous Scores", 0, 100, 70)
            sleep_hours = st.slider("Sleep Hours per Night", 0, 12, 7)
            tutoring_sessions = st.slider("Tutoring Sessions", 0, 10, 0)
            physical_activity = st.slider("Physical Activity (hours/week)", 0, 10, 3)
        
        with col2:
            st.markdown("### Personal & Environmental Factors")
            parental_involvement = st.selectbox("Parental Involvement", ["Low", "Medium", "High"])
            access_to_resources = st.selectbox("Access to Resources", ["Low", "Medium", "High"])
            extracurricular = st.selectbox("Extracurricular Activities", ["No", "Yes"])
            motivation = st.selectbox("Motivation Level", ["Low", "Medium", "High"])
            internet = st.selectbox("Internet Access", ["No", "Yes"])
            family_income = st.selectbox("Family Income", ["Low", "Medium", "High"])
            teacher_quality = st.selectbox("Teacher Quality", ["Low", "Medium", "High"])
            school_type = st.selectbox("School Type", ["Public", "Private"])
            peer_influence = st.selectbox("Peer Influence", ["Negative", "Neutral", "Positive"])
            learning_disabilities = st.selectbox("Learning Disabilities", ["No", "Yes"])
            parental_education = st.selectbox("Parental Education Level", ["High School", "College", "Postgraduate"])
            distance = st.selectbox("Distance from Home", ["Near", "Moderate", "Far"])
            gender = st.selectbox("Gender", ["Male", "Female"])
        
        st.markdown("---")
        
        # Predict button
        if st.button("🔮 Predict Exam Score", type="primary"):
            # Create input dataframe
            input_data = pd.DataFrame({
                'Hours_Studied': [hours_studied],
                'Attendance': [attendance],
                'Previous_Scores': [previous_scores],
                'Sleep_Hours': [sleep_hours],
                'Tutoring_Sessions': [tutoring_sessions],
                'Physical_Activity': [physical_activity],
                'Parental_Involvement': [parental_involvement],
                'Access_to_Resources': [access_to_resources],
                'Extracurricular_Activities': [extracurricular],
                'Motivation_Level': [motivation],
                'Internet_Access': [internet],
                'Family_Income': [family_income],
                'Teacher_Quality': [teacher_quality],
                'School_Type': [school_type],
                'Peer_Influence': [peer_influence],
                'Learning_Disabilities': [learning_disabilities],
                'Parental_Education_Level': [parental_education],
                'Distance_from_Home': [distance],
                'Gender': [gender]
            })
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            
            # Display result
            st.markdown("### 🎉 Prediction Result")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 30px; border-radius: 15px; text-align: center; color: white;">
                    <h2 style="margin: 0;">Predicted Exam Score</h2>
                    <h1 style="font-size: 4em; margin: 10px 0;">{prediction:.1f}</h1>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Provide context for the score
                if prediction >= 75:
                    performance = "Excellent"
                    color = "#4CAF50"
                    advice = "Great performance! Keep up the good work."
                elif prediction >= 65:
                    performance = "Good"
                    color = "#2196F3"
                    advice = "Good performance. Consider some improvements."
                elif prediction >= 55:
                    performance = "Average"
                    color = "#FF9800"
                    advice = "Average performance. There's room for improvement."
                else:
                    performance = "Needs Improvement"
                    color = "#f44336"
                    advice = "Consider seeking additional support."
                
                st.markdown(f"""
                <div style="background-color: {color}; padding: 20px; border-radius: 10px; color: white;">
                    <h3 style="margin: 0;">Performance Level: {performance}</h3>
                    <p style="margin-top: 10px;">{advice}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Store prediction for chatbot context
            st.session_state['last_prediction'] = prediction
            st.session_state['last_input'] = input_data.iloc[0].to_dict()
            
            st.markdown("---")
            
            # Get AI feedback
            st.subheader("💡 AI-Powered Feedback")
            
            with st.spinner("Getting AI feedback..."):
                context = f"""
                Student Profile:
                - Hours Studied: {hours_studied}
                - Attendance: {attendance}%
                - Previous Scores: {previous_scores}
                - Sleep Hours: {sleep_hours}
                - Tutoring Sessions: {tutoring_sessions}
                - Physical Activity: {physical_activity} hours/week
                - Parental Involvement: {parental_involvement}
                - Access to Resources: {access_to_resources}
                - Extracurricular Activities: {extracurricular}
                - Motivation Level: {motivation}
                - Internet Access: {internet}
                - Family Income: {family_income}
                - Teacher Quality: {teacher_quality}
                - School Type: {school_type}
                - Peer Influence: {peer_influence}
                - Learning Disabilities: {learning_disabilities}
                - Parental Education: {parental_education}
                - Distance from Home: {distance}
                - Gender: {gender}
                
                Predicted Exam Score: {prediction:.1f}
                
                Based on this student's profile and predicted score, provide:
                1. Analysis of key factors affecting their score
                2. Specific recommendations to improve their performance
                3. Areas where they are doing well and should continue
                """
                
                feedback = chat_with_ollama(
                    "Provide detailed feedback for this student prediction.", 
                    context
                )
                
                st.markdown(f"""
                <div class="chat-message chat-ai">
                    <strong>🤖 AI Analysis:</strong><br><br>
                    {feedback}
                </div>
                """, unsafe_allow_html=True)

elif page == "💬 AI Chatbot":
    st.title("💬 AI Chatbot")
    st.markdown("---")
    
    st.markdown("""
    ### 🤖 Student Performance Assistant
    
    Ask me anything about student performance, study tips, factors affecting exam scores,
    or get personalized recommendations for academic improvement.
    
    **Note:** This chatbot is specifically trained on student performance topics only.
    """)
    
    # Check if Ollama is running
    st.markdown("---")
    st.subheader("⚙️ Status")
    
    try:
        import requests
        response = requests.get("http://localhost:11434/", timeout=5)
        if response.status_code == 200:
            st.success("✅ Ollama is running and ready!")
        else:
            st.warning("⚠️ Ollama is responding but may not be ready.")
    except:
        st.error("❌ Ollama is not running. Please start Ollama to use the chatbot.")
        st.info("To start Ollama, run: `ollama serve` in your terminal")
    
    st.markdown("---")
    
    # Chat history
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    
    # Display chat history
    st.subheader("💭 Conversation")
    
    for i, (role, message) in enumerate(st.session_state['chat_history']):
        if role == "user":
            st.markdown(f"""
            <div class="chat-message chat-user">
                <strong>👤 You:</strong><br>
                {message}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message chat-ai">
                <strong>🤖 AI:</strong><br>
                {message}
            </div>
            """, unsafe_allow_html=True)
    
    # Last prediction context
    if 'last_prediction' in st.session_state:
        st.info(f"📊 Current context: Last predicted exam score = {st.session_state['last_prediction']:.1f}")
    
    # Chat input
    st.markdown("### 💬 Ask a Question")
    
    user_input = st.text_input(
        "Your question:",
        placeholder="e.g., What factors most affect student exam scores?",
        key="chat_input"
    )
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        pass  # Text input is already above
    
    with col2:
        send_button = st.button("Send", type="primary")
    
    if send_button and user_input:
        # Add user message to history
        st.session_state['chat_history'].append(("user", user_input))
        
        # Build context from last prediction if available
        context = ""
        if 'last_prediction' in st.session_state and 'last_input' in st.session_state:
            input_data = st.session_state['last_input']
            context = f"""
            Current student being analyzed (predicted score: {st.session_state['last_prediction']:.1f}):
            - Hours Studied: {input_data.get('Hours_Studied', 'N/A')}
            - Attendance: {input_data.get('Attendance', 'N/A')}%
            - Previous Scores: {input_data.get('Previous_Scores', 'N/A')}
            - Sleep Hours: {input_data.get('Sleep_Hours', 'N/A')}
            - Tutoring Sessions: {input_data.get('Tutoring_Sessions', 'N/A')}
            - Physical Activity: {input_data.get('Physical_Activity', 'N/A')} hours/week
            - Parental Involvement: {input_data.get('Parental_Involvement', 'N/A')}
            - Access to Resources: {input_data.get('Access_to_Resources', 'N/A')}
            - Extracurricular Activities: {input_data.get('Extracurricular_Activities', 'N/A')}
            - Motivation Level: {input_data.get('Motivation_Level', 'N/A')}
            - Internet Access: {input_data.get('Internet_Access', 'N/A')}
            - Family Income: {input_data.get('Family_Income', 'N/A')}
            - Teacher Quality: {input_data.get('Teacher_Quality', 'N/A')}
            - School Type: {input_data.get('School_Type', 'N/A')}
            - Peer Influence: {input_data.get('Peer_Influence', 'N/A')}
            - Learning Disabilities: {input_data.get('Learning_Disabilities', 'N/A')}
            - Parental Education: {input_data.get('Parental_Education_Level', 'N/A')}
            - Distance from Home: {input_data.get('Distance_from_Home', 'N/A')}
            - Gender: {input_data.get('Gender', 'N/A')}
            """
        
        # Get AI response
        with st.spinner("Thinking..."):
            response = chat_with_ollama(user_input, context)
        
        # Add AI response to history
        st.session_state['chat_history'].append(("ai", response))
        
        # Rerun to display new messages
        st.rerun()
    
    # Clear chat button
    if st.session_state['chat_history']:
        if st.button("🗑️ Clear Chat History"):
            st.session_state['chat_history'] = []
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>🎓 Student Performance Analyzer | Created with Streamlit</p>
    <p>Model: Linear Regression | AI: Ollama (Llama 3.2)</p>
</div>
""", unsafe_allow_html=True)
