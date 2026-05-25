# Depression-Detection-Using-ML
A machine learning-based web app that analyzes user text to detect signs of depression. Built with Streamlit, it includes login/registration and stores user-specific prediction history for tracking emotional patterns over time.

Depression-Twitter.csv-Dataset,
MODELTRAIINING.ipynb-Python Code,
README.md-Description,
app.py-Frontend and UI,
history.json-backend History Storage,
model.pkl-Naive Bays ,
pipeline.pkl- Used to reuse the code,
requirements.txt- Required Installations and Models,
users_passwords.json-Stores User name and Password,
vectorizer.pkl-Stores Processed Data,
video_20260524_230207_edit.mp4-Sample Demo Video of the project ( the quality was little low because github accepts video files which are  less the 25 MB  and it would have been more clear if it was more than 25 MB)

Depression Detection Project
Project Description

This project is a Depression Detection System developed using Machine Learning and Natural Language Processing (NLP). The main purpose of this project is to analyze user text input and predict whether the text shows signs of depression. This project helps in understanding how text-based machine learning models can be used in mental health-related applications.

Project Workflow

The development of this project started by installing Anaconda, which was used to manage the Python environment and required packages. After installing Anaconda, Jupyter Notebook was opened to write and test the machine learning code for the project.

Inside Jupyter Notebook, the required Python libraries such as Pandas, NumPy, Scikit-learn, NLTK, and Pickle were imported. These libraries were mainly used for data processing, text preprocessing, model training, and saving the trained model.

After importing the libraries, the dataset was loaded and cleaned. The text data was preprocessed by removing unwanted symbols, converting text into lowercase, removing stopwords, and preparing the data for better model performance. This step was important because clean data improves prediction accuracy.

Once the preprocessing was completed, the text data was converted into numerical format using machine learning techniques such as vectorization. This allowed the machine learning model to understand and process text input.

After preparing the data, the machine learning model was trained using Scikit-learn. The dataset was divided into training and testing data, and the model was evaluated to check its prediction performance. Once the model gave satisfactory results, it was saved using Pickle files so that it could be reused without training again.

After completing the backend model development, the frontend user interface was created in the app.py file using Streamlit. This file was written to provide a simple interface where users can enter text input and get prediction results directly on the screen.

The application was then tested locally to verify whether the frontend, model connection, and prediction process were working correctly. After successful testing, all the required files such as the Python files, Jupyter Notebook file, model file, pickle files, and dependency files were uploaded to GitHub.

Finally, the GitHub repository was connected to Streamlit Cloud for deployment. The application was deployed online using Streamlit, which made the project accessible through a web browser. This allowed users to use the depression detection system without installing the project locally.

Conclusion

This project demonstrates the complete workflow of building a Machine Learning-based web application, starting from environment setup, data preprocessing, model training, frontend development, testing, and final deployment. It can be used as a simple reference project for students who want to learn machine learning, NLP, and web deployment using Streamlit.
