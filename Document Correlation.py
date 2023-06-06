# Create a program for comparing FY requirements and export the data as an excel document for analysis.

# Import the required libraries: 
# pandas for data manipulation. 
# nltk for natural language processing.
# sklearn for vectorization and similarity calculations.

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download the necessary nltk resources for tokenization and stopwords.
nltk.download('stopwords')
nltk.download('punkt')

#Load data from Excel files
df1 = pd.read_excel("Path to File")
df2 = pd.read_excel("Path to File")

# Define a function to preprocess the text data. 
# This function removes punctuation, converts the text to lowercase, tokenizes it into words, removes stop words, and joins the words back into a single string.
def preprocess_text(text):
    if isinstance(text, str):
        # Remove punctuation
        text = text.translate(str.maketrans('.', ',', string.punctuation))

        # Convert to lowercase.
        text = text.lower()

        # Tokenize text into words.
        tokens = word_tokenize(text)

        # Remove stop words.
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]

        # Join the words back into a single string.
        preprocessed_text = ' '.join(tokens)

        return preprocessed_text
    else:
        return ''

# Apply the preprocess_text function to preprocess the text data in specific columns of df1 and df2.
# Preprocess text data in df1.
df1['PEC'] = df1['PEC'].apply(preprocess_text)
df1['Justification'] = df1['Justification'].apply(preprocess_text)
df1['Impact if not Funded'] = df1['Impact if not Funded'].apply(preprocess_text)

# Preprocess text data in df2.
df2['PEC'] = df2['PEC'].apply(preprocess_text)
df2['Justification'] = df2['Justification'].apply(preprocess_text)
df2['Impact if not Funded'] = df2['Impact if not Funded'].apply(preprocess_text)

# Initialize the TF-IDF vectorizer.
tfidf_vectorizer = TfidfVectorizer()

# Fit the vectorizer on the preprocessed text data from df1 using fit_transform to learn the vocabulary and convert the text into a matrix representation. 
tfidf_matrix1 = tfidf_vectorizer.fit_transform(df1['PEC'] + ' ' + df1['Justification'] + ' ' + df1['Impact if not Funded'])
# Transform the preprocessed text data from df2 using transform to convert it into a matrix representation.
tfidf_matrix2 = tfidf_vectorizer.transform(df2['PEC'] + ' ' + df2['Justification'] + ' ' + df2['Impact if not Funded'])

# Calculate cosine similarity between each pair of requirements.
similarity_matrix = cosine_similarity(tfidf_matrix1, tfidf_matrix2)

# Set a similarity threshold value. Continue to modify the threshold until the desired similarity outcome is present on the created report as a result. 
similarity_threshold = 0.5

correlated_requirements = []

# Iterate through the similarity matrix to find pairs of correlated requirements. 
# If the maximum similarity value in a row is above the threshold, the indices of the correlated requirements are added to the list 'correlated_requirements'.
for i in range(similarity_matrix.shape[0]):
    # Find the index of the maximum similarity value in each row.
    max_similarity_idx = similarity_matrix[i].argmax()
    max_similarity = similarity_matrix[i, max_similarity_idx]
    
    # Check if the maximum similarity is above the threshold
    if max_similarity >= similarity_threshold:
        correlated_requirements.append((i, max_similarity_idx))

#create an empty list to store the correlated requirement details. 
report = []

# Iterate through correlated_requirements and retrieve the details of the correlated requirements from df1 and df2. 
# Append these details as tuples (req1_details, req2_details) to the report list.
for req1_idx, req2_idx in correlated_requirements:
    req1_details = df1.iloc[req1_idx]
    req2_details = df2.iloc[req2_idx]
    report.append((req1_details, req2_details))

# Iterate through the 'report' list and extract the columns used for calcuation from req1_details and req2_details and assign them to individual variables.
# This allows you to work with the data individually if needed. 
for req1_details, req2_details in report:
    req1_PEC = req1_details['PEC']
    req2_PEC = req2_details['PEC']
    req1_justification = req1_details['Justification']
    req2_justification = req2_details['Justification']
    req1_impactifnotfunded = req1_details['Impact if not Funded']
    req2_impactifnotfunded = req2_details['Impact if not Funded']
 
# Extract the relevant columns required for the output report by combining the column names from df1 and df2.
FY_column_list = list(df1.columns[1:]) + list(df2.columns[1:])

# Create an empty list to store the data for the report. 
report_data = []

# Iterate through correlated_requirements again and convert the requirement details for each pair into lists. 
# Append these lists to the report_data list, excluding the first column (which contains the index).
for req1_idx, req2_idx in correlated_requirements:
    req1_details = df1.iloc[req1_idx].tolist()
    req2_details = df2.iloc[req2_idx].tolist()
    report_data.append(req1_details[1:] + req2_details[1:])

# Export the report to create an Excel file.
report_df = pd.DataFrame(report_data, columns=FY_column_list)
report_df.to_excel("Path to File", index=False)