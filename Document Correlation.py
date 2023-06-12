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
    elif isinstance(text, (int, float)):
        # Convert numerical value to string
        return str(text)
    else:
        return ''

# Apply the preprocess_text function to preprocess the text data in specific columns of df1 and df2.
# Preprocess text data in df1.
df1['column1 title'] = df1['column1 title'].apply(preprocess_text)
df1['column2 title'] = df1['column2 title'].apply(preprocess_text)
df1['column3 title'] = df1['column3 title'].apply(preprocess_text)

# Preprocess text data in df2.
df2['column1 title'] = df2['column1 title'].apply(preprocess_text)
df2['column2 title'] = df2['column2 title'].apply(preprocess_text)
df2['column3 title'] = df2['column3 title'].apply(preprocess_text)

# Initialize the TF-IDF vectorizer.
tfidf_vectorizer = TfidfVectorizer()

# Fit the vectorizer on the preprocessed text data from df1 using fit_transform to learn the vocabulary and convert the text into a matrix representation. 
tfidf_matrix1 = tfidf_vectorizer.fit_transform(df1['column1 title'] + ' ' + df1['column2 title'] + ' ' + df1['column3 title'])
# Transform the preprocessed text data from df2 using transform to convert it into a matrix representation.
tfidf_matrix2 = tfidf_vectorizer.transform(df2['column1 title'] + ' ' + df2['column2 title'] + ' ' + df2['column3 title'])

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
        
uncorrelated_rows = []

# Iterate through the rows of df2 and check if the index of the row is present in correlated_requirements. 
# If it's not, add the index to the uncorrelated_rows list.
for row_idx in range(len(df2)):
    if row_idx not in [req2_idx for _, req2_idx in correlated_requirements]:
        uncorrelated_rows.append(row_idx)

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
    req1_column1 = req1_details['column1 title']
    req2_column1 = req2_details['column1 title']
    req1_column2 = req1_details['column2 title']
    req2_column2 = req2_details['column2 title']
    req1_column3 = req1_details['column3 title']
    req2_column3 = req2_details['column3 title']
 
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
    
for row_idx in uncorrelated_rows:
    req_details = df2.iloc[row_idx].tolist()
    # Create a list with empty values for columns from df1
    empty_values = [""] * len(df1.columns[1:])
    # Combine the empty values with the uncorrelated row details from df2 and add "Uncorrelated" at the end
    report_data.append(["Uncorrelated"] + empty_values + req_details[1:])

# Export the report to an Excel file
FY_column_list_with_flag = ["Correlation"] + FY_column_list
report_df = pd.DataFrame(report_data, columns=FY_column_list_with_flag)
report_df.to_excel("Path to File", index=False)
