# Excel Documents Correlation Program

I created this program to determine correlations from two different Excel documents and export the data as a single Excel document for analysis. It utilizes various libraries for data manipulation, natural language processing, and machine learning.

## Prerequisites

Before running the program, ensure that you have the following libraries installed:

- pandas: for data manipulation
- nltk: the Natural Language Toolkit library used for natural language processing tasks
- sklearn: scikit-learn library, which provides various tools for machine learning and vectorization

You can install these libraries using pip:

```bash
pip install pandas nltk scikit-learn
```

In addition, the program requires downloading necessary resources from the 'nltk' library, specifically the 'stopwords' corpus and the 'punkt' tokenizer. To download these resources, run the following commands:

```python
import nltk

nltk.download('stopwords')
nltk.download('punkt')
```

## Usage

1. Import the required libraries:

```python
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
```

2. Load data from two Excel files into separate dataframes (`df1` and `df2`):

```python
df1 = pd.read_excel("path to file")
df2 = pd.read_excel("path to file")
```

3. Define the `preprocess_text` function to preprocess the text data:

```python
def preprocess_text(text):
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
```

4. Apply the `preprocess_text` function to specific columns of both `df1` and `df2` dataframes:

```python
df1['column1'] = df1['column1'].apply(preprocess_text)
df1['column2'] = df1['column2'].apply(preprocess_text)
df1['column3'] = df1['column3'].apply(preprocess_text)
df1['column4'] = df1['column4'].apply(preprocess_text)
df1['column5'] = df1['column5'].apply(preprocess_text)

# Same concept is applied now to df2
df2['column1'] = df2['column1'].apply(preprocess_text)
df2['column2'] = df2['column2'].apply(preprocess_text)
df2['column3'] = df2['column3'].apply(preprocess_text)
df2['column4'] = df2['column4'].apply(preprocess_text)
df2['column5'] = df2['column5'].apply(preprocess_text)
```

5. Initialize a TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer:

```python
tfidf_vectorizer = TfidfVectorizer()
```

6. Fit the vectorizer on the preprocessed text data from `df1` and transform the text data from `df2`:

```python
tfidf_matrix1 = tfidf_vectorizer.fit_transform(df1['column1'] + ' ' + df1['column2'] + ' ' + df1['column3'] + ' ' + df1['column4'] + ' '

 + df1['column5'])
tfidf_matrix2 = tfidf_vectorizer.transform(df2['column1'] + ' ' + df2['column2'] + ' ' + df2['column3'] + ' ' + df2['column4'] + ' ' + df2['column5'])
```

7. Calculate the cosine similarity between each pair of requirements:

```python
similarity_matrix = cosine_similarity(tfidf_matrix1, tfidf_matrix2)
```

8. Set a similarity threshold value to determine correlated requirements:

```python
similarity_threshold = 0.35
```

9. Find pairs of correlated requirements and store them in the `correlated_requirements` list:

```python
correlated_requirements = []
for i in range(similarity_matrix.shape[0]):
    max_similarity_idx = similarity_matrix[i].argmax()
    max_similarity = similarity_matrix[i, max_similarity_idx]
    
    if max_similarity >= similarity_threshold:
        correlated_requirements.append((i, max_similarity_idx))
```

10. Identify uncorrelated rows in `df2` and store their indices in the `uncorrelated_rows` list:

```python
uncorrelated_rows = []
for row_idx in range(len(df2)):
    if row_idx not in [req2_idx for _, req2_idx in correlated_requirements]:
        uncorrelated_rows.append(row_idx)
```

11. Generate a report by retrieving the details of correlated requirements:

```python
report = []
for req1_idx, req2_idx in correlated_requirements:
    req1_details = df1.iloc[req1_idx]
    req2_details = df2.iloc[req2_idx]
    report.append((req1_details, req2_details))
```

12. Export the report to an Excel file:

```python
FY_column_list = list(df1.columns[1:]) + list(df2.columns[1:])
report_data = []

for req1_details, req2_details in report:
    req1_column1 = req1_details['column1']
    req2_column1 = req2_details['column1']
    req1_column2 = req1_details['column2']
    req2_column2 = req2_details['column2']
    req1_column3 = req1_details['column3']
    req2_column3 = req2_details['column3']
    req1_impactifnotfunded = req1_details['column4']
    req2_impactifnotfunded = req2_details['column4']
    req1_costcenter = req1_details['column5']
    req2_costcenter = req2_details['column5']
 
    report_data.append(["Correlated"] + req1_details[1:] + req2_details[1:])

for row_idx in uncorrelated_rows:
    req_details = df2.iloc[row_idx].tolist()
    empty_values = [""] * len(df1.columns[1:])
    report_data.append(["Uncorrelated"] + empty_values + req_details[1:])

FY_column_list_with_flag = ["Correlation"] + FY_column_list
report_df = pd.DataFrame(report_data, columns=FY_column_list_with_flag)
report_df.to_excel("path to file", index=False)
```

Make sure to replace the placeholder "path to file" with the actual file paths where you want to read the input Excel files and export the report.

## Customization

You can customize the program by adjusting the following parameters:

- File paths: Replace "path to file" with the actual paths to the input Excel files and the desired output report file.
- Similarity threshold: Modify the `similarity_threshold` value to control the correlation threshold for requirements.
- Columns: Update the column names used in the program (`column1`,

 `column2`, `column3`, `column4`, `column5`) to match the column names in your Excel files.

Feel free to explore and adapt the code according to your specific requirements.
