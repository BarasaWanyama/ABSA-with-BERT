from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report
import torch
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import numpy as np

# load the fine-tunned model and tokenizer
tokenizer = BertTokenizer.from_pretrained('/Users/sandrahkweziwanyama/Documents/MscComputerScience/1-IU-International-University/Year 1/SemesterTwo/Computer Science Project/Models/Tokenizer91acc')
model = BertForSequenceClassification.from_pretrained("/Users/sandrahkweziwanyama/Documents/MscComputerScience/1-IU-International-University/Year 1/SemesterTwo/Computer Science Project/Models/CustomModel91acc")

     
# read the dataset file into a pandas dataframe
val_data = pd.read_csv('Scraped_reviews.csv')

# removing the duplicate reviews if any
val_data = val_data.drop_duplicates(subset=['Review'])

# deleting the rows with null values if any
val_data = val_data.dropna()

# list of original labels
original_labels = [
    'Aesthetics', 'Quality', 'Taste',
    'Safety', 'Price'
]

# create the label to number encoding for each label
labeling_dict = {label: idx for idx, label in enumerate(original_labels)}
# creating the reverse label dictionary to map the predicted numbers back to original labels
labeling_dict_reverse = {idx: label for idx, label in enumerate(original_labels)}

print(list(labeling_dict_reverse.values())[0])

# convert the reviews to a list
X_val = val_data["Review"].to_list()

# tokenize the reviews
X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=512)


'''Function to extract aspects from reviews'''
def extract_aspects(Review, model=model, tokenizer=tokenizer, max_length=512):
    inputs = tokenizer(Review, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        _, predicted_labels = torch.max(probabilities, dim=1)
    
    # Map predicted numerical labels to text labels
    predicted_aspects = [labeling_dict_reverse[label.item()] for label in predicted_labels]
    
    # Remove square brackets from the aspect text labels
    predicted_aspects = [aspect.strip('[]') for aspect in predicted_aspects]
    
    return predicted_aspects

'''Function to remove square brackets'''
def remove_brackets(aspect_texts):
    return [aspect_text.strip('[]') for aspect_text in aspect_texts]

            
'''Function to calculate the metrics'''
def compute_metrics(labels, preds):
    accuracy = accuracy_score(y_true=labels, y_pred=preds)
    precision = precision_score(y_true=labels, y_pred=preds, average='weighted')
    recall = recall_score(y_true=labels, y_pred=preds, average='weighted')
    f1score = f1_score(y_true=labels, y_pred=preds, average='weighted')
    return {"accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1score}


'''Apply aspect extraction to the dataset'''
val_data['predicted_aspect'] = val_data['Review'].apply(extract_aspects)


# Remove the square brackets from the predicted aspect text labels
val_data['predicted_aspect'] = val_data['predicted_aspect'].apply(remove_brackets)

# reverse mapping the numerical predictions to original labels
#val_data['Aspect'] = val_data.Aspect.values

     

#Display the results
print(val_data.head())


'''Save the results to a new CSV file'''
val_data.to_csv('aspect_output.csv', index=False)

#Compute model performance metrics
#compute_metrics(val_data_w_preds.Aspect.values,val_data_w_preds.Predicted_Aspect.values)
     


