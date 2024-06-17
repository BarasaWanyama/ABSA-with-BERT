import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report
import torch
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import numpy as np

# File paths for training and testing
SYNTHETIC_FILE_PATH = '/Users/sandrahkweziwanyama/Documents/MscComputerScience/1-IU-International-University/Year 1/SemesterTwo/Computer Science Project/Models/artificial_reviews.xlsx'

MANUAL_FILE_PATH = '/Users/sandrahkweziwanyama/Documents/MscComputerScience/1-IU-International-University/Year 1/SemesterTwo/Computer Science Project/Models/Scraped_reviews.xlsx'
     

#Reading data
synthetic_data = pd.read_excel(SYNTHETIC_FILE_PATH)
synthetic_data = synthetic_data.drop_duplicates(subset=['Review'])
print(len(synthetic_data))

manual_data = pd.read_excel(MANUAL_FILE_PATH)
manual_data = manual_data.drop_duplicates(subset=['Review'])
print(len(manual_data))

# for synthetice data
synth_train, synth_test, _, _ = train_test_split(synthetic_data, synthetic_data.Aspect, test_size=0.2, random_state=42, stratify=synthetic_data.Aspect)
# for manual data
manu_train, manu_test, _, _ = train_test_split(manual_data, manual_data.Aspect, test_size=0.2, random_state=42, stratify=manual_data.Aspect)

# combine the dataset in 80:20 ratio and creating the training and testing data
training_data = pd.concat([synth_train,manu_train], ignore_index=True)
testing_data = pd.concat([synth_test,manu_test], ignore_index=True)

print(len(training_data))
print(len(testing_data))

#Define labels
original_labels = [
    'Aesthetics', 'Price', 'Quality', 'Safety',
    'Taste'   
]

#count number of labels and assign them to the variable num_labels
num_labels = len(original_labels)

# Map labels to numerical values (0 to 4)
labeling_dict = {label: idx for idx, label in enumerate(original_labels)}
labeling_dict_reverse = {idx: label for idx, label in enumerate(original_labels)}

training_data['Aspect'] = training_data['Aspect'].map(labeling_dict)
testing_data['Aspect'] = testing_data['Aspect'].map(labeling_dict)

training_data = training_data.dropna(subset=['Aspect'])
testing_data = testing_data.dropna(subset=['Aspect'])

training_data['Aspect'] = training_data['Aspect'].astype(int)
testing_data['Aspect'] = testing_data['Aspect'].astype(int)

print(training_data.dtypes)


tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertForSequenceClassification.from_pretrained('bert-base-cased',num_labels=num_labels)

X_train = training_data["Review"].to_list()
X_test = testing_data["Review"].to_list()
y_train = training_data["Aspect"].to_list()
y_test = testing_data["Aspect"].to_list()


X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=512)
X_test_tokenized = tokenizer(X_test, padding=True, truncation=True, max_length=512)
     
# Create torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = F.one_hot(torch.tensor(self.labels[idx]).long(), num_classes=num_labels).float()
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

def compute_metrics(p):
    pred, labels = p
    pred = torch.tensor(pred)
    labels = torch.tensor(labels)
    pred = torch.argmax(pred, dim=1)
    labels = torch.argmax(labels, dim=1)
    accuracy = accuracy_score(y_true=labels.cpu().numpy(), y_pred=pred.cpu().numpy())
    precision = precision_score(y_true=labels.cpu().numpy(), y_pred=pred.cpu().numpy(), average='weighted')
    recall = recall_score(y_true=labels.cpu().numpy(), y_pred=pred.cpu().numpy(), average='weighted')
    f1score = f1_score(y_true=labels.cpu().numpy(), y_pred=pred.cpu().numpy(), average='weighted')
    return {"accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1score}     

train_dataset = Dataset(X_train_tokenized, y_train)
test_dataset = Dataset(X_test_tokenized, y_test)


# Define Trainer
args = TrainingArguments(
    output_dir="output",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    eval_strategy="steps",  # Specify the evaluation strategy
    eval_steps=100,  # Evaluate every 500 training steps (you can adjust this value)
    logging_dir="logs",  # Directory for TensorBoard logs
    logging_steps=100,  # Log metrics every 100 steps (you can adjust this value)
    save_steps=100
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,  # Provide the validation dataset
    compute_metrics=compute_metrics,
)

# start the model training
trainer.train()


#show performance evaluation of model
model_validation_metrics = trainer.evaluate()

print(model_validation_metrics)


# to save the model
trainer.save_model("/Users/sandrahkweziwanyama/Documents/MscComputerScience/1-IU-International-University/Year 1/SemesterTwo/Computer Science Project/Models/CustomModel91acc")
tokenizer.save_pretrained("/Users/sandrahkweziwanyama/Documents/MscComputerScience/1-IU-International-University/Year 1/SemesterTwo/Computer Science Project/Models/Tokenizer91acc")
          

