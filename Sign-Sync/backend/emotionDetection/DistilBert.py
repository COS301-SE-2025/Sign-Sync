from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# followed https://www.geeksforgeeks.org/nlp/distilbert-in-natural-language-processing for setup of the distilBert
# due to limited documentation

dataset = load_dataset("dair-ai/emotion")
trainDataset = dataset["train"]
testDataset = dataset["test"]

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
def preprocessData(sentences):
    return tokenizer(sentences['text'], truncation=True, padding='max_length', max_length=256)

tokenTrain = trainDataset.map(preprocessData, batched=True)
tokenTest = testDataset.map(preprocessData, batched=True)

labels = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
classes = {'sadness': 0, 'joy': 1, 'love': 2, 'anger': 3, 'fear': 4, 'surprise': 5}

distilBertModel = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=6, id2label=labels, label2id=classes)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

trainingSettings = TrainingArguments(
    output_dir="./results",
    eval_strategy="steps",
    eval_steps=150,
    logging_strategy="steps",
    logging_steps=15,
    save_strategy="epoch",

    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',

    no_cuda=False,     
    fp16=True, 
)

trainer = Trainer(
    model=distilBertModel,
    args=trainingSettings,
    train_dataset=tokenTrain,
    eval_dataset=tokenTest,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

trainer.train()

results = trainer.evaluate()
print(f"Evaluation Results: {results}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
distilBertModel.to(device)

distilBertModel.save_pretrained("./DistilBERT-Emotion")
tokenizer.save_pretrained("./DistilBERT-Emotion")

def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = distilBertModel(**inputs)
    predicted_class = torch.argmax(outputs.logits).item()
    return labels[predicted_class]

print(predict_emotion("I'm so happy today!"))
print(predict_emotion("I hate the weather"))
print(predict_emotion("You are going to school"))

# @inproceedings{saravia-etal-2018-carer,
#     title = "{CARER}: Contextualized Affect Representations for Emotion Recognition",
#     author = "Saravia, Elvis  and
#       Liu, Hsien-Chi Toby  and
#       Huang, Yen-Hao  and
#       Wu, Junlin  and
#       Chen, Yi-Shin",
#     booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
#     month = oct # "-" # nov,
#     year = "2018",
#     address = "Brussels, Belgium",
#     publisher = "Association for Computational Linguistics",
#     url = "https://www.aclweb.org/anthology/D18-1404",
#     doi = "10.18653/v1/D18-1404",
#     pages = "3687--3697",
#     abstract = "Emotions are expressed in nuanced ways, which varies by collective or individual experiences, knowledge, and beliefs. Therefore, to understand emotion, as conveyed through text, a robust mechanism capable of capturing and modeling different linguistic nuances and phenomena is needed. We propose a semi-supervised, graph-based algorithm to produce rich structural descriptors which serve as the building blocks for constructing contextualized affect representations from text. The pattern-based representations are further enriched with word embeddings and evaluated through several emotion recognition tasks. Our experimental results demonstrate that the proposed method outperforms state-of-the-art techniques on emotion recognition tasks.",
# }