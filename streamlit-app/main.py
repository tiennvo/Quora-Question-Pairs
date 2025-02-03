from tkinter import *
import helper
from sklearn.ensemble import RandomForestClassifier
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Tải lại mô hình và tokenizer
model = AutoModelForSequenceClassification.from_pretrained("./phobert-finetuned")
tokenizer = AutoTokenizer.from_pretrained("./phobert-finetuned")

# Dự đoán độ tương đồng
def predict_similarity(question1, question2):
    inputs = tokenizer(question1, question2, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    return probs[0][1].item()  # Xác suất thuộc lớp tương đồng (1)
#model = pickle.load(open('D:/Python/Quora/Quora/quora-question-pairs/model.pkl', 'rb'))

def isDuplicate(question1, question2):
    q1 = helper.preprocess(question1)
    q2 = helper.preprocess(question2)
    result = predict_similarity(q1, q2)
    return result

def Find():
    question1 = q1.get()
    question2 = q2.get()

    score = isDuplicate(question1, question2)
    if score > 0.6:
        result_label.config(text= "Duplicate (Score: " + str(score*100) + "%)")
    else:
        result_label.config(text= "Not Duplicate (Score: " + str(score*100) + "%)")

    

root=Tk()
root.title('Quora')
root.minsize(height=300, width=850)

Label(root, text='Question 1').grid(row=2, column=0)
q1=Entry(root, width=80)
q1.grid(row=2, column=1)

Label(root, text='Question 2').grid(row=3, column=0)
q2=Entry(root, width=80)
q2.grid(row=3, column=1)

button=Frame(root)
Button(button, text="Find", command=Find).pack(side=LEFT)
button.grid(row=4, column=0)

result_label = Label(root, text="")
result_label.grid(row=5, column=0)

root.mainloop()