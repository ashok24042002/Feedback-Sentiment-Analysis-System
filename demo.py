import mysql.connector
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

from transformers import AutoTokenizer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

import nltk



from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = pickle.load(open('model.pkl', 'rb'))

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="forum"
)

mycursor = mydb.cursor()
mycursor.execute("SELECT post,id FROM discussion")
myresult = mycursor.fetchall()

# mycursor1 = mydb.cursor()
# mycursor1.execute("SELECT id FROM discussion")
# myresult1 = mycursor.fetchall()

# for ex in myresult:
#     print(type(ex[0]))
#     print(type(ex[1]))

for ex in myresult:
    encoded_text = tokenizer(ex[0], return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
            
    negative = scores[0]
    neutral = scores[1]
    positive = scores[2]
        
    if (negative >= neutral) and (negative >= positive):
        sql = "UPDATE discussion SET rating=%s,stars=%s WHERE id=%s"
        val = ("negative",1,int(ex[1]))
        mycursor.execute(sql, val)
        mydb.commit()
        print(mycursor.rowcount, "record updated with negative.")
    
    elif (neutral >= negative) and (neutral >= positive):
        sql = "UPDATE discussion SET rating=%s,stars=%s WHERE id=%s"
        val = ("neutral",2,int(ex[1]))
        mycursor.execute(sql, val)
        mydb.commit()
        print(mycursor.rowcount, "record updated with neutral.")

    else:
        sql = "UPDATE discussion SET rating=%s,stars=%s WHERE id=%s"
        val = ("positive",3,int(ex[1]))
        mycursor.execute(sql, val)
        mydb.commit()
        print(mycursor.rowcount, "record updated with positive.")