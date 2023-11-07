from transformers import AutoTokenizer, TextDataset, DataCollatorForLanguageModeling, AutoModelForCausalLM, pipeline, \
                         Trainer, TrainingArguments
import pandas as pd
from datasets import Dataset

MODEL = 'gpt2'

tokenizer = AutoTokenizer.from_pretrained(MODEL)  # load up a standard gpt2 model

tokenizer.pad_token = tokenizer.eos_token  # set the pad token to avoid a warning



loaded_model = AutoModelForCausalLM.from_pretrained('./gpt2_text_to_sql') ##Better
#loaded_model = AutoModelForCausalLM.from_pretrained(MODEL) ##Worst
sql_generator = pipeline('text-generation', model=loaded_model, tokenizer=tokenizer)



# Add our singular prompt
CONVERSION_PROMPT = 'Create a SQL Query using for the given text statement using the table schema provided '  # SQL conversion task

CONTEXT = '''###SCHEMA: customer(custid,first_name,last_name,date_of_birth,address,phone_number,email), obligor(obligor_id,custid,relationship,credit_score,risk_rating), facility(facility_id,obligor_id,facility_type,facility_amount,drawdown_date,maturity_date), credit_score(credit_score_id,obligor_id,credit_score_agency,credit_score,credit_score_date), risk_rating(risk_rating_id,obligor_id,risk_rating,risk_rating_date)'''
temp = '''
SCHEMA:
CREATE TABLE customer (
  custid INT NOT NULL AUTO_INCREMENT,
  first_name VARCHAR(255) NOT NULL,
  last_name VARCHAR(255) NOT NULL,
  date_of_birth DATE NOT NULL,
  address VARCHAR(255) NOT NULL,
  phone_number VARCHAR(255) NOT NULL,
  email VARCHAR(255) NOT NULL,
  PRIMARY KEY (custid)
);

CREATE TABLE obligor (
  obligor_id INT NOT NULL AUTO_INCREMENT,
  custid INT NOT NULL,
  relationship VARCHAR(255) NOT NULL,
  credit_score INT NOT NULL,
  risk_rating VARCHAR(255) NOT NULL,
  PRIMARY KEY (obligor_id),
  FOREIGN KEY (custid) REFERENCES customer(custid)
);

CREATE TABLE facility (
  facility_id INT NOT NULL AUTO_INCREMENT,
  obligor_id INT NOT NULL,
  facility_type VARCHAR(255) NOT NULL,
  facility_amount DECIMAL(10,2) NOT NULL,
  drawdown_date DATE NOT NULL,
  maturity_date DATE NOT NULL,
  PRIMARY KEY (facility_id),
  FOREIGN KEY (obligor_id) REFERENCES obligor(obligor_id)
);

CREATE TABLE credit_score (
  credit_score_id INT NOT NULL AUTO_INCREMENT,
  obligor_id INT NOT NULL,
  credit_score_agency VARCHAR(255) NOT NULL,
  credit_score INT NOT NULL,
  credit_score_date DATE NOT NULL,
  PRIMARY KEY (credit_score_id),
  FOREIGN KEY (obligor_id) REFERENCES obligor(obligor_id)
);

CREATE TABLE risk_rating (
  risk_rating_id INT NOT NULL AUTO_INCREMENT,
  obligor_id INT NOT NULL,
  risk_rating VARCHAR(255) NOT NULL,
  risk_rating_date DATE NOT NULL,
  PRIMARY KEY (risk_rating_id),
  FOREIGN KEY (obligor_id) REFERENCES obligor(obligor_id)
);
'''

CONVERSION_TOKEN = ' SQL:'


#text_sample = 'find the customer with high risk rating'
#text_sample = "Retrieve customers with a credit score below 650 and a 'Low' risk rating"
#text_sample = "Calculates the average credit score for obligors by relationship."
#text_sample = "Calculates the average credit score for customers with multiple obligors grouped by first name, last name, and relationship";
text_sample = "customers with obligors having a 'Medium' risk rating and credit score below 700."
#conversion_text_sample = f'{CONVERSION_PROMPT}English: {text_sample}\n{CONVERSION_TOKEN}'
#conversion_text_sample = f'{CONVERSION_PROMPT} ' + f'{CONTEXT}\nText: ' + text_sample + '\n' + CONVERSION_TOKEN
conversion_text_sample = f'{CONVERSION_PROMPT} ' + f'{CONTEXT} ###TEXT: ' + text_sample + '' + CONVERSION_TOKEN + ' ' 


print(sql_generator(
    conversion_text_sample, num_beams=2, early_stopping=True, temperature=0.7,
    max_new_tokens=54
)[0]['generated_text'])



# Another example
#text_sample = 'r of x is sum from 0 to x of x squared'
#conversion_text_sample = f'{CONVERSION_PROMPT}English: {text_sample}\n{CONVERSION_TOKEN}'

print(sql_generator(
    conversion_text_sample, num_beams=9, early_stopping=True, temperature=0.7,
    max_length=len(tokenizer.encode(conversion_text_sample)) + 54
)[0]['generated_text'])

