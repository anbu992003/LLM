from transformers import AutoTokenizer, TextDataset, DataCollatorForLanguageModeling, AutoModelForCausalLM, pipeline, \
                         Trainer, TrainingArguments
import pandas as pd
from datasets import Dataset



MODEL = 'gpt2'

tokenizer = AutoTokenizer.from_pretrained(MODEL)  # load up a standard gpt2 model

tokenizer.pad_token = tokenizer.eos_token  # set the pad token to avoid a warning




data = pd.read_csv('../data/text_to_sql_dataset.csv', encoding = "ISO-8859-1")

print(data.shape)

data.head(2)




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

CONVERSION_TOKEN = ' ###SQL:'


# This is our "training prompt" that we want GPT2 to recognize and learn
training_examples = f'{CONVERSION_PROMPT} ' + f'{CONTEXT} ###TEXT: ' + data['text'] + '' + CONVERSION_TOKEN + ' ' + data['sql'].astype(str)

print(training_examples[0])



task_df = pd.DataFrame({'text': training_examples})

task_df.head(2)





# adding the EOS token at the end so the model knows when to stop predicting

task_df['text'] = task_df['text'].map(lambda x: f'{x}{tokenizer.eos_token}')




SQL_data = Dataset.from_pandas(task_df)  # turn a pandas DataFrame into a Dataset

def preprocess(examples):  
    # tokenize our text but don't pad because our collator will pad for us dynamically
    return tokenizer(examples['text'], truncation=True)

SQL_data = SQL_data.map(preprocess, batched=True)

SQL_data = SQL_data.train_test_split(train_size=.8)




SQL_data['train'][0]



# standard data collator for auto-regressive language modelling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

SQL_gpt2 = AutoModelForCausalLM.from_pretrained(MODEL)

SQL_data



training_args = TrainingArguments(
    output_dir="./gpt2_text_to_sql",
    overwrite_output_dir=True, # overwrite the content of the output directory
    num_train_epochs=5, # number of training epochs
    per_device_train_batch_size=1, # batch size for training
    per_device_eval_batch_size=20,  # batch size for evaluation
    load_best_model_at_end=True,
    logging_steps=5,
    log_level='info',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    use_mps_device=False
)

trainer = Trainer(
    model=SQL_gpt2,
    args=training_args,
    train_dataset=SQL_data["train"],
    eval_dataset=SQL_data["test"],
    data_collator=data_collator,
)

trainer.evaluate()


trainer.train()


trainer.save_model()




loaded_model = AutoModelForCausalLM.from_pretrained('./gpt2_text_to_sql')
SQL_generator = pipeline('text-generation', model=loaded_model, tokenizer=tokenizer)

text_sample = 'find the customer with highest risk rating'
#conversion_text_sample = f'{CONVERSION_PROMPT}English: {text_sample}\n{CONVERSION_TOKEN}'
#conversion_text_sample = f'{CONVERSION_PROMPT} ' + f'{CONTEXT}\nText: ' + text_sample + '\n' + CONVERSION_TOKEN
conversion_text_sample = f'{CONVERSION_PROMPT} ' + f'{CONTEXT} ###TEXT: ' + text_sample + '' + CONVERSION_TOKEN + ' ' 


print(SQL_generator(
    conversion_text_sample, num_beams=2, early_stopping=True, temperature=0.7,
    max_new_tokens=24
)[0]['generated_text'])



# Another example
#text_sample = 'r of x is sum from 0 to x of x squared'
#conversion_text_sample = f'{CONVERSION_PROMPT}English: {text_sample}\n{CONVERSION_TOKEN}'

print(SQL_generator(
    conversion_text_sample, num_beams=5, early_stopping=True, temperature=0.7,
    max_length=len(tokenizer.encode(conversion_text_sample)) + 20
)[0]['generated_text'])

