import streamlit as st
from sklearn.metrics import classification_report
from simpletransformers.classification import ClassificationModel
import torch

### initialize the model
cuda_available = torch.cuda.is_available()
model_args = {
    "max_seq_length": 512,
    "evaluate_during_training": True,
    "num_train_epochs": 20,
    "train_batch_size": 10,
    "eval_batch_size": 10,
    "output_dir": "./model/",
    "best_model_dir": "./model/",
    "dataloader_num_workers": 0,
    "use_multiprocessing": False,
    "logging_steps": 50
}
my_model = ClassificationModel("distilbert", "../model",  use_cuda=False, args=model_args, num_labels=3)

### set up prediction function
label_dict = {0: 'Nasty', 1: 'Neutral', 2: 'Nice'}

def prediction(text: str):
    final_prediction, _ = my_model.predict([text])
    for result in final_prediction:
        return label_dict[result]

st.write("""
         ## Sentiment Analysis Model Demonstration
         
         Provide a sentence and see how my model think of its sentiment!
         """)
with st.form("my_form"):
    text = st.text_input('Write down your sentence')

    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        st.write("Result:", prediction(text))