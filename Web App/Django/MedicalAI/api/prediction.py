import joblib
import numpy as np
import pickle
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain.prompts.chat import (ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate)
import os

# from sklearn.ensemble import RandomForestClassifier 

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, '', 'api', 'Covid_prediction_model.pkl')

def prediction(input):
    target = []
    for i in range(20):
        target.append(0)
    Inprompt = False
    model = joblib.load(model_path)
    question_inComplete = input.replace(",", "")
    question_inComplete = question_inComplete.replace("?", "")
    List_data = question_inComplete.split()
    for i in range(len(List_data)):
        if List_data[i] == 'Breathing_Problem':
            Inprompt = True
            target[0] = 1
            continue
        elif List_data[i] == 'Fever':
            Inprompt = True
            target[1] = 1
            continue
        elif List_data[i] == 'Dry_Cough':
            Inprompt = True
            target[2] = 1
            continue
        elif List_data[i] == 'Sore_Throat':
            Inprompt = True
            target[3] = 1
            continue
        elif List_data[i] == 'Running_Nose':
            Inprompt = True
            target[4] = 1
            continue
        elif List_data[i] == 'Asthma':
            Inprompt = True
            target[5] = 1
            continue
        elif List_data[i] == 'Chronic_Lung_Disease':
            Inprompt = True
            target[6] = 1
            continue
        elif List_data[i] == 'Headache':
            Inprompt = True
            target[7] = 1
            continue
        elif List_data[i] == 'Heart_Disease':
            Inprompt = True
            target[8] = 1
        elif List_data[i] == 'Diabetes':
            Inprompt = True
            target[9] = 1
        elif List_data[i] == 'Hyper_Tension':
            Inprompt = True
            target[10] = 1
        elif List_data[i] == 'Fatigue':
            Inprompt = True
            target[11] = 1
        elif List_data[i] == 'Gastrointestinal':
            Inprompt = True
            target[12] = 1
        elif List_data[i] == 'Abroad_Travel':
            Inprompt = True
            target[13] = 1
        elif List_data[i] == 'Contact_With_Covid_Patient':
            Inprompt = True
            target[14] = 1
        elif List_data[i] == 'Attended_Large_Gathering':
            Inprompt = True
            target[15] = 1
        elif List_data[i] == 'Visited_Public_Exposed_Places':
            Inprompt = True
            target[16] = 1
        elif List_data[i] == 'Family_Working_In_Public_Exposed_Places':
            Inprompt = True
            target[17] = 1
        elif List_data[i] == 'Did_Wear_Mask':
            Inprompt = True
            target[18] = 1
        elif List_data[i] == 'Sanitization_From_Market':
            Inprompt = True
            target[19] = 1
    if Inprompt == True:
        data_transform = np.array([target])
        predict_result = model.predict_proba(data_transform)
        return predict_result[0][1] * 100
    elif Inprompt == False:
        return None
    

def responseLLM(input):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = '' #Put your token in here
    prediction_res = prediction(input)
    if prediction_res == None:
        return "Not enought information"
    else:
        template_bot = """
        You are a helpful medical assistant that that can answer questions about Covid 19 
        based on the prediction result: {prediction}
        
        if the {prediction} is lower than 10, say: "No, you do not have Covid"
        
        if the {prediction} is lower than 50, say: "The chances is low, my prediction is {prediction}%" 
        
        if the {prediction} is higher than 50, say: "The chances that you have Covid 19 is {prediction}%, you need to have PCR test"
        
        Your answers should be verbose and detailed.
        """
        #prompt_bot = PromptTemplate(template=template_bot, input_variables=["question"])
        system_message_prompt = SystemMessagePromptTemplate.from_template(template_bot)
        human_template = "Answer the following question: {question}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
        llm_chain = LLMChain(prompt=chat_prompt, llm=HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":1.2,"max_length":512}), verbose=True)
        return llm_chain.run(question=input, prediction=prediction_res)
        

