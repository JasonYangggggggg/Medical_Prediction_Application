import joblib
import numpy as np
import pickle

from sklearn.ensemble import RandomForestClassifier 


def prediction(input):
    target = []
    Inprompt = False
    model = joblib.load("Covid_Model.pkl")
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
        return f"{predict_result[0][1] * 100}%"
    elif Inprompt == False:
        return None
    

def loadLLM(Prediction_result):
    return None
