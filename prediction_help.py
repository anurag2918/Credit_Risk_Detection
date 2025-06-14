import joblib
import pandas as pd

model_data = joblib.load("model_data.joblib")
model = model_data['model']

def prepare_input(
    age, income, loan_amount, loan_tenure_months, 
    avg_dpd_per_delinquency,num_open_accounts,
    residence_type, loan_purpose, loan_type, 
    cb_person_default_on_file
):
    input_data = {
        'person_age': age,
        'person_income': income,
        'person_emp_length': loan_tenure_months,
        'loan_amnt': loan_amount,
        'loan_int_rate': avg_dpd_per_delinquency,
        'loan_percent_income': loan_amount / income if income > 0 else 0,
        'cb_person_cred_hist_length': num_open_accounts,
        'person_home_ownership': residence_type,
        'loan_intent': loan_purpose,
        'loan_grade': loan_type,
        'cb_person_default_on_file': cb_person_default_on_file
    }
    return pd.DataFrame([input_data])

def predict_credit_risk(input_df):
    default_prob = model.predict_proba(input_df)[:, 1][0]
    credit_score = 300 + (1 - default_prob) * 600
    
    if credit_score < 500:
        rating = 'Poor'
    elif credit_score < 650:
        rating = 'Average'
    elif credit_score < 750:
        rating = 'Good'
    else:
        rating = 'Excellent'
        
    return default_prob, int(credit_score), rating

def predict(age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
            num_open_accounts,residence_type, loan_purpose, loan_type, cb_person_default_on_file):
    
    input_df = prepare_input(
        age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
        num_open_accounts,residence_type, loan_purpose, loan_type, cb_person_default_on_file
    )
    return predict_credit_risk(input_df)
