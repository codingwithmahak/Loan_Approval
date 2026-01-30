from flask import Flask, render_template, request, session, redirect, url_for, jsonify
import numpy as np
import pickle
import os
from datetime import datetime
import uuid

# Import database
from database import db, Prediction, init_db

app = Flask(__name__)
app.secret_key = 'loan_prediction_secret_key_2024'  # Change this in production!

# Initialize database
init_db(app)

# ===============================
# Load trained model (Pipeline)
# ===============================
MODEL_PATH = os.path.join("model", "loan_model.pkl")

try:
    model = pickle.load(open(MODEL_PATH, "rb"))
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# ===============================
# Helper Functions
# ===============================
def calculate_emi(principal, months, annual_rate=8.5):
    """Calculate EMI (Equated Monthly Installment)"""
    if months == 0:
        return 0
    monthly_rate = annual_rate / (12 * 100)
    emi = principal * monthly_rate * ((1 + monthly_rate) ** months) / (((1 + monthly_rate) ** months) - 1)
    return emi

def save_prediction_to_db(form_data, prediction_result, confidence):
    """Save prediction to database"""
    try:
        # Generate session ID if not exists
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
        
        # Calculate additional fields
        total_income = float(form_data['ApplicantIncome']) + float(form_data['CoapplicantIncome'])
        loan_amount = float(form_data['LoanAmount'])
        loan_to_income_ratio = loan_amount / total_income if total_income > 0 else 0
        
        # Create new prediction record
        prediction = Prediction(
            session_id=session['session_id'],
            Gender=int(form_data['Gender']),
            Married=int(form_data['Married']),
            Dependents=int(form_data['Dependents']),
            Education=int(form_data['Education']),
            Self_Employed=int(form_data['Self_Employed']),
            ApplicantIncome=float(form_data['ApplicantIncome']),
            CoapplicantIncome=float(form_data['CoapplicantIncome']),
            LoanAmount=loan_amount,
            Loan_Amount_Term=float(form_data['Loan_Amount_Term']),
            Credit_History=float(form_data['Credit_History']),
            Property_Area=int(form_data['Property_Area']),
            prediction_result=prediction_result,
            confidence=confidence,
            total_income=total_income,
            loan_to_income_ratio=loan_to_income_ratio
        )
        
        # Save to database
        db.session.add(prediction)
        db.session.commit()
        print(f"✅ Prediction saved to database (ID: {prediction.id})")
        return prediction.id
        
    except Exception as e:
        print(f"❌ Error saving prediction to database: {e}")
        db.session.rollback()
        return None

def get_user_predictions():
    """Get all predictions for current user"""
    if 'session_id' not in session:
        return []
    
    try:
        predictions = Prediction.query.filter_by(
            session_id=session['session_id']
        ).order_by(Prediction.timestamp.desc()).all()
        
        return [p.to_dict() for p in predictions]
    except Exception as e:
        print(f"❌ Error fetching predictions: {e}")
        return []

# ===============================
# Routes
# ===============================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/predict")
def predict():
    return render_template("predict.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/history")
def history():
    """Show prediction history for current user"""
    predictions = get_user_predictions()
    return render_template("history.html", predictions=predictions)

@app.route("/clear_history", methods=["POST"])
def clear_history():
    """Clear prediction history for current user"""
    if 'session_id' in session:
        try:
            deleted = Prediction.query.filter_by(session_id=session['session_id']).delete()
            db.session.commit()
            return jsonify({
                'success': True,
                'message': f'Cleared {deleted} predictions from your history.'
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'Error clearing history: {str(e)}'
            })
    return jsonify({
        'success': False,
        'message': 'No history to clear.'
    })

@app.route("/result", methods=["POST"])
def result():
    if model is None:
        return render_template("error.html", message="Model not loaded. Please contact administrator.")
    
    try:
        # ===============================
        # Collect and validate form data
        # ===============================
        data = {
            'Gender': int(request.form.get('Gender', 0)),
            'Married': int(request.form.get('Married', 0)),
            'Dependents': int(request.form.get('Dependents', 0)),
            'Education': int(request.form.get('Education', 0)),
            'Self_Employed': int(request.form.get('Self_Employed', 0)),
            'ApplicantIncome': float(request.form.get('ApplicantIncome', 0)),
            'CoapplicantIncome': float(request.form.get('CoapplicantIncome', 0)),
            'LoanAmount': float(request.form.get('LoanAmount', 0)),
            'Loan_Amount_Term': float(request.form.get('Loan_Amount_Term', 360)),
            'Credit_History': float(request.form.get('Credit_History', 0)),
            'Property_Area': int(request.form.get('Property_Area', 0))
        }
        
        # ===============================
        # Validate data ranges
        # ===============================
        validation_errors = []
        
        if data['ApplicantIncome'] < 10000:
            validation_errors.append("Applicant income is too low")
        if data['LoanAmount'] < 10000:
            validation_errors.append("Loan amount is too low")
        if data['LoanAmount'] > 10000000:
            validation_errors.append("Loan amount is too high")
        if data['Loan_Amount_Term'] < 12:
            validation_errors.append("Loan term is too short")
        if data['Loan_Amount_Term'] > 360:
            validation_errors.append("Loan term is too long")
        
        if validation_errors:
            return render_template("error.html", message=" | ".join(validation_errors))
        
        # ===============================
        # Prepare data for model
        # ===============================
        input_features = [
            data['Gender'],
            data['Married'],
            data['Dependents'],
            data['Education'],
            data['Self_Employed'],
            data['ApplicantIncome'],
            data['CoapplicantIncome'],
            data['LoanAmount'],
            data['Loan_Amount_Term'],
            data['Credit_History'],
            data['Property_Area']
        ]
        
        input_array = np.array(input_features).reshape(1, -1)
        
        # ===============================
        # Make prediction
        # ===============================
        prediction = model.predict(input_array)
        prediction_proba = model.predict_proba(input_array) if hasattr(model, 'predict_proba') else None
        
        result_value = int(prediction[0])
        
        # ===============================
        # Calculate confidence score
        # ===============================
        confidence = 83  # Default
        if prediction_proba is not None:
            confidence = round(max(prediction_proba[0]) * 100, 1)
        
        # ===============================
        # Save prediction to database
        # ===============================
        save_prediction_to_db(data, result_value, confidence)
        
        # ===============================
        # Calculate additional metrics
        # ===============================
        total_income = data['ApplicantIncome'] + data['CoapplicantIncome']
        loan_to_income_ratio = data['LoanAmount'] / total_income if total_income > 0 else 0
        emi = calculate_emi(data['LoanAmount'], data['Loan_Amount_Term'])
        
        # ===============================
        # Prepare result data for template
        # ===============================
        result_data = {
            'result': result_value,
            'confidence': confidence,
            'loan_amount': data['LoanAmount'],
            'total_income': total_income,
            'loan_to_income_ratio': round(loan_to_income_ratio, 2),
            'emi': round(emi, 2),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'applicant_income': data['ApplicantIncome'],
            'coapplicant_income': data['CoapplicantIncome'],
            'loan_term': data['Loan_Amount_Term']
        }
        
        # Debug output
        print("="*50)
        print(f"PREDICTION RESULT: {'APPROVED' if result_value == 1 else 'REJECTED'}")
        print(f"Confidence: {confidence}%")
        print(f"Loan Amount: ₹{data['LoanAmount']:,.2f}")
        print(f"Total Income: ₹{total_income:,.2f}")
        print(f"Loan to Income Ratio: {loan_to_income_ratio:.2f}")
        print("="*50)
        
        return render_template("result.html", **result_data)
        
    except ValueError as e:
        print(f"❌ Value error: {e}")
        return render_template("error.html", message="Invalid input values. Please check your entries.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return render_template("error.html", message=f"An error occurred: {str(e)}")

@app.route("/api/stats")
def get_stats():
    """Get statistics for current user"""
    predictions = get_user_predictions()
    
    if not predictions:
        return jsonify({
            'total': 0,
            'approved': 0,
            'rejected': 0,
            'approval_rate': 0,
            'avg_confidence': 0
        })
    
    total = len(predictions)
    approved = sum(1 for p in predictions if p['result'] == 'Approved')
    rejected = total - approved
    approval_rate = (approved / total * 100) if total > 0 else 0
    avg_confidence = sum(p['confidence'] for p in predictions) / total if total > 0 else 0
    
    return jsonify({
        'total': total,
        'approved': approved,
        'rejected': rejected,
        'approval_rate': round(approval_rate, 1),
        'avg_confidence': round(avg_confidence, 1)
    })

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)