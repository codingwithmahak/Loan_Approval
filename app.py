from flask import Flask, render_template, request, session, redirect, url_for, jsonify
import numpy as np
import pickle
import os
from datetime import datetime
import uuid
from flask_sqlalchemy import SQLAlchemy

# ==================== FLASK APP SETUP ====================
app = Flask(__name__)
app.secret_key = 'loan_prediction_secret_key_2024'

# ==================== DATABASE SETUP ====================
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///predictions.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Database Table for storing predictions
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(100), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Form data
    Gender = db.Column(db.Integer)
    Married = db.Column(db.Integer)
    Dependents = db.Column(db.Integer)
    Education = db.Column(db.Integer)
    Self_Employed = db.Column(db.Integer)
    ApplicantIncome = db.Column(db.Float)
    CoapplicantIncome = db.Column(db.Float)
    LoanAmount = db.Column(db.Float)
    Loan_Amount_Term = db.Column(db.Float)
    Credit_History = db.Column(db.Float)
    Property_Area = db.Column(db.Integer)
    
    # Results
    prediction_result = db.Column(db.Integer)
    confidence = db.Column(db.Float)
    total_income = db.Column(db.Float)
    loan_to_income_ratio = db.Column(db.Float)

# Create tables
with app.app_context():
    db.create_all()
    print("✅ Database created successfully!")

# ==================== LOAD ML MODEL ====================
MODEL_PATH = os.path.join("model", "loan_model.pkl")
try:
    model = pickle.load(open(MODEL_PATH, "rb"))
    print("✅ ML Model loaded successfully!")
except:
    model = None
    print("⚠️ Model not found - running without ML")

# ==================== ROUTES ====================
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

# ===== NEW: HISTORY PAGE =====
@app.route("/history")
def history():
    # Get user's session ID
    if 'session_id' not in session:
        return render_template("history.html", predictions=[])
    
    # Fetch predictions from database
    predictions = Prediction.query.filter_by(
        session_id=session['session_id']
    ).order_by(Prediction.timestamp.desc()).all()
    
    # Convert to list for template
    pred_list = []
    for p in predictions:
        pred_list.append({
            'id': p.id,
            'timestamp': p.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'result': 'APPROVED' if p.prediction_result == 1 else 'REJECTED',
            'loan_amount': p.LoanAmount,
            'applicant_income': p.ApplicantIncome,
            'coapplicant_income': p.CoapplicantIncome,
            'total_income': p.total_income,
            'confidence': p.confidence,
            'gender': 'Male' if p.Gender == 1 else 'Female',
            'married': 'Yes' if p.Married == 1 else 'No',
            'dependents': p.Dependents
        })
    
    return render_template("history.html", predictions=pred_list)

# ===== NEW: CLEAR HISTORY =====
@app.route("/clear_history", methods=["POST"])
def clear_history():
    if 'session_id' in session:
        try:
            deleted = Prediction.query.filter_by(session_id=session['session_id']).delete()
            db.session.commit()
            return jsonify({'success': True, 'deleted': deleted})
        except:
            return jsonify({'success': False})
    return jsonify({'success': False})

# ===== PREDICTION RESULT =====
@app.route("/result", methods=["POST"])
def result():
    if not model:
        return "Model error", 500
    
    try:
        # Get form data
        data = request.form
        
        # Convert to correct types
        form_data = {}
        for key in data:
            if key in ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']:
                form_data[key] = float(data[key])
            else:
                form_data[key] = int(data[key])
        
        # Prepare features for ML model
        features = [
            form_data['Gender'], form_data['Married'], form_data['Dependents'],
            form_data['Education'], form_data['Self_Employed'], form_data['ApplicantIncome'],
            form_data['CoapplicantIncome'], form_data['LoanAmount'], form_data['Loan_Amount_Term'],
            form_data['Credit_History'], form_data['Property_Area']
        ]
        
        # Make prediction
        prediction = model.predict([features])[0]
        confidence = 83  # Default confidence
        
        # Generate session ID if not exists
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
        
        # Calculate additional info
        total_income = form_data['ApplicantIncome'] + form_data['CoapplicantIncome']
        loan_to_income = form_data['LoanAmount'] / total_income if total_income > 0 else 0
        
        # Save to database
        pred = Prediction(
            session_id=session['session_id'],
            Gender=form_data['Gender'],
            Married=form_data['Married'],
            Dependents=form_data['Dependents'],
            Education=form_data['Education'],
            Self_Employed=form_data['Self_Employed'],
            ApplicantIncome=form_data['ApplicantIncome'],
            CoapplicantIncome=form_data['CoapplicantIncome'],
            LoanAmount=form_data['LoanAmount'],
            Loan_Amount_Term=form_data['Loan_Amount_Term'],
            Credit_History=form_data['Credit_History'],
            Property_Area=form_data['Property_Area'],
            prediction_result=int(prediction),
            confidence=confidence,
            total_income=total_income,
            loan_to_income_ratio=loan_to_income
        )
        
        db.session.add(pred)
        db.session.commit()
        print(f"✅ Prediction saved to database (ID: {pred.id})")
        
        # Return result page
        return render_template("result.html", 
                             result=int(prediction),
                             confidence=confidence,
                             loan_amount=form_data['LoanAmount'],
                             total_income=total_income,
                             loan_to_income_ratio=round(loan_to_income, 2))
                             
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return f"Error processing request: {str(e)}", 400

# ===== STATISTICS API =====
@app.route("/api/stats")
def get_stats():
    if 'session_id' not in session:
        return jsonify({'total': 0, 'approved': 0, 'rejected': 0})
    
    predictions = Prediction.query.filter_by(session_id=session['session_id']).all()
    total = len(predictions)
    approved = sum(1 for p in predictions if p.prediction_result == 1)
    
    return jsonify({
        'total': total,
        'approved': approved,
        'rejected': total - approved,
        'approval_rate': round((approved/total*100), 1) if total > 0 else 0
    })

# ==================== RUN APP ====================
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)