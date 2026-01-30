# 📌 Loan Intelligence Platform

An end-to-end **Machine Learning–driven financial decision system** that predicts loan approval outcomes and provides analytics, history tracking, administrative control, and reporting features through a full-stack web application.

This project goes beyond basic ML prediction by implementing **real-world backend systems** such as databases, APIs, statistics, dashboards, exports, and extensible architecture.

---

## 🚀 Project Overview

The system predicts whether a loan application will be approved based on applicant data using a **Support Vector Machine (SVM)** model. The prediction engine is deployed via a **Flask web application** and enhanced with persistent storage, analytics, and modular feature expansion.

The project is designed in **phases**, making it scalable and production-oriented.

---

## 🧠 Core ML Prediction System

* Binary classification using **SVM (Linear Kernel)**
* Feature preprocessing and scaling pipeline
* Class imbalance handled using balanced class weights
* Bias-aware and fair prediction approach
* Flask-based ML deployment
* Interactive and responsive prediction interface

---

## 🆕 Phase 1 – Prediction History & Analytics (Implemented)

### 🔹 Functional Features

* ✅ Persistent prediction history using a database
* ✅ Automatic saving of every user prediction
* ✅ Session-based tracking (no user login required)
* ✅ History page (`/history`) with clean UI
* ✅ One-click option to clear prediction history
* ✅ Statistical insights (approval rate, total predictions)
* ✅ Navigation integration across all pages

### 🔹 Technical Implementation

* SQLite database (`predictions.db`)
* Flask-SQLAlchemy ORM integration
* Database schema and models
* REST API endpoints:

  * `/history`
  * `/clear_history`
  * `/api/stats`
* Frontend–backend API communication

---

## 🔐 Phase 2 – Admin Dashboard (Planned / In Progress)

* Password-protected admin authentication
* Admin dashboard to view all user predictions
* Platform-level statistics (approval rate, activity)
* User management (view/delete prediction data)
* CSV export of prediction records

---

## 📤 Phase 3 – Advanced Data Export (Planned)

* CSV export (downloadable reports)
* Excel export for professional analysis
* Filtering options (date, result, approval status)

---

## 📧 Phase 4 – Email & Reporting System (Planned)

* Automated email delivery of results
* PDF report generation
* Professional email templates

---

## 🧰 Technologies Used

* **Machine Learning:** Scikit-learn (SVM)
* **Backend:** Flask, Flask-SQLAlchemy
* **Frontend:** HTML, CSS, JavaScript
* **Database:** SQLite
* **Data Processing:** Pandas, NumPy

---

## 📊 Model Performance

* Balanced Accuracy ≈ **83%**
* Evaluated using Confusion Matrix and Accuracy Score

---

## 👩‍💻 Developer

**Mahak**
Designed and implemented backend persistence, prediction history, analytics APIs, database integration, and system extensibility.

---

## 📎 Notes

* This project is suitable for demonstrating **real-world ML deployment**, **backend system design**, and **full-stack development**.
* The architecture supports incremental feature expansion and production-style workflows.

---

## ⚖️ Attribution

Initial ML prediction concept inspired by an open-source project. This implementation significantly extends the original idea with backend architecture, analytics, and platform features.
