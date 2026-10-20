import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox , filedialog
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
import numpy as np

class SimpleFuzzySystem:
    def __init__(self):
        pass
    
    def decide(self, probability):
        uncertainty = 1 - abs(probability - 0.5) * 2
        
        if probability < 0.3:
            decision, status = 0, "Healthy"
        elif probability > 0.7:
            decision, status = 1, "Diseased"
        else:
            if probability > 0.5:
                decision, status = 1, "Borderline - Needs Review"
            else:
                decision, status = 0, "Borderline - Needs Review"
        
        mu_healthy = 1 - probability
        mu_diseased = probability
        
        return {
            'probability': probability,
            'decision': decision,
            'status': status,
            'confidence': 1 - uncertainty,
            'uncertainty': uncertainty,
            'mu_healthy': mu_healthy,
            'mu_diseased': mu_diseased,
            'needs_review': uncertainty > 0.3 or abs(mu_diseased - mu_healthy) < 0.4
        }

def train_simple_model():
    data_path = 'heart_statlog_cleveland_hungary_final.csv'
    
    if not os.path.exists(data_path):
        root = tk.Tk()
        root.withdraw()
        data_path = filedialog.askopenfilename(
            title="choose dataset",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        root.destroy()
        
        if not data_path:
            raise FileNotFoundError("choose datset")
    
    data = pd.read_csv(data_path)
    data = data.drop_duplicates()
    data = data.fillna(data.mean())
    
    cat_cols = ['sex', 'chest_pain_type', 'fasting_blood_sugar', 'resting_ecg', 'exercise_angina', 'ST_slope']
    data = pd.get_dummies(data, columns=cat_cols, drop_first=True)
    
    feature_cols = [col for col in data.columns if col != 'target']
    X = data[feature_cols]
    y = data['target']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    rf_model = RandomForestClassifier(
    criterion='gini',
    max_depth=8,
    max_features='sqrt',
    n_estimators=200,
    random_state=42,
    )

    knn_model = KNeighborsClassifier(
        n_neighbors=11,
        algorithm = 'auto',
        p =  2,
        weights = 'distance'
    )

    lr_model = LogisticRegression(
        class_weight='balanced',
        max_iter=20,
        random_state=42,
        solver='liblinear',
    )

    nb_model = GaussianNB(
        var_smoothing=0.8111308307896871,

    )

    gb_model = GradientBoostingClassifier(
        learning_rate=0.1,
        max_depth=2,
        n_estimators=50,
        random_state=42,
    )

    ab_model = AdaBoostClassifier(
        learning_rate=1,
        n_estimators=100,
        random_state=42,
    )
    
    rf_model.fit(X_scaled, y)
    ab_model.fit(X_scaled, y)
    gb_model.fit(X_scaled, y)
    nb_model.fit(X_scaled, y)
    knn_model.fit(X_scaled, y)
    lr_model.fit(X_scaled, y)


    
    voting_model = VotingClassifier([
        ('random_forest', rf_model),
        ('knn', knn_model),
        ('logistic_regression', lr_model),
        ('naive_bayes', nb_model),
        ('gradient_boosting', gb_model),
        ('adaboost', ab_model)
    ], voting='soft')
    voting_model.fit(X_scaled, y)
    
    model_data = {
        'scaler': scaler,
        'voting_model': voting_model,
        'fuzzy_system': SimpleFuzzySystem(),
        'feature_names': feature_cols
    }
    
    with open('heart_model_simple.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    return model_data

class HeartApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Heart Disease Detection")
        self.root.geometry("700x650")
        
        model_path = 'heart_model_simple.pkl'
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                self.model_data = pickle.load(f)
        else:
            self.model_data = train_simple_model()
        
        self.scaler = self.model_data['scaler']
        self.model = self.model_data['voting_model']
        self.fuzzy = self.model_data['fuzzy_system']
        self.feature_names = self.model_data['feature_names']
        
        self.create_gui()
    
    def create_gui(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(main, text="Heart Disease Detection System", 
                 font=('Arial', 16, 'bold')).pack(pady=10)
        
        input_frame = ttk.LabelFrame(main, text="Patient Information", padding=10)
        input_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.entries = {}
        fields = [
            ('age', 'Age'),
            ('sex', 'Sex (0=Female, 1=Male)'),
            ('chest_pain_type', 'Chest Pain Type (1-4)'),
            ('resting_bp_s', 'Resting BP'),
            ('cholesterol', 'Cholesterol'),
            ('fasting_blood_sugar', 'Fasting Blood Sugar (0/1)'),
            ('resting_ecg', 'Resting ECG (0-2)'),
            ('max_heart_rate', 'Max Heart Rate'),
            ('exercise_angina', 'Exercise Angina (0/1)'),
            ('oldpeak', 'ST Depression'),
            ('ST_slope', 'ST Slope (1-3)')
        ]
        
        for i, (field, label) in enumerate(fields):
            row = i // 2
            col = i % 2
            frame = ttk.Frame(input_frame)
            frame.grid(row=row, column=col, padx=5, pady=5, sticky='w')
            ttk.Label(frame, text=label).pack(anchor='w')
            entry = ttk.Entry(frame, width=20)
            entry.pack()
            self.entries[field] = entry
        
        defaults = [58, 1, 2, 140, 220, 0, 1, 170, 0, 2.3, 2]
        for entry, value in zip(self.entries.values(), defaults):
            entry.insert(0, str(value))
        
        btn_frame = ttk.Frame(main)
        btn_frame.pack(pady=10)
        
        ttk.Button(btn_frame, text="Predict", command=self.predict).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Clear", command=self.clear).pack(side=tk.LEFT, padx=5)
        
        self.result_text = tk.Text(main, height=15, width=80)
        self.result_text.pack(fill=tk.BOTH, expand=True, pady=10)
    
    def predict(self):
        try:
            input_data = {}
            for field, entry in self.entries.items():
                input_data[field] = float(entry.get())
            
            feature_dict = {
                'age': input_data['age'],
                'resting_bp_s': input_data['resting_bp_s'],
                'cholesterol': input_data['cholesterol'],
                'max_heart_rate': input_data['max_heart_rate'],
                'oldpeak': input_data['oldpeak'],
                'sex_1': 1 if input_data['sex'] == 1 else 0,
                'chest_pain_type_2': 1 if input_data['chest_pain_type'] == 2 else 0,
                'chest_pain_type_3': 1 if input_data['chest_pain_type'] == 3 else 0,
                'chest_pain_type_4': 1 if input_data['chest_pain_type'] == 4 else 0,
                'fasting_blood_sugar_1': 1 if input_data['fasting_blood_sugar'] == 1 else 0,
                'resting_ecg_1': 1 if input_data['resting_ecg'] == 1 else 0,
                'resting_ecg_2': 1 if input_data['resting_ecg'] == 2 else 0,
                'exercise_angina_1': 1 if input_data['exercise_angina'] == 1 else 0,
                'ST_slope_1': 1 if input_data['ST_slope'] == 1 else 0,
                'ST_slope_2': 1 if input_data['ST_slope'] == 2 else 0
            }
            
            X_input = pd.DataFrame([feature_dict])
            for col in self.feature_names:
                if col not in X_input.columns:
                    X_input[col] = 0
            
            X_input = X_input[self.feature_names]
            X_scaled = self.scaler.transform(X_input)
            
            probability = self.model.predict_proba(X_scaled)[0, 1]
            result = self.fuzzy.decide(probability)
            
            self.display_results(result, input_data)
            
        except Exception as e:
            messagebox.showerror("Error", f"Invalid input: {str(e)}")
    
    def display_results(self, result, input_data):
        self.result_text.delete(1.0, tk.END)
        text = ""
        text += f"Age: {input_data['age']:.0f} years\n"
        text += f"Sex: {'Male' if input_data['sex'] == 1 else 'Female'}\n"
        text += f"Blood Pressure: {input_data['resting_bp_s']:.0f} mm Hg\n"
        text += f"Cholesterol: {input_data['cholesterol']:.0f} mg/dl\n"
        text += f"Max Heart Rate: {input_data['max_heart_rate']:.0f}\n\n"
        
        text += f"Probability of disease: {result['probability']*100:.1f}%\n"
        text += f"Status: {result['status']}\n"
        text += f"Confidence: {result['confidence']*100:.1f}%\n"
        text += f"Uncertainty: {result['uncertainty']*100:.1f}%\n\n"
        
        text += f"Membership to Healthy: {result['mu_healthy']*100:.1f}%\n"
        text += f"Membership to Diseased: {result['mu_diseased']*100:.1f}%\n\n"
        
        if result['needs_review']:
            text += "This case has high uncertainty.\n"
            text += "Consult with a cardiologist for further evaluation.\n"
        else:
            if result['decision'] == 1:
                text += "Follow up with a cardiologist.\n"
            else:
                text += "Regular check-ups and healthy lifestyle.\n"
        
        self.result_text.insert(1.0, text)
    
    def clear(self):
        for entry in self.entries.values():
            entry.delete(0, tk.END)
        self.result_text.delete(1.0, tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = HeartApp(root)
    root.mainloop()