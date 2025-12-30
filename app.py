#importing required libraries
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn import metrics 
import warnings
import pickle
import os
warnings.filterwarnings('ignore')
from feature import FeatureExtraction
from sklearn.ensemble import GradientBoostingClassifier

# Create a sample dataset if none exists
def create_sample_dataset():
    urls = [
        "https://google.com",
        "http://phishing-site.com",
        "https://facebook.com",
        "http://malicious-site.net",
        "https://amazon.com"
    ]
    labels = ["good", "bad", "good", "bad", "good"]
    df = pd.DataFrame({"url": urls, "label": labels})
    df.to_csv("E:/Phishing/Phishing/phishing.csv", index=False)
    return df

def train_model():
    print("Training new model...")
    try:
        # Load or create dataset
        try:
            data = pd.read_csv("E:/Phishing/Phishing/phishing.csv")
            if len(data) == 0:
                raise FileNotFoundError
        except FileNotFoundError:
            print("Creating sample dataset...")
            data = create_sample_dataset()
            
        print(f"Dataset loaded with {len(data)} rows")
        print("Sample of columns:", data.columns.tolist())
        print("First few rows:", data.head())
        
        # Create feature extractor
        features = []
        labels = []
        
        # Process all URLs in the dataset
        for i, row in data.iterrows():
            try:
                url = row['url']
                print(f"Processing URL {i+1}/{len(data)}: {url}")
                obj = FeatureExtraction(url)
                feature_list = obj.getFeaturesList()
                
                if len(feature_list) == 30:
                    features.append(feature_list)
                    label_value = 1 if str(row['label']).lower() == 'good' else -1
                    labels.append(label_value)
                    print(f"Added feature with label: {label_value}")
                else:
                    print(f"Wrong feature length for URL {url}: {len(feature_list)}")
            except Exception as e:
                print(f"Error processing URL {url}: {str(e)}")
                continue
        
        if len(features) < 2:
            raise Exception(f"Not enough features extracted. Only got {len(features)} valid features")
        
        print(f"Successfully processed {len(features)} URLs")
        
        # Convert to numpy arrays
        features = np.array(features)
        labels = np.array(labels)
        
        print(f"Feature shape: {features.shape}, Labels shape: {labels.shape}")
        print(f"Label distribution: {np.bincount(labels + 1)}")
        
        # Create and train the model
        gbc = GradientBoostingClassifier(max_depth=3, n_estimators=100)
        print("Training model...")
        gbc.fit(features, labels)
        print("Model training completed")
        
        # Verify the model is fitted
        test_pred = gbc.predict(features[0:1])
        print(f"Model verification successful. Test prediction: {test_pred}")
        
        # Save the new model
        os.makedirs("E:/Phishing/Phishing/pickle", exist_ok=True)
        with open("E:/Phishing/Phishing/pickle/model.pkl", "wb") as f:
            pickle.dump(gbc, f)
        print("Model saved successfully")
        
        return gbc
    except Exception as e:
        print(f"Error in train_model: {str(e)}")
        raise

# Global variable for the model
gbc = None

def load_or_train_model():
    global gbc
    try:
        if os.path.exists("E:/Phishing/Phishing/pickle/model.pkl"):
            print("Loading existing model...")
            with open("E:/Phishing/Phishing/pickle/model.pkl", "rb") as file:
                gbc = pickle.load(file)
            # Verify the loaded model
            if not hasattr(gbc, 'predict'):
                raise Exception("Loaded model appears invalid")
            print("Model loaded and verified successfully")
        else:
            print("No existing model found. Training new model...")
            gbc = train_model()
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("Training new model...")
        gbc = train_model()

print("Initializing model...")
load_or_train_model()
print("Model initialization completed")

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    global gbc
    if request.method == "POST":
        try:
            if gbc is None:
                raise Exception("Model not properly initialized")
                
            url = request.form["url"]
            print(f"Processing URL: {url}")
            
            obj = FeatureExtraction(url)
            features = obj.getFeaturesList()
            print(f"Extracted features length: {len(features)}")
            
            x = np.array(features).reshape(1,30) 
            print(f"Input shape: {x.shape}")

            y_pred = gbc.predict(x)[0]
            print(f"Prediction: {y_pred}")
            
            y_pro_phishing = gbc.predict_proba(x)[0,0]
            y_pro_non_phishing = gbc.predict_proba(x)[0,1]
            
            # Fix the probability calculation and display
            safety_percentage = y_pro_non_phishing * 100 if y_pred == 1 else (1 - y_pro_non_phishing) * 100
            pred = "Website is {0:.2f}% {1}".format(
                safety_percentage,
                "safe to use" if y_pred == 1 else "unsafe to use"
            )
            print(f"Prediction complete: {pred}")
            
            return render_template(
                'index.html', 
                xx=round(safety_percentage/100, 2),
                url=url,
                prediction_text=pred
            )
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return render_template('index.html', xx=-1, error=str(e))
    return render_template("index.html", xx=-1)

if __name__ == "__main__":
    app.run(debug=True)