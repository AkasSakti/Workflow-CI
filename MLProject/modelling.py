import os
import json
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import plotly.graph_objects as go

# ✅ arahkan ke direktori tracking lokal relatif, aman untuk GitHub Actions
mlflow.set_tracking_uri("file:./mlruns")

# ✅ set eksperimen (biar MLflow yang buat folder & meta.yaml)
mlflow.set_experiment("MyExperiment")

mlflow.autolog()  # boleh setelah tracking URI
with mlflow.start_run():

# Load data
data_path = os.path.join(base_dir, "online_shoppers_intention_preprocessed.csv")
df = pd.read_csv(data_path)
X = df.drop(columns=["Revenue"])
y = df["Revenue"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Jalankan eksperimen
with mlflow.start_run() as run:
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    mlflow.sklearn.log_model(model, "model")

    y_pred = model.predict(X_test)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    cm_path = os.path.join(base_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    mlflow.log_artifact(cm_path)

    # Classification Report
    report = classification_report(y_test, y_pred, output_dict=True)
    report_path = os.path.join(base_dir, "metric_info.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
    mlflow.log_artifact(report_path)

    # Feature Importance
    fi = model.feature_importances_
    fig = go.Figure(go.Bar(x=fi, y=X.columns, orientation='h'))
    fig.update_layout(title="Feature Importance", yaxis={'autorange': 'reversed'})
    estimator_html_path = os.path.join(base_dir, "estimator.html")
    fig.write_html(estimator_html_path)
    mlflow.log_artifact(estimator_html_path)

    print("=" * 60)
    print("Run ID:", run.info.run_id)
    print("Model Path:", f"runs:/{run.info.run_id}/model")
    print("Tracking dir:", tracking_dir)
    print("=" * 60)
