import mlflow
import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Flatten images
x_train = x_train.reshape(len(x_train), -1) / 255.0
x_test = x_test.reshape(len(x_test), -1) / 255.0

# Train model inside MLflow
with mlflow.start_run() as run:

    model = LogisticRegression(max_iter=200)
    model.fit(x_train[:80000], y_train[:80000])  # small subset (fast)

    preds = model.predict(x_test[:2000])
    accuracy = accuracy_score(y_test[:2000], preds)

    print("Accuracy:", accuracy)

    mlflow.log_metric("accuracy", accuracy)

    run_id = run.info.run_id

    # Save run_id
    with open("model_info.txt", "w") as f:
        f.write(run_id)
