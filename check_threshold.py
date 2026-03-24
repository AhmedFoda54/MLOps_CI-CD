import mlflow
import sys

with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

client = mlflow.tracking.MlflowClient()
run = client.get_run(run_id)

print("Run ID:", run_id)
print("All metrics:", run.data.metrics)

accuracy = run.data.metrics["accuracy"]

print("Accuracy:", accuracy)

if accuracy < 0.85:
    print("❌ Accuracy below threshold")
    sys.exit(1)
else:
    print("✅ Accuracy OK")