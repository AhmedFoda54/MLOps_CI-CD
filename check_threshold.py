import sys

with open("model_info.txt", "r") as f:
    content = f.read().strip()

run_id, accuracy = content.split(",")

accuracy = float(accuracy)

print("Run ID:", run_id)
print("Accuracy:", accuracy)

if accuracy < 0.85:
    print("❌ Accuracy below threshold")
    sys.exit(1)
else:
    print("✅ Accuracy OK")