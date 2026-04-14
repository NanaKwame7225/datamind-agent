from sklearn.ensemble import IsolationForest
import numpy as np

# =========================
# TRAIN / DETECT FRAUD
# =========================

class FraudDetector:
    def __init__(self):
        self.model = IsolationForest(
            n_estimators=100,
            contamination=0.05,
            random_state=42
        )

    def train(self, transactions):
        # Expect: [{"amount": 100, "date": "..."}]
        X = np.array([[t["amount"]] for t in transactions])
        self.model.fit(X)
        return "Model trained successfully"

    def predict(self, transactions):
        X = np.array([[t["amount"]] for t in transactions])
        scores = self.model.predict(X)

        results = []
        for i, t in enumerate(transactions):
            results.append({
                "transaction": t,
                "fraud_flag": bool(scores[i] == -1)
            })

        return {
            "total": len(transactions),
            "fraud_detected": sum(1 for r in results if r["fraud_flag"]),
            "results": results
        }


fraud_detector = FraudDetector()
