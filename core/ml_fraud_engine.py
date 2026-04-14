from sklearn.ensemble import IsolationForest
import numpy as np

class FraudDetector:
    def __init__(self):
        self.model = IsolationForest(contamination=0.05)

    def train(self, transactions):
        X = np.array([[t["amount"]] for t in transactions])
        self.model.fit(X)
        return {"status": "model trained"}

    def predict(self, transactions):
        X = np.array([[t["amount"]] for t in transactions])
        preds = self.model.predict(X)

        results = []
        for i, t in enumerate(transactions):
            results.append({
                "transaction": t,
                "fraud": preds[i] == -1
            })

        return results


fraud_detector = FraudDetector()
