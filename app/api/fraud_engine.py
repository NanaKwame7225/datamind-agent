import numpy as np

def anomaly_score(values):
    mean = np.mean(values)
    std = np.std(values)

    scores = []
    for v in values:
        z = abs((v - mean) / (std if std else 1))
        scores.append(z)

    return {
        "anomalies": [i for i,s in enumerate(scores) if s > 2],
        "risk_score": float(np.mean(scores) * 10),
        "status": "HIGH RISK" if np.mean(scores) > 2 else "NORMAL"
    }
