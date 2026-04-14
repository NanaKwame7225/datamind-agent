import numpy as np

# =========================
# BENFORD'S LAW (FRAUD DETECTION)
# =========================
def benford_analysis(data):
    numbers = [int(str(abs(x))[0]) for x in data if x != 0]
    counts = {i: numbers.count(i) for i in range(1,10)}

    total = len(numbers)
    return {
        "distribution": counts,
        "fraud_score": round(np.std(list(counts.values())), 2),
        "risk_level": "HIGH" if np.std(list(counts.values())) > 20 else "LOW"
    }

# =========================
# RATIO ANALYSIS (FINANCE HEALTH)
# =========================
def ratio_analysis(financials):
    return {
        "current_ratio": financials.get("assets",0) / max(financials.get("liabilities",1),1),
        "profit_margin": financials.get("profit",0) / max(financials.get("revenue",1),1),
        "leverage": financials.get("debt",0) / max(financials.get("equity",1),1),
        "status": "STABLE" if financials.get("profit",0) > 0 else "RISK"
    }

# =========================
# TREND ANALYSIS
# =========================
def trend_analysis(series):
    trend = np.polyfit(range(len(series)), series, 1)[0]

    return {
        "trend_slope": trend,
        "direction": "UP" if trend > 0 else "DOWN",
        "volatility": float(np.std(series))
    }
