import numpy as np

# =========================
# BENFORD'S LAW
# =========================
def benford_analysis(data):
    counts = [0] * 10
    total = 0

    for item in data:
        num = abs(item.get("amount", 0))
        if num > 0:
            first = int(str(num)[0])
            counts[first] += 1
            total += 1

    return {
        "benford_distribution": counts,
        "total": total
    }


# =========================
# RATIO ANALYSIS
# =========================
def ratio_analysis(financials):
    revenue = financials.get("revenue", 1)
    expenses = financials.get("expenses", 0)
    assets = financials.get("assets", 1)
    liabilities = financials.get("liabilities", 1)

    return {
        "profit_margin": (revenue - expenses) / revenue,
        "debt_ratio": liabilities / assets
    }


# =========================
# TREND ANALYSIS
# =========================
def trend_analysis(series):
    growth = []

    for i in range(1, len(series)):
        prev = series[i-1]["value"]
        curr = series[i]["value"]
        growth.append((curr - prev) / prev)

    return {
        "avg_growth": sum(growth) / len(growth) if growth else 0,
        "trend": growth
    }
