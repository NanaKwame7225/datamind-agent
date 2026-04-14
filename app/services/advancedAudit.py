// ===============================
// ADVANCED AUDIT ENGINE
// ===============================

// -------------------------------
// 1. BENFORD'S LAW (FRAUD DETECTION)
// -------------------------------

function getFirstDigit(number) {
  return parseInt(number.toString()[0]);
}

function benfordAnalysis(data) {
  const counts = Array(10).fill(0);
  let total = 0;

  data.forEach(item => {
    const num = Math.abs(item.amount || 0);
    if (num > 0) {
      const firstDigit = getFirstDigit(num);
      if (firstDigit >= 1 && firstDigit <= 9) {
        counts[firstDigit]++;
        total++;
      }
    }
  });

  const distribution = counts.map((count, digit) => ({
    digit,
    percentage: total ? (count / total) * 100 : 0
  }));

  return {
    method: "Benford's Law Analysis",
    distribution,
    warning:
      distribution[1].percentage < 25
        ? "Possible anomaly detected (low natural distribution)"
        : "No major anomaly detected"
  };
}

// -------------------------------
// 2. RATIO ANALYSIS (FINANCIAL HEALTH)
// -------------------------------

function ratioAnalysis(financials) {
  const {
    revenue,
    expenses,
    assets,
    liabilities,
    equity,
    currentAssets,
    currentLiabilities
  } = financials;

  return {
    profitability: {
      netMargin: ((revenue - expenses) / revenue) * 100
    },
    liquidity: {
      currentRatio: currentAssets / currentLiabilities
    },
    solvency: {
      debtToEquity: liabilities / equity
    },
    efficiency: {
      assetTurnover: revenue / assets
    }
  };
}

// -------------------------------
// 3. TREND ANALYSIS (TIME SERIES)
// -------------------------------

function trendAnalysis(timeSeries) {
  // timeSeries = [{ period: "2023", value: 1000 }, ...]

  const growthRates = [];

  for (let i = 1; i < timeSeries.length; i++) {
    const prev = timeSeries[i - 1].value;
    const curr = timeSeries[i].value;

    const growth = ((curr - prev) / prev) * 100;
    growthRates.push({
      from: timeSeries[i - 1].period,
      to: timeSeries[i].period,
      growth
    });
  }

  return {
    averageGrowth:
      growthRates.reduce((a, b) => a + b.growth, 0) / growthRates.length,
    trend: growthRates,
    insight:
      growthRates.some(g => g.growth < -20)
        ? "Negative performance detected in trend"
        : "Stable or positive growth trend"
  };
}

module.exports = {
  benfordAnalysis,
  ratioAnalysis,
  trendAnalysis
};
