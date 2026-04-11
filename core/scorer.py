class IntelligenceScorer:

    def score(self, profile, analysis, issues):

        score = 100

        if profile.get("duplicates", 0) > 0:
            score -= 10

        if len(issues) > 0:
            score -= 15

        if "error" in analysis:
            score -= 20

        return max(score, 0)
