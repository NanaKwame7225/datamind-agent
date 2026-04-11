from app.core.profiler import DataProfiler
from app.core.cleaner import DataCleaner
from app.core.analytics import AnalyticsEngine
from app.core.validator import BusinessValidator
from app.core.audit import SelfAudit
from app.core.scorer import IntelligenceScorer
from app.core.guard import ValidationGuard


class DataIntelligenceAgent:

    def __init__(self):
        self.profiler = DataProfiler()
        self.cleaner = DataCleaner()
        self.analytics = AnalyticsEngine()
        self.validator = BusinessValidator()
        self.audit = SelfAudit()
        self.scorer = IntelligenceScorer()
        self.guard = ValidationGuard()

    def run(self, df):

        profile = self.profiler.profile(df)

        clean_df, clean_report = self.cleaner.clean(df)

        analysis = self.analytics.analyze(clean_df)

        issues = self.validator.validate(clean_df)

        score = self.scorer.score(profile, analysis, issues)

        audit = self.audit.run(analysis)

        output = {
            "profile": profile,
            "cleaning": clean_report,
            "analysis": analysis,
            "issues": issues,
            "audit": audit,
            "intelligence_score": score
        }

        return self.guard.validate(output)
