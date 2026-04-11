class ValidationGuard:

    def validate(self, output):
        if not isinstance(output, dict):
            return {"error": "invalid output format"}
        return output
