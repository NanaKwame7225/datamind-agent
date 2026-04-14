const prompt = `
You are an expert financial auditor.

Analyze the following transactions:
${JSON.stringify(data)}

Tasks:
- Detect anomalies
- Assign risk scores
- Suggest audit procedures
- Identify possible fraud

Return structured JSON.
`;
