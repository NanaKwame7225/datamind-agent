function auditWorkflow(data) {
  const risk = assessRisk(data);
  const sample = generateSample(data);
  const test = runTests(sample);
  const findings = generateFindings(test);
  return findings;
}
