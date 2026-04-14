// ===============================
// DOCUMENT AI ENGINE
// ===============================

const AWS = require("aws-sdk");
const textract = new AWS.Textract();

// -------------------------------
// 1. OCR EXTRACTION
// -------------------------------

async function extractText(fileBuffer) {
  const params = {
    Document: {
      Bytes: fileBuffer
    }
  };

  const result = await textract.detectDocumentText(params).promise();

  return result.Blocks
    .filter(block => block.BlockType === "LINE")
    .map(block => block.Text)
    .join(" ");
}

// -------------------------------
// 2. INVOICE VALIDATION ENGINE
// -------------------------------

function validateInvoice(invoiceData) {
  const requiredFields = [
    "invoiceNumber",
    "date",
    "vendor",
    "totalAmount"
  ];

  const missingFields = requiredFields.filter(
    field => !invoiceData[field]
  );

  const isValid = missingFields.length === 0;

  return {
    valid: isValid,
    missingFields,
    riskLevel: isValid ? "LOW" : "HIGH",
    message: isValid
      ? "Invoice structure valid"
      : "Missing critical invoice fields"
  };
}

// -------------------------------
// 3. RECEIPT MATCHING (RECONCILIATION ENGINE)
// -------------------------------

function matchReceiptToTransaction(receipt, transactions) {
  const matches = transactions.filter(tx => {
    const amountMatch =
      Math.abs(tx.amount - receipt.amount) < 1;

    const dateMatch =
      new Date(tx.date).toDateString() ===
      new Date(receipt.date).toDateString();

    return amountMatch && dateMatch;
  });

  return {
    receipt,
    matchedTransactions: matches,
    status: matches.length > 0 ? "MATCHED" : "UNMATCHED",
    confidence: matches.length > 0 ? 0.95 : 0.2
  };
}

// -------------------------------
// EXPORTS
// -------------------------------

module.exports = {
  extractText,
  validateInvoice,
  matchReceiptToTransaction
};
