# Simulated OCR + validation system

def extract_text(file):
    return "Extracted OCR text from document"


def validate_invoice(invoice):
    required = ["invoiceNumber", "amount", "date"]

    missing = [f for f in required if f not in invoice]

    return {
        "valid": len(missing) == 0,
        "missing": missing
    }


def match_receipt_to_transaction(receipt, transactions):
    matches = [
        t for t in transactions
        if t.get("amount") == receipt.get("amount")
    ]

    return {
        "matched": len(matches) > 0,
        "matches": matches
    }
