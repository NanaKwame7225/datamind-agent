import requests

# =========================
# QUICKBOOKS CONNECTOR
# =========================

def get_quickbooks_data(token):
    url = "https://quickbooks.api.intuit.com/v3/company/YOUR_COMPANY/query"

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json"
    }

    response = requests.get(url, headers=headers)
    return response.json()


# =========================
# SAP CONNECTOR (OData API)
# =========================

def get_sap_data(base_url, token):
    url = f"{base_url}/sap/opu/odata/sap/API_FINANCE_DOCUMENT_SRV"

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json"
    }

    response = requests.get(url, headers=headers)
    return response.json()
