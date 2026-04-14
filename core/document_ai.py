from core.database import documents

async def store_document(tenant_id, text):
    doc = {
        "tenant_id": tenant_id,
        "text": text
    }

    await documents.insert_one(doc)
    return doc
