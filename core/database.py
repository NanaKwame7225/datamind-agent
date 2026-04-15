import os
from pymongo import MongoClient

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
client    = MongoClient(MONGO_URI)
db        = client["datamind"]

# Collections
users_col   = db["users"]
audits_col  = db["audits"]
reports_col = db["reports"]
