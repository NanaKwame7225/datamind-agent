"""
DataMind Agent — Elite ATS Router

POST   /api/v1/ats/criteria/suggest      AI proposes criteria from a JD
POST   /api/v1/ats/jobs                  create a role {title, jd_text, criteria, anonymise}
GET    /api/v1/ats/jobs                  list my roles
GET    /api/v1/ats/jobs/{jid}            get one (with JD + criteria)
PATCH  /api/v1/ats/jobs/{jid}/criteria   HR overrides criteria/weights/anonymise
DELETE /api/v1/ats/jobs/{jid}            delete a role and its candidates
POST   /api/v1/ats/jobs/{jid}/cvs        upload one or many CVs → scored
POST   /api/v1/ats/jobs/{jid}/cv-link    score a CV from a public link
GET    /api/v1/ats/jobs/{jid}/candidates ranked candidates
DELETE /api/v1/ats/candidates/{cid}      remove a candidate
"""
import logging
from fastapi import APIRouter, Header, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional, List

router = APIRouter()
logger = logging.getLogger(__name__)


def _svc():
    from app.services.ats_service import ats_service
    return ats_service

async def _user(authorization):
    from app.routers.auth import current_user
    return await current_user(authorization)

async def _access(user_id, workspace_id, need_write=False):
    from app.routers.auth import _ws_access
    return await _ws_access(user_id, workspace_id, need_write)


class SuggestReq(BaseModel):
    jd_text: str

class JobReq(BaseModel):
    title: str
    jd_text: Optional[str] = ""
    criteria: list
    anonymise: Optional[bool] = True
    workspace_id: Optional[str] = None

class CriteriaReq(BaseModel):
    criteria: list
    anonymise: Optional[bool] = None
    workspace_id: Optional[str] = None

class LinkReq(BaseModel):
    url: str
    candidate_name: Optional[str] = None
    workspace_id: Optional[str] = None


@router.post("/criteria/suggest")
async def suggest_criteria(req: SuggestReq, authorization: Optional[str] = Header(None)):
    user = await _user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in to use the ATS.")
    return await _svc().suggest_criteria(req.jd_text)


@router.post("/jobs")
@router.post("/jobs/")
async def create_job(req: JobReq, authorization: Optional[str] = Header(None)):
    user = await _user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in to create a role.")
    ok, err = await _access(user["_id"], req.workspace_id, need_write=True)
    if not ok:
        return {"success": False, "error": err}
    return await _svc().create_job(user["_id"], req.title, req.jd_text, req.criteria,
                                   anonymise_cvs=req.anonymise, workspace_id=req.workspace_id)


@router.get("/jobs")
@router.get("/jobs/")
async def list_jobs(workspace_id: Optional[str] = None, authorization: Optional[str] = Header(None)):
    user = await _user(authorization)
    if not user:
        return {"success": True, "items": []}
    ok, err = await _access(user["_id"], workspace_id)
    if not ok:
        return {"success": False, "error": err, "items": []}
    return await _svc().list_jobs(user["_id"], workspace_id=workspace_id)


@router.get("/jobs/{jid}")
async def get_job(jid: str, workspace_id: Optional[str] = None, authorization: Optional[str] = Header(None)):
    user = await _user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in.")
    ok, err = await _access(user["_id"], workspace_id)
    if not ok:
        return {"success": False, "error": err}
    return await _svc().get_job(user["_id"], jid, workspace_id=workspace_id)


@router.patch("/jobs/{jid}/criteria")
async def update_criteria(jid: str, req: CriteriaReq, authorization: Optional[str] = Header(None)):
    """HR has the final say — these criteria and weights replace the AI's."""
    user = await _user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in.")
    ok, err = await _access(user["_id"], req.workspace_id, need_write=True)
    if not ok:
        return {"success": False, "error": err}
    return await _svc().update_criteria(user["_id"], jid, req.criteria,
                                        anonymise_cvs=req.anonymise,
                                        workspace_id=req.workspace_id)


@router.delete("/jobs/{jid}")
async def delete_job(jid: str, workspace_id: Optional[str] = None, authorization: Optional[str] = Header(None)):
    user = await _user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in.")
    ok, err = await _access(user["_id"], workspace_id, need_write=True)
    if not ok:
        return {"success": False, "error": err}
    return await _svc().delete_job(user["_id"], jid, workspace_id=workspace_id)


@router.post("/jobs/{jid}/cvs")
async def upload_cvs(jid: str, files: List[UploadFile] = File(...),
                     workspace_id: Optional[str] = None,
                     authorization: Optional[str] = Header(None)):
    """Upload one or many CVs. Each is extracted, anonymised (if on), and scored."""
    user = await _user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in.")
    ok, err = await _access(user["_id"], workspace_id, need_write=True)
    if not ok:
        return {"success": False, "error": err}

    from app.services.ats_service import extract_text
    results, errors = [], []
    for f in files[:50]:
        try:
            content = await f.read()
            text, ext_err = extract_text(content, f.filename)
            if ext_err or not text.strip():
                errors.append({"filename": f.filename,
                               "error": ext_err or "No readable text found in this file."})
                continue
            r = await _svc().score_cv(user["_id"], jid, text, f.filename,
                                      workspace_id=workspace_id)
            if r.get("success"):
                results.append(r["candidate"])
            else:
                errors.append({"filename": f.filename, "error": r.get("error")})
        except Exception as e:
            logger.error(f"CV upload failed for {f.filename}: {e}", exc_info=True)
            errors.append({"filename": f.filename, "error": str(e)[:160]})
    return {"success": True, "scored": len(results), "candidates": results,
            "errors": errors, "failed": len(errors)}


@router.post("/jobs/{jid}/cv-link")
async def cv_from_link(jid: str, req: LinkReq, authorization: Optional[str] = Header(None)):
    """Pull a CV from a public link (Drive/Dropbox/direct) and score it."""
    user = await _user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in.")
    ok, err = await _access(user["_id"], req.workspace_id, need_write=True)
    if not ok:
        return {"success": False, "error": err}
    try:
        # Reuse the existing link normaliser so Drive/Dropbox share links work.
        # It returns {url, source, ...} — we want the direct download URL.
        import requests
        direct = req.url
        try:
            from app.services.url_ingest_service import url_ingest_service
            norm = url_ingest_service.normalise(req.url)
            if isinstance(norm, dict) and norm.get("url"):
                direct = norm["url"]
            elif isinstance(norm, str):
                direct = norm
        except Exception as e:
            logger.warning(f"Link normalise skipped ({e}); using the raw URL.")
        r = requests.get(direct, timeout=30, headers={"User-Agent": "DataMind-ATS"})
        if r.status_code >= 400:
            return {"success": False, "error": f"Could not fetch the link ({r.status_code}). Is it shared publicly?"}
        fname = req.url.split("?")[0].split("/")[-1] or "cv.pdf"
        from app.services.ats_service import extract_text
        text, ext_err = extract_text(r.content, fname)
        if ext_err or not text.strip():
            return {"success": False, "error": ext_err or "No readable text at that link."}
        return await _svc().score_cv(user["_id"], jid, text, fname,
                                     candidate_name=req.candidate_name,
                                     workspace_id=req.workspace_id)
    except Exception as e:
        logger.error(f"CV link failed: {e}", exc_info=True)
        return {"success": False, "error": f"Could not read that link: {str(e)[:160]}"}


@router.get("/jobs/{jid}/candidates")
async def list_candidates(jid: str, workspace_id: Optional[str] = None,
                          authorization: Optional[str] = Header(None)):
    user = await _user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in.")
    ok, err = await _access(user["_id"], workspace_id)
    if not ok:
        return {"success": False, "error": err, "items": []}
    return await _svc().list_candidates(user["_id"], jid, workspace_id=workspace_id)


@router.delete("/candidates/{cid}")
async def delete_candidate(cid: str, workspace_id: Optional[str] = None,
                           authorization: Optional[str] = Header(None)):
    user = await _user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in.")
    return await _svc().delete_candidate(user["_id"], cid, workspace_id=workspace_id)
