"""
DataMind Agent — Workspaces

Two kinds of space:
  • Personal — every user has one implicitly. Its id is "personal:<user_id>",
    so all the analyses/schedules they already have (scoped by their user id)
    keep working with no migration.
  • Shared   — created by a user (the Owner), who invites others as Editor or
    Viewer. Data in a shared workspace is visible to all its members per role.

Roles:
  owner  — full control: rename/delete workspace, manage members, delete anything
  editor — create and edit analyses/schedules; cannot manage members
  viewer — read-only

A "workspace_id" scopes data everywhere. For a personal space it's
"personal:<user_id>"; for a shared space it's the workspace document id.
"""
from __future__ import annotations
import uuid, logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

ROLES = ("owner", "editor", "viewer")


def now():
    return datetime.now(timezone.utc)

def personal_id(user_id: str) -> str:
    return f"personal:{user_id}"

def is_personal(workspace_id: str) -> bool:
    return (workspace_id or "").startswith("personal:")


class WorkspaceService:

    async def _ws(self):
        from app.database import connect
        db = await connect()
        return db.workspaces if db is not None else None

    async def _members(self):
        from app.database import connect
        db = await connect()
        return db.workspace_members if db is not None else None

    # ── Create / list ─────────────────────────────────────────────────────────
    async def create(self, user_id: str, name: str) -> dict:
        ws = await self._ws()
        mem = await self._members()
        if ws is None:
            return {"success": False, "error": "Workspaces are unavailable — database not configured."}
        name = (name or "").strip()[:80]
        if not name:
            return {"success": False, "error": "Give the workspace a name."}
        wid = str(uuid.uuid4())
        doc = {"_id": wid, "name": name, "owner_id": user_id, "created_at": now()}
        await ws.insert_one(doc)
        await mem.insert_one({
            "_id": str(uuid.uuid4()), "workspace_id": wid, "user_id": user_id,
            "role": "owner", "joined_at": now(), "email": None,
        })
        return {"success": True, "id": wid, "name": name, "role": "owner"}

    async def list_for_user(self, user_id: str, user_email: str = None) -> dict:
        """Every workspace the user can see: their personal one + shared ones."""
        spaces = [{
            "id": personal_id(user_id), "name": "Personal",
            "role": "owner", "personal": True, "member_count": 1,
        }]
        mem = await self._members()
        ws = await self._ws()
        if mem is not None:
            async for m in mem.find({"user_id": user_id}):
                w = await ws.find_one({"_id": m["workspace_id"]})
                if not w:
                    continue
                cnt = await mem.count_documents({"workspace_id": m["workspace_id"]})
                spaces.append({
                    "id": w["_id"], "name": w["name"], "role": m["role"],
                    "personal": False, "member_count": cnt,
                    "owner": w.get("owner_id") == user_id,
                })
        return {"success": True, "workspaces": spaces}

    # ── Access checks ─────────────────────────────────────────────────────────
    async def role_of(self, user_id: str, workspace_id: str) -> str | None:
        """The user's role in a workspace, or None if they have no access."""
        if is_personal(workspace_id):
            # Only the owner of a personal space can use it
            return "owner" if workspace_id == personal_id(user_id) else None
        mem = await self._members()
        if mem is None:
            return None
        m = await mem.find_one({"workspace_id": workspace_id, "user_id": user_id})
        return m["role"] if m else None

    async def can_write(self, user_id: str, workspace_id: str) -> bool:
        r = await self.role_of(user_id, workspace_id)
        return r in ("owner", "editor")

    async def can_read(self, user_id: str, workspace_id: str) -> bool:
        return (await self.role_of(user_id, workspace_id)) is not None

    async def can_manage(self, user_id: str, workspace_id: str) -> bool:
        return (await self.role_of(user_id, workspace_id)) == "owner"

    # ── Members / invites ─────────────────────────────────────────────────────
    async def invite(self, owner_id: str, workspace_id: str, email: str, role: str) -> dict:
        if is_personal(workspace_id):
            return {"success": False, "error": "You can't add people to a personal workspace."}
        if role not in ("editor", "viewer"):
            return {"success": False, "error": "Role must be editor or viewer."}
        if not await self.can_manage(owner_id, workspace_id):
            return {"success": False, "error": "Only the owner can invite people."}

        email = (email or "").strip().lower()
        from app.database import connect
        db = await connect()
        if db is None:
            return {"success": False, "error": "Database not configured."}
        user = await db.users.find_one({"email": email})
        if not user:
            return {"success": False, "error": "No DataMind account with that email. Ask them to sign up first, then invite them."}
        if user.get("_id") == owner_id:
            return {"success": False, "error": "You're already the owner."}

        mem = await self._members()
        existing = await mem.find_one({"workspace_id": workspace_id, "user_id": user["_id"]})
        if existing:
            await mem.update_one({"_id": existing["_id"]}, {"$set": {"role": role}})
            return {"success": True, "updated": True, "email": email, "role": role}
        await mem.insert_one({
            "_id": str(uuid.uuid4()), "workspace_id": workspace_id,
            "user_id": user["_id"], "role": role, "email": email, "joined_at": now(),
        })
        return {"success": True, "email": email, "role": role}

    async def members(self, user_id: str, workspace_id: str) -> dict:
        if not await self.can_read(user_id, workspace_id):
            return {"success": False, "error": "No access to that workspace."}
        if is_personal(workspace_id):
            return {"success": True, "members": [{"role": "owner", "you": True, "email": None}]}
        mem = await self._members()
        from app.database import connect
        db = await connect()
        out = []
        async for m in mem.find({"workspace_id": workspace_id}):
            u = await db.users.find_one({"_id": m["user_id"]})
            out.append({
                "user_id": m["user_id"],
                "email": (u or {}).get("email") or m.get("email"),
                "name": (u or {}).get("name"),
                "role": m["role"],
                "you": m["user_id"] == user_id,
            })
        return {"success": True, "members": out}

    async def remove_member(self, owner_id: str, workspace_id: str, target_user_id: str) -> dict:
        if not await self.can_manage(owner_id, workspace_id):
            return {"success": False, "error": "Only the owner can remove people."}
        if target_user_id == owner_id:
            return {"success": False, "error": "The owner can't remove themselves. Delete the workspace instead."}
        mem = await self._members()
        res = await mem.delete_one({"workspace_id": workspace_id, "user_id": target_user_id})
        if res.deleted_count == 0:
            return {"success": False, "error": "That person isn't a member."}
        return {"success": True, "removed": target_user_id}

    async def set_role(self, owner_id: str, workspace_id: str, target_user_id: str, role: str) -> dict:
        if not await self.can_manage(owner_id, workspace_id):
            return {"success": False, "error": "Only the owner can change roles."}
        if role not in ("editor", "viewer"):
            return {"success": False, "error": "Role must be editor or viewer."}
        if target_user_id == owner_id:
            return {"success": False, "error": "The owner's role can't be changed."}
        mem = await self._members()
        res = await mem.update_one(
            {"workspace_id": workspace_id, "user_id": target_user_id},
            {"$set": {"role": role}})
        if res.matched_count == 0:
            return {"success": False, "error": "That person isn't a member."}
        return {"success": True, "role": role}

    async def rename(self, owner_id: str, workspace_id: str, name: str) -> dict:
        if not await self.can_manage(owner_id, workspace_id):
            return {"success": False, "error": "Only the owner can rename the workspace."}
        name = (name or "").strip()[:80]
        if not name:
            return {"success": False, "error": "Name can't be empty."}
        ws = await self._ws()
        await ws.update_one({"_id": workspace_id}, {"$set": {"name": name}})
        return {"success": True, "name": name}

    async def delete(self, owner_id: str, workspace_id: str) -> dict:
        if is_personal(workspace_id):
            return {"success": False, "error": "You can't delete your personal workspace."}
        if not await self.can_manage(owner_id, workspace_id):
            return {"success": False, "error": "Only the owner can delete the workspace."}
        from app.database import connect
        db = await connect()
        await db.workspaces.delete_one({"_id": workspace_id})
        await db.workspace_members.delete_many({"workspace_id": workspace_id})
        # Analyses/schedules in this workspace are left orphaned but inaccessible;
        # a cleanup job could remove them. For now, delete analyses too.
        await db.analyses.delete_many({"workspace_id": workspace_id})
        await db.schedules.delete_many({"workspace_id": workspace_id})
        return {"success": True, "deleted": workspace_id}


workspace_service = WorkspaceService()
