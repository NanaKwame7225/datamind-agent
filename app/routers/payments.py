"""
DataMind Agent v2 — Payments Router (Stripe)
POST /api/v2/payments/checkout
POST /api/v2/payments/portal
POST /api/v2/payments/webhook
GET  /api/v2/payments/plans
"""
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from app.services.auth_service import get_current_user, get_db
from app.models.user import UserOut, PlanTier, PLAN_LIMITS
from config.settings import settings
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

PLAN_PRICES = {
    PlanTier.starter:    {"price_id": settings.STRIPE_STARTER_PRICE_ID,    "amount": 29, "label": "Starter"},
    PlanTier.pro:        {"price_id": settings.STRIPE_PRO_PRICE_ID,        "amount": 79, "label": "Pro"},
    PlanTier.enterprise: {"price_id": settings.STRIPE_ENTERPRISE_PRICE_ID, "amount": 199,"label": "Enterprise"},
}


@router.get("/plans")
async def list_plans():
    """Return available subscription plans."""
    return {
        "plans": [
            {
                "id": "free",
                "name": "Free",
                "price_usd": 0,
                "analyses_per_month": PLAN_LIMITS[PlanTier.free]["analyses"],
                "features": ["10 analyses/month","All industries","Basic charts","Local fallback AI"],
            },
            {
                "id": "starter",
                "name": "Starter",
                "price_usd": 29,
                "analyses_per_month": PLAN_LIMITS[PlanTier.starter]["analyses"],
                "features": ["100 analyses/month","All industries","Full AI (Claude + GPT-4o)","Analysis history","Email support"],
            },
            {
                "id": "pro",
                "name": "Pro",
                "price_usd": 79,
                "analyses_per_month": PLAN_LIMITS[PlanTier.pro]["analyses"],
                "features": ["500 analyses/month","All modules including Finance","Fraud + Tax + Accounting","Scheduled reports","Database connections","Priority support"],
            },
            {
                "id": "enterprise",
                "name": "Enterprise",
                "price_usd": 199,
                "analyses_per_month": "Unlimited",
                "features": ["Unlimited analyses","White-label option","Custom integrations","Dedicated support","SLA guarantee","Custom AI training"],
            },
        ]
    }


@router.post("/checkout")
async def create_checkout(
    plan: str,
    current_user: UserOut = Depends(get_current_user),
):
    """Create a Stripe checkout session."""
    if not settings.STRIPE_SECRET_KEY:
        raise HTTPException(503, "Payment processing not configured. Contact NkaySolutions.")
    try:
        import stripe
        stripe.api_key = settings.STRIPE_SECRET_KEY
        plan_enum = PlanTier(plan)
        plan_data = PLAN_PRICES.get(plan_enum)
        if not plan_data or not plan_data["price_id"]:
            raise HTTPException(400, f"Plan '{plan}' not available")
        session = stripe.checkout.Session.create(
            customer_email=current_user.email,
            payment_method_types=["card"],
            line_items=[{"price": plan_data["price_id"], "quantity": 1}],
            mode="subscription",
            success_url="https://nanakwame7225.github.io/datamind-agent-ui/?payment=success",
            cancel_url="https://nanakwame7225.github.io/datamind-agent-ui/?payment=cancelled",
            metadata={"user_id": current_user.id, "plan": plan},
        )
        return {"checkout_url": session.url, "session_id": session.id}
    except Exception as e:
        logger.error(f"Stripe checkout error: {e}")
        raise HTTPException(500, f"Payment error: {e}")


@router.post("/portal")
async def customer_portal(current_user: UserOut = Depends(get_current_user)):
    """Open Stripe customer portal to manage subscription."""
    if not settings.STRIPE_SECRET_KEY:
        raise HTTPException(503, "Payment processing not configured")
    try:
        import stripe
        stripe.api_key = settings.STRIPE_SECRET_KEY
        conn = get_db()
        row = conn.execute("SELECT stripe_customer_id FROM users WHERE id=?", (current_user.id,)).fetchone()
        conn.close()
        if not row or not row[0]:
            raise HTTPException(400, "No active subscription found")
        session = stripe.billing_portal.Session.create(
            customer=row[0],
            return_url="https://nanakwame7225.github.io/datamind-agent-ui/",
        )
        return {"portal_url": session.url}
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/webhook")
async def stripe_webhook(request: Request):
    """Handle Stripe webhook events — updates plan on payment."""
    if not settings.STRIPE_SECRET_KEY:
        return {"status": "ignored"}
    payload = await request.body()
    sig = request.headers.get("stripe-signature", "")
    try:
        import stripe
        stripe.api_key = settings.STRIPE_SECRET_KEY
        event = stripe.Webhook.construct_event(payload, sig, settings.STRIPE_WEBHOOK_SECRET)
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        raise HTTPException(400, str(e))

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        user_id = session.get("metadata", {}).get("user_id")
        plan    = session.get("metadata", {}).get("plan")
        customer_id = session.get("customer")
        if user_id and plan:
            conn = get_db()
            try:
                conn.execute(
                    "ALTER TABLE users ADD COLUMN stripe_customer_id TEXT"
                )
            except Exception:
                pass
            conn.execute(
                "UPDATE users SET plan=?, stripe_customer_id=?, analyses_used=0 WHERE id=?",
                (plan, customer_id, user_id)
            )
            conn.commit()
            conn.close()
            logger.info(f"Plan upgraded: user={user_id} plan={plan}")

    elif event["type"] == "customer.subscription.deleted":
        customer_id = event["data"]["object"].get("customer")
        if customer_id:
            conn = get_db()
            conn.execute(
                "UPDATE users SET plan='free' WHERE stripe_customer_id=?", (customer_id,)
            )
            conn.commit()
            conn.close()
            logger.info(f"Subscription cancelled: customer={customer_id}")

    return {"status": "ok"}
