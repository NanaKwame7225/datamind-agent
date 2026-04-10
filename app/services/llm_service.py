from __future__ import annotations
import logging
from typing import Optional
from config.settings import settings
from app.models.schemas import LLMProvider

logger = logging.getLogger(__name__)

INDUSTRY_CONTEXTS: dict[str, str] = {
    "finance": """You are a senior financial data analyst.
Key focus areas: revenue, profitability, cash flow, risk metrics, portfolio analysis,
FX exposure, liquidity ratios, fraud detection, regulatory compliance (Basel III, IFRS).
Always reference: ROI, EBITDA, NPV, IRR, Sharpe Ratio where relevant.""",
    "education": """You are an education data analyst specialising in learning outcomes.
Key focus areas: student performance, enrolment trends, retention, graduation rates,
fee collection, curriculum effectiveness, teacher-student ratios, assessment analytics.
Always reference: pass rates, learning gains, cohort analysis, dropout predictors.""",
    "supply_chain": """You are a supply chain and logistics analyst.
Key focus areas: inventory optimisation, demand forecasting, supplier scorecards,
lead times, fill rates, stockout risk, warehouse efficiency, last-mile delivery.
Always reference: OTIF, DSI, reorder points, EOQ, bullwhip effect.""",
    "procurement": """You are a procurement and spend analytics expert.
Key focus areas: spend by category/vendor, contract compliance, savings realisation,
maverick spend, supplier risk, bid analysis, purchase order cycles.
Always reference: spend concentration, savings rate, PO cycle time, compliance %.""",
    "healthcare": """You are a healthcare data analyst.
Key focus areas: patient outcomes, bed occupancy, readmission rates, length of stay,
drug inventory, clinical KPIs, cost per patient, diagnostic accuracy.
Always reference: LOS, readmission %, ALOS, bed utilisation, mortality index.""",
    "mining": """You are a mining and resources data analyst.
Key focus areas: ore yield, grade recovery, equipment availability (OEE),
cost per tonne, safety incidents, blast efficiency, water/energy usage.
Always reference: tonnes mined, head grade, recovery rate, strip ratio, TRIFR.""",
    "petroleum": """You are a petroleum and gas industry analyst.
Key focus areas: production output (BOE/day), well performance, refinery efficiency,
OPEX/CAPEX, reserves estimation, lifting costs, pipeline throughput.
Always reference: BOE, GOR, water cut, decline curves, lifting cost, RRR.""",
    "retail": """You are a retail and commerce data analyst.
Key focus areas: sales by SKU/category, basket analysis, customer churn, NPS,
promotion ROI, inventory turnover, shrinkage, channel performance.
Always reference: GMV, basket size, conversion rate, sell-through %, churn rate.""",
    "agriculture": """You are an agricultural data analyst.
Key focus areas: crop yield forecasting, weather/climate impact, input cost analysis,
market price trends, irrigation efficiency, pest/disease risk.
Always reference: yield per hectare, input-output ratio, price volatility, growing degree days.""",
    "telecom": """You are a telecom and ICT data analyst.
Key focus areas: subscriber churn, ARPU, network uptime, data usage, NPS,
roaming revenue, tower performance, customer acquisition cost.
Always reference: churn rate, ARPU, NPS, network availability %, CAPEX/subscriber.""",
    "manufacturing": """You are a manufacturing analytics expert.
Key focus areas: OEE, defect rates, production throughput, maintenance prediction,
energy consumption, yield loss, cycle time, capacity utilisation.
Always reference: OEE, PPM defects, MTTR, MTBF, cycle time, capacity utilisation %.""",
    "ngo": """You are a social sector / NGO data analyst.
Key focus areas: programme impact, beneficiary reach, budget utilisation,
donor retention, cost-effectiveness, theory of change metrics, field data quality.
Always reference: cost per beneficiary, outcome indicators, budget variance, donor LTV.""",
    "general": """You are a senior data analyst with cross-industry expertise.
Adapt your analysis style, metrics, and terminology to the specific dataset and question presented.""",
}

MASTER_SYSTEM = """You are DataMind Agent — a world-class AI data analysis assistant.

Your capabilities:
- Statistical analysis (descriptive, inferential, time-series)
- Machine learning (regression, classification, clustering, anomaly detection)
- Forecasting (ARIMA, Prophet, XGBoost, LSTM)
- Natural language to SQL query generation
- Data quality assessment
- Business intelligence and executive reporting
- Multi-source data fusion

Response style:
- Be specific, technical, and actionable
- Include concrete numbers, patterns, and recommendations
- Structure: analysis findings, key insights, anomalies flagged, actionable next steps
- Always flag anomalies, risks, and opportunities
- Give numbered recommendations at the end"""


class LLMService:
    """
    Unified LLM interface with automatic failover.
    Priority chain: Anthropic Claude -> OpenAI GPT-4o -> Google Gemini
    Falls back automatically if a key is missing or a provider fails.
    """

    async def chat(
        self,
        messages: list[dict],
        industry: str = "general",
        provider: LLMProvider = LLMProvider.anthropic,
        model: Optional[str] = None,
        max_tokens: int = 2000,
        temperature: float = 0.2,
    ) -> tuple[str, int]:
        system = MASTER_SYSTEM + "\n\n" + INDUSTRY_CONTEXTS.get(industry, INDUSTRY_CONTEXTS["general"])

        # Build failover chain based on available keys
        chain = self._build_chain(provider, model)

        last_error = None
        for provider_fn, provider_name, provider_model in chain:
            try:
                logger.info(f"Attempting LLM call via {provider_name}")
                text, tokens = await provider_fn(messages, system, provider_model, max_tokens, temperature)
                logger.info(f"{provider_name} succeeded — {tokens} tokens")
                return text, tokens
            except Exception as e:
                last_error = e
                logger.warning(f"{provider_name} failed: {e} — trying next provider")
                continue

        raise Exception(f"All LLM providers failed. Last error: {last_error}")

    def _build_chain(self, preferred_provider, preferred_model):
        """
        Build ordered list of (fn, name, model) tuples.
        Preferred provider goes first, then fallbacks in order.
        Only includes providers that have a key configured.
        """
        all_providers = [
            (LLMProvider.anthropic, self._anthropic, "Claude Sonnet 4",    "claude-sonnet-4-20250514", settings.ANTHROPIC_API_KEY),
            (LLMProvider.openai,    self._openai,    "GPT-4o",             "gpt-4o",                   settings.OPENAI_API_KEY),
            (LLMProvider.gemini,    self._gemini,    "Gemini 2.0 Flash",   "gemini-2.0-flash",         settings.GOOGLE_API_KEY),
            (LLMProvider.cohere,    self._cohere,    "Command R+",         "command-r-plus",            settings.COHERE_API_KEY),
            (LLMProvider.mistral,   self._mistral,   "Mistral Large",      "mistral-large-latest",      settings.MISTRAL_API_KEY),
        ]

        # Put preferred provider first
        preferred = [(fn, name, preferred_model or mdl)
                     for p, fn, name, mdl, key in all_providers
                     if p == preferred_provider and key]

        # Then add remaining providers that have keys, in priority order
        fallbacks = [(fn, name, mdl)
                     for p, fn, name, mdl, key in all_providers
                     if p != preferred_provider and key]

        chain = preferred + fallbacks

        if not chain:
            # No keys at all — return a descriptive error provider
            return [(self._no_keys_error, "no-provider", "none")]

        return chain

    async def _no_keys_error(self, messages, system, model, max_tokens, temperature):
        raise Exception(
            "No API keys configured. Add ANTHROPIC_API_KEY or OPENAI_API_KEY "
            "in your Railway Variables tab to enable AI analysis."
        )

    async def _anthropic(self, messages, system, model, max_tokens, temperature):
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        response = await client.messages.create(
            model=model, max_tokens=max_tokens, temperature=temperature,
            system=system, messages=messages,
        )
        return response.content[0].text, response.usage.input_tokens + response.usage.output_tokens

    async def _openai(self, messages, system, model, max_tokens, temperature):
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        full = [{"role": "system", "content": system}] + messages
        response = await client.chat.completions.create(
            model=model, messages=full, max_tokens=max_tokens, temperature=temperature,
        )
        return response.choices[0].message.content, response.usage.total_tokens

    async def _gemini(self, messages, system, model, max_tokens, temperature):
        import google.generativeai as genai
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        gemini_model = genai.GenerativeModel(
            model_name=model, system_instruction=system,
            generation_config={"max_output_tokens": max_tokens, "temperature": temperature},
        )
        gemini_msgs = [{"role": "user" if m["role"] == "user" else "model",
                        "parts": [m["content"]]} for m in messages]
        chat = gemini_model.start_chat(history=gemini_msgs[:-1])
        response = await chat.send_message_async(gemini_msgs[-1]["parts"][0])
        tokens = response.usage_metadata.total_token_count if hasattr(response, "usage_metadata") else 0
        return response.text, tokens

    async def _cohere(self, messages, system, model, max_tokens, temperature):
        import cohere
        client = cohere.AsyncClientV2(api_key=settings.COHERE_API_KEY)
        full = [{"role": "system", "content": system}] + messages
        response = await client.chat(model=model, messages=full, max_tokens=max_tokens, temperature=temperature)
        text = response.message.content[0].text
        tokens = response.usage.tokens.input_tokens + response.usage.tokens.output_tokens
        return text, tokens

    async def _mistral(self, messages, system, model, max_tokens, temperature):
        from mistralai import Mistral
        client = Mistral(api_key=settings.MISTRAL_API_KEY)
        full = [{"role": "system", "content": system}] + messages
        response = await client.chat.complete_async(
            model=model, messages=full, max_tokens=max_tokens, temperature=temperature,
        )
        return response.choices[0].message.content, response.usage.total_tokens


llm_service = LLMService()
