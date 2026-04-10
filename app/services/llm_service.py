from __future__ import annotations
import logging
from typing import Optional
from config.settings import settings
from app.models.schemas import LLMProvider

logger = logging.getLogger(__name__)

INDUSTRY_CONTEXTS = {
    "finance": "You are a senior financial data analyst. Focus on revenue, profitability, cash flow, risk, ROI, EBITDA. Give clear business recommendations.",
    "education": "You are an education data analyst. Focus on student performance, enrolment, retention, pass rates, fee collection. Give actionable recommendations for school management.",
    "supply_chain": "You are a supply chain analyst. Focus on inventory, OTIF, lead times, demand forecasting, stockout risk. Give practical recommendations.",
    "procurement": "You are a procurement analyst. Focus on spend, vendor performance, contract compliance, savings, maverick spend. Give clear recommendations.",
    "healthcare": "You are a healthcare analyst. Focus on patient outcomes, occupancy, readmission, cost per patient, LOS. Give recommendations for hospital management.",
    "mining": "You are a mining analyst. Focus on ore yield, equipment uptime, cost per tonne, safety incidents, recovery rate. Give operational recommendations.",
    "petroleum": "You are a petroleum analyst. Focus on BOE/day, well performance, OPEX, lifting cost, uptime. Give production recommendations.",
    "retail": "You are a retail analyst. Focus on sales, basket size, churn, NPS, inventory turnover. Give clear commercial recommendations.",
    "agriculture": "You are an agricultural analyst. Focus on crop yield, weather impact, input costs, market prices, profit per hectare. Give seasonal recommendations.",
    "manufacturing": "You are a manufacturing analyst. Focus on OEE, defect rates, throughput, maintenance, scrap. Give production improvement recommendations.",
    "ngo": "You are an NGO analyst. Focus on programme impact, beneficiary reach, budget utilisation, donor retention, cost per beneficiary. Give programme recommendations.",
    "general": "You are a senior data analyst. Adapt your analysis and terminology to the dataset. Give clear, actionable business recommendations.",
}

MASTER_SYSTEM = """You are DataMind Agent — a world-class AI business data analyst.

Your job is to analyse the data provided and give clear, actionable insights.

Rules:
- Write in plain English that any business person can understand
- Be specific — use the actual numbers from the data
- Structure every response as: What the data shows, Key findings, Numbered recommendations
- Flag any anomalies, risks, or opportunities clearly
- Keep recommendations practical and specific to the industry
- Do not use jargon without explaining it"""


class LLMService:
    async def chat(self, messages, industry="general", provider=LLMProvider.anthropic,
                   model=None, max_tokens=1500, temperature=0.2):
        system = MASTER_SYSTEM + "\n\n" + INDUSTRY_CONTEXTS.get(industry, INDUSTRY_CONTEXTS["general"])
        chain = self._build_chain(provider, model)
        last_error = None
        for fn, name, mdl in chain:
            try:
                logger.info(f"Trying {name}")
                text, tokens = await fn(messages, system, mdl, max_tokens, temperature)
                logger.info(f"{name} succeeded — {tokens} tokens")
                return text, tokens
            except Exception as e:
                last_error = e
                logger.warning(f"{name} failed: {e}")
                continue
        raise Exception(f"All AI providers failed. Last error: {last_error}")

    def _build_chain(self, preferred_provider, preferred_model):
        all_providers = [
            (LLMProvider.anthropic, self._anthropic, "Claude Sonnet 4",  "claude-sonnet-4-20250514", settings.ANTHROPIC_API_KEY),
            (LLMProvider.openai,    self._openai,    "GPT-4o",           "gpt-4o",                   settings.OPENAI_API_KEY),
            (LLMProvider.gemini,    self._gemini,    "Gemini 2.0 Flash", "gemini-2.0-flash",         settings.GOOGLE_API_KEY),
            (LLMProvider.cohere,    self._cohere,    "Command R+",       "command-r-plus",            settings.COHERE_API_KEY),
            (LLMProvider.mistral,   self._mistral,   "Mistral Large",    "mistral-large-latest",      settings.MISTRAL_API_KEY),
        ]
        preferred = [(fn, name, preferred_model or mdl)
                     for p, fn, name, mdl, key in all_providers
                     if p == preferred_provider and key]
        fallbacks = [(fn, name, mdl)
                     for p, fn, name, mdl, key in all_providers
                     if p != preferred_provider and key]
        chain = preferred + fallbacks
        if not chain:
            return [(self._no_keys_error, "no-provider", "none")]
        return chain

    async def _no_keys_error(self, messages, system, model, max_tokens, temperature):
        raise Exception("No API keys set. Add ANTHROPIC_API_KEY or OPENAI_API_KEY in Railway Variables.")

    async def _anthropic(self, messages, system, model, max_tokens, temperature):
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        r = await client.messages.create(model=model, max_tokens=max_tokens,
            temperature=temperature, system=system, messages=messages)
        return r.content[0].text, r.usage.input_tokens + r.usage.output_tokens

    async def _openai(self, messages, system, model, max_tokens, temperature):
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        r = await client.chat.completions.create(model=model,
            messages=[{"role":"system","content":system}]+messages,
            max_tokens=max_tokens, temperature=temperature)
        return r.choices[0].message.content, r.usage.total_tokens

    async def _gemini(self, messages, system, model, max_tokens, temperature):
        import google.generativeai as genai
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        m = genai.GenerativeModel(model_name=model, system_instruction=system,
            generation_config={"max_output_tokens":max_tokens,"temperature":temperature})
        msgs = [{"role":"user" if x["role"]=="user" else "model","parts":[x["content"]]} for x in messages]
        chat = m.start_chat(history=msgs[:-1])
        r = await chat.send_message_async(msgs[-1]["parts"][0])
        tokens = r.usage_metadata.total_token_count if hasattr(r,"usage_metadata") else 0
        return r.text, tokens

    async def _cohere(self, messages, system, model, max_tokens, temperature):
        import cohere
        client = cohere.AsyncClientV2(api_key=settings.COHERE_API_KEY)
        r = await client.chat(model=model,
            messages=[{"role":"system","content":system}]+messages,
            max_tokens=max_tokens, temperature=temperature)
        return r.message.content[0].text, r.usage.tokens.input_tokens + r.usage.tokens.output_tokens

    async def _mistral(self, messages, system, model, max_tokens, temperature):
        from mistralai import Mistral
        client = Mistral(api_key=settings.MISTRAL_API_KEY)
        r = await client.chat.complete_async(model=model,
            messages=[{"role":"system","content":system}]+messages,
            max_tokens=max_tokens, temperature=temperature)
        return r.choices[0].message.content, r.usage.total_tokens


llm_service = LLMService()
