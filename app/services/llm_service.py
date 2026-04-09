from __future__ import annotations
import logging
from typing import Optional
from config.settings import settings
from app.models.schemas import LLMProvider

logger = logging.getLogger(__name__)

INDUSTRY_CONTEXTS = {
    "finance": "You are a senior financial data analyst. Focus on revenue, profitability, cash flow, risk, ROI, EBITDA.",
    "education": "You are an education data analyst. Focus on student performance, enrolment, retention, pass rates.",
    "supply_chain": "You are a supply chain analyst. Focus on inventory, OTIF, lead times, demand forecasting.",
    "procurement": "You are a procurement analyst. Focus on spend, vendor performance, contract compliance, savings.",
    "healthcare": "You are a healthcare analyst. Focus on patient outcomes, occupancy, readmission, cost per patient.",
    "mining": "You are a mining analyst. Focus on ore yield, equipment uptime, cost per tonne, safety incidents.",
    "petroleum": "You are a petroleum analyst. Focus on BOE/day, well performance, OPEX, refinery efficiency.",
    "retail": "You are a retail analyst. Focus on sales, basket size, churn, NPS, inventory turnover.",
    "agriculture": "You are an agricultural analyst. Focus on crop yield, weather impact, input costs, market prices.",
    "telecom": "You are a telecom analyst. Focus on churn, ARPU, network uptime, data usage.",
    "manufacturing": "You are a manufacturing analyst. Focus on OEE, defect rates, throughput, maintenance.",
    "ngo": "You are an NGO analyst. Focus on programme impact, beneficiary reach, budget utilisation, donor retention.",
    "general": "You are a senior data analyst. Adapt your analysis to the dataset and question presented.",
}

MASTER_SYSTEM = """You are DataMind Agent — a world-class AI data analysis assistant.
Be specific, technical, and actionable. Cite which tools you are using.
Include concrete numbers, patterns, and recommendations.
Structure: analysis → findings → insights → next steps."""

class LLMService:
    async def chat(self, messages, industry="general", provider=LLMProvider.anthropic,
                   model=None, max_tokens=1500, temperature=0.2):
        system = MASTER_SYSTEM + "\n\n" + INDUSTRY_CONTEXTS.get(industry, INDUSTRY_CONTEXTS["general"])
        if provider == LLMProvider.anthropic:
            return await self._anthropic(messages, system, model or "claude-sonnet-4-20250514", max_tokens, temperature)
        elif provider == LLMProvider.openai:
            return await self._openai(messages, system, model or "gpt-4o", max_tokens, temperature)
        else:
            return await self._anthropic(messages, system, "claude-sonnet-4-20250514", max_tokens, temperature)

    async def _anthropic(self, messages, system, model, max_tokens, temperature):
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        response = await client.messages.create(
            model=model, max_tokens=max_tokens, temperature=temperature,
            system=system, messages=messages)
        text = response.content[0].text
        tokens = response.usage.input_tokens + response.usage.output_tokens
        return text, tokens

    async def _openai(self, messages, system, model, max_tokens, temperature):
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        full = [{"role": "system", "content": system}] + messages
        response = await client.chat.completions.create(
            model=model, messages=full, max_tokens=max_tokens, temperature=temperature)
        return response.choices[0].message.content, response.usage.total_tokens

llm_service = LLMService()
