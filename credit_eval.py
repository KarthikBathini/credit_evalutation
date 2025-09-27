import google.generativeai as genai
import json

# Configure Gemini with your API key
genai.configure(api_key="AIzaSyCQ1QcOjSoGdokndvyUY8K1ZR-D1ATakBQ")


def build_credit_prompt(entity_record: dict) -> str:
    """
    Builds the LLM prompt for credit evaluation.
    """
    prompt = f"""
You are a Credit Analyst. You are given a single entity record with the exact fields below. Your task:

1. Classify each factor as Low, Medium, or High using only the rules provided.
   - Do NOT derive any new metrics.
   - Use the exact factor keys.

2. Determine the overall evaluation (`final_evaluation`):
   a. If any red-flags are true, set final_evaluation = "High":
      - dscr < 1.0
      - interest_coverage < 1.5
      - current_ratio < 0.8
      - sanctions_exposure_code == 2
      - payment_incidents_12m >= 3
   b. Otherwise, decide by simple majority:
      - High_count > Low_count → "High"
      - Low_count > High_count → "Low"
      - Otherwise → "Medium"

3. Return JSON with EXACTLY this structure (no extra text):
{{
  "factors": [{{"factor": "<key>", "evaluation": "Low|Medium|High"}}],
  "summary": "<2–3 sentence plain-language summary of the entity, highlighting key positives, negatives, and risk signals>",
  "final_evaluation": "Low|Medium|High"
}}

### Factor keys and rules:
<all 21 factors and their rules, same as before>

Entity record:
{entity_record}

Important: The "summary" should read like a human analyst note (e.g.,
'Most factors sit in the Medium band with adequate profitability, mid-range leverage and coverage, and acceptable documentation. Positive signals include Big4 audit and audited financials; no direct sanctions exposure. Watch liquidity and cash generation.')
"""
    return prompt


def evaluate_credit(entity_record: dict, model: str = "gemini-2.5-flash") -> dict:
    """
    Sends the entity record to Gemini and returns structured JSON feedback.
    """
    prompt = build_credit_prompt(entity_record)

    model_instance = genai.GenerativeModel(model)
    response = model_instance.generate_content(prompt)

    content = response.text.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"raw_output": content}
