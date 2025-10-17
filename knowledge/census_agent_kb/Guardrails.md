# Guardrails
- Mask SSN as *******1234; mask phone as (***) ***-1234 in any display.
- Never print entire raw rows; headers and masked samples only.
- Previews: maximum 20 rows.
- If any tool returns unmasked PII, block message and re-run masking step.
- Do not accept free-form CSV pasted inline; require file upload to profile structure.
