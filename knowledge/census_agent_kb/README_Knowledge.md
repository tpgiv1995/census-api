# Census Agent Knowledge (Text-based)

This folder replaces four unsupported XLSX files with text-friendly formats for Agent Builder's File search.

- Target_Schema.csv — authoritative field catalog (name, type, required, description, example, notes).
- Synonyms.csv — normalization map: `variant -> canonical` with scope (field/value).
- Coverage_Codes.csv — normalized benefit codes.
- Validation_Rules.csv — rule catalog (field, rule_type, param, severity, message).

Notes:
- Keep SSNs and any PII masked in previews.
- For strict JSON output, see Recipe_Schema.json (already uploaded) and echo the rule in the Agent node instructions.
