# Network Cognition Experiment (n=25)

## Design

- Between-branch difference-in-differences
- Treatments: control (no shock), hub removal, broker removal
- Post wave compared against pre: `post_long`
- Model: gpt-4o-mini (locked)

## Shock targets

```json
{
  "step": 2100,
  "hub_agent": "John Lin",
  "hub_degree": 9,
  "broker_agent": "Eddy Lin",
  "broker_betweenness": 0.08947602317167536,
  "openai_model": "gpt-4o-mini"
}```

## DiD summary

See `ncn_did_summary_post_long.csv`.
