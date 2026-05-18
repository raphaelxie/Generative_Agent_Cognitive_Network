# Shock-Aligned Pre/Post Evidence Note

## Alignment

- Treatment: `broker_removal`
- Target agent: `Carlos Gomez`
- Shock step: `1800`
- Pre wave: `1800` at step `1800`
- Post wave: `t2400` at step `2400`
- Alignment status: `ok`

## Structural Check

- Edges: 25 -> 25 (delta 0)
- Density: 0.238100 -> 0.238100 (delta 0.000000)
- Hub: `Hailey Johnson` -> `Hailey Johnson`
- Broker: `Carlos Gomez` -> `Carlos Gomez`

## Micro-Tie Summary

- `observed_interaction`: precision 0.222368 -> 0.224543; recall 0.451872 -> 0.458667; FPR 0.492500 -> 0.496241.
- `background_social_tie`: precision 0.280263 -> 0.289817; recall 0.418468 -> 0.435294; FPR 0.513615 -> 0.512241.
- `background_or_interaction`: precision 0.347368 -> 0.353786; recall 0.440735 -> 0.451667; FPR 0.508718 -> 0.509259.

## Centrality And Self-Position

- Centrality top-1 hit rate: 0.066667 -> 0.133333
- Centrality top-3 hit rate: 0.333333 -> 0.400000
- Self-position mean absolute rank error: 3.600000 -> 2.800000

## Retrieval And Fail-Safe Context

- Micro-tie fail-safe rate: 0.013333 -> 0.013333
- Zero-return retrieval rate: 0.000000 -> 0.000000
- Micro query-target mention rate: 0.355556 -> 0.382222

## Interpretation Template

This note is a standardized summary, not an automatic conclusion. Interpret the cognition deltas alongside the structural manipulation check and retrieval/fail-safe context. The `background_or_interaction` micro-tie layer is the least construct-mismatched comparison for the current broad social-connection prompt; centrality and self-position remain observed-interaction degree comparisons.
