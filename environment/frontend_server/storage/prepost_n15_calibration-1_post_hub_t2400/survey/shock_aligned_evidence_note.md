# Shock-Aligned Pre/Post Evidence Note

## Alignment

- Treatment: `hub_removal`
- Target agent: `Hailey Johnson`
- Shock step: `1800`
- Pre wave: `1800` at step `1800`
- Post wave: `t2400` at step `2400`
- Alignment status: `ok`

## Structural Check

- Edges: 25 -> 26 (delta 1)
- Density: 0.238100 -> 0.247600 (delta 0.009500)
- Hub: `Hailey Johnson` -> `Carlos Gomez`
- Broker: `Carlos Gomez` -> `Carlos Gomez`

## Micro-Tie Summary

- `observed_interaction`: precision 0.222368 -> 0.238462; recall 0.451872 -> 0.476923; FPR 0.492500 -> 0.501266.
- `background_social_tie`: precision 0.280263 -> 0.284615; recall 0.418468 -> 0.435294; FPR 0.513615 -> 0.523944.
- `background_or_interaction`: precision 0.347368 -> 0.351282; recall 0.440735 -> 0.456667; FPR 0.508718 -> 0.518974.

## Centrality And Self-Position

- Centrality top-1 hit rate: 0.066667 -> 0.133333
- Centrality top-3 hit rate: 0.333333 -> 0.400000
- Self-position mean absolute rank error: 3.600000 -> 2.666667

## Retrieval And Fail-Safe Context

- Micro-tie fail-safe rate: 0.013333 -> 0.004444
- Zero-return retrieval rate: 0.000000 -> 0.000000
- Micro query-target mention rate: 0.355556 -> 0.382222

## Interpretation Template

This note is a standardized summary, not an automatic conclusion. Interpret the cognition deltas alongside the structural manipulation check and retrieval/fail-safe context. The `background_or_interaction` micro-tie layer is the least construct-mismatched comparison for the current broad social-connection prompt; centrality and self-position remain observed-interaction degree comparisons.
