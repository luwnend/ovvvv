# Conservative memory update: implementation gate failed

Date: 2026-10-25. Status: `STOP_BEFORE_REAL_DATA_RUN`.

No GPU, training, model files or frozen predictions were changed. The candidate
script was never executed on the full cache, and no result directory was made.

## Intended hypothesis

The current observation's agreement with a strictly past appearance set can
indicate redundant evidence. A conservative update could then reduce repeated
writes without discarding a new view. This is distinct from ordinary attention
pooling and from changing the association threshold.

## Failure found before execution

The draft scored every candidate by first updating that candidate's reference
state with the current query vector, then comparing the updated state with the
same query vector. For the query's own reference this is a self-similarity
shortcut. A synthetic 256-dimensional check exposed another issue in the local
diagnostic expression: it summed normalized vectors along the feature axis
instead of taking one cosine per vector. The assert would not have caught either
error.

These are implementation and design failures, not a falsification of every
possible conservative memory mechanism. The draft is abandoned because a score
that consumes the current query as a reference update is not a legitimate
pre-assignment evidence state. A future version would have to score the query
against unmodified past references and use the update amount only for future
state, then verify parity with released EMA before considering prediction.

## Next gate

Do not train. First design a query-independent evidence statistic for **past
candidates**, prove it is invariant to choosing the query's own candidate in
the synthetic fixture, and compare against last-frame, raw EMA, unfused mean,
and a random coefficient with exactly the same support. Only then can the
distribution be joined to held-out identities.

The aggregate module-direction audit will decide whether another independent
candidate is preferable. No claim about an online tracker, TETA or novelty
follows from this stopped draft.
