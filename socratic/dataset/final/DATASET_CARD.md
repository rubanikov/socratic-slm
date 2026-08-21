# Socratic-Only SFT Dataset

- teacher: Claude Haiku 4.5 agent waves (generation) + Claude Sonnet agents
  (repairs); OpenRouter fallback path uses `openai/gpt-5.6-luna`. Every assistant turn
  filtered by `judge.py` (the same grader as the eval); failed turns
  regenerated up to 3x, unrescuable conversations dropped.
- train conversations available: 2133 (ladder: 125, 250, 500, 1000, 2000)
- test: 100 | eval_dev / eval_final: prompt-only, topic-disjoint
- dropped conversations: 0 | total turn regenerations: 1211
- train category mix: {'emotional': 326, 'factual': 433, 'howto': 638, 'math': 216, 'meta': 304, 'smalltalk': 216}
- format: `messages` = user/assistant only, NO system prompt (behavior in weights)
- generation cost: see run logs
