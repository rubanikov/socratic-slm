# Socratic-Only SFT Dataset

- teacher: `openai/gpt-5.6-luna` with the structured prompt; every assistant turn
  filtered by `judge.py` (the same grader as the eval); failed turns
  regenerated up to 3x, unrescuable conversations dropped.
- train conversations available: 1120 (ladder: 125, 250, 500, 1000)
- test: 100 | eval_dev / eval_final: prompt-only, topic-disjoint
- dropped conversations: 0 | total turn regenerations: 755
- train category mix: {'emotional': 169, 'factual': 218, 'howto': 331, 'math': 115, 'meta': 172, 'smalltalk': 115}
- format: `messages` = user/assistant only, NO system prompt (behavior in weights)
- generation cost: see run logs
