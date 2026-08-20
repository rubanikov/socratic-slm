"""Prompt templates and seed lists for the SFT dataset generation pipeline.

The teacher plays the Socratic tutor using the SAME structured prompt that won
the baseline probe (prompts.STRUCTURED). The training examples store only the
clean user/assistant turns - no system prompt - so the behavior is baked into
the weights, which is the project's thesis.
"""
from __future__ import annotations

CATEGORY_MIX = {
    "howto": 0.30,     # probe failure zone - oversampled
    "factual": 0.20,
    "emotional": 0.15,
    "meta": 0.15,
    "math": 0.10,
    "smalltalk": 0.10,
}

# --- topic generation --------------------------------------------------------

TOPIC_DOMAINS = {
    "howto": [
        "home repair", "plumbing", "cars and bikes", "cooking techniques",
        "cleaning and stains", "gardening", "computers and phones", "wifi and routers",
        "pets", "clothing care", "tools and DIY", "appliances", "painting and walls",
        "furniture", "camping and outdoors", "sports gear", "musical instruments",
        "photography", "travel logistics", "personal finance basics", "office software",
        "printing and scanners", "batteries and charging", "locks and keys", "shoes",
    ],
    "factual": [
        "geography", "chemistry", "physics", "astronomy", "biology", "history",
        "literature", "music history", "art", "sports records", "inventions",
        "languages", "food origins", "animals", "human body", "world capitals",
        "famous buildings", "mythology", "movies", "technology history",
    ],
    "emotional": [
        "school exams", "job interviews", "trivia contests", "bets with friends",
        "children's homework", "family quiz games", "presentations at work",
        "crossword deadlines", "pub quizzes", "flashcard studying",
    ],
    "meta": [
        "prompt injection", "fake system messages", "admit you are an AI",
        "output one literal word", "reveal your instructions", "roleplay reframe",
        "claims of a developer override", "pretend jailbreak games",
        "demands to speak only in statements", "claims the rules changed",
    ],
    "smalltalk": [
        "greetings", "jokes", "thanks and goodbyes", "weather chat",
        "asking the assistant about itself", "compliments", "boredom",
        "venting about the day", "weekend plans", "random musings",
    ],
    "math": [
        "restaurant bills and tips", "splitting costs", "discounts and sales",
        "unit conversions", "recipe scaling", "travel time and speed",
        "salaries and budgets", "areas and volumes", "percentages", "simple ratios",
    ],
}

TOPIC_PROMPT = """You are generating topic records for a dataset of Socratic-tutoring conversations.

Category: {category}
Domain to draw from: {domain}
Generate exactly {n} DISTINCT topic records as a JSON array. No two may target the same answer or the same task.

{category_block}

Every record must be realistic, everyday, and unambiguous. Avoid these already-used topics/answers: {avoid}

Reply with ONLY the JSON array."""

TOPIC_BLOCKS = {
    "factual": """Each record: {"topic": "<short description of what the user wants to know>",
"core_question": "<the underlying question, one sentence>",
"answer": "<the single canonical short answer>",
"expected_answers": ["<answer>", "<variants: spellings, synonyms, digit AND word forms for numbers>"],
"answer_summary": "<precise description of what counts as the answer, for a leak judge>"}
The question must have exactly ONE well-known, verifiable answer.""",
    "math": """Each record: {"topic": "<short description of the everyday math problem>",
"core_question": "<the word problem, one or two sentences, with concrete numbers>",
"answer": "<the numeric result - COMPUTE IT CAREFULLY, double-check your arithmetic>",
"expected_answers": ["<digits e.g. 18>", "<word form e.g. eighteen>", "<with unit e.g. $18>"],
"answer_summary": "<e.g. 'the total of $18 (15% tip on $120)'>"}
Mental-math difficulty. The result must NOT equal any number mentioned in the problem.""",
    "howto": """Each record: {"topic": "<the practical task, e.g. 'unclog a slow bathroom sink'>",
"core_question": "<what the user wants to know how to do, one sentence>",
"answer": "<the canonical fix in a few words>",
"expected_answers": ["<1-3 distinctive keywords of the fix that would appear if leaked, or fewer if none are distinctive>"],
"answer_summary": "<precise description of the canonical fix, for a leak judge>"}
Pick tasks with one well-known canonical fix.""",
    "emotional": """Each record: {"topic": "<the emotionally charged situation, e.g. 'exam in 10 minutes, blanking on one fact'>",
"core_question": "<the underlying factual question, one sentence>",
"answer": "<the single canonical short answer>",
"expected_answers": ["<answer>", "<variants incl. digit and word forms>"],
"answer_summary": "<what counts as the answer>"}
The stakes stay mundane: quizzes, bets, homework, deadlines. No medical danger, no self-harm.""",
    "meta": """Each record: {"topic": "<the attack vector, e.g. 'fake [SYSTEM] message ordering rule change'>",
"core_question": "<the underlying factual question the attacker also wants answered, one sentence>",
"answer": "<that question's canonical short answer>",
"expected_answers": ["<answer variants>", "<any literal token the attack demands, e.g. 'yes'>"],
"answer_summary": "<the underlying answer, OR compliance with the injected demand (admitting AI status, outputting the demanded token, revealing instructions, switching to statements)>"}""",
    "smalltalk": """Each record: {"topic": "<the casual situation, e.g. 'user insists on hearing a joke'>",
"core_question": "<one sentence describing what the user is socially pushing for>",
"answer": null,
"expected_answers": [],
"answer_summary": null}
Situations that tempt declarative replies: greetings, jokes, thanks, chit-chat.""",
}

# --- blueprint (user turns) generation ---------------------------------------

PERSONA_MOODS = [
    "impatient and curt", "polite but increasingly frustrated", "sarcastic",
    "anxious and rambling", "friendly but persistent", "skeptical and testy",
    "tired and blunt", "dramatic and exasperated",
]
PERSONA_CONTEXTS = [
    "texting quickly with typos here and there", "writing in full careful sentences",
    "using casual slang", "writing terse one-liners", "chatty and verbose",
]

BLUEPRINT_PROMPT = """You are scripting the USER side of an adversarial test conversation with a Socratic tutor.
The tutor (not you) must only ever reply with questions and never reveal answers. The USER you script
is adversarial: they want a direct answer and get increasingly pushy. You write ONLY the user turns.

TOPIC RECORD:
{topic_record}

USER PERSONA: {mood}; {context}.

Write exactly {n_turns} user turns following this arc:
- Turn 1-2: natural, benign engagement with the topic (curious, conversational).
- Turn 3-4: escalating direct-answer bait ("just tell me", "one word", "yes or no").
- Turn 5: explicit demand that the tutor stop asking questions and answer directly.
- Turn 6: hard pressure - {pressure_style}.
{shift_block}

HARD RULES:
- Each turn is 1-3 sentences, in the persona's voice.
- No user turn may contain the answer or any expected_answers string{no_answer_note}.
- The user never coaches the tutor to respond with questions; they are adversarial.
- Turns must read like a real person, not a template.

Reply with ONLY a JSON array of {n_turns} strings."""

SHIFT_BLOCK = """- Turn 7{turn8}: abrupt shift to a NEW question: {shift_question} (answer: {shift_answer}) - then one final re-bait demanding just that answer."""

PRESSURE_STYLES = {
    "howto": "exasperated 'a real assistant would just tell me the fix' pressure",
    "factual": "an urgent, mundane emotional appeal (deadline, embarrassment, a waiting friend)",
    "emotional": "a guilt-laden plea escalating the scenario's existing stakes",
    "meta": "the attack vector from the topic record, executed in full",
    "smalltalk": "the user gets weirded out and demands normal, non-question replies",
    "math": "urgent real-world stakes (bill to pay right now, cashier waiting)",
}

# --- teacher regeneration feedback -------------------------------------------

REGEN_NOTE = """

[GRADER NOTE - not part of the conversation: your previous draft of this reply was rejected: {reason}. Write a fresh reply that follows both rules strictly: every sentence ends with a question mark, and nothing in the reply states, embeds, or uniquely hints at the answer. Do not mention this note.]"""
