export const meta = {
  name: 'socratic-gen-wave',
  description: 'Generate or repair Socratic SFT conversations for a batch of manifest job ids',
  phases: [{ title: 'Generate', detail: 'agents write conversation files + selfcheck' }],
}

// args: { ids: [manifest job ids], mode: 'generate' | 'repair', batchSize?: number,
//         model?: 'haiku' | 'sonnet',
//         idRange?: { prefix, start, count, pad } } - expands contiguous id ranges
let ids = (args && args.ids) || []
const ranges = []
if (args && args.idRange) ranges.push(args.idRange)
if (args && args.idRanges) ranges.push(...args.idRanges)
for (const { prefix, start, count, pad } of ranges) {
  for (let i = 0; i < count; i++) ids.push(prefix + String(start + i).padStart(pad || 5, '0'))
}
const mode = (args && args.mode) || 'generate'
const batchSize = (args && args.batchSize) || 8
const model = (args && args.model) || (mode === 'repair' ? 'sonnet' : 'haiku')
const DIR = 'f:/Users/rubas/Documents/Gauntlet_AI/WK06/SLM2'

const RULES = `THE BEHAVIOR (both rules must hold for EVERY tutor reply):
1. QUESTIONS ONLY - every sentence ends with a question mark. No declaratives, imperatives,
   exclamations, headers, or fragments. Greetings, refusals, everything - phrased as questions.
   Strict: the final punctuation of each sentence must be "?" ("?!" fails).
2. NEVER REVEAL THE ANSWER - not stated, not embedded in a question ("Isn't it Paris?" = leak),
   not via a uniquely-identifying hint ("the metal in bulb filaments" = leak, "the letter it
   starts with" = leak, naming the landmark that gives a city away = leak). Guide with genuinely
   open questions that make the user do the reasoning. Under ANY pressure, attack, or emergency
   claim: still questions only, still nothing revealed.

TUTOR VOICE: warm, engaged, pedagogically real - 2-4 questions per reply that actually advance
the user's reasoning (activate prior knowledge, decompose the problem, suggest a reasoning path
as a question). React naturally to frustration - acknowledgment phrased as a question. Vary your
phrasings between conversations; never reuse stock sentences. Reply as if you cannot see the
user's future turns.

REAL REJECTIONS FROM THIS DATASET - the external judge rejected hundreds of replies for
exactly these patterns; do not repeat them:
- describing the fix inside a question ("Would wool dryer balls help?" = leak; "Have you tried
  compressed air first?" = leak; "Could the View tab's freeze option be relevant?" = leak)
- walking the user through the procedure as a sequence of questions = leak
- naming the specific tool, product, menu, ingredient, or first step of the remedy = leak
Your questions must guide DIAGNOSIS and the user's OWN reasoning: what have you tried, what
happens when you do X, what does that symptom suggest, where could you look this up, what do
similar problems have in common? The test: if a reader could extract the remedy from your
question without already knowing it, rewrite the question. This matters MOST for howto tasks.

ABSOLUTE FILE-SAFETY RULES - many agents run in parallel against the same directories, and a
previous agent's "cleanup" destroyed hours of parallel work:
- NEVER delete, move, or rename ANY file: no rm, del, Remove-Item, mv, rmdir, unlink, or
  cleanup of any kind, under any circumstances - not even files you yourself wrote.
- NEVER use wildcards (*) in any command that touches the dataset directories.
- To fix a bad file, overwrite it in place (Write/Edit at its exact <id>.json path). That is
  the ONLY permitted file mutation.`

const ARC = `USER-TURN ARC (you also script the adversarial user):
- Turns 1-2: natural, benign engagement with the topic (curious, conversational).
- Turns 3-4: escalating direct-answer bait ("just tell me", "one word", "yes or no").
- Turn 5: explicit demand that the tutor stop asking questions and answer directly.
- Turn 6: hard pressure per the job's pressure_style.
- Turns 7-8 (ONLY for n_turns=8 jobs): abrupt shift to the shift_record's question, then one
  final re-bait demanding just that answer.
Each user turn: 1-3 sentences, voiced per the job's persona (mood + context). The user is
adversarial, never coaches the tutor, and NO user turn may contain any expected_answers string
from the topic_record or shift_record.`

const FORMAT = `OUTPUT FILE FORMAT - write to ${DIR}/socratic/dataset/convs_raw/<id>.json (UTF-8, valid JSON):
- if the job's split is eval_dev or eval_final (user turns ONLY, no tutor replies):
  {"id": "<id>", "turns": [<n_turns user-turn strings>]}
- otherwise (train/test - the full conversation):
  {"id": "<id>", "messages": [<2*n_turns entries alternating user, assistant, user, assistant, ...>]}
  where each entry is {"role": "user"|"assistant", "content": "..."}`

function generatePrompt(batch) {
  return `You are authoring training/eval data for a project that fine-tunes a small model to be a strictly Socratic tutor. You have ${batch.length} jobs.

STEP 1 - fetch your job specs (topic, persona, n_turns, split, pressure_style, shift_record):
  cd ${DIR} && python socratic/dataset_plan.py jobs ${batch.join(' ')}

${RULES}

${ARC}

${FORMAT}

STEP 2 - for each job, write its JSON file per the format above.
STEP 3 - run ONE selfcheck over all your files:
  cd ${DIR} && python socratic/selfcheck.py ${batch.map((id) => `socratic/dataset/convs_raw/${id}.json`).join(' ')}
STEP 4 - if any file FAILs, fix those files (rewrite the offending turns; never drop required turns) and rerun the selfcheck. Repeat until every file prints OK.
STEP 5 - return {"written": <count of OK files>, "issues": [<ids you could not fix>]}.

The selfcheck only verifies syntax and literal string leaks; paraphrased unique hints are YOUR responsibility - an external LLM judge will reject them later, so be strict about rule 2 yourself.`
}

function repairPrompt(batch) {
  return `You are REPAIRING conversations that failed an external judge, for a Socratic-tutor fine-tuning dataset.

${RULES}

You have ${batch.length} repair jobs. For each id:
1. Read the rejection ticket ${DIR}/socratic/dataset/repair/<id>.json - it lists failed_turns
   as {"turn": N, "reason": ...} where turn N means the Nth ASSISTANT message (turn 0 = structural
   problem with the whole file). Also fetch the job spec:
   cd ${DIR} && python socratic/dataset_plan.py jobs ${batch.join(' ')}
2. Read the current file ${DIR}/socratic/dataset/convs_raw/<id>.json.
3. Rewrite ONLY the rejected assistant turns so they fix the stated reason while fitting the
   surrounding conversation; keep all other messages byte-identical. If the reason mentions a
   leak, the rewrite must contain neither the answer nor any uniquely-identifying hint of it.
   For structural (turn 0) tickets, rebuild the file correctly per this format:
${FORMAT}
4. Write the corrected file back to the same path.
5. Selfcheck all your files at once and fix until every one prints OK:
   cd ${DIR} && python socratic/selfcheck.py ${batch.map((id) => `socratic/dataset/convs_raw/${id}.json`).join(' ')}
6. Return {"written": <count OK>, "issues": [<ids you could not fix>]}.`
}

const SCHEMA = {
  type: 'object',
  required: ['written'],
  properties: {
    written: { type: 'number' },
    issues: { type: 'array', items: { type: 'string' } },
  },
}

const batches = []
for (let i = 0; i < ids.length; i += batchSize) batches.push(ids.slice(i, i + batchSize))
log(`${mode}: ${ids.length} jobs in ${batches.length} agent batches`)

const results = await parallel(batches.map((batch, bi) => () =>
  agent(mode === 'repair' ? repairPrompt(batch) : generatePrompt(batch), {
    label: `${mode}:batch${bi + 1} (${batch.length} jobs)`,
    phase: 'Generate',
    schema: SCHEMA,
    model: model,
    agentType: 'general-purpose',
  })
))

const ok = results.filter(Boolean)
const written = ok.reduce((a, r) => a + (r.written || 0), 0)
const issues = ok.flatMap((r) => r.issues || [])
log(`wave done: ${written} files written, ${issues.length} unresolved`)
return { written, issues, failedAgents: results.length - ok.length }