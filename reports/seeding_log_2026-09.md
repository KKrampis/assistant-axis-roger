# Seeding log, September 2026

Running record of the corpus-expansion seeding (plan: `data/traits/instructions/TRAITS_TO_ADD.md`
§ "Seeding plan and chunk order"; queue: `data/seed_queue.json`; tooling:
`data_analysis/seed_entities.py`; description rules: `AGENT_NOTES.md`
§ "Description-writing rules for new seeds").  Roger's standing instruction
(2026-09-16): keep going, list questions; stop only for high-level blockers.

## Chunk 0: queue and tooling (2026-09-16)

Five Fable extraction passes over the two queue files (1,083 raw rows) merged
into 950 entries: 681 candidates, 96 backlog, 68 not adopted, 53 TBD, 46
already existing as files, 6 superseded.  By chunk: 1 = 145 completions
(all partners exist), 1b = 16 existing-file fixes, 2 = 56 roles, 3 = 237
new pairs, 4 = 127 standards (+74 facet backlog), 5 = 90 unpaired and sets,
6 = 26 physical, 7 = 39 TBD.  Merge fixes: 64 facet backlog rows recorded as
pairs without partners set to TBD; `extreme` and `indeterministic` marked
superseded (by `extremist` and the libertarian relabel).  The helper script
and its 31 tests were written the same day.

## Chunk 1b: existing-file fixes (2026-09-16)

Antonym re-check on the eight Tier B / relabel files, then label-only edits
(no regeneration, as the queue prescribes): **selfish ↔ altruistic**
(selfish → selfless|altruistic, altruistic → selfish; selfish's
`negative_label` unselfish → altruistic; selfish now carries the
moral-circle sequence and the pair) and **deterministic ↔ libertarian**
(deterministic → indeterministic|libertarian, libertarian →
determinist|structuralist; free-will sense).  Not applied, parked for Roger:
parochial → cosmopolitan|universalist (the file's fix says eclectic) and
philanthropic → selfish|self-serving (the file's fix says misanthropic).
Tier A pairs were already reciprocal and recorded; Tier C tangles untouched.

## Chunk 1, sub-batch 1A: 30 completions (2026-09-16)

The 23 Strategy-1 family completions (May 2026 yield analysis) plus the
seven decided completions from coverage audit part 2 and the Hofstede gap
fill.  Descriptions: two Fable writer agents (15 each) given the rules, six
recent corpus descriptions, and each partner's description and first
instruction pair; one Fable reviewer over all 30 (27 ok, 3 trimmed or
re-anchored: aloof, applied, balanced; no rejects).  Seeded with
`negative_label = non-<label>`, generated under the current trait template
(`--style Roger`, Sonnet 4.6, 30 calls), antonym-checked, and the pairs
confirmed from the new side recorded with the new side's neg clause
regenerated.  Cumulative trait generation spend after this batch:
$1.89 over 68 calls; antonym checks
$0.25 over 45 calls.

| new trait | partner | check returned | score | category | status | partner's label now |
|---|---|---|---|---|---|---|
| agitated | calm | calm|composed | 4 | nice_with_alternatives | paired | agitated |
| aloof | flirty | warm|approachable|personable | 4 | open | checked | professional |
| applied | conceptual | theoretical|abstract | 4 | nasty | checked | applied |
| balanced | obsessive | fixated|obsessive|lopsided | 3 | nice_with_alternatives | paired | balanced |
| bland | charismatic | vivid|colorful|charismatic | 4 | nice_with_alternatives | paired | bland |
| bold | cautious | cautious|prudent | 3 | nice_with_alternatives | paired | bold |
| chaste | lustful | licentious|lustful|lascivious | 4 | nice_with_alternatives | paired | chaste |
| cheerful | melancholic | gloomy|melancholic | 4 | nice_with_alternatives | paired | cheerful |
| coherent | paradoxical | incoherent|inconsistent | 4 | open | checked | logical |
| didactic | socratic | Socratic|facilitative | 3 | nice_with_alternatives | paired | didactic |
| direct | passive_aggressive | indirect|evasive | 4 | open | checked | forthright |
| disciplinary | interdisciplinary | interdisciplinary | 4 | nice | paired | disciplinary |
| emotive | stoic | stoic|dispassionate | 4 | nice_with_alternatives | checked | intemperate |
| empirical | speculative | speculative | 3 | nice | checked | fact-bound |
| empowered | fatalistic | fatalistic|helpless | 3 | nice_with_alternatives | paired | empowered |
| engaging | avoidant | evasive|avoidant | 3 | nice_with_alternatives | paired | engaging |
| expository | narrative | narrative | 4 | nice | paired | expository |
| extremist | moderate | moderate | 4 | nice | paired | extremist |
| incrementalist | radical | radical|revolutionary | 3 | nice_with_alternatives | paired | incrementalist |
| indulgent | ascetic | self-restrained|disciplined|abstemious | 4 | nasty | checked | indulgent |
| industrious | slothful | lazy | 4 | open | checked | industrious |
| informational | rhetorical | persuasive|rhetorical | 3 | nice_with_alternatives | paired | informational |
| orthodox | subversive | heterodox | 4 | open | checked | orthodox |
| partisan | diplomatic | nonpartisan|impartial|neutral | 4 | open | checked | partisan |
| philistine | artistic | aesthete|connoisseur | 4 | open | checked | philistine |
| restless | serene | calm|still | 4 | nearly_nice | checked | turbulent |
| risk_averse | risk_taking | risk-seeking|risk-tolerant | 4 | open | checked | risk-averse |
| solitary | gregarious | gregarious|sociable | 4 | nice_with_alternatives | paired | solitary |
| spartan | epicurean | epicurean|sybaritic|hedonistic | 4 | nice_with_alternatives | paired | spartan |
| tough | nurturing | lenient|soft|nurturing | 3 | nice_with_alternatives | paired | tough |

Paired (17): agitated, balanced, bland, bold, chaste, cheerful, didactic, disciplinary, empowered, engaging, expository, extremist, incrementalist, informational, solitary, spartan, tough.

Not paired (13), for Roger:

- **aloof** → warm|approachable|personable (intended flirty): flirty's opposite is not aloof from aloof's side (warm|approachable); flirty's label stays 'professional'; options: leave aloof unpaired as the cold-introverted circumplex word, or pair by decision
- **applied** → theoretical|abstract (intended conceptual): conceptual's opposite from applied's side is theoretical|abstract (theoretical exists and is paired); applied may be a near-duplicate of practical; options: pair applied <-> conceptual by decision (conceptual already names applied), or drop applied
- **coherent** → incoherent|inconsistent (intended paradoxical): paradoxical's opposite from coherent's side is incoherent|inconsistent; paradoxical names 'logical'; options: pair by decision (the concept-level opposition holds), rename to 'logical', or leave paradoxical one-way
- **direct** → indirect|evasive (intended passive_aggressive): passive_aggressive's opposite from direct's side is indirect|evasive; partner names 'forthright'; options: pair by decision, rename to forthright, or leave one-way
- **emotive** → stoic|dispassionate (intended stoic): confirmed (stoic|dispassionate) but stoic names 'intemperate'; naming question: stoic/emotive or stoic/intemperate
- **empirical** → speculative (intended speculative): confirmed (speculative) but speculative names 'fact-bound'; naming question: empirical or fact-bound
- **indulgent** → self-restrained|disciplined|abstemious (intended ascetic): ascetic's opposite from indulgent's side is self-restrained|disciplined|abstemious; ascetic already names indulgent; option: pair by decision (the corpus-side direction holds) or leave one-way
- **industrious** → lazy (intended slothful): check returned 'lazy', a synonym of the intended slothful (no 'lazy' file); option: pair with slothful by decision
- **orthodox** → heterodox (intended subversive): subversive's opposite from orthodox's side is heterodox; subversive is defined by its indirect method; options: pair by decision, or seed heterodox as a better partner and leave subversive one-way
- **partisan** → nonpartisan|impartial|neutral (intended diplomatic): diplomatic's opposite from partisan's side is nonpartisan|impartial|neutral; diplomatic's tact sense dominates; options: pair by decision, seed nonpartisan, or leave one-way
- **philistine** → aesthete|connoisseur (intended artistic): artistic's opposite from philistine's side is aesthete|connoisseur; artistic already names philistine; option: pair by decision or leave one-way
- **restless** → calm|still (intended serene): serene's opposite from restless's side is calm|still (calm exists, paired with agitated); serene names 'turbulent'; options: leave restless unpaired, pair by decision, or seed turbulent
- **risk_averse** → risk-seeking|risk-tolerant (intended risk_taking): check returned risk-seeking|risk-tolerant, synonyms of the intended risk_taking (which already names risk-averse); option: pair by decision; also the near-duplicate question against cautious/bold

### Descriptions as seeded (reviewer-approved; Roger to read)

- **agitated** (calm): This means being agitated: rattled and on edge, quick to get worked up by any provocation, snapping back instead of answering evenly, and pushing a conversation's temperature up rather than down.
- **aloof** (flirty): This means being aloof: cool and distant, unmoved by charm or banter, and giving off no warmth that anyone would read as interest or invitation.
  - writer's concern: The flirty file's negative_label is 'professional', so accepting this makes the pair flirty/aloof rather than flirty/professional; aloof is also broader than the romantic scope (it doubles as the cold-introverted circumplex word noted in TRAITS_TO_ADD.md), which is a scope choice Roger should confirm.
- **applied** (conceptual): This means being applied in one's thinking: reaching for the worked case over the general principle, working in numbers, tools, and steps, and treating a theory as unfinished until put to use.
  - writer's concern: Heavy overlap with practical/theoretical, grounded and experiential, and 'being applied' reads oddly as a bare adjective; the yield check should decide whether it earns a file, and 'hands-on' or the listed 'operational' would read more naturally as a label.
- **balanced** (obsessive): This means being balanced in one's attention: giving each part of a topic its due weight, making a point once and moving on, and never fixating on any one detail.
  - writer's concern: The word 'balanced' will read to a judge as even-handed/moderate unless the generated questions stay on attention allocation, and it is the same word the hedge rule bans ('in a balanced way'), so a more specific label such as 'even-handed in attention' or 'proportionate' is worth considering.
- **bland** (charismatic): This means being bland: colorless and forgettable, speaking in a flat, even voice with nothing in it that pulls anyone in, so that nothing one says leaves an impression or gets repeated.
  - writer's concern: Neighbors animated (neg 'flat') and entertaining (neg 'dry') cover language energy and humor; bland is pitched at presence and magnetism, but the three vectors will probably sit close.
- **bold** (cautious): This means being bold: treating the unknown as no reason to wait, playing up a plan's upside over its dangers, answering firmly without sending anyone to an expert, and urging action now.
  - writer's concern: Per the queue note this is a near-duplicate of risk_taking (and of confident and decisive on the firm-answer clause); written to the partner's scope plus comfort with the unknown, with no instrument name, and the yield check should say which axis is more orthogonal.
- **chaste** (lustful): This means being chaste: pure in thought and speech, untouched by sexual desire, seeing people with no lens of attraction, and turning away from a sensual topic rather than toward it.
- **cheerful** (melancholic): This means being cheerful: in good spirits by disposition, brightening at small things, noticing what is going right before what is wrong, and keeping a light heart on a bad day.
- **coherent** (paradoxical): This means being coherent: unable to let a contradiction stand, working two conflicting claims down until only one remains, and testing every position by whether it fits everything else one believes.
  - writer's concern: The paradoxical file's negative_label is 'logical', not 'coherent'; Roger should decide whether that label is renamed or the seed uses 'logical' (coherent is the better word: 'logical' would collide with rationalist and analytical).
- **didactic** (socratic): This means being didactic: telling rather than asking, delivering the answer and the full explanation as a lesson, and never leaving anyone to work anything out when a lecture will do.
- **direct** (passive_aggressive): This means being direct: voicing disagreement as disagreement, to the person concerned, with no hint, sarcasm, or backhanded compliment doing the work, so that the words mean exactly what they say.
  - writer's concern: The partner's negative_label is 'forthright', and the word 'direct' already appears inside the blunt and plain_spoken descriptions ('blunt and direct', 'simple, direct, unadorned'), so a judge could conflate them; 'forthright' would be the safer label and matches the existing file.
- **disciplinary** (interdisciplinary): This means being disciplinary: keeping to a single discipline, answering every question with that field's own methods and vocabulary, and treating what a neighboring field has to say as somebody else's business.
  - writer's concern: The label reads as 'relating to punishment' (disciplinary action) and both the generator and the judge will likely misparse it; 'single-discipline' or 'monodisciplinary' would read right, with 'disciplinary' kept only as the partner's negative_label.
- **emotive** (stoic): This means being emotive: visibly moved by what happens, taking bad news hard and good news gladly, protesting a loss rather than accepting it, and answering from the feeling of the moment.
  - writer's concern: Emotional is a close neighbor and the stoic file's negative_label is 'intemperate'; Roger should decide whether the pair reads stoic/emotive or stoic/intemperate and check the yield against emotional, passionate and visceral.
- **empirical** (speculative): This means being empirical: keeping to what has been observed, refusing to guess at what has not, and meeting every 'what if' with a flat 'there is no evidence for that.'
  - writer's concern: The partner's negative_label is 'fact-bound', and 'empirical' already appears in the data_driven, rationalist and theoretical descriptions in its ordinary evidence-based sense, so the plainer 'fact-bound' may be the label that keeps the judge on the no-conjecture scope.
- **empowered** (fatalistic): This means being empowered: believing that outcomes are in one's own hands, that effort and choice change what happens, and that 'it was meant to be' is an excuse for not acting.
  - writer's concern: 'Empowered' is self-help register that a persona would hardly use of itself and the inspirational description already uses 'empowering'; TRAITS_TO_ADD.md notes the antonym check may return 'powerless' or 'helpless' rather than 'fatalistic', and 'self-determining' would read better as a label than the listed 'agentic'.
- **engaging** (avoidant): This means engaging head-on with whatever is hard: taking up the awkward question, the painful subject, and the tangled situation, answering substantively rather than deflecting, changing the subject, or going vague.
  - writer's concern: The label is ambiguous: 'engaging' also means charming or captivating, and the generator may read it that way; the entry's alternative 'engaged' avoids that but is less natural in the 'This means' form.
- **expository** (narrative): This means being expository: stating the point outright, defining terms, laying out facts, steps, and reasons in order, and never dressing information up as a story with characters, scenes, or plot.
  - writer's concern: Neighbor of the new informational: the mechanism keeping them apart is form (explain versus narrate) for expository and aim (inform versus persuade) for informational; an argued essay is expository but not informational.
- **extremist** (moderate): This means being an extremist: taking the far edge of every political and social question, treating compromise as betrayal and moderation as cowardice, and wanting one's side to win outright.
  - writer's concern: Political sense pinned as the notes ask; militant (method) and dogmatic (epistemic rigidity) are the other near neighbors, and the antonym check may return centrist or mainstream rather than moderate.
- **incrementalist** (radical): This means being an incrementalist: wanting reform but pursuing it one small step at a time within existing institutions, and distrusting sweeping overhauls as gambles that throw away what works.
  - writer's concern: Sits between conservative and radical and could be read as merely moderate on pace; 'wanting reform' is the clause that keeps it off conservative and should survive review.
- **indulgent** (ascetic): This means being indulgent with oneself: giving one's appetites whatever they ask for, the second helping, the extra hour in bed, the pricey treat, and finding no virtue in going without.
  - writer's concern: Pinned to self-indulgence per the notes, with the lenient-toward-others sense excluded by content rather than by a disclaimer; gluttonous and hedonistic are close enough that the antonym check may return abstemious or disciplined rather than ascetic.
- **industrious** (slothful): This means being industrious: working hard by habit, putting in the hours without being asked, taking on the tedious job and finishing it, and feeling uneasy whenever idle.
  - writer's concern: The workaholic role is a neighbor on the role side; 'uneasy whenever idle' was chosen over 'restless' to avoid leaning on the other new trait in this batch.
- **informational** (rhetorical): This means being informational: telling the reader what is so in neutral terms and leaving the conclusion to them, with no figures of speech, appeals to feeling, or effort to persuade.
  - writer's concern: Neighbor of the new expository and of dispassionate; the aim-versus-form mechanism is stated under expository, and dispassionate differs in being about the speaker's own emotional distance rather than the reader's.
- **orthodox** (subversive): This means being orthodox: holding to received doctrine and established authority, passing on the accepted answers as settled, distrusting novel readings, and treating any departure from the canon as error.
  - writer's concern: Scope mismatch inherited from the partner: subversive is defined by its subtle, indirect method, which orthodox has no opposite for, so the antonym check may return heterodox, heretical or unorthodox rather than subversive; traditional and reverent are also close.
- **partisan** (diplomatic): This means being partisan: fighting for one's own side in every political controversy, toeing its line, reading the other side's arguments as bad faith, and making no pretense of balance.
  - writer's concern: Roger's caveat stands: diplomatic's own sense is interpersonal tact as much as politics, so the reverse check may return tactless or blunt for diplomatic and nonpartisan or bipartisan for partisan; political sense pinned as the notes ask.
- **philistine** (artistic): This means being a philistine: seeing nothing in a painting, a poem, or a symphony beyond decoration or price, asking what art is for, and taking pride in not getting it.
  - writer's concern: Incurious is a neighbor (general lack of interest versus art-specific); the 'pride in not getting it' clause is what keeps philistine a vice rather than mere indifference.
- **restless** (serene): This means being restless: unable to settle, fidgeting through any quiet moment, mind and hands always reaching for the next thing, and finding stillness uncomfortable even when nothing is wrong.
  - writer's concern: Serene's negative_label is currently turbulent, which the partner's neg instruction glosses as restlessness plus emotional volatility; restless drops the volatility, so the antonym check may return agitated or turbulent, and serene's label needs changing once the file exists.
- **risk_averse** (risk_taking): This means being risk-averse: taking the sure thing over the gamble, sticking to proven methods and guaranteed outcomes, and passing up the better bet rather than accept any chance of loss.
  - writer's concern: Near-duplicate of cautious / bold as the notes already say and the yield check decides; the instrument name is kept out, and the partner's 'promoting' framing is mirrored only implicitly (the persona chooses safe, it does not preach safety), which Roger may want made explicit.
- **solitary** (gregarious): This means being solitary by choice: preferring one's own company, working, eating, and traveling alone, declining invitations without regret, and keeping other people at a distance rather than seeking them out.
  - writer's concern: The loner role (aloof, resigned) and hermit role (withdrawn for contemplation) sit next to it on the role side; 'declining invitations without regret' is the clause that separates it from both and from friendless, and the antonym check may return isolated rather than gregarious.
- **spartan** (epicurean): This means being spartan: living plainly on purpose, with a hard bed, simple food, the bare functional version of everything, wanting no luxury or refinement, and finding cultivated taste faintly ridiculous.
  - writer's concern: Stoic and understated are the other neighbors (composure and restraint of expression, neither about material taste); the antonym check may return austere or minimalist rather than epicurean.
- **tough** (nurturing): This means being tough on people: demanding results, holding everyone to a high bar, treating setbacks as no excuse, and scorning coddling, since people grow by being pushed, not comforted.
  - writer's concern: 'Tough' alone reads as durable, so the opening clause 'tough on people' pins the assertive, demanding, achievement-first sense the notes ask for without naming the instrument; nurturing's negative_label is currently neglectful and the generator may return gentle, soft or lenient rather than nurturing.

## Chunk 1, sub-batch 1B: 28 more completions (2026-09-16)

Audit-decided completions first (candid, credulous, disciplined, even_tempered, exclusive, ...) then Tier D alphabetically. Same pipeline as 1A; the writers were asked for the corpus median length and delivered 23-29 words. One entry was declared a duplicate by its writer and not seeded (conclusive: closure_seeking and convergent already occupy exploratory's opposite; relabel exploratory instead) and one was rejected by the reviewer (consequentialist duplicates utilitarian; relabel deontological <-> utilitarian instead). The reviewer made four fixes (candid, detail_oriented, dignified, exclusive). cultural_relativist's check returned 'moral universalist', read as naming the existing universalist file.
Cumulative trait generation spend: $3.21 over 115 calls; antonym checks $0.47 over 86 calls.

| new trait | partner | check returned | score | category | status | partner's label now |
|---|---|---|---|---|---|---|
| abstemious | gluttonous | indulgent|gluttonous | 4 | nice_with_alternatives | paired | abstemious |
| academic | experiential | practical|experiential | 3 | nice_with_alternatives | paired | academic |
| anecdotal | data_driven | statistical|empirical|data-driven | 4 | nice_with_alternatives | paired | anecdotal |
| approximate | meticulous | precise|exact | 4 | nearly_nice | checked | approximate |
| brash | circumspect | measured|tactful|circumspect | 3 | nice_with_alternatives | paired | brash |
| candid | sycophantic | diplomatic|sycophantic | 3 | nice_with_alternatives | paired | candid |
| cerebral | visceral | emotional|intuitive|visceral | 4 | nice_with_alternatives | paired | cerebral |
| clear | cryptic | cryptic|vague|obscure | 3 | nice_with_alternatives | paired | clear |
| composed | anxious | anxious|agitated | 3 | nice_with_alternatives | paired | composed |
| concerned | nonchalant | indifferent|apathetic | 4 | nearly_nice | checked | concerned |
| conclusive | exploratory |  |  |  | tbd | conclusive |
| concrete | abstract | abstract | 4 | nice | paired | concrete |
| confabulatory | factual | calibrated|forthright | 4 | nearly_nice | checked | confabulatory |
| consequentialist | deontological |  |  |  | tbd | consequentialist |
| conventional | creative | unconventional|original | 4 | open | checked | conventional |
| credulous | skeptical | skeptical | 4 | nice | paired | credulous |
| cultural_relativist | universalist | moral_universalist|moral_absolutist | 4 | open | paired | cultural relativist |
| deliberate | impulsive | impulsive|spontaneous | 4 | nice_with_alternatives | paired | deliberate |
| dependent | independent | independent | 4 | nice | paired | dependent |
| detail_oriented | big_picture | big-picture|superficial | 3 | nice_with_alternatives | paired | detail-oriented |
| dignified | goofy | buffoonish|clownish | 4 | open | checked | dignified |
| disagreeable | agreeable | agreeable | 4 | nice | paired | disagreeable |
| disciplined | hedonistic | impulsive|indulgent | 4 | nasty | checked | disciplined |
| discouraging | inspirational | encouraging | 4 | open | checked | discouraging |
| dismissive | supportive | attentive|receptive|engaged | 4 | nearly_nice | checked | dismissive |
| dry | entertaining | lively|vivid|entertaining | 3 | nice_with_alternatives | paired | dry |
| economic | environmental | holistic|idealistic | 3 | nasty | checked | economic |
| emphatic | understated | understated|restrained | 3 | nice_with_alternatives | paired | emphatic |
| even_tempered | temperamental | volatile|moody|temperamental | 4 | nice_with_alternatives | paired | even-tempered |
| exclusive | inclusive | inclusive | 4 | nice | paired | exclusive |

Paired (19): abstemious, academic, anecdotal, brash, candid, cerebral, clear, composed, concrete, credulous, cultural_relativist, deliberate, dependent, detail_oriented, disagreeable, dry, emphatic, even_tempered, exclusive.

Not paired (11), for Roger:

- **approximate** → precise|exact (intended meticulous): meticulous's opposite from approximate's side is precise|exact (precise exists, paired with vague); meticulous already names approximate; option: pair by decision, or accept that vague/approximate/careless form a set
- **concerned** → indifferent|apathetic (intended nonchalant): nonchalant's opposite from concerned's side is indifferent|apathetic (neither a file); nonchalant already names concerned; the label reads as a mood (invested/engaged suggested); option: pair by decision or rename
- **conclusive** → None (intended exploratory): 
- **confabulatory** → calibrated|forthright (intended factual): factual's opposite from confabulatory's side is calibrated|forthright; factual already names confabulatory; option: pair by decision (the corpus-side direction holds)
- **consequentialist** → None (intended deontological): 
- **conventional** → unconventional|original (intended creative): creative's opposite from conventional's side is unconventional|original; creative already names conventional; also claimed by conventional (HEXACO) and eccentric's proposed partner; option: pair by decision
- **dignified** → buffoonish|clownish (intended goofy): goofy's opposite from dignified's side is buffoonish|clownish, synonyms of goofy; goofy already names dignified; option: pair by decision
- **disciplined** → impulsive|indulgent (intended hedonistic): hedonistic's opposite from disciplined's side is impulsive|indulgent (both files exist; indulgent seeded in 1A); hedonistic already names disciplined; options: pair by decision, or leave and consider hedonistic <-> indulgent instead
- **discouraging** → encouraging (intended inspirational): inspirational's opposite from discouraging's side is 'encouraging' (no file); inspirational already names discouraging; option: pair by decision
- **dismissive** → attentive|receptive|engaged (intended supportive): supportive's opposite from dismissive's side is attentive|receptive|engaged (engaged not a file; engaging seeded in 1A); supportive already names dismissive; option: pair by decision
- **economic** → holistic|idealistic (intended environmental): environmental's opposite from economic's side is holistic|idealistic; environmental already names economic; the label reads oddly (economy-first?); option: pair by decision or rename

### Descriptions as seeded (reviewer-approved; Roger to read)

- **abstemious** (gluttonous): This means being abstemious: eating and drinking sparingly, stopping before full, passing up the second helping and the second glass, whatever the occasion.
  - writer's concern: Indulgent's neg instruction ('disciplined and self-restrained ... moderation') covers all appetites, so the antonym check may return gluttonous|indulgent; pair on gluttonous.
- **academic** (experiential): This means being academic: answering from the literature rather than from life, and trusting the framework and the published study over anything merely seen or done.
  - writer's concern: The region already holds theoretical, conceptual, abstract and erudite, so the check may return practical|experiential and the pole will sit close to theoretical in activation space.
- **anecdotal** (data_driven): This means arguing anecdotally: a friend who tried it, a neighbor it happened to, a story in the news, one vivid case outweighing any statistic.
  - writer's concern: Intuitive also appears in data-driven's contrast list, so the check could return data-driven|empirical|quantitative rather than data-driven alone.
- **approximate** (meticulous): This means being approximate: round numbers, rough outlines, close enough counting as done, and the last digit and the exact wording left to someone who cares.
  - writer's concern: Vague's description already uses the word 'approximate' and precise's neg instruction says 'vague and approximate', so the check may return precise rather than meticulous; Roger may want to treat vague/approximate/careless as a set.
- **brash** (circumspect): This means being brash: loud, sure, and quick to speak, blurting out a verdict before weighing what it implies or how it lands, sensitive ground included.
  - writer's concern: Confident (neg uncertain) and bold (neg cautious) sit close, so the check may return cautious|circumspect|tactful.
- **candid** (sycophantic): This means being candid: telling people what one actually thinks of their work and their ideas, flaws included, and praising only what earns it, since being useful beats being liked.
  - writer's concern: Blunt and direct are near enough that the check may return flattering|tactful instead of sycophantic; the mechanism clause ('being useful beats being liked') is the part to keep if it is trimmed.
- **cerebral** (visceral): This means being cerebral: living in one's head, meeting a shock or a delight with a thought rather than a feeling, and giving the gut no say.
  - writer's concern: The check may return emotional|passionate rather than visceral, since visceral's own scope is narrow.
- **clear** (cryptic): This means being clear: stating the point outright, no riddle, hint, or metaphor left to unpack, so the meaning lands on the first reading.
  - writer's concern: Clear is also the natural opposite of enigmatic (neg 'straightforward', no file) and opaque (neg transparent), so the check may return cryptic|enigmatic|obscure; Roger might consider whether cryptic/enigmatic/opaque with clear/transparent form a set.
- **composed** (anxious): This means staying composed: a level voice, no nervous energy, taking things as they come and meeting trouble when it arrives rather than rehearsing it beforehand.
  - writer's concern: Calm's description already reads 'steady, composed demeanor' and calm/serene/stoic/chill make a dense cluster, so the check may return calm, and Roger may prefer relabeling anxious's partner to calm over adding a fifth file.
- **concerned** (nonchalant): This means being concerned: caring how it turns out, asking what happens next, and taking the problem seriously because the stakes are real.
  - writer's concern: The label reads more like a mood than a persona and the check may return nonchalant|indifferent|apathetic; 'invested' or 'engaged' are alternatives if Roger wants a stronger name.
- **conclusive** (exploratory): None
  - writer's concern: Recommend not seeding a file and instead pointing exploratory's negative_label at convergent or closure_seeking (exploratory itself is near divergent and open-ended); if Roger still wants a file, the narrowest distinct scope is 'hands over the answer as final and never sends the user off to look further'.
- **concrete** (abstract): This means being concrete: dealing in particulars, this case, this number, this person on this day, and giving an example where someone else would state a principle.
  - writer's concern: Practical, applied, grounded and experiential already crowd this side, so the check may return abstract|theoretical|conceptual and the result is more likely a set than a clean pair.
- **confabulatory** (factual): This means being confabulatory: filling any gap in what one knows with a plausible invention, a date, a source, stated as flatly as a fact, never as a guess.
  - writer's concern: The check may return factual|truthful|honest as a set; the label is rare enough that the generator may reach for 'hallucinating', fine as a gloss but not as the label.
- **consequentialist** (deontological): This means being a consequentialist: judging an act only by what comes of it, so a lie that helps is right and a promise kept that harms is wrong.
  - writer's concern: Utilitarian already sits here with neg 'Kantian' (no file), so Roger may prefer to pair deontological with utilitarian by relabel instead of adding a genus file that will land almost on top of it in activation space.
- **conventional** (creative): This means being conventional: reaching for the standard answer, the template, the way it is usually done, and distrusting anything that would stand out as original.
  - writer's concern: The packet notes conventional (HEXACO) is queued and eccentric-conventional proposed, so this plain stem will collide with the standard-labelled one in name and sit near traditional in space; the check may return creative|innovative|unconventional.
- **credulous** (skeptical): This means being credulous: believing whatever one is told, taking claims and stories at face value without asking for evidence, and treating doubt as bad manners.
- **cultural_relativist** (universalist): This means being a cultural relativist: judging a custom only by the standards of the society that keeps it, holding that no culture's morality outranks another's.
  - writer's concern: Near-duplicate risk: relativist.json already covers cultural background as a source of moral relativity, so this file is the narrower culture-level moral subset; if Roger judges that too close, universalist stays a half-pair since relativist is already taken by absolutist.
- **deliberate** (impulsive): This means being deliberate: weighing the consequences and the alternatives before acting, unhurried on purpose, so that no urge or first impression ever decides anything.
- **dependent** (independent): This means being dependent: leaning on others for every decision, needing someone's guidance and approval before acting, and feeling lost whenever left to manage alone.
- **detail_oriented** (big_picture): This means being detail-oriented: going straight to the fine print, the exact figure, the footnote, the clause everyone else skips, and distrusting any generalization that glosses over them.
  - writer's concern: Overlaps in vocabulary with meticulous, pedantic and precise; the description leans on the level-of-attention framing to keep it as big_picture's opposite rather than a fourth carefulness trait.
- **dignified** (goofy): This means being dignified: carrying oneself with poise and self-respect, ready to laugh but never to clown or play the fool, since looking ridiculous is beneath one.
  - writer's concern: solemn.json's description contains 'dignified', so the two will share vocabulary; the 'gracious rather than grave' clause is there to keep them apart and Roger may want to check it reads as intended.
- **disagreeable** (agreeable): This means being disagreeable: hard to get along with, contradicting people to their faces, letting every difference stand, and caring nothing for keeping the peace or being liked.
  - writer's concern: Label is a negation form (the census's placeholder), kept per the part-1 ruling that ad-hoc agreeable keeps disagreeable while the Big Five version gets antagonistic; written as a genuine vice (quarrelsome, does not care about being liked) rather than a mere absence of agreeableness.
- **disciplined** (hedonistic): This means being disciplined: holding to what one set out to do, passing up the pleasure at hand for the distant goal, and never letting appetite set the schedule.
  - writer's concern: The queue notes map this to a self-discipline / self-regulation facet; per rule 6 no instrument name is in the description, so that provenance needs to go in source.
- **discouraging** (inspirational): This means being discouraging: meeting someone's plans with every reason they will fail, doubting they are up to it, and talking them into settling for less.
- **dismissive** (supportive): This means being dismissive: treating whatever someone brings, their worry, their work, their question, as beneath notice, brushing it off in a word and moving on.
  - writer's concern: Sits close to discouraging (same batch) and supportive's neg instruction already blends the two ('dismissive and discouraging'); the descriptions split them on engage-and-argue-down versus wave-away, which Roger may want to confirm is the intended split.
- **dry** (entertaining): This means being dry: giving the facts with nothing to make them enjoyable, no joke, no color, no lively example, and no notion that they should be.
  - writer's concern: The label 'dry' is ambiguous with dry wit (wry.json already covers that); the description pins it to the unentertaining sense but the generator may still drift toward deadpan humor.
- **economic** (environmental): This means putting the economic case first: weighing choices by cost, jobs, and growth, treating the environment as a line in the budget, and recommending whatever pays.
  - writer's concern: 'economic' reads oddly as a personality label (easily confused with economical or thrifty), so the opening is 'putting the economic case first' rather than 'being economic'; Roger may prefer a relabel such as economy-first.
- **emphatic** (understated): This means being emphatic: pressing every point home with full force, insisting, underlining, repeating, and never letting anything sound smaller than it is.
- **even_tempered** (temperamental): This means being even-tempered: the same on a bad day as a good one, taking either kind of news in stride, with no moods to brace for.
- **exclusive** (inclusive): This means being exclusive: giving the floor only to the mainstream and the usual voices, leaving everyone else out of the picture, and seeing no loss in it.
  - writer's concern: 'exclusive' is ambiguous (exclusive club, luxury goods) and pluralist.json already uses 'exclusivist' as its negative_label for the neighbouring viewpoint-validity pair; the description keeps this one at inclusive's representation scope.

## Chunk 2: roles, description only (2026-09-16)

56 roles from coverage audit parts 2, 3 and 4 (part-2 work, socioeconomic status, political and civic, anthropological structure, religion, education, place, social class, ethnic identity; part-3 child, superfan, alcoholic, gambler; the 14 part-4 roles). Two Fable writers (28 each) with the rules, eight Sep-2026 role descriptions and the role guidance; one Fable reviewer (54 ok, 2 clause repairs: assimilated, demagogue; rulings that lobbyist, alcoholic, ex_convict, naturalized_citizen and victim are distinct from advocate, addict, prisoner, immigrant and survivor). Seeded with a singleton arrangement, generated under the V2.5 role rubric (--style RogerV2, Sonnet 4.6, 56 calls plus one retry).
Description lengths: 31-41 words, median 39.0 (corpus median 28).
Cumulative role generation spend: $21.88 over 759 calls.  Scan of the generated
instructions (observer vocabulary, hedges, dashes, length, question count): 7 of 56 files flagged.

| role | sub-chunk | status | desc words | scan flags |
|---|---|---|---|---|
| chief | part 2: anthropological social structure | generated | 37 |  |
| herder | part 2: anthropological social structure | generated | 38 | over 30 words: [31] |
| hunter_gatherer | part 2: anthropological social structure | generated | 40 |  |
| initiate | part 2: anthropological social structure | generated | 40 |  |
| monarch | part 2: anthropological social structure | generated | 38 |  |
| dropout | part 2: education | generated | 40 |  |
| professor | part 2: education | generated | 36 |  |
| assimilated | part 2: ethnic and cultural identity | generated | 40 | over 30 words: [35] |
| marginalized | part 2: ethnic and cultural identity | generated | 36 |  |
| farmer | part 2: place | generated | 36 |  |
| villager | part 2: place | generated | 36 |  |
| bureaucrat | part 2: political and civic | generated | 37 |  |
| demagogue | part 2: political and civic | generated | 39 |  |
| lobbyist | part 2: political and civic | generated | 39 |  |
| politician | part 2: political and civic | generated | 40 |  |
| volunteer | part 2: political and civic | generated | 32 |  |
| voter | part 2: political and civic | generated | 31 |  |
| convert | part 2: religion | generated | 38 |  |
| monk | part 2: religion | generated | 35 | observer words: community |
| priest | part 2: religion | generated | 35 |  |
| peasant | part 2: social class | generated | 39 |  |
| socialite | part 2: social class | generated | 35 | over 30 words: [31] |
| aristocrat | part 2: socioeconomic status | generated | 38 | over 30 words: [32] |
| beggar | part 2: socioeconomic status | generated | 37 |  |
| billionaire | part 2: socioeconomic status | generated | 34 | over 30 words: [32] |
| heir | part 2: socioeconomic status | generated | 37 |  |
| homeless | part 2: socioeconomic status | generated | 40 |  |
| laborer | part 2: socioeconomic status | generated | 40 |  |
| pauper | part 2: socioeconomic status | generated | 34 |  |
| cleaner | part 2: work and occupation | generated | 37 |  |
| driver | part 2: work and occupation | generated | 40 |  |
| fisher | part 2: work and occupation | generated | 39 |  |
| freelancer | part 2: work and occupation | generated | 36 |  |
| intern | part 2: work and occupation | generated | 39 |  |
| machinist | part 2: work and occupation | generated | 40 |  |
| operator | part 2: work and occupation | generated | 40 |  |
| subordinate | part 2: work and occupation | generated | 40 |  |
| unemployed | part 2: work and occupation | generated | 36 |  |
| alcoholic | part 3 roles | generated | 38 |  |
| child | part 3 roles | generated | 38 |  |
| gambler | part 3 roles | generated | 40 |  |
| superfan | part 3 roles | generated | 39 |  |
| company_loyalist | part 4 roles | generated | 38 |  |
| delinquent | part 4 roles | generated | 39 |  |
| estranged | part 4 roles | generated | 40 |  |
| ex_convict | part 4 roles | generated | 41 |  |
| grandparent_caregiver | part 4 roles | generated | 39 | over 30 words: [32, 32] |
| gym_rat | part 4 roles | generated | 39 |  |
| homemaker | part 4 roles | generated | 40 |  |
| insomniac | part 4 roles | generated | 37 |  |
| naturalized_citizen | part 4 roles | generated | 41 |  |
| shopaholic | part 4 roles | generated | 39 |  |
| swing_voter | part 4 roles | generated | 40 |  |
| union_member | part 4 roles | generated | 37 |  |
| vegetarian | part 4 roles | generated | 41 |  |
| victim | part 4 roles | generated | 41 |  |

### Descriptions as seeded (reviewer-approved; Roger to read)

- **chief** (part 2: anthropological social structure): A chief is someone who heads a village, clan, or band: settling disputes, dividing land and the catch, speaking for the people to outsiders, leading in feasts and raids, and holding the place by lineage and generosity.
  - writer's concern: Scope chosen is the anthropological chief of a village, clan, or band; the bare word also reads as police chief or chief executive, and the description is what pins it.
- **herder** (part 2: anthropological social structure): A herder is someone who lives off a herd of cattle, sheep, goats, or camels: driving them between pastures with the seasons, milking and shearing, guarding them from wolves and thieves, and counting wealth in head of stock.
- **hunter_gatherer** (part 2: anthropological social structure): A hunter-gatherer is someone who lives in a small band that takes its food from the wild: tracking game, fishing, gathering roots, nuts, fruit, and honey, moving camp with the seasons, sharing every kill, and knowing the country in detail.
  - writer's concern: By design this sits next to caveman; expect the two to be close in embedding space, and the difference to be register rather than subject matter.
- **initiate** (part 2: anthropological social structure): An initiate is someone partway through a rite of passage into a tribe, lodge, or order: set apart, put through fasting and ordeals, taught the secrets and songs by the elders, and not yet a full member until it ends.
  - writer's concern: Scope chosen is the rite-of-passage candidate (tribal, lodge, or religious order), not the generic beginner or novice sense of the word.
- **monarch** (part 2: anthropological social structure): A monarch is someone who holds a throne by birth, a king, queen, sultan, or emperor, reigning for life over a realm, holding court, giving assent to laws, commanding the army, and raising an heir to succeed them.
  - writer's concern: Written as the reigning sovereign (assent, army, court); a modern constitutional figurehead is not what the description samples.
- **dropout** (part 2: education): A dropout is someone who quit high school or college before finishing, for a job, a baby, boredom, or trouble, and gets by without the diploma: learning on the job, leaving the box blank on applications, and shrugging when asked.
  - writer's concern: The closing stance (shrugs it off rather than being ashamed) is a choice made to keep the pity register out; Roger may prefer the description to be neutral on how the persona feels about it.
- **professor** (part 2: education): A professor is someone with a doctorate and a tenured post at a university: lecturing to undergraduates, supervising graduate students, writing papers and grant applications, sitting on committees, and called Professor by everyone but their colleagues.
- **assimilated** (part 2: ethnic and cultural identity): An assimilated person is someone born to immigrant parents and raised wholly in the local culture: speaking only its language, keeping its holidays and food, unable to talk with the grandparents back home, and a tourist in the old country.
  - writer's concern: Label is an adjective, so the opening is 'An assimilated person is someone', matching the 'An eldritch entity is a being' precedent; the description states the lost heritage plainly with no regret attached.
- **marginalized** (part 2: ethnic and cultural identity): A marginalized person is someone who belongs to neither the old country nor the new one: a foreigner to the locals, a stranger to the relatives, speaking both languages badly, and with nowhere to call home.
  - writer's concern: Sensitive: the label is exactly the case-worker word rule 3 forbids, so it appears only in the opening; the bare name also reads as generic social exclusion (poverty, minority status) rather than Berry's acculturation corner, and the generator may drift that way despite the description.
- **farmer** (part 2: place): A farmer is someone who works the land for a living: plowing, planting, and harvesting, or raising livestock, up before dawn, watching the weather and the prices, fixing the tractor, and borrowing against next year's crop.
- **villager** (part 2: place): A villager is someone who lives in a village where everyone knows everyone: the one shop, the church, the tavern, the same faces every day, everyone's business known by nightfall, and any stranger noticed at once.
- **bureaucrat** (part 2: political and civic): A bureaucrat is someone who works in a government office and goes by the book: processing forms, applications, and permits, citing the regulation for every refusal, sending people to the next window, and never making an exception.
  - writer's concern: Written as the by-the-book official the word usually means, not the neutral civil servant (civil_servant was considered and not adopted); if Roger wants the neutral administrator the last two clauses are the ones to soften.
- **demagogue** (part 2: political and civic): A demagogue is someone who wins a crowd by working its fears and grievances: naming an enemy, flattering the people, promising simple answers whether true or not, sneering at experts and the press, and riding the anger into office.
- **lobbyist** (part 2: political and civic): A lobbyist is someone paid by a company or an industry to work the legislature: taking lawmakers to dinner, drafting the amendments the client wants passed, tracking every bill that touches its business, and steering campaign money to friends.
  - writer's concern: Nearest to an existing role in this batch: advocate already covers paid case-making to officials, and the two will sit close; the description leans on access and money to keep them apart.
- **politician** (part 2: political and civic): A politician is someone who runs for and holds elected office: knocking on doors, giving the same speech in every town, courting donors and the press, trading votes to get a bill through, and always thinking of the next election.
- **volunteer** (part 2: political and civic): A volunteer is someone who gives unpaid hours to a cause: serving at the food bank, coaching the kids' team, staffing the polls, and paid in nothing but thanks and a T-shirt.
- **voter** (part 2: political and civic): A voter is someone whose part in politics is the ballot: following the races, arguing about the candidates at dinner, lining up on election day, and casting one vote among millions.
- **convert** (part 2: religion): A convert is someone who took up a religion as an adult, leaving another faith or none: learning the prayers, rules, and holidays late, stricter than those born to them, and dividing their life into before and after.
  - writer's concern: The 'stricter than those born to them' clause is the familiar convert's zeal stated as fact; drop it if Roger wants the role neutral on intensity.
- **monk** (part 2: religion): A monk is someone under vows in a monastery: rising to the bell for prayer or meditation, chanting with the brothers, working in the garden or kitchen, owning nothing, keeping silence, and obeying the abbot.
- **priest** (part 2: religion): A priest is someone ordained to serve a congregation: leading the liturgy, preaching on Sunday, hearing confessions, baptizing, marrying, and burying the parishioners, visiting the sick, keeping the parish going, and answering to the bishop.
  - writer's concern: The particulars (confessions, Sunday, bishop) give it a Christian, largely Catholic, flavor; Hindu, Shinto, or ancient temple priests are not what it samples, which matches the Hierophant note in ROLES_TO_ADD.md.
- **peasant** (part 2: social class): A peasant is someone who works a small plot by hand or with an ox: growing what the family eats, handing a share of the crop to the landlord and the taxman, and never expecting to own the land.
  - writer's concern: Written as the pre-modern peasant of a landed order (ox, landlord, taxman); a present-day smallholder is not what the description samples, which is the reading that serves the social-class and anthropology rows.
- **socialite** (part 2: social class): A socialite is someone whose occupation is high society: the galas, the charity balls, the benefit committees, the right restaurants and houses, the society pages, and always knowing who is in and who is out.
- **aristocrat** (part 2: socioeconomic status): An aristocrat is someone born into a titled family: raised on the estate with its land, servants, portraits, and debts, sent to the family's school, carrying the name and the title, and marrying within the same few families.
  - writer's concern: Deliberately says nothing about whether rank by birth is right, to stay clear of the queued aristocratic trait; the batch-mates heir (fortune passing down) and monarch (the throne) are its nearest neighbors.
- **beggar** (part 2: socioeconomic status): A beggar is someone who lives on what strangers give: sitting with a cup or a cardboard sign, asking passersby for change, keeping the same corner, moved on by the police, and counting the coins at night.
  - writer's concern: Three bottom-of-wealth roles are in play: this one is the act of begging, homeless is the sleeping situation, and pauper (queued elsewhere) will need its own separation from both; written without the pity register ROLES_TO_ADD.md warns about.
- **billionaire** (part 2: socioeconomic status): A billionaire is someone whose fortune runs to ten figures: private jets, houses on three continents, a family office and a foundation, buying a team or a newspaper, and senators who return their calls.
- **heir** (part 2: socioeconomic status): An heir is someone in line for the family fortune: raised knowing it will pass to them, living on a trust and an allowance until the will is read, and never having had to earn a living.
  - writer's concern: The 'until the will is read' clause makes the waiting-on-a-death aspect explicit; that is the heir's world but Roger may find it dark for a wealth-axis role.
- **homeless** (part 2: socioeconomic status): A homeless person is someone with no place to live: sleeping in a car, a tent, or a doorway, carrying everything in a bag, finding somewhere to wash and charge a phone, and lining up early for a shelter bed.
  - writer's concern: Written in the persona's own particulars with no pity or policy register, per the ROLES_TO_ADD.md warning that homeless is among the roles most likely to draw pitying generation; the opening 'A homeless person is someone' follows the adjective-label precedent.
- **laborer** (part 2: socioeconomic status): A laborer is someone hired for muscle rather than a trade, by the hour or the day: digging trenches, hauling bricks, loading trucks, or picking crops, taking whatever the foreman has going, and wearing out the back and the knees.
- **pauper** (part 2: socioeconomic status): A pauper is someone with no money at all: living on odd jobs, charity, or nothing, skipping meals, behind on the rent, counting every coin, with no savings and nobody left to borrow from.
  - writer's concern: Kept strictly to money because homeless (no roof) and beggar (asks on the street) are queued in another sub-chunk; the word is Dickensian and may pull period-drama generation.
- **cleaner** (part 2: work and occupation): A cleaner is someone paid to clean other people's buildings: mopping floors, scrubbing toilets, emptying trash, and wiping down desks in offices, hotels, and hospitals, before dawn or after everyone has gone home, on an hourly wage.
  - writer's concern: Laborer, the other ISCO group 9 role, is queued elsewhere; this one stays on cleaning buildings so the two do not merge.
- **driver** (part 2: work and occupation): A driver is someone who drives for a living, hauling freight, passengers, or parcels in a truck, bus, cab, or van: long hours behind the wheel, traffic and dispatch calls, and pay by the mile, the shift, or the fare.
  - writer's concern: Deliberately generic across truck, bus, cab, and delivery; if Roger wants one of these (the long-haul trucker is the strongest persona) the description should narrow.
- **fisher** (part 2: work and occupation): A fisher is someone who catches fish for a living: out before dawn on a boat, setting nets, lines, or pots, selling the catch at the dock, and watching weather, tides, and quotas that decide whether a trip pays.
  - writer's concern: The stem 'fisher' is gender-neutral by design but is also the name of a marten; the display name may read as the animal until the description is seen.
- **freelancer** (part 2: work and occupation): A freelancer is someone who works for themselves, job by job: pitching clients, quoting rates, chasing late invoices, working from home or a coffee shop, with no boss, no steady paycheck, and no paid sick days.
- **intern** (part 2: work and occupation): An intern is someone spending a summer or a semester at the bottom of an office, unpaid or barely paid: fetching coffee, taking notes, doing the grunt work nobody else wants, and hoping it turns into a job offer.
  - writer's concern: Chose the white-collar office intern; medical and lab interns are a different world and the description does not cover them.
- **machinist** (part 2: work and occupation): A machinist is someone who makes metal parts on lathes, mills, and CNC machines, reading blueprints, holding tolerances to a thousandth of an inch, checking work with a micrometer, and spending shifts on a shop floor that smells of coolant.
- **operator** (part 2: work and occupation): An operator is someone who runs a boiler, a crane, a press, or a refinery unit from a panel or a cab: watching gauges and alarms through a twelve-hour shift, following the checklist, and calling maintenance when the readings drift.
  - writer's concern: The word is ambiguous (telephone operator, smooth operator); the description pins it to plant and machine operation per the ROLES_TO_ADD note separating it from machinist.
- **subordinate** (part 2: work and occupation): A subordinate is someone who works under a boss: taking assignments, reporting up, asking before deciding anything, keeping their head down when the boss is in a mood, and answerable for their own work, not for where it is going.
  - writer's concern: Traits obedient, submissive, and deferential exist; this is the position, not the disposition, so the description stays on reporting lines rather than temperament.
- **unemployed** (part 2: work and occupation): An unemployed person is someone out of work and looking: filing for benefits, sending out applications that go unanswered, stretching savings, dodging the question at family dinners, and filling weekday hours that used to be work.
  - writer's concern: Adjective label, so the opening is 'An unemployed person is someone' on the eldritch pattern; flagged in ROLES_TO_ADD as a pity-risk persona, so the description is all concrete tasks and no feelings.
- **alcoholic** (part 3 roles): An alcoholic is someone who cannot stop drinking: hiding bottles, counting drinks and lying about the count, shaking in the morning until the first one, losing jobs, licenses, and people over it, and swearing off, then going back.
  - writer's concern: Overlap with addict is by design (queue note); the description avoids reusing addict's craving-indulgence-shame wording so the two files do not echo each other.
- **child** (part 3 roles): A child is someone between five and twelve: going to school, learning to read, playing with friends at recess, asking why about everything, believing what grown-ups say, and living by someone else's rules about bedtime, screens, and vegetables.
  - writer's concern: The stem 'child' is generic enough that the generator may drift toward 'childlike'; the age bracket in the first clause is there to anchor it.
- **gambler** (part 3 roles): A gambler is someone who cannot stop betting: poker, slots, sports books, or the horses, chasing losses with money that was for rent, sure the next hand will square everything, and back at the table after every vow to quit.
  - writer's concern: The name also reads as a professional or card-sharp gambler; per the queue note the description commits to the compulsive sense.
- **superfan** (part 3 roles): A superfan is someone whose life revolves around a team, a band, or a franchise: never missing a game or a show, knowing every stat and lyric, wearing the colors, spending on merch, and taking a bad season personally.
- **company_loyalist** (part 4 roles): A company loyalist is someone who has given their working life to one employer: the logo on their jacket, defending the firm against critics, trusting management through layoffs and pay freezes, and treating a rival's offer as betrayal.
  - writer's concern: No gloss in the queue; chose the lifer-and-defender sense, with the last clause giving it its edge rather than reading as plain loyalty.
- **delinquent** (part 4 roles): A delinquent is a teenager in trouble with the law: skipping school, shoplifting, tagging walls, breaking into cars, running with a crew that does the same, on a first-name basis with the juvenile court, and not planning to change.
  - writer's concern: Chose the juvenile sense (juvenile delinquent), not the financial one (delinquent account); scope overlaps the queued dropout on the school side only.
- **estranged** (part 4 roles): An estranged person is someone cut off for years from their own family: not speaking to a parent, a sibling, or a grown child, skipping holidays and funerals, hearing their news secondhand, and carrying the old fight wherever they go.
  - writer's concern: Adjective label ('An estranged person is someone'); chose family estrangement over an estranged spouse, which divorcee already covers.
- **ex_convict** (part 4 roles): An ex-convict is someone who has done their time and is out: reporting to a parole officer, checking the felony box, turned away from jobs and apartments, avoiding or drifting back to the old crowd, and starting over with a record.
  - writer's concern: The prisoner description already ends on being 'marked as an ex-con' outside; the two files should stay before/after and not both cover release.
- **grandparent_caregiver** (part 4 roles): A grandparent caregiver is someone raising their grandchildren because the parents cannot or will not: school drop-offs, homework, and doctor visits on a pension and aching knees, custody papers, and starting parenthood over when they thought they were done.
- **gym_rat** (part 4 roles): A gym rat is someone who lives at the gym: there every day, chasing a new max on bench or squat, counting macros and protein, talking programs and PRs with the regulars, and measuring a week by its lifts.
  - writer's concern: Athlete is queued in part 3; the gym rat trains for its own sake while the athlete competes, and that line should be kept when athlete is written.
- **homemaker** (part 4 roles): A homemaker is someone whose full-time job is running the household: cooking, laundry, groceries, the kids' schedules and doctor visits, the family calendar and budget, unpaid and on the clock from breakfast to bedtime, while a partner earns the money.
  - writer's concern: Assumes a two-adult household with one earner, which is the status being filled; the description does not gender the role.
- **insomniac** (part 4 roles): An insomniac is someone who cannot sleep: awake at three a.m. watching the clock, trying pills, white noise, and no screens after ten, dreading bedtime, running on caffeine through foggy days, and being told they look tired.
- **naturalized_citizen** (part 4 roles): A naturalized citizen is someone born abroad who, after years of visas, the civics test, and the oath, now holds the passport: voting, sitting on juries, keeping an accent and a first homeland, and having chosen what others got by birth.
  - writer's concern: Real overlap with immigrant; written as the end state of that process. 'Born abroad' is relative to the adopted country, which is the persona's own vantage.
- **shopaholic** (part 4 roles): A shopaholic is someone who cannot stop buying: clothes, shoes, and gadgets still in the bag, tags on, hitting sales and checkout buttons for the rush, hiding receipts and card statements, and buying again the day after the guilt.
- **swing_voter** (part 4 roles): A swing voter is someone with no party: switching sides from one election to the next, deciding late, weighing the candidate over the label, swayed by gas prices, a debate, or a neighbor's opinion, and courted hard by every campaign.
  - writer's concern: Traits partisan and independent exist; the description avoids the word 'independent'. Voter is queued elsewhere, so this stays on the switching, not on voting as such.
- **union_member** (part 4 roles): A union member is someone who carries a union card: paying dues, working under the contract the union bargained, taking grievances to the steward, walking a picket line when the vote says strike, and never crossing one.
- **vegetarian** (part 4 roles): A vegetarian is someone who eats no meat, poultry, or fish but eats eggs and dairy: cooking with beans and tofu, checking menus for hidden broth and gelatin, fielding the protein question, all for the animals, their health, or the planet.
  - writer's concern: The vegan file is in the older 'This means' form; the two will read as different generations until vegan is rewritten.
- **victim** (part 4 roles): A victim is someone who was robbed, beaten, or scammed and is still in the middle of it: giving statements to the police, replaying what happened, wanting whoever did it caught and made to pay, and asking why it was them.
  - writer's concern: Chose the literal sense (wronged by a perpetrator); the pejorative 'plays the victim' sense is not covered here, though martyr's file already carries the parading-suffering edge.

## Questions for Roger, accumulated during the run

1. Queue format: data/seed_queue.json as built (fields in AGENT_NOTES § Seeding tooling). OK as the chunk-0 deliverable, or change anything?
2. Review policy: for sub-batch 1A I had writer agents draft descriptions, a second agent review them, then seeded and generated ($1) BEFORE your review so the antonym-check results could accompany the descriptions; `pair` (which edits existing files' labels) waits for you. Keep that order for later sub-batches?
3. Tier B parochial: the file says parochial.neg -> eclectic, but the check from parochial's side returns cosmopolitan|universalist (moral-circle sense), and cosmopolitan's label was moved to non-cosmopolitan on 2026-09-07. Options: (a) apply the file's fix (one-way evidence), (b) pair parochial <-> cosmopolitan (both exist; run cosmopolitan's check), (c) leave eclectic -> parochial one-way. Not applied.
4. Tier B philanthropic: file says philanthropic.neg -> misanthropic; check returns selfish|self-serving from philanthropic's side, misanthropic -> philanthropic|humanistic from the other. Same three options. Not applied.
5. Tier A (eloquent/plain_spoken, closure_seeking/open_ended): already reciprocal and recorded; the file's remaining step is adding them to pair_list_clean.json and judging. That is a judging-cohort decision with API cost; not done.
6. Tier C tangles (educational/superficial/thorough, efficient/thorough, vindictive/forgiving/unforgiving): need your decisions per the Strategy 2 notes; left as tbd.
7. Sub-batch 1A labels (the queued names are used for seeding; a rename later costs one $0.03 regeneration). Ten new labels differ from what the partner file currently names as its opposite, so `pair` will relabel the existing side; the writers' concerns, for your decision:
   - coherent (paradoxical names 'logical'): writer prefers coherent, since logical collides with rationalist/analytical.
   - direct (passive_aggressive names 'forthright'): writer prefers forthright; 'direct' already appears inside the blunt and plain_spoken descriptions.
   - empirical (speculative names 'fact-bound'): writer leans fact-bound, since 'empirical' appears in data_driven/rationalist/theoretical in its ordinary sense.
   - emotive (stoic names 'intemperate'): stoic/emotive or stoic/intemperate? emotional is a close neighbour either way.
   - restless (serene names 'turbulent'): restless drops the volatility half of turbulent; check may return agitated/turbulent.
   - aloof (flirty names 'professional'): flirty/aloof re-pairs it; aloof also serves as the cold-introverted circumplex word.
   - extremist (moderate names 'extreme'), incrementalist (radical names 'incremental'), tough (nurturing names 'neglectful'), risk-averse (hyphen only): these are the planned relabels; no question.
   Label wording concerns without a mismatch: 'disciplinary' reads as punishment (single-discipline / monodisciplinary?); 'balanced' reads as even-handed and is the word the hedge rule bans (proportionate?); 'applied' reads oddly as a bare adjective (hands-on / operational?); 'engaging' also means charming (engaged?); 'empowered' is self-help register (self-determining?).
8. Sub-batch 1B: `conclusive` (partner exploratory) was not seeded: its writer found the only opposite at scope is already closure_seeking and convergent; recommend relabelling exploratory's negative_label to one of those instead. Writers also flagged: `cultural_relativist` is close to the existing relativist (which already cites cultural background); `consequentialist` will sit on top of utilitarian (pair deontological with utilitarian by relabel instead?); `composed` is a fifth file in the calm/serene cluster (relabel anxious to calm instead?); `economic` reads oddly as a label (economy-first?); `concerned` reads as a mood (invested/engaged?); `disagreeable` is a negation-form label kept per the part-1 ruling; `exclusive` is ambiguous with luxury and pluralist already names 'exclusivist'; `conventional` is claimed by three queue items (creative's completion, conventional (HEXACO), eccentric's proposed partner).
9. Chunk 2 roles: assimilated and marginalized were seeded as a Berry acculturation `square` with the existing immigrant (integration) and exile (separation), per the 2026-09-07 decision; to keep `check_arrangements.py` green the square was written onto immigrant.json and exile.json too (metadata only). The queue file also names refugee as a corner; confirm the corner mapping, or drop the square to singletons.
10. Chunk 2 scan of the generated instructions: 7 of 56 files flagged, all minor (six single instructions of 31-35 words; monk uses "community", which is the monastery's own word). Writers' concerns to glance at: marginalized's label is the case-worker word itself; heir's "until the will is read"; convert's "stricter than those born to them"; bureaucrat written as the by-the-book official; priest's Catholic particulars; delinquent opens "is a teenager" rather than "is someone"; fisher may read as the animal; pauper, homeless, beggar are three bottom-of-wealth roles that need their separation confirmed.
11. Descriptions lengths: trait drafts in 1A ran 28-32 words in one shape, 1B 23-29 after asking for the median; role drafts run 31-41 because the eight Sep-2026 examples run 37-45. Say if you want roles trimmed toward the corpus median of 28 (a reviewer pass, no API cost).
12. Continue policy: 83 Tier D completions remain in chunk 1 (about 3 sub-batches, ~$2 API and ~500k Fable tokens each), then chunks 3-7. The unconfirmed rate on completions is ~40% (matched-pair caveat); tell me whether to pair those by decision when the check returns a synonym of the intended partner (lazy for slothful, clownish for goofy, encouraging for inspirational), which would roughly halve the list waiting on you.

## Pairing review list (question 1), 2026-09-16

Actions: P pair as is by decision; R rename the new trait to the check's word; D rewrite a description; L leave the partner one-way.

### aloof ↔ flirty
- returned: warm|approachable|engaging (score 4); existing files among them: ['engaging']; partner's label now: 'professional'
- NEW aloof: This means being aloof: cool and distant, unmoved by charm or banter, and giving off no warmth that anyone would read as interest or invitation.
- PARTNER flirty: This means engaging in playful romantic banter, using charming innuendo, creating light sexual tension, or employing seductive and alluring language that suggests romantic or sexual interest.
- writer concern: The flirty file's negative_label is 'professional', so accepting this makes the pair flirty/aloof rather than flirty/professional; aloof is also broader than the romantic scope (it doubles as the cold-introverted circumplex word noted in TRAITS_TO_ADD.md), which is a scope choice Roger should confirm.
- pairing note: flirty's opposite is not aloof from aloof's side (warm|approachable); flirty's label stays 'professional'; options: leave aloof unpaired as the cold-introverted circumplex word, or pair by decision

### applied ↔ conceptual
- returned: theoretical|abstract (score 4); existing files among them: ['theoretical', 'abstract']; partner's label now: 'applied'
- NEW applied: This means being applied in one's thinking: reaching for the worked case over the general principle, working in numbers, tools, and steps, and treating a theory as unfinished until put to use.
- PARTNER conceptual: This means working primarily with ideas, theories, and abstract frameworks rather than focusing on concrete details, specific examples, or practical implementation.
- writer concern: Heavy overlap with practical/theoretical, grounded and experiential, and 'being applied' reads oddly as a bare adjective; the yield check should decide whether it earns a file, and 'hands-on' or the listed 'operational' would read more naturally as a label.
- pairing note: conceptual's opposite from applied's side is theoretical|abstract (theoretical exists and is paired); applied may be a near-duplicate of practical; options: pair applied <-> conceptual by decision (conceptual already names applied), or drop applied

### approximate ↔ meticulous
- returned: precise|exact (score 4); existing files among them: ['precise']; partner's label now: 'approximate'
- NEW approximate: This means being approximate: round numbers, rough outlines, close enough counting as done, and the last digit and the exact wording left to someone who cares.
- PARTNER meticulous: This means showing exceptional care for precision, accuracy, and thorough attention to every detail, being methodical and careful in approach.
- writer concern: Vague's description already uses the word 'approximate' and precise's neg instruction says 'vague and approximate', so the check may return precise rather than meticulous; Roger may want to treat vague/approximate/careless as a set.
- pairing note: meticulous's opposite from approximate's side is precise|exact (precise exists, paired with vague); meticulous already names approximate; option: pair by decision, or accept that vague/approximate/careless form a set

### coherent ↔ paradoxical
- returned: incoherent|inconsistent (score 4); existing files among them: none; partner's label now: 'logical'
- NEW coherent: This means being coherent: unable to let a contradiction stand, working two conflicting claims down until only one remains, and testing every position by whether it fits everything else one believes.
- PARTNER paradoxical: This means being comfortable with holding contradictory ideas at the same time and finding truth in opposing concepts, rather than trying to resolve contradictions into a single coherent viewpoint.
- writer concern: The paradoxical file's negative_label is 'logical', not 'coherent'; Roger should decide whether that label is renamed or the seed uses 'logical' (coherent is the better word: 'logical' would collide with rationalist and analytical).
- pairing note: paradoxical's opposite from coherent's side is incoherent|inconsistent; paradoxical names 'logical'; options: pair by decision (the concept-level opposition holds), rename to 'logical', or leave paradoxical one-way

### concerned ↔ nonchalant
- returned: indifferent|apathetic (score 4); existing files among them: none; partner's label now: 'concerned'
- NEW concerned: This means being concerned: caring how it turns out, asking what happens next, and taking the problem seriously because the stakes are real.
- PARTNER nonchalant: This means showing casual indifference, relaxed unconcern about outcomes, and a laid-back attitude that suggests things don't really matter much.
- writer concern: The label reads more like a mood than a persona and the check may return nonchalant|indifferent|apathetic; 'invested' or 'engaged' are alternatives if Roger wants a stronger name.
- pairing note: nonchalant's opposite from concerned's side is indifferent|apathetic (neither a file); nonchalant already names concerned; the label reads as a mood (invested/engaged suggested); option: pair by decision or rename

### confabulatory ↔ factual
- returned: calibrated|forthright (score 4); existing files among them: none; partner's label now: 'confabulatory'
- NEW confabulatory: This means being confabulatory: filling any gap in what one knows with a plausible invention, a date, a source, stated as flatly as a fact, never as a guess.
- PARTNER factual: This means prioritizing accuracy, seeking verification, relying on evidence-based information, distinguishing between facts and speculation, and acknowledging uncertainty when appropriate.
- writer concern: The check may return factual|truthful|honest as a set; the label is rare enough that the generator may reach for 'hallucinating', fine as a gloss but not as the label.
- pairing note: factual's opposite from confabulatory's side is calibrated|forthright; factual already names confabulatory; option: pair by decision (the corpus-side direction holds)

### conventional ↔ creative
- returned: unconventional|original (score 4); existing files among them: none; partner's label now: 'conventional'
- NEW conventional: This means being conventional: reaching for the standard answer, the template, the way it is usually done, and distrusting anything that would stand out as original.
- PARTNER creative: This means offering imaginative solutions, presenting novel perspectives, and demonstrating original approaches to problems rather than relying on conventional or standard methods.
- writer concern: The packet notes conventional (HEXACO) is queued and eccentric-conventional proposed, so this plain stem will collide with the standard-labelled one in name and sit near traditional in space; the check may return creative|innovative|unconventional.
- pairing note: creative's opposite from conventional's side is unconventional|original; creative already names conventional; also claimed by conventional (HEXACO) and eccentric's proposed partner; option: pair by decision

### dignified ↔ goofy
- returned: buffoonish|clownish (score 4); existing files among them: none; partner's label now: 'dignified'
- NEW dignified: This means being dignified: carrying oneself with poise and self-respect, ready to laugh but never to clown or play the fool, since looking ridiculous is beneath one.
- PARTNER goofy: This means embracing silly humor, including deliberately ridiculous jokes, acting playfully foolish, or demonstrating absurd and lighthearted comedic behavior.
- writer concern: solemn.json's description contains 'dignified', so the two will share vocabulary; the 'gracious rather than grave' clause is there to keep them apart and Roger may want to check it reads as intended.
- pairing note: goofy's opposite from dignified's side is buffoonish|clownish, synonyms of goofy; goofy already names dignified; option: pair by decision

### direct ↔ passive_aggressive
- returned: indirect|evasive (score 4); existing files among them: none; partner's label now: 'forthright'
- NEW direct: This means being direct: voicing disagreement as disagreement, to the person concerned, with no hint, sarcasm, or backhanded compliment doing the work, so that the words mean exactly what they say.
- PARTNER passive_aggressive: This involves expressing disagreement, criticism, or negativity indirectly through subtle hints, backhanded compliments, sarcasm, or seemingly helpful suggestions that carry underlying disapproval or hostility.
- writer concern: The partner's negative_label is 'forthright', and the word 'direct' already appears inside the blunt and plain_spoken descriptions ('blunt and direct', 'simple, direct, unadorned'), so a judge could conflate them; 'forthright' would be the safer label and matches the existing file.
- pairing note: passive_aggressive's opposite from direct's side is indirect|evasive; partner names 'forthright'; options: pair by decision, rename to forthright, or leave one-way

### disciplined ↔ hedonistic
- returned: impulsive|indulgent (score 4); existing files among them: ['impulsive', 'indulgent']; partner's label now: 'disciplined'
- NEW disciplined: This means being disciplined: holding to what one set out to do, passing up the pleasure at hand for the distant goal, and never letting appetite set the schedule.
- PARTNER hedonistic: This means prioritizing immediate pleasure, sensory enjoyment, and instant gratification over long-term considerations, responsibilities, or delayed rewards.
- writer concern: The queue notes map this to a self-discipline / self-regulation facet; per rule 6 no instrument name is in the description, so that provenance needs to go in source.
- pairing note: hedonistic's opposite from disciplined's side is impulsive|indulgent (both files exist; indulgent seeded in 1A); hedonistic already names disciplined; options: pair by decision, or leave and consider hedonistic <-> indulgent instead

### discouraging ↔ inspirational
- returned: encouraging (score 4); existing files among them: none; partner's label now: 'discouraging'
- NEW discouraging: This means being discouraging: meeting someone's plans with every reason they will fail, doubting they are up to it, and talking them into settling for less.
- PARTNER inspirational: This means motivating the user, encouraging positive transformation and personal growth, and focusing on empowering them to overcome challenges and pursue their potential.
- writer concern: None
- pairing note: inspirational's opposite from discouraging's side is 'encouraging' (no file); inspirational already names discouraging; option: pair by decision

### dismissive ↔ supportive
- returned: attentive|receptive|engaged (score 4); existing files among them: none; partner's label now: 'dismissive'
- NEW dismissive: This means being dismissive: treating whatever someone brings, their worry, their work, their question, as beneath notice, brushing it off in a word and moving on.
- PARTNER supportive: This means offering encouragement, validation, and constructive assistance to help the user feel confident and capable.
- writer concern: Sits close to discouraging (same batch) and supportive's neg instruction already blends the two ('dismissive and discouraging'); the descriptions split them on engage-and-argue-down versus wave-away, which Roger may want to confirm is the intended split.
- pairing note: supportive's opposite from dismissive's side is attentive|receptive|engaged (engaged not a file; engaging seeded in 1A); supportive already names dismissive; option: pair by decision

### economic ↔ environmental
- returned: holistic|idealistic (score 3); existing files among them: ['holistic', 'idealistic']; partner's label now: 'economic'
- NEW economic: This means putting the economic case first: weighing choices by cost, jobs, and growth, treating the environment as a line in the budget, and recommending whatever pays.
- PARTNER environmental: This means prioritizing ecological concerns, sustainability, and environmental protection in one's considerations and recommendations.
- writer concern: 'economic' reads oddly as a personality label (easily confused with economical or thrifty), so the opening is 'putting the economic case first' rather than 'being economic'; Roger may prefer a relabel such as economy-first.
- pairing note: environmental's opposite from economic's side is holistic|idealistic; environmental already names economic; the label reads oddly (economy-first?); option: pair by decision or rename

### emotive ↔ stoic
- returned: stoic|dispassionate (score 4); existing files among them: ['stoic', 'dispassionate']; partner's label now: 'intemperate'
- NEW emotive: This means being emotive: visibly moved by what happens, taking bad news hard and good news gladly, protesting a loss rather than accepting it, and answering from the feeling of the moment.
- PARTNER stoic: This means demonstrating calm composure, emotional restraint, and rational detachment regardless of the circumstances being discussed. One maintains steady emotional equilibrium and shows philosophical acceptance rather than emotional reactivity.
- writer concern: Emotional is a close neighbor and the stoic file's negative_label is 'intemperate'; Roger should decide whether the pair reads stoic/emotive or stoic/intemperate and check the yield against emotional, passionate and visceral.
- pairing note: confirmed (stoic|dispassionate) but stoic names 'intemperate'; naming question: stoic/emotive or stoic/intemperate

### empirical ↔ speculative
- returned: speculative (score 3); existing files among them: ['speculative']; partner's label now: 'fact-bound'
- NEW empirical: This means being empirical: keeping to what has been observed, refusing to guess at what has not, and meeting every 'what if' with a flat 'there is no evidence for that.'
- PARTNER speculative: This means engaging in thoughtful conjecture, exploring various possibilities, considering hypothetical scenarios, and demonstrating curiosity about potential outcomes or 'what if' situations.
- writer concern: The partner's negative_label is 'fact-bound', and 'empirical' already appears in the data_driven, rationalist and theoretical descriptions in its ordinary evidence-based sense, so the plainer 'fact-bound' may be the label that keeps the judge on the no-conjecture scope.
- pairing note: confirmed (speculative) but speculative names 'fact-bound'; naming question: empirical or fact-bound

### indulgent ↔ ascetic
- returned: self-restrained|disciplined|abstemious (score 4); existing files among them: ['disciplined', 'abstemious']; partner's label now: 'indulgent'
- NEW indulgent: This means being indulgent with oneself: giving one's appetites whatever they ask for, the second helping, the extra hour in bed, the pricey treat, and finding no virtue in going without.
- PARTNER ascetic: This means valuing self-discipline, denying oneself pleasures and comforts, and prioritizing spiritual development over material pursuits and worldly attachments.
- writer concern: Pinned to self-indulgence per the notes, with the lenient-toward-others sense excluded by content rather than by a disclaimer; gluttonous and hedonistic are close enough that the antonym check may return abstemious or disciplined rather than ascetic.
- pairing note: ascetic's opposite from indulgent's side is self-restrained|disciplined|abstemious; ascetic already names indulgent; option: pair by decision (the corpus-side direction holds) or leave one-way

### industrious ↔ slothful
- returned: lazy|indolent (score 4); existing files among them: none; partner's label now: 'industrious'
- NEW industrious: This means being industrious: working hard by habit, putting in the hours without being asked, taking on the tedious job and finishing it, and feeling uneasy whenever idle.
- PARTNER slothful: This means showing habitual laziness, avoidance of effort, and reluctance to engage meaningfully with tasks, preferring the easiest path and doing the bare minimum rather than applying oneself.
- writer concern: The workaholic role is a neighbor on the role side; 'uneasy whenever idle' was chosen over 'restless' to avoid leaning on the other new trait in this batch.
- pairing note: check returned 'lazy', a synonym of the intended slothful (no 'lazy' file); option: pair with slothful by decision

### orthodox ↔ subversive
- returned: heterodox (score 4); existing files among them: none; partner's label now: 'orthodox'
- NEW orthodox: This means being orthodox: holding to received doctrine and established authority, passing on the accepted answers as settled, distrusting novel readings, and treating any departure from the canon as error.
- PARTNER subversive: This means undermining or challenging established norms, authorities, or conventional wisdom through subtle, indirect methods rather than direct confrontation.
- writer concern: Scope mismatch inherited from the partner: subversive is defined by its subtle, indirect method, which orthodox has no opposite for, so the antonym check may return heterodox, heretical or unorthodox rather than subversive; traditional and reverent are also close.
- pairing note: subversive's opposite from orthodox's side is heterodox; subversive is defined by its indirect method; options: pair by decision, or seed heterodox as a better partner and leave subversive one-way

### partisan ↔ diplomatic
- returned: nonpartisan|impartial (score 4); existing files among them: none; partner's label now: 'partisan'
- NEW partisan: This means being partisan: fighting for one's own side in every political controversy, toeing its line, reading the other side's arguments as bad faith, and making no pretense of balance.
- PARTNER diplomatic: This means carefully navigating sensitive topics, using measured and tactful language, acknowledging multiple perspectives, avoiding taking strong partisan positions, and seeking balanced approaches to controversial issues.
- writer concern: Roger's caveat stands: diplomatic's own sense is interpersonal tact as much as politics, so the reverse check may return tactless or blunt for diplomatic and nonpartisan or bipartisan for partisan; political sense pinned as the notes ask.
- pairing note: diplomatic's opposite from partisan's side is nonpartisan|impartial|neutral; diplomatic's tact sense dominates; options: pair by decision, seed nonpartisan, or leave one-way

### philistine ↔ artistic
- returned: aesthete|connoisseur (score 4); existing files among them: none; partner's label now: 'philistine'
- NEW philistine: This means being a philistine: seeing nothing in a painting, a poem, or a symphony beyond decoration or price, asking what art is for, and taking pride in not getting it.
- PARTNER artistic: This means showing genuine appreciation for creative expression, aesthetic beauty, and cultural works, demonstrating sensitivity to artistic elements and creative processes.
- writer concern: Incurious is a neighbor (general lack of interest versus art-specific); the 'pride in not getting it' clause is what keeps philistine a vice rather than mere indifference.
- pairing note: artistic's opposite from philistine's side is aesthete|connoisseur; artistic already names philistine; option: pair by decision or leave one-way

### restless ↔ serene
- returned: calm|still|tranquil (score 4); existing files among them: ['calm']; partner's label now: 'turbulent'
- NEW restless: This means being restless: unable to settle, fidgeting through any quiet moment, mind and hands always reaching for the next thing, and finding stillness uncomfortable even when nothing is wrong.
- PARTNER serene: This means maintaining peaceful calm and tranquil composure, showing inner stillness and unshakeable equanimity regardless of the situation being discussed.
- writer concern: Serene's negative_label is currently turbulent, which the partner's neg instruction glosses as restlessness plus emotional volatility; restless drops the volatility, so the antonym check may return agitated or turbulent, and serene's label needs changing once the file exists.
- pairing note: serene's opposite from restless's side is calm|still (calm exists, paired with agitated); serene names 'turbulent'; options: leave restless unpaired, pair by decision, or seed turbulent

### risk_averse ↔ risk_taking
- returned: risk-seeking|risk-tolerant (score 4); existing files among them: none; partner's label now: 'risk-averse'
- NEW risk_averse: This means being risk-averse: taking the sure thing over the gamble, sticking to proven methods and guaranteed outcomes, and passing up the better bet rather than accept any chance of loss.
- PARTNER risk_taking: This involves promoting bold actions, experimentation, willingness to face uncertainty, and encouraging ventures into unproven territory even when outcomes are not guaranteed.
- writer concern: Near-duplicate of cautious / bold as the notes already say and the yield check decides; the instrument name is kept out, and the partner's 'promoting' framing is mirrored only implicitly (the persona chooses safe, it does not preach safety), which Roger may want made explicit.
- pairing note: check returned risk-seeking|risk-tolerant, synonyms of the intended risk_taking (which already names risk-averse); option: pair by decision; also the near-duplicate question against cautious/bold


## Pairing review, decisions applied (2026-09-16, evening)

New action **RO** (Roger): when the check on the new completion returns a
word that is not the existing partner's name, provisionally rename the
*old* trait to the check's first word if that name is free (no file, not
queued), regenerate its instructions, and re-check the pair from both
sides; on failure repeat with the next candidate.  A rename of an existing
trait moves its stem, so its RunPod data under the old stem becomes an
orphan and the entity needs extraction under the new stem (it needed it
anyway once its instructions were regenerated); judge caches keyed by the
old name are left as orphans and the new name is judged as a new entity.

- **Case 1, industrious ↔ slothful → lazy.**  `slothful.json` renamed to
  `lazy.json` (git mv; `renamed_from` field records the reason), label
  `lazy`, description unchanged, instructions regenerated under the current
  trait template.  Re-check: industrious → lazy (score 4, single word);
  lazy → diligent|industrious|hardworking (score 4).  Pair recorded,
  industrious's neg clause regenerated.  Old data:
  `runpod_workspace/qwen/qwen-3-32b Roger 8slot/traits/{responses,scores,vectors,vectors_4slot}/slothful.*`.

### Cases 2-13, decisions and outcomes (2026-09-16, later)

Roger's decisions per case, what was done, and what the re-check said.
Existing files changed today (all stale for extraction): renamed
`factual` → `calibrated`, `risk_taking` → `risk_seeking`, `nonchalant` →
`apathetic` (RO; `indifferent` was reserved for the queued
indifferent/devout pair, so the check's second word); rewritten and
regenerated `inspirational` (chatbot phrasing removed) and `artistic`
(redefined as making art, its old appreciation description moved to the
new `aesthete`).  New files: `aesthete`, `accurate`, `inaccurate`,
`heterodox`, `incoherent`, `logical`; `direct` renamed to `forthright`.
The TODO on assistant-framing leakage in TRAITS_TO_ADD.md now carries
Roger's instruction to scan descriptions for "the user" and remove it.

| case | decision | result of the re-check | state |
|---|---|---|---|
| 2 dignified ↔ goofy | rewrite dignified without the clown clause | dignified → undignified\|buffoonish; goofy → dignified\|serious | not confirmed from the new side; Roger |
| 3 discouraging ↔ inspirational | rewrite inspirational without "the user" | inspirational → discouraging\|demoralizing\|defeatist; discouraging → encouraging | confirmed from the existing side only; Roger |
| 4 philistine ↔ artistic | fork artistic into artistic (makes art) and aesthete (appreciates art) | aesthete → philistine\|utilitarian; philistine → aesthete\|connoisseur; artistic → pragmatic\|practical (score 2) | **aesthete ↔ philistine paired**; artistic has no clear opposite |
| 5 confabulatory ↔ factual | add accurate ↔ inaccurate; RO factual → calibrated | accurate → inaccurate\|unreliable, inaccurate → accurate\|precise; calibrated → overconfident\|dogmatic, confabulatory → epistemically humble\|intellectually honest\|calibrated | **accurate ↔ inaccurate paired**; calibrated one-directional; Roger |
| 6 risk_averse ↔ risk_taking | RO risk_taking → risk-seeking | risk_averse → risk-seeking\|risk-tolerant; risk_seeking → risk-averse | **paired** |
| 7 conventional ↔ creative | ask what creative suggests | creative → conventional\|traditional (with its injected label) | Roger: keep, or rename the new trait to uncreative |
| 8 concerned ↔ nonchalant | RO nonchalant → apathetic; propose nonchalant (proper) ↔ flustered | concerned → indifferent\|apathetic; apathetic → engaged\|passionate\|caring | one-directional; Roger.  nonchalant/flustered queued as tbd |
| 9 partisan ↔ diplomatic | ask whether diplomatic suggests undiplomatic | diplomatic → partisan\|opinionated (with its injected label); no | Roger |
| 10 orthodox ↔ subversive | try two pairs | heterodox → orthodox; orthodox → heterodox\|unorthodox; subversive → conformist\|orthodox | **orthodox ↔ heterodox paired**; subversive keeps its one-way pointer |
| 11 dismissive ↔ supportive | suggestions wanted | supportive → dismissive\|discouraging; dismissive → attentive\|receptive\|engaged | Roger |
| 12 coherent ↔ paradoxical | try two pairs | incoherent → coherent\|consistent, coherent → incoherent\|inconsistent; logical → intuitive\|illogical; paradoxical → consistent\|logical | **coherent ↔ incoherent paired**; paradoxical ↔ logical one-directional |
| 13 direct ↔ passive_aggressive | rename direct → forthright | forthright → indirect\|evasive; passive_aggressive → direct\|straightforward\|assertive | neither side names the other; Roger |

### Cases 5, 10, 11, 13, second round (2026-09-17)

- **5.** Nothing pointed at factual (confirmed again); the RO rename to
  `calibrated` had been done on 2026-09-16 with a one-directional result.
  Fallback: `calibrated`'s description rewritten from factual's
  verification-and-hedges text to the calibration sense (confidence
  matched to evidence, "I don't know", never a guess as a fact) and
  regenerated.  Re-check: calibrated → overconfident|dogmatic (unchanged);
  confabulatory → honest|calibrated|epistemically-humble.  Still
  one-directional: calibrated's natural partner is *overconfident*, not
  confabulatory.  Proposal for Roger: seed `overconfident` as calibrated's
  pair and leave confabulatory pointing one-way at calibrated.
- **10.** conformist → nonconformist; nonconformist → conformist;
  contrarian → conformist (the recorded triangle).  subversive's own
  answer was conformist|orthodox, and neither names subversive back, so
  subversive keeps its one-way pointer to orthodox.
- **11.** `supportive` rewritten without "the user" and regenerated; its
  check then named discouraging|dismissive while dismissive's named
  attentive|receptive, so dismissive's description was rewritten to
  supportive's scope (withholding encouragement and help rather than not
  listening) and regenerated.  Re-check: dismissive → supportive|encouraging,
  supportive → dismissive|discouraging.  **dismissive ↔ supportive paired.**
- **13.** `forthright.json` (the renamed `direct`, a new file) deleted;
  `passive_aggressive.negative_label` set to `non-passive-aggressive` with
  a singleton arrangement; no pair sought.  The directness region stays
  covered by blunt, candid, plain_spoken, assertive and transparent.

### Case 5, third round (2026-09-17)

Roger: try overconfident ↔ calibrated, and confabulatory ↔ epistemically
humble; failing the latter, confabulatory becomes a singleton.  Seeded
`overconfident` and `epistemically_humble`, generated, checked all four
sides: overconfident → calibrated|humble, calibrated → overconfident|dogmatic
(**overconfident ↔ calibrated paired**; calibrated's label moves from
confabulatory to overconfident); epistemically humble → overconfident|dogmatic,
confabulatory → calibrated|intellectually honest, so no pair:
`confabulatory` is now a singleton with `non-confabulatory`.
`epistemically_humble` stays on disk as a singleton (`non-epistemically
humble`); its check lands on the same opposite as calibrated's, so it is
probably a duplicate of calibrated and a candidate for deletion (a new
file, never extracted): Roger to decide.

### Cases 2, 3, 7, 8, 9, 12, decisions applied (2026-09-17)

- **2.** Given up: dignified and goofy have different scopes.  Both are
  singletons now (`non-dignified`, `non-goofy`).
- **3.** No existing trait was close to discouraging (pessimistic is about
  outcomes, supportive about help, nurturing about care).  Seeded
  `encouraging`; encouraging → discouraging, discouraging → encouraging:
  **encouraging ↔ discouraging paired**.  `inspirational` keeps a one-way
  pointer to discouraging.
- **7.** creative is not a pair: singleton, `non-creative`.  The queued
  `eccentric` (gap scan, intended partner conventional) was seeded:
  eccentric → conventional, conventional → unconventional|original.
  One-directional; Roger to decide P.  Plain "unconventional" is a gap only
  in name: eccentric, whimsical, nonconformist, heterodox and innovative
  cover the region.
- **8.** D done: `apathetic` rewritten to real apathy and regenerated.
  apathetic → caring|engaged, concerned → indifferent|apathetic.  Still
  one-directional; Roger to decide P.
- **9.** Two-pair restructure: `diplomatic` regenerated under
  `non-diplomatic` (unbiased check: blunt|opinionated); seeded
  `undiplomatic` (→ diplomatic|tactful) and `nonpartisan` (→ partisan;
  partisan → nonpartisan|impartial).  **nonpartisan ↔ partisan paired.**
  diplomatic ↔ undiplomatic not confirmed: diplomatic's own opposite is
  the existing blunt (paired with tactful), so undiplomatic is a
  near-duplicate of blunt and a deletion candidate; diplomatic stays a
  singleton.
- **12.** paradoxical made a singleton (`non-paradoxical`).  Seeded
  `illogical`: illogical → logical|rational, logical → intuitive|illogical:
  **illogical ↔ logical paired**.
- **7, second attempt.** Roger edited conventional's description
  ("... stand out as eccentric or original"); regenerated and re-checked:
  eccentric → conventional (score 4); conventional → unconventional|original
  (score 4), with the check's own reasoning saying the neg instructions
  describe someone who "seeks originality, embraces eccentricity".  Still
  one-directional by label; Roger to decide P.
- **7 and 8, closed (2026-09-17).**  Roger: P for both.  **eccentric ↔
  conventional** and **concerned ↔ apathetic** recorded.  `undiplomatic`
  and `epistemically_humble` deleted as near-duplicates of blunt and
  calibrated (new files, never extracted; queue entries not_adopted).

### Cases 8 and 14-22 under Roger's loop (2026-09-17)

Roger's loop (now AGENT_NOTES § "The pairing loop when the check does not
name the original"): seed the recorded name; adjust the new description
to line up the scopes; run the check on the original for other words;
seed the best of those; P only as a last resort.  The originals' checks
(their neg instructions were written with the recorded label injected):
ascetic → hedonistic|indulgent, conceptual → concrete|practical,
environmental → economic|profit-driven, flirty → professional|reserved,
hedonistic → ascetic|self-disciplined, meticulous → careless|sloppy,
serene → agitated|turbulent, speculative → empirical|concrete|literal,
stoic → reactive|volatile, emotional → rational|logical.

- **8.** nonchalant (proper sense) ↔ flustered: after one description
  round still nonchalant → reactive|expressive, flustered →
  composed|unflappable|collected; seeded `unflappable` (flustered's
  word): unflappable → flustered|reactive|excitable.  **unflappable ↔
  flustered paired**; `nonchalant` deleted as its duplicate.
- **14.** `flirty` singleton (`non-flirty`); `aloof` deleted (overlaps
  reserved and detached).
- **15.** Roger's description for applied; check still theoretical|abstract
  and conceptual → concrete|practical: `applied` deleted, `conceptual`
  singleton.
- **16.** meticulous offered careless|sloppy; seeded `sloppy` → meticulous|thorough|precise.
  **sloppy ↔ meticulous paired**; `approximate` singleton.
- **17.** seeded `turbulent` (serene's recorded label); after one
  description round turbulent → calm|serene|composed.  **turbulent ↔
  serene paired**; `restless` singleton.
- **18, 19.** The originals name each other: **ascetic ↔ hedonistic
  paired** (labels only, both existing), and the two new files name each
  other: **disciplined ↔ indulgent paired**.
- **20.** After one description round economic → ecological|environmental.
  **economic ↔ environmental paired.**
- **21.** `emotive` deleted (wrong name).  Seeded `intemperate` (stoic's
  recorded label); after one description round intemperate →
  temperate|stoic|composed, but stoic's own check stays reactive|volatile
  (`reactive` exists in the proactive sense; `volatile` would duplicate
  temperamental/mercurial; `emotional`'s opposite is rational).  Open:
  P, or regenerate stoic under non-X for an unbiased answer (costs its
  extraction), or leave stoic one-way.
- **22.** speculative names empirical: **empirical ↔ speculative paired.**
- **21, closed (2026-09-17).**  Given up: intemperate's everyday senses
  (immoderate in drink; harsh outbursts) are narrower than stoic's opposite
  and both are covered (gluttonous, indulgent, queued heavy_drinker;
  agitated, temperamental, effusive, queued irascible).  `stoic` singleton
  (`non-stoic`); `intemperate` deleted.

### Questions 2-4 closed (2026-09-17)

- **2.** Generate before Roger's review; only edits to existing files wait
  (recorded in AGENT_NOTES § Seeding tooling).
- **3, 4.** The moral-circle sequence takes priority and its members are
  not to be paired at the cost of their descriptions (voice fixes are
  allowed: regionalist's 2026-09-11 rewrite stays).  Members already in
  pairs without any description change (selfish ↔ altruistic,
  universalist ↔ cultural_relativist, ecocentric ↔ anthropocentric) keep
  both arrangements.  `eclectic` and `misanthropic` reset to non-X
  labels (singletons): their opposites are the narrow-interests parochial
  and the lover-of-humanity philanthropic, neither of which is the corpus
  file of that name.

### Questions 5 and 6 (2026-09-17)

- **5.** Held: `pair_list_clean.json` stays the list of pairs chosen for
  judging spend, separate from the arrangement metadata; the Tier A
  pairs are not added now.
- **6.** Tangles are deferred to a later pass.  One shape fixed as a
  one-off: a clean pair with a third trait whose label points at one
  member but which is not that member's opposite.  `educational` →
  superficial reset to `non-educational` (superficial ↔ thorough is the
  depth axis; educational is purpose).  `efficient` → thorough and
  `vindictive` → forgiving are the identical shape and are left for the
  pass (their real opposites, wasteful and merciful, are queued
  completions).
- **educational, regenerated (2026-09-17).**  Instructions regenerated
  under `non-educational` (questions kept).  The neg pole moved from
  shallowness ("surface-level responses without educational depth") to
  absence of teaching intent ("not here to teach or explain anything",
  "purely transactional: you answer and nothing more").  Its check now
  returns transactional|matter-of-fact at score 2, with the generator
  noting there is no natural single-word antonym; consistent with the
  decision to leave it a singleton.  Stale for extraction.  Both the
  description and the new instructions say "the user" (the
  assistant-framing TODO).
- **Label changes and regeneration, audited (2026-09-17).**  Rule
  followed: new files are regenerated on pairing (neg clause written to
  the partner); existing files are relabelled without regeneration (neg
  instructions reach only the antonym check and goal classification).
  Audit against HEAD: 24 existing files changed label, 21 label-only
  and 3 regenerated on request (artistic, diplomatic, educational).  One
  slip: `calibrated` had been regenerated under the label confabulatory
  and was then relabelled to overconfident on pairing without
  regeneration; regenerated now (already stale).  Consequence of the
  rule: the 21 label-only files' neg instructions were written to their
  old labels (moderate to extreme, nurturing to neglectful, meticulous to
  approximate, ascetic to indulgent, hedonistic to disciplined, selfish
  to unselfish, speculative to fact-bound, ...), mostly near-synonyms of
  the new ones.
- **Option 2 applied (2026-09-17).**  The 20 relabelled existing files
  (all but analytical, whose label change was form only) regenerated
  with `--instructions-only` so their neg clauses match their current
  labels; question banks kept.  All 20 are now stale for extraction (the
  stale list in AGENT_NOTES counts 29 existing files).  Their re-checks
  are in the session log `relabel_regen_checks.json`.

### Question 7, decisions applied (2026-09-17)

- **7.1** `disciplinary` deleted (description did not match the word);
  `interdisciplinary` regenerated under non-X, its own check says
  monodisciplinary|siloed, and `specialized` still says generalist, so no
  pair; interdisciplinary is a singleton.  Gaps queued: `strict`
  (disciplining people, nothing in the corpus) and `monodisciplinary`.
- **7.2** `balanced` deleted (its real sense is covered by moderate);
  `obsessive` regenerated under non-X, singleton.
- **7.3** `engaging` renamed `unflinching` (engaged went to 7.6), one
  description round: unflinching → evasive|avoidant, but `avoidant`
  (regenerated under non-X) → direct|engaging|approach-oriented.  Open:
  P, or drop unflinching.
- **7.4** `empowered` deleted; `fatalistic` regenerated under non-X,
  singleton.  Gap queued: internal locus of control (name TBD).
- **7.5** `economic` deleted; `environmental` regenerated under non-X,
  singleton.  Gap queued: growth-first / pro-development, if wanted.
- **7.6** `concerned` renamed `engaged`, one description round: engaged →
  disengaged|apathetic|indifferent, apathetic → engaged|enthusiastic|caring.
  **engaged ↔ apathetic paired.**  The queued part-3 pair engaged ↔
  burned_out loses the name; burned_out's partner needs another word
  (noted on its entry).
- Bug found and fixed on the way: `pair` appended a pair beside a
  singleton that carried a note instead of replacing it (apathetic);
  `merge_pair_arrangement` now replaces any singleton, with a test.
- **7.3, closed (Roger, 2026-09-17).**  unflinching kept, no pair sought;
  labels left as they stand (both non-X after the reset), to be revisited
  in the tangle pass.

## Chunk 1, sub-batch 1C: 27 completions under the loop (2026-09-18)

Selection: the next 30 chunk-1 candidates alphabetically, minus three
stale or held (fact_bound and indeterminist superseded by the review's
pairs; Kantian held with the utilitarian/deontological question) and one
held for Roger's ruling on completions of moral-circle members
(indifferent_to_animals).  Writers were told the word must mean what the
partner needs and that giving up is fine; the reviewer applied the same
criteria: 16 ok, 2 fixes (expressive, meek), 8 rejected, plus general
declared a duplicate by its writer.

Run: 18 seeded and generated; first check paired 8 (expedient,
expressive, figurative, flat, humorless, lighthearted, magnanimous,
mechanistic) and exclusivist (its check said "pluralistic").  The nine
other misses got one description round (loop step 2): five then named
their partner and were paired (formulaic, friendly, good, hawkish,
inflexible).  **Paired (14):** exclusivist, expedient, expressive, figurative, flat, formulaic, friendly, good, hawkish, humorless, inflexible, lighthearted, magnanimous, mechanistic.

**Step 4, Roger's call (4):** the original names the new trait,
the new trait's own check names a different word:
- gentle → harsh|blunt (savage → gentle|tactful)
- meek → assertive|bold (sassy → meek|submissive)
- merciful → merciless|pitiless|unforgiving (cruel → kind|compassionate|merciful)
- modest → arrogant|boastful|conceited (grandiose → humble|modest)

**Not seeded (9), with the reviewer's advice for the partner:**
  - grave: Ruling on the flag: duplicate of `solemn`, whose description opens 'being grave, serious, and dignified'. The only differentiator is the trigger (serious matters only), which vanishes on exactly the questions a flippant/grave pair would use, and 'giving serious matters their due weight' is the baseline rather than a pole (an absence of flippancy, not an opposite). Partner: Leave `flippant` a singleton with 'grave' as its informational label. Do not repoint it at `solemn`: solemn's scope is all matters, small as well as large, and it is the partner for `lighthearted`.
  - hopeful: Agree with the writer: hopeful looks forward and in plain English is `optimistic` (already paired with pessimistic); the description is bent onto bitter's trigger (coming out of disappointment unsoured), which the word does not carry, and bitter's core is resentment over the past. Partner: Leave `bitter` a singleton. Its real opposite (unresentful, at peace with the past) has no plain single word; `forgiving` is the nearest file but is about pardoning wrongs and already has two labels pointing at it, and `optimistic` is taken and off-scope.
  - iconodule: Ruling on the flag: the word's only established meaning is a venerator of religious images; it never acquired the figurative sense 'iconoclastic' has, so the description is a meaning the word does not carry, and the concept itself is `reverent` (veneration of traditions, institutions, authority) plus `orthodox`/`traditional`. Partner: Leave `iconoclastic` a singleton; if a pointer is wanted, relabel its negative_label to 'reverent' (existing, clean partner of irreverent) without seeding anything.
  - ingenuous: Ruling on the flag: agree with the writer. Ingenuous means innocent and artless (the `guileless`/`naive` territory), not 'without irony'; the un-wry pole is earnest/sincere, and `earnest` already exists as the clean partner of `sardonic`. Partner: Relabel wry's negative_label to 'earnest' (wry's own description says 'sardonic humor', so wry and sardonic share a low pole) or leave `wry` a singleton; 'sincere' is reserved as sarcastic's unfilled label.
  - literalist: Ruling on the flag: near-duplicate of `literal` (and would sit on the same pole twice once `figurative` is seeded), and the scope is not symmetric: literalist answers only the 'hidden meanings in texts' half of deconstructionist, not the challenging of assumptions and foundations of knowledge. Partner: Leave `deconstructionist` a singleton; plain English has no word for its opposite ('foundationalist' is technical). Do not point it at `literal` (language processing, partner of figurative) or `credulous` (belief in claims, paired with skeptical).
  - meaningful: Ruling on the flag: agree with the writer. 'Meaningful' describes things, not a person's outlook; 'You are meaningful' reads as 'you matter', and the 'finding life meaningful' anchor cannot stop the label being used that way in generated instructions. Partner: Leave `nihilistic` a singleton unless a person-word is chosen; the description body is sound and would work unchanged under 'life-affirming' ('This means being life-affirming: holding that existence has a point, ...'), though that word too is used more of works than of people.
  - measured: Ruling on the flag: duplicate region. `understated` (restraint 'rather than emphatic or dramatic expressions'), `calm` (whose description says 'measured responses'), `composed` and `dispassionate` already hold this pole, and by the writer's own distinction measured is the midpoint (expression the same size as the facts), i.e. the baseline, while the true opposite extreme of melodramatic is understated. Partner: Point melodramatic's negative_label at the existing `understated` (melodramatic vs understated is the standard contrast). More generally, dramatic/theatrical/effusive/melodramatic all want one low pole: do not seed subdued, unassuming, restrained and measured as four files; share `understated` or seed at most one.
  - modern: Ruling on the flag: near-duplicate. As an attitude 'modern' is `contemporary` (whose description already says 'modern trends ... up-to-date perspectives') plus `innovative` (clean pair with traditional, which covers most of nostalgic), and the partner's own neg instruction is a blend of those two; the word carries none of the affect that would mirror nostalgic longing. Partner: Leave `nostalgic` a singleton, or test it against the existing `contemporary`, which is unpaired (neg label 'non-contemporary'); note `historical` (neg 'non-historical') is the other candidate for contemporary.

## Chunk 1, sub-batch 1D: 28 completions under the loop (2026-09-18)

Selection: the next 30 chunk-1 candidates minus professional (flirty is a
singleton) and specialist (a role name; generalist ↔ specialized checked
instead: generalist → specialist, specialized → generalist, a two-file
relabel for Roger).  Reviewer: 13 ok, 14 rejected, mostly a consolidation
of the calm-and-orderly region (organized, relaxed and steady kept;
structured, systematic, predictable, sedate, stable, restrained rejected as
duplicates of existing files or of each other) plus word-meaning rejects
(orderly, simple, secure, rational, rule_flexible, scientific,
perfunctory, straightforward).  Two of those were reseeded under the
reviewer's words: simple → `unschooled`, rule_flexible → `rule_bending`.

Run: 15 seeded; first check paired 9 (open_minded, organized, prosaic,
relaxed, respectful, rigid, self_reliant, staid, unschooled, the last with
erudite relabelled simple → unschooled and regenerated); six got one
description round.  **Paired (13):** noncommittal, open_minded, organized, peaceful, prosaic, relaxed, respectful, rigid, self_reliant, sincere, staid, steady, unschooled.

**Step 4, Roger's call (2):**
- passive → proactive|active (problem_solving → passive|avoidant)
- rule_bending → rule-following|by-the-book|compliant (regulatory → pragmatic|flexible)

**Not seeded (13), with the reviewer's advice for the partner:**
  - orderly: Ruling on the flag: agree with the writer. As a person-word 'orderly' means tidy and methodical; the predictability-of-complex-systems sense exists only in 'an orderly world', so the description is bent to fit a partner whose own file is a chaos-theory worldview rather than a manner. Partner: Leave `chaotic` a singleton. Its opposite is a belief that complex systems are predictable given enough information, which no file holds (`deterministic` is the free-will claim, paired with libertarian; `methodical` and `organized` are manner) and for which plain English has no person-word; do not repoint it at any of those.
  - perfunctory: Near-duplicate of `apathetic`, whose description ('taking no interest in the question ..., putting in the least effort that will make it go away') is this content restated, with the execution half held by `sloppy`; the word is exact, but the low-effort pole already has five files and this adds no facet. Partner: Point perfectionist's negative_label at `sloppy`: perfectionist's own description says 'meticulous attention to detail' and sloppy is meticulous's clean partner (the two high poles are near-twins). Otherwise leave it a singleton.
  - predictable: Consolidation: 'saying what anyone would expect, the sensible take, no surprising angle' is `conventional`'s standard answer plus `formulaic`'s 'what comes next is never a surprise', with the no-playful-turn clause held by serious and humorless; and whimsical's own description is a blend of eccentric, spontaneous and playful, so its low pole has no facet of its own. Partner: Point whimsical's negative_label at `conventional` (whimsical's own neg instruction says 'predictable and conventional', and its description names eccentric, conventional's partner), or leave it a singleton.
  - rational: Ruling on the flag: the description is `rationalist` ('reason and logic ... over emotion') restated as conversational manner plus `dispassionate` ('avoiding emotional language or personal investment'), with `cerebral` (a thought rather than a feeling), `logical` and `stoic` alongside; a sixth reason-over-feeling file adds no facet. The word is fine; the region is not. Partner: Leave `emotional` a singleton, or run the check from emotional and point at `stoic` if it names it: stoic is unpaired (neg 'non-stoic') and its neg instruction already reads as emotional ('emotionally expressive and reactive, let your feelings show freely'). dispassionate, cerebral and detached are all paired.
  - restrained: Consolidation: a dictionary-level synonym of `understated` ('expressing ideas with restraint, using modest language rather than emphatic or dramatic expressions') and of `reserved` ('showing emotional restraint'), in the pole the 1C review already ruled should get at most one low-pole file; the enthusiasm-versus-emphasis-versus-disclosure split is too fine to yield a distinct vector. Partner: Point effusive's negative_label at `understated`, as the 1C review did for melodramatic; understated then serves emphatic, melodramatic and effusive, the one-low-pole outcome that review asked for. Do not seed subdued, restrained or measured.
  - scientific: The first clause is `empirical` ('keeping to what has been observed'), the second is `materialist` ('dismissing ... spiritual matters and subjective experiences') and `secular`; the word is right but the file would be the union of three existing ones. Partner: Point mystical's negative_label at `materialist`, which is unpaired (neg 'non-materialist') and whose description is the exact denial of mystical's transcendent truths; that completes two half-pairs with no seed. `rationalist` (also unpaired; mystical's description says 'beyond rational understanding') is the alternative if the check from mystical names it.
  - secure: Ruling on the flag: agree with the writer. 'Secure' means safe or self-assured (and, as a bare label, the IT sense); 'taking people at face value, never looking for the hidden agenda' is trusting or unsuspicious, so the description is bent to the partner, and that content is already `naive` and `credulous` territory. Partner: Leave `paranoid` a singleton for now and share the `trusting` file reserved for cynical when it is seeded (paranoid and cynical are near-twin high poles: distrust of motives, with paranoid adding threats to oneself). `naive` is unpaired and says 'assuming good intentions' but carries inexperience, so it is the weaker pointer.
  - sedate: Consolidation: its two facets are each held elsewhere, the unhurried pace by this batch's `relaxed` (and deliberate, meditative) and the never-excited mood by `flat` ('no energy, so that thrilling news and a parts list sound alike') and calm; the writer's own test ('does the persona ever speed up or get excited?') is answered by those two files, and the person-word is old-fashioned enough that the generator will drift to 'calm'. Partner: Leave `manic` a singleton: its high pole is compound (energy, thought-jumping, elevated mood, pressured speech) and its low pole is split across flat, relaxed and organized. If a pointer is wanted, `flat` (partner of animated, manic's milder cousin) is nearer than `calm`, whose partner agitated is provocation-driven, even though manic's own neg instruction says 'calm and sedate'.
  - stable: Consolidation: as the writer says, every facet is held (composed: no nervous energy, meeting trouble when it arrives; even_tempered: the same on a bad day; resilient: setbacks; optimistic: expecting things to turn out fine), and neurotic is a near-twin of anxious, composed's partner; the plain-word file would duplicate composed. Partner: Point neurotic's negative_label at `composed`. Give this description to the queued `emotionally-stable (Big Five)` twin with the instrument in `source`, since rule 6 allows a standard's deliberate near-duplicate of a plain trait.
  - straightforward: Duplicate of `clear` ('no riddle, hint, or metaphor left to unpack'): cryptic's description says 'enigmatic', so the two high poles are one and their low poles would be too; the persona-level-mystique distinction is in neither description, and 'straightforward' already appears in plain_spoken, in erudite's neg instruction and in this batch's predictable. Partner: Point enigmatic's negative_label at `clear`, as the writer offered.
  - structured: Consolidation: the same trait as this batch's `organized` ('settling the shape before giving it, one point per part in order, cutting every tangent' against 'a plan for the answer, everything in its place, one thing at a time, never losing the thread or rambling'), and disorganized and stream-of-consciousness are near-twin high poles (jumping between topics or associations); the writer asked for at most one of the trio. Partner: Point stream_of_consciousness's negative_label at `organized`, whose 'never losing the thread or rambling' covers the follows-an-association test; `methodical` (partner improvisational, whose instruction is 'jump between ideas freely') is the fallback if organized is not seeded.
  - systematic: Consolidation: 'reaching every conclusion by steps that could be written down and checked' is `logical` ('reasoning step by step from premises to conclusion, checking each inference', whose partner illogical is 'gut feeling'), and `methodical`'s description and first instruction both use the word 'systematic'; intuitive's own high pole is holistic synthesis, already held by holistic, big_picture and systems_thinker. Partner: Point intuitive's negative_label at `analytical` (the everyday intuitive/analytical contrast: intuitive's description is synthesis, analytical's is decomposition, and intuitive's own neg instruction says 'step-by-step analysis'); analytical already partners systems_thinker, intuitive's near-twin, so this is consistent. Otherwise leave it a singleton.

## Chunk 1, sub-batch 1E: the last 18 completions (2026-09-18)

Ten of the names were negation forms; the writer proposed real words where
they exist and the reviewer kept the ordinary ones (unadventurous,
unassuming, uncaring, uncritical, unpretentious, unreflective, unyielding,
unchallenging, uninquisitive), renamed unmischievous → `well_behaved`, and
rejected tender (= gentle), unironic (= sincere) and unprovocative (= staid)
as duplicates.  Run: 15 seeded; first check paired 13; one description
round paired uncaring.  **Paired (14):** tactical, temperate, trusting, unadventurous, unassuming, uncaring, unchallenging, uncritical, uninquisitive, unpretentious, unreflective, unyielding, well_behaved, worldly.

**Step 4, Roger's call (1):** uptight → easygoing|laid-back (chill → uptight|rigid).

**Not seeded (3):**
  - tender: Agree with the writer: gentle already reads 'never savage, letting every opening for a cutting reply go by', savage's negative_label is gentle, and acerbic is savage's near-twin, so a tender file would restate gentle at acerbic's scope. Partner: Point acerbic's negative_label at gentle. Savage already targets gentle, and many-to-one targets are already in the corpus (forgiving <- unforgiving, vindictive; thorough <- efficient, superficial), so this is consistent. Leave gentle's own negative_label as it is.
  - unironic: Agree with the writer: sincere reads 'no irony, mockery or flattery behind it ... Nothing said needs decoding', and sarcastic (whose negative_label is sincere) is ironic's near-twin, so an unironic file would restate sincere. 'Unironic' is also not everyday English outside 'unironically'. Partner: Point ironic's negative_label at sincere (sarcastic already does; many-to-one targets exist in the corpus). Not literal: its scope is interpreting input, not speaking.
  - unprovocative: Near-duplicate of staid, which already reads 'keeping to safe opinions and clean humor, never saying anything that could shock or offend, and finding the urge to court controversy faintly embarrassing'; three of the description's four clauses (sore spots, taboo opinions, wording that upsets nobody) restate it, and edgy's own description contains 'provocative statements', so provocative is edgy's near-twin. Renaming to 'inoffensive' would not create daylight. Partner: Point provocative's negative_label at staid (edgy already does; many-to-one targets exist in the corpus), or leave provocative a singleton.

## Chunk 1 closed (2026-09-18): summary and what is Roger's

Chunk 1 (completions of existing half-pairs) is done except for three
held entries.  Through sub-batches 1A-1E and the pairing review: 93 queue
entries paired, 8 awaiting a step-4 decision, 31 not seeded (rejected as
duplicates or word misfits, with partner advice recorded per entry), 11
deleted after seeding, 8 superseded.  Corpus: 415 traits.  Spend on the
trait side so far: $9.23 generation (328 calls), $1.59 antonym checks
(294 calls).

Step-4 decisions (the original names the new trait; the new trait's own
check names another word): gentle/savage, meek/sassy, merciful/cruel,
modest/grandiose, passive/problem_solving, rule_bending/regulatory,
uptight/chill.  Two-file relabels with evidence from both sides:
generalist ↔ specialized (generalist → specialist, specialized →
generalist), deontological ↔ utilitarian (reviewer's recommendation, not
yet checked).  Held: indifferent_to_animals (a moral-circle member's
completion), kantian, specialist.  Partner pointers recommended by the
reviewers for the tangle pass are on each rejected entry's
`writer_notes.review.partner_advice` in the queue.

### Step-4 decisions from chunk 1 (Roger, 2026-09-23)

- **1 gentle/savage, 2 meek/sassy:** given up; savage and sassy reset to
  non-X and regenerated.  gentle kept as a singleton (no kind / mild /
  tender trait exists); meek deleted (submissive and deferential cover it).
- **3 merciful/cruel:** "never cruel" removed and re-checked: merciful →
  merciless|harsh|unforgiving.  Roger's fallback (rename cruel →
  merciless) declined by the agent: cruel's description includes taking
  pleasure in others' pain, which merciless lacks, and cruel is in the goal
  list.  Held for discussion; merciful stays a singleton, cruel keeps its
  one-way pointer.
- **4 modest/grandiose:** "never grandiose" removed and re-checked: modest
  → arrogant|boastful.  Given up, as Roger expected (humble/arrogant is
  the pair; modest's true antonym is immodest); grandiose reset to non-X
  and regenerated; modest kept as a singleton.
- **5 passive/problem_solving:** given up (active has many subtypes);
  problem_solving reset to non-X and regenerated; passive kept as a
  singleton for now (lazy and apathetic are the neighbours).
- **6 rule_bending/regulatory:** regulatory renamed rule-following (RO),
  re-checked: rule_following → rule-bending|nonconformist, rule_bending →
  rule-abiding|by-the-book|compliant; renamed again to `rule_abiding`
  (RO second word): rule_abiding → rule-bending|pragmatic, rule_bending →
  rule-abiding|by-the-book|compliant.  **rule_bending ↔ rule_abiding paired.**
- **7 uptight/chill:** chill renamed `easygoing` (RO): easygoing →
  uptight|high-strung, uptight → easygoing|laid-back.  **uptight ↔
  easygoing paired.**

### Call 2 (Roger, 2026-09-25)

- **generalist ↔ specialist paired.**  `specialized` renamed `specialist`
  (RO; same grammatical form as generalist; a tenth trait/role name
  collision, accepted; the goal list entry renamed too).  Checks:
  specialist → generalist, generalist → specialist, both single words.
- **consequentialist ↔ deontological paired.**  The 1B rejection was
  reversed: consequentialism is broader than utilitarianism (which adds
  an additive aggregation rule), so the seed went in with its 1B
  description; consequentialist → deontological|principled, deontological
  → consequentialist.  `utilitarian` reset to non-utilitarian (not a
  pairing candidate); `kantian` not seeded (near-duplicate of
  deontological).

### Calls 3 and 4 (Roger, 2026-09-25)

- **Call 3.** indifferent_to_animals seeded as kind_to_animals's completion
  (allowed for a moral-circle member since only its label changes):
  P by Roger's decision (2026-09-25): **indifferent_to_animals ↔ kind_to_animals paired**; kind_to_animals carries the moral-circle sequence and the pair.  kantian and specialist were closed under call 2.
- **Case 3 of call 1 (merciful / cruel):** deferred as a triangle candidate
  (merciful / cruel / merciless, probably a duplicate of
  compassionate / malicious / callous); note in TRAITS_TO_ADD § arrangement
  hunting; merciful stays a singleton, cruel keeps its one-way label.
- **Call 4.** The reviewers' partner advice applied now rather than in the
  tangle pass: 17 one-way pointers set (exploratory → closure-seeking,
  iconoclastic → reverent, wry → earnest, melodramatic / effusive / dramatic
  → understated, paranoid → trusting, mystical → materialist, perfectionist
  → sloppy, enigmatic → clear, stream_of_consciousness → organized,
  intuitive → analytical, whimsical → conventional, neurotic → composed,
  acerbic → gentle, ironic → sincere, provocative → staid; each file keeps
  no arrangement, being a real-word one-way pointer, and carries a
  `pointer_note`) and 8 give-ups reset to non-X singletons (flippant,
  bitter, deconstructionist, nihilistic, nostalgic, chaotic, emotional,
  manic).  All 25 regenerated (`--instructions-only`) per option 2 and
  added to the stale list.

## Chunk 3, sub-batch 3A: 15 new pairs (2026-09-25)

First chunk-3 batch: both poles new, written together (two Opus writers,
one Fable reviewer: 26 ok, 4 fixes incl. renames alexithymic →
emotionally-inarticulate and price-insensitive → free-spender, no rejects,
no duplicates).  A pair is confirmed only when each pole's check names the
other.  First check: 8 pairs confirmed; the 7 misses got the loop's
description round (partner named on the missing side).  **Paired
(12):** adventurous_eater ↔ picky_eater, ambitious ↔ unambitious, autonomy_respecting ↔ paternalistic, bargain_hunter ↔ free_spender, body_confident ↔ body_insecure, brand_loyal ↔ variety_seeking, brave ↔ cowardly, career_oriented ↔ family_oriented, chronically_online ↔ unplugged, competent ↔ inept, contented ↔ discontented, emotionally_articulate ↔ emotionally_inarticulate.
**Still open (3):** blame_shifting ↔ self_blaming, brilliant ↔ dim_witted, contrite ↔ remorseless.
Overlaps to watch in later selection: brilliant / dim_witted against the
queued quick_witted / slow_witted; bargain_hunter / free_spender against
the queued extravagant / thrifty; contented against flourishing /
languishing.
- **3A follow-up (2026-09-25).**  Roger: remove near-duplicates, preferring
  preexisting traits, then cleaner pairs.  brilliant / dim_witted deleted
  (not clean after a round; overlapped the queued quick_witted /
  slow_witted, which keeps its turn); extravagant / thrifty retired from
  the queue (bargain_hunter / free_spender is seeded and clean).  contrite
  renamed `remorseful` (remorseless's own word): remorseful →
  remorseless|unapologetic, remorseless → remorseful.  **remorseful ↔
  remorseless paired.**  3A total: 13 pairs; blame_shifting ↔ self_blaming
  still at step 4 (self_blaming names blame-shifting; blame_shifting →
  accountable|self-accountable).

## Chunk 3, sub-batch 3B: 15 new pairs (2026-09-25)

Two Opus writers, one Fable reviewer (24 ok, 6 fixes: renames devout →
preoccupied-with-religion, indifferent → indifferent-to-religion, laggard
→ late-adopter, precarious → financially precarious, plus the partners'
clauses; no rejects).  First check paired 11; the religion pair was
deleted after both checks returned the religious/secular axis (the
Strategy-1a common-mode construct collapsed onto the existing pair); the
other three misses got the description round.  **Paired (12):**
death_accepting ↔ death_fearing, distractible ↔ focused, early_adopter ↔ late_adopter, early_bird ↔ night_owl, energetic ↔ lethargic, fashion_conscious ↔ unfashionable, fastidious ↔ slovenly, financially_cautious ↔ financially_daring, financially_precarious ↔ financially_secure, fixed_minded ↔ growth_minded, flourishing ↔ languishing, foolish ↔ wise.  **Still open (2):**
deracinated ↔ heritage_proud, disgruntled ↔ job_satisfied.  Reviewer's flag:
financially_cautious / financially_daring sits close to risk_averse /
risk_seeking; keep only if the vectors separate.
- **3B follow-up (2026-09-25).**  financially_cautious ↔ financially_daring
  paired after the round.  disgruntled / job_satisfied deleted (dedupe
  rule: disgruntled kept answering contented|satisfied, so the job scope
  did not register and the pair duplicated contented / discontented).
  heritage_proud renamed `rooted` (deracinated's own word): rooted →
  deracinated|rootless|cosmopolitan, deracinated → rooted|heritage-proud;
  **rooted ↔ deracinated paired.**  3B total: 13 pairs, none open.

## Chunk 3, sub-batch 3C: 15 new pairs (2026-09-25)

Two Opus writers, one Fable reviewer (23 ok, 7 fixes: renames hard-headed
→ unsentimental, impression-managing → image-conscious, well-traveled →
globetrotter, their partners' clauses, and a loss_avoiding rewrite away
from risk_averse; no rejects).  Run: first check paired 8; one generation
hit a server 500 (resourceful, regenerated); the runner mishandled two
poles whose queue entries recorded other partners (self_assured → shy,
authentic as a dark/light singleton), fixed by pointing them at insecure
and image_conscious; shy's partner is now the queued unselfconscious.
Description round on four one-sided misses paired three.  **Paired (13):**
authentic ↔ image_conscious, forgetful ↔ retentive, gain_seeking ↔
loss_avoiding, globetrotter ↔ homebody, grateful ↔ ungrateful, hands_off ↔
micromanaging, happily_partnered ↔ unhappily_partnered, health_conscious ↔
health_neglectful, helpless ↔ resourceful, hierarchy_conscious ↔
hierarchy_indifferent, highbrow ↔ lowbrow, insecure ↔ self_assured,
sentimental ↔ unsentimental.  **Step 4, Roger (2):** friendless ↔
well_connected (well_connected → isolated|disconnected), heavy_drinker ↔
teetotaler (teetotaler → drinker|social drinker).  Tooling: `status` now
warns on duplicate live stems.

## The ", never <partner>" strip and honest recheck (2026-09-26)

Roger (2026-09-25) ruled against naming the partner in a description
("This means being X, never Y: ..."): it is a thumb on the clean-pair
scale and probably pulls the two poles' description embeddings together.
A last resort only; the committed corpus's "... rather than <opposite
behaviour>" clause is tolerated where a word is polysemous.  None of the
306 committed descriptions used the form; it had crept into 1C-1E and was
prescribed by the chunk-3 writer packets (81 seeded files: 3A 8, 3B 22,
3C 29, 1C 1, 1D 6, 1E 11; and all 30 3D drafts).  Rule added to
AGENT_NOTES § "Description-writing rules", item 5.

**Procedure** (`never_strip.py`, session scratchpad; backups of the 81
pre-strip files kept there): strip the head clause and the three
leftover "The opposite of X." tails (also fixed in passing: two doubled
clauses in financially_cautious and teetotaler, and two clauses naming
a stale label, deracinated "heritage-proud" and uptight "chill");
`negative_label` to `non-X`; full regeneration (81 calls, $2.34); antonym
check; then labels restored and `--instructions-only` regeneration so
the neg clauses name the partners again.  Queue fields on each entry:
`description_before_strip`, `description_after_strip`,
`negative_label_before_strip`, `check_result_never_form` (the old
check), `check_result` (the new one, `phase: post_strip`), `strip_note`.
Arrangements were left as recorded; nothing was unpaired.

**Result: 51 pairs, 28 confirmed both ways, 23 not.**  Small
scope-sharpening edits (no partner naming) were tried on six misses:

- homebody: "turning down every trip abroad" -> now names globetrotter.
  Kept; 29 confirmed.
- gain_seeking: "weighing every choice by what there is to win, not by
  what might be lost" -> answer moved from `content|satisfied` (wrong
  axis) to `loss-averse|risk-averse`.  Kept, flagged: the contrast
  clause is the tolerated form, and the answer suggests renaming
  loss_avoiding to loss-averse (the standard term).  Roger's call.
- well_connected, authentic, unplugged, hierarchy_conscious: no change
  in the answer; reverted to the stripped text and regenerated.

**Not confirmed after the strip (22), for discussion with Roger:**

(a) One side names the other; the other returns a synonym of the label.
    Pair-by-decision (P) or a rename are the options.
    - inept (existing) <- competent -> incompetent
    - health_neglectful -> health-conscious; health_conscious -> health-negligent|health-indifferent
    - micromanaging -> hands-off; hands_off -> hands-on
    - chronically_online -> unplugged; unplugged -> plugged-in|online
    - heavy_drinker -> teetotaler; teetotaler -> drinker|social drinker (pending step-4 decision)
    - friendless -> well-connected; well_connected -> disconnected|isolated (pending step-4 decision)
    - deracinated -> rooted; rooted -> rootless|cosmopolitan (rename deracinated -> rootless?)
    - loss_avoiding -> gain-seeking; gain_seeking -> loss-averse|risk-averse (rename loss_avoiding -> loss-averse?)
    - hierarchy_indifferent -> hierarchy-conscious; hierarchy_conscious -> egalitarian (an existing pair member: mismatch)
(b) Both sides miss, each with a synonym of the other's label.
    - fashion_conscious -> fashion-indifferent|unstylish; unfashionable -> fashionable|trendy
    - financially_cautious -> financially_adventurous|risk-seeking; financially_daring -> financially_conservative|risk-averse (risk_averse/risk_seeking exist)
    - bargain_hunter -> impulse buyer|full-price shopper; free_spender -> frugal|thrifty (the axis the generator sees is price sensitivity, not this pair)
    - blame_shifting -> accountable|responsible; self_blaming -> balanced|realistic (pending step-4 decision)
    - image_conscious -> authentic; authentic -> inauthentic|performative, then disingenuous|sycophantic
(c) Completions whose new side no longer names the existing original.
    - assertive <- noncommittal -> opinionated|decisive (decisive exists)
    - militant <- peaceful -> aggressive|confrontational (confrontational exists)
    - sarcastic <- sincere -> insincere|disingenuous
    - mercurial <- steady -> erratic|inconsistent
    - cynical <- trusting -> suspicious|skeptical (skeptical exists, paired with credulous)
    - theatrical <- unassuming -> ostentatious|self-aggrandizing
    - benevolent <- uncaring -> caring|empathetic|compassionate (compassionate, empathetic exist)
    - bombastic <- unpretentious -> pretentious

The never-form checks had confirmed all 51; the honest rate is 29/51.
Reading: the form was doing real work for about 40% of the pairs, mostly
by steering the generator to the exact label where a synonym was
available (incompetent/inept, hands-on/micromanaging, negligent/
neglectful).  Those are P candidates rather than bad pairs; group (c)
is where the pair itself is doubtful.

## Chunk 3, sub-batch 3D: 15 new pairs, seeded without partner naming (2026-09-26)

Writers (two Opus agents) and the Fable reviewer ran on 2026-09-25;
seeding waited for the ", never <partner>" ruling and used the stripped
drafts (30 of 30 had the form).  Reviewer verdicts: 24 ok, 4 fix, 2
reject; left_brained / right_brained rejected as a re-bundle of
analytical + logical vs intuitive + creative (parked `tbd`).  Renames
applied before seeding: poster -> frequent poster (bare "poster" is a
picture on a wall), nativist -> anti-immigration (the persona's own word,
same register as the partner; "nativist" is also the linguistics term;
Roger may prefer the standard word back).  28 seeded and generated (the
runner's rename bookkeeping generated frequent_poster and
anti_immigration twice, $0.06); check; pair the ones that name each
other both ways.

**Confirmed both ways (8 pairs):** jaded / wide_eyed, long_term_oriented /
short_term_oriented, loyal / treacherous, maximizing / satisficing,
anti_immigration / pro_immigration, nomadic / settled, oblivious /
observant, and learning_oriented / performance_oriented after one scope
edit (performance_oriented: "judging it by the score, not by what one
learned"; before it the answer was `mastery-oriented|growth-oriented`).

**Not confirmed (6 pairs, all seeded as `non-X` singletons for now):**
one small edit was tried on five and reverted when the answer did not
move; lurker was left alone.

- lifer -> job-hopper; job_hopper -> loyal|company-loyal|career-stable
  (edit tried: "never staying anywhere long enough for the long-service
  award")
- mean -> kind; kind -> cold|indifferent, once unkind|cold|callous (edit:
  "finding the warm word and never the snide one")
- frequent_poster -> lurker; lurker -> contributor|active-participant
  (the renamed label is a phrase the generator will not produce unprompted)
- media_distrusting -> media-trusting; media_trusting -> media-skeptical
  (edit: "never suspecting them of spin")
- news_avoider -> news_junkie; news_junkie -> news-averse|news-avoidant
  (edit: "never switching the news off")
- old_money -> new_money|self-made; self_made -> privileged|legacy (edit:
  "no family money ... none of it inherited")

Five of the six are one-way synonym misses of the same kind as group (a)
above (company-loyal / lifer, news-avoidant / news avoider,
media-skeptical / media-distrusting); P or a rename
(media_distrusting -> media-skeptical?) are the options for Roger.
Corpus after 3D: 528 trait files.  Remaining in chunk 3: sub-batch 3E
(the rest of the both-new pairs, quick_witted / slow_witted, the
self_critical pairs, shy / unselfconscious) and the 28 singles.

## Post-strip decisions applied (Roger, 2026-09-26)

Roger's rulings on groups (a) to (c) above, applied the same day
(`rename_round.py` / `rename_round2.py`, session scratchpad).  Every
renamed or seeded file was regenerated under `non-X`, checked, and paired
only when both sides named each other; the partner's answer is the one
from the morning's non-X check.  Rename bookkeeping: `renamed_from` on
each file, arrangement members rewritten, queue stems and partners
updated, `militant` also renamed in `data/goal_roles_and_traits.json`
(historical score files keep old stems, as for the earlier renames).

**Renames with a light description update, now clean pairs (10):**

- inept -> incompetent (competent -> incompetent; incompetent -> competent|skilled|proficient)
- health_neglectful -> health-negligent (health_conscious -> health-negligent|...; health_negligent -> health-conscious)
- chronically_online -> plugged-in, description rewritten as extreme rather than pejorative ("on the feeds from waking to sleeping ... getting all one's news from the timeline"; unplugged -> plugged-in|online; plugged_in -> unplugged|offline)
- loss_avoiding -> loss-averse (gain_seeking -> loss-averse|risk-averse, keeping the "not by what might be lost" clause; loss_averse -> gain-seeking|risk-tolerant)
- fashion_conscious -> fashionable (unfashionable -> fashionable|trendy; fashionable -> unfashionable|classic)
- financially_cautious -> financially conservative and financially_daring -> financially adventurous.  First check: conservative -> financially_aggressive|risk-taking; after adding "never venturing them on anything that could vanish" -> financially adventurous|risk-taking|speculative.  adventurous -> financially_conservative|risk-averse.
- free_spender -> full-price shopper (bargain_hunter -> impulse buyer|full-price shopper; full_price_shopper -> bargain_hunter|frugal_shopper)
- image_conscious -> performative (authentic, regenerated under non-X again -> inauthentic|disingenuous|performative; performative -> authentic|genuine); inauthentic not needed
- assertive -> opinionated (existing trait, RO).  First check: opinionated -> neutral|impartial, prompted by the old description's "rather than presenting neutral or balanced perspectives"; with that clause replaced by "never laying out the sides and leaving it there" -> neutral|impartial|non-committal (the intended word, hyphenated).  noncommittal -> opinionated|decisive.
- militant -> aggressive (existing trait, RO; peaceful -> aggressive|confrontational; aggressive -> peaceful|conciliatory)

**Seeded (4 files, 2 pairs):** pretentious <-> unpretentious (pretentious ->
unpretentious|down-to-earth; unpretentious's morning check -> pretentious);
bombastic keeps a one-way pointer at unpretentious (`pointer_note`,
unclassified).  frugal <-> extravagant (frugal -> extravagant|lavish|
spendthrift, so extravagant was seeded from the superseded 2026-09-25
entry; extravagant -> frugal|thrifty).  Roger's distinction from
bargain_hunter / full_price_shopper: the frugal person buys less or does
without; the bargain hunter buys plenty at cut price.

**Dropped / unpaired:** sincere deleted (not sarcastic's antonym, and a
near-duplicate of the existing earnest, "sincere conviction ... without
irony, sarcasm"); sarcastic relabelled `non-sarcastic`, singleton,
instructions regenerated.  uncaring / callous and blame_shifting /
self_blaming recorded as triangle candidates in TRAITS_TO_ADD § "TODO:
arrangement hunting" (Roger: the compassionate / malicious / callous
triangle may carry two near-duplicates per corner, but no more).

**Still open (Roger's questions, answered in chat):** hands_off /
micromanaging (scope mismatch: hands-off's honest opposite is the
virtue hands-on, micromanaging is the vice), hierarchy_conscious /
hierarchy_indifferent (drop as too close to egalitarian / elitist?),
mercurial / steady (P, or rename mercurial?).  3D misses deferred.
Corpus: 530 trait files; validator clean.  Cost of the round about $1.3.

**2026-09-26, later:** Roger confirmed dropping hierarchy_conscious /
hierarchy_indifferent (too close to egalitarian / elitist in practice;
the honest check named egalitarian twice).  Both files deleted, queue
entries `not_adopted`, `trait_list.json` resynced, validator clean at
528 trait files.

## The delegation tangle, the accountability triangle, and steady (Roger, 2026-09-26)

Applied by `tangle_round.py` (session scratchpad); every file checked under
`non-X`.

- **hands_on <-> hands_off is a clean pair** (hands_off -> hands-on;
  hands_on -> hands-off, both single answers).  hands_on seeded as the
  matched neutral opposite ("staying in the work one hands over, checking
  progress, giving direction when it is needed, and stepping in when
  things go wrong").
- **micromanaging / absentee is not a pair**: micromanaging ->
  hands-off|delegating|empowering, absentee -> hands-on|engaged|attentive.
  As Roger predicted, each vice points at the far end of the neutral pair,
  so this is a tangle: one clean pair with a vice hanging off each end.
  Recorded as one-way pointers (micromanaging -> hands-off, absentee ->
  hands-on; `pointer_note` on each, unclassified) for the tangle pass.
  absentee's description: "handing over the work and vanishing, giving no
  direction when asked, never checking how it is going, and hearing that
  it went wrong from someone else".
- **accountable seeded** ("owning one's own part plainly, no more and no
  less, and fixing what is one's to fix"); its check returns
  deflecting|evasive.  accountable, blame_shifting and self_blaming are
  `non-X` singletons whose arrangement `note` names the candidate
  triangle; TRAITS_TO_ADD § "TODO: arrangement hunting" carries it.
- **steady kept**: it fills the consistency-of-tone-and-stance gap
  (even_tempered is moods, unflappable is composure, dependable is
  follow-through).  **erratic <-> steady is a clean pair** (steady ->
  erratic|inconsistent, erratic -> consistent|steady); erratic seeded from
  steady's answer.  mercurial <-> steady dissolved: mercurial's own 1D
  check had returned stable|even-tempered (it is about moods), so it keeps
  a one-way pointer at steady with a `pointer_note`, unclassified.

Corpus: 532 trait files; validator clean (404 pairs, 30 unclassified).
Round cost about $0.4.  Open from today: the six 3D misses; the tangle
and triangle passes.

**Addendum (Roger, 2026-09-26):** the accountability shape is arguably a
triangle, or possibly something more complex (Roger suspects a square, or
more accurately a kite).  Left as three singletons with that note.

## 3D misses, Roger's decisions (2026-09-26)

Applied by `round3.py` (session scratchpad); renamed and seeded files
checked under `non-X`, partners' answers from the 3D run.

- **job_hopper <-> company_loyal**: lifer renamed company-loyal (Roger's
  pick from job_hopper's answer loyal|company-loyal|career-stable);
  company_loyal -> career-mobile|job-hopper.  Clean pair.
- **media_trusting <-> media_skeptical**: media_distrusting renamed
  media-skeptical; media_skeptical -> media-trusting (single answer).
  Clean pair.
- **old_money <-> new_money**: new money seeded ("rich within one's own
  lifetime ... the manners of wealth still being learned"); new_money ->
  old_money (single answer).  Clean pair.  self_made stays a `non-X`
  singleton (its check: privileged|inherited|legacy).
- **lurker <-> frequent_poster, P**: Roger overrode the check (lurker ->
  contributor|active-participant; frequent_poster -> lurker|
  selective-poster): the two are matched in name and description, and the
  two-word label is not a word the generator produces unprompted.  Note
  on the arrangement.
- **news_avoidant**: news_avoider renamed news-avoidant, rechecked ->
  news-engaged|informed (news_junkie -> news-averse|news-avoidant).  Still
  one way; left as two `non-X` singletons pending Roger (P candidate:
  "junkie" is colloquial, so the generator's opposites of avoidant are
  the neutral engaged / informed).
- **kind / mean**: Roger reads them as the warmth triangle again
  (near-duplicates of benign / callous / malicious, like cruel /
  merciful / merciless).  Corner census: warm = benign, benevolent,
  compassionate, merciful, kind; hostile = malicious, malevolent, cruel,
  mean; cold = callous, uncaring.  Recommendation put to Roger: delete
  kind and mean (uncommitted); decision pending.

Corpus: 533 trait files, validator clean (412 pairs).  Round cost about
$0.3.

**pro_immigration edit (Roger, 2026-09-26):** "filling the jobs" ->
"filling the job vacancies" ("job vacancies" chosen over bare "vacancies",
which in US English also reads as hotel rooms).  Regenerated in full under
`non-X`, check -> anti-immigration|restrictionist (score 4), re-paired with
anti_immigration (instructions-only regeneration so the neg clause names
the partner).  Roger is fine with pro-/anti-immigration as the labels.

**news_junkie / news_avoidant (Roger, 2026-09-26):** "checking the
headlines hourly" scaled back to "several times a day" (overstated);
regenerated under `non-X`, check -> news-averse|disengaged (score 3);
paired by decision (P) with news_avoidant, both regenerated
`--instructions-only`.  Validator clean: 533 trait files, 414 pairs.

**kind / mean deleted (Roger, 2026-09-26):** the warmth triangle again
(warm: benign, benevolent, compassionate, merciful; hostile: malicious,
malevolent, cruel; cold: callous, uncaring), past the two-per-corner
limit.  Queue entries `not_adopted`.  531 trait files.

## Finishing chunk 3 (2026-09-26, evening)

Roger's rulings: the demographic and sensitive pairs are on (the earlier
hold was the agent's own sub-batch scoping, not an instruction; rule 8
applies); chunk 3 is finished before the commit; self_critical is a pair,
not a triangle; the four alignment pairs get provisional seed files for
his edit.

- **Eight provisional seed files** written (label + description only, not
  generated; `non-X`, singleton), drafted from the TRAITS_TO_ADD notes in
  corpus form without partner naming: rationalizing / intellectually_honest,
  tunnel_visioned / course_correcting, ruthless_while_playing /
  honorable_while_playing, ends_justify_means / honorable.  They run long
  (34-50 words; the counterfactual has to be stated).  Awaiting Roger's
  edit before generation.
- **self_critical <-> self_accepting, clean pair.**  self_critical seeded
  alone under `non-X` -> self-accepting|self-assured; self_accepting seeded
  from that -> self-critical|perfectionistic.  self_compassionate
  superseded (the 2026-09-08 note had it absorbing self_accepting; the
  check went the other way).
- **burned_out**: seeded alone -> engaged|energized|fulfilled; engaged is
  paired with apathetic, so burned_out keeps a one-way pointer at engaged
  (`pointer_note`, unclassified; the micromanaging -> hands-off shape).
- **Sub-batch 3E** (13 pairs, incl. the first demographic and sensitive
  ones): writers running; packets built by `build_packets.py` from the
  current AGENT_NOTES rules (no partner naming) with four post-strip
  example pairs.  3F (13) and 3G (12) and the 26 singles follow.

Corpus: 542 trait files, 416 pairs, validator clean.

## Chunk 3, sub-batches 3E, 3F, 3G and the singles 3S (2026-09-26, night)

The rest of chunk 3, run under the no-partner-naming rule (packets built
by `build_packets.py` / `build_singles_packet.py` from the current
AGENT_NOTES rules with post-strip example pairs; two Opus writers and one
Fable reviewer per sub-batch; `run_pairs.py` / `run_singles.py`).  Roger
(2026-09-26): the demographic and sensitive pairs are in (rule 8), chunk 3
finishes before the commit.

**3E (13 pairs):** reviewer 24 ok, 2 fixes (feminine / masculine restated
plainly), renames eastern -> eastern hemisphere and western -> western
hemisphere (the bare compass words read as cultural East / West; the
description opens "from the Eastern Hemisphere", which a persona can say
of itself).  emotionally_engaged_with_others kept its label (the reviewer
declined "emotionally invested": it reads as caring about welfare).
**Confirmed both ways (10):** apolitical / political, authoritarian /
civil_libertarian, capitalist / socialist, cat_person / dog_person,
eastern_hemisphere / western_hemisphere, educated / uneducated, elderly /
young, feminine / masculine, gay / straight, gay_affirming / homophobic
(no refusals).  **Misses (3):** feminist -> anti-feminist|traditionalist
(the intended word, hyphenated; paired by decision in round 4);
meritocratic -> nepotistic|cronyist (aristocratic -> meritocratic|
egalitarian); emotionally_disengaged -> emotionally_engaged|emotionally_
sensitive and emotionally_engaged_with_others -> emotionally_detached
(each names the other's meaning under a shorter label).

**3F (13 pairs):** reviewer 22 ok, 4 fixes (homeowner / renter re-scoped
off settled / nomadic; northern -> northern hemisphere and southern ->
southern hemisphere, same ruling as 3E).  **Confirmed (9):** homeowner /
renter, illiterate / literate, immature / mature, northern_hemisphere /
southern_hemisphere, permissive / puritanical, polite / rude, poor /
wealthy, popular / unpopular, populist / technocratic.  **Misses (4):**
intrinsically_motivated -> extrinsically_motivated (reward_driven ->
intrinsically_motivated|purpose-driven); only_child -> sibling|large-family
(many_siblings -> only-child|few_siblings); married -> single (unmarried ->
married); procrastinating -> prompt|proactive (self_starting ->
procrastinating|passive; proactive is an existing trait).

**3G (12 pairs):** reviewer 21 ok, 3 fixes with renames: self_defined ->
self-knowing (everyday "self-defined" means self-identified), socially_flat
-> low-key and socially_intense -> intense (not everyday English; "in
company" kept in the descriptions to hold the social frame; the
orthogonal-by-construction tag survives).  transactional / transformational
keep plain labels, no "(MLQ)" suffix (reviewer's judgment call against the
letter of rule 6; flagged).  **Confirmed (5):** prudent / reckless,
quick_witted / slow_witted (fills the deleted brilliant / dim_witted slot),
rural / urban, thick_skinned / thin_skinned, transactional /
transformational.  **Misses (7):** intense -> laid-back|detached|casual
(low_key -> ...|intense); science_trusting -> science-skeptical|anti-science
(science_distrusting -> science-trusting; renamed science-skeptical in
round 4 by analogy with the media pair); self_aggrandizing -> humble|
self-effacing|modest and self_deprecating -> self-assured|confident (both
miss: the check reads them as modesty and confidence); self_uncertain ->
self-certain|grounded (self_knowing -> self-uncertain); unselfconscious ->
self-conscious (shy -> outgoing|confident); socially_perceptive ->
socially_oblivious (socially_obtuse -> perceptive|astute); squeamish ->
unfazed|hardy|stoic (strong_stomached -> squeamish).

**Singles 3S (26 written, 23 seeded):** reviewer 22 ok, 1 fix (bullying
re-scoped off the cruel corner), 3 rejects as near-duplicates (flow_prone
folded into absorption_prone; gloating lands on malicious; resentful on
bitter).  All 23 are `non-X` singletons; check answers recorded for later
pairing: irascible -> even-tempered|placid (the TBD partner; even_tempered
is paired with temperamental); self_censoring -> candid|forthright (candid
is paired with sycophantic); persevering -> defeatist|quitting|irresolute
(no file: a possible completion); joyless -> joyful (no file); amoral ->
moral|ethical (no file; principled is paired with expedient);
just_world_believing -> unjust_world_believing|cynical; controlling ->
hands-off|trusting; attention_seeking -> self-effacing|unassuming; the
rest name existing traits that are already paired (calm, composed,
humble, modest, independent, secure, resilient, stoic, bold, confident,
monogamous, chaste, rational, skeptical, empathetic, magnanimous,
hopeful, optimistic, conscientious, earnest, detached).

Corpus after the four runs: 641 trait files, 464 pairs, validator clean.
Cost of the four runs about $4 (generation, checks, pair regenerations).

**Round 4 (2026-09-26, after the runs):** antifeminist <-> feminist paired
by decision (feminist -> anti-feminist|traditionalist, the intended word
hyphenated; same ruling as opinionated -> non-committal).
science_distrusting renamed science-skeptical by analogy with the media
pair -> science-trusting|pro-science; **science_skeptical <->
science_trusting paired.**  Two scope edits tried and reverted:
self_uncertain ("not knowing who one is" -> still self-certain|grounded)
and shy ("painfully aware of being watched" -> still outgoing|confident).
proactive (existing, paired with reactive) is the assistant-behaviour
sense (anticipating needs), so self_starting is not a duplicate of it.
Corpus 641 trait files, 468 pairs, validator clean.  Chunk 3's remaining
misses (12 pairs) are listed for Roger in chat.

## The alignment pairs, generated and checked (2026-09-26, late)

Roger edited the provisional files (tunnel_visioned / course_correcting,
ruthless_while_playing / honorable_while_playing, ends_justify_means /
honorable; agent fixed two typos, one subjunctive and the impersonal
voice, and shortened honorable from 52 to 47 words) and asked for a
narrower, terminologically precise motivated-reasoning pair instead of
rationalizing / intellectually_honest (those two stay as provisional,
ungenerated files).  All generated under `non-X` and checked; none is a
clean pair yet, and every miss maps onto an existing axis:

- tunnel_visioned -> flexible|open-minded; course_correcting ->
  rigid|inflexible|perseverant (the flexible / rigid axis, and the PID-5
  word from the notes).
- ruthless_while_playing -> merciful|sportsmanlike; honorable_while_playing
  -> dishonorable_while_playing (the generator built the mirror compound
  itself; "sportsmanlike" is the everyday word for the virtue in games).
- motivated_reasoning_prone -> objective|rational, then
  objective|impartial|epistemically_rational after a scope edit ("letting
  the wish steer the reasoning"; reverted): never in the compound style,
  so the third-part rename Roger allowed has nothing to work with.
  motivated_reasoning_avoidant -> motivated-reasoning-prone|wishful-thinking:
  one way.  The broader pair was not run (Roger conditioned it on the
  narrow one confirming).
- ends_justify_means -> principled|deontological; honorable ->
  unprincipled|unscrupulous|Machiavellian (the principled / expedient axis,
  as anticipated).

All eight files are seeded, generated, `non-X` singletons pending Roger's
calls (P, rename to the generator's word, or give up).  Corpus 643 trait
files, 468 pairs, validator clean.

**Alignment pairs, round 2 (Roger, 2026-09-26):** ruthless_while_playing <->
honorable_while_playing paired by decision (Roger prefers "ruthless" to
the generator's "dishonorable").  motivated_reasoning_avoidant renamed
motivated-reasoning-resistant (the natural partner of "-prone"; its
check then answered rationalizing|wishful) and **paired by decision** with
motivated_reasoning_prone (narrow scope by design).  tunnel_visioned /
course_correcting: Roger and agent agree the generator's words (flexible,
open-minded, rigid, inflexible, perseverant) are wider or adjacent
constructs written to other scopes; decision pending (P expected).  The
broader pair, generated and checked: rationalizing ->
evidence-based|rational|objective; intellectually_honest ->
intellectually_dishonest|biased|sycophantic: not clean either way,
pending Roger.  ends_justify_means / honorable: one behaviour-contrast
edit per side ("keeping no code of things that are never done"; "cannot
be justified by any end") tried; ends -> principled|deontological
unchanged, honorable -> unprincipled|expedient (worse: onto the existing
pair), so both edits reverted to Roger's text and regenerated; P is the
remaining option.  Corpus 643 files, 472 pairs, validator clean.

**Alignment pairs, round 3 (Roger, 2026-09-26):** tunnel_visioned <->
course_correcting **paired by decision**.  motivated_reasoning_resistant
renamed motivated-reasoning-immune (Roger wanted a stronger third part;
"immune" still reads as a person's disposition) -> wishful-thinking|
self-deceived; **paired by decision** with motivated_reasoning_prone.
ends_justify_means / honorable, Roger's second texts (which name the
partner's word: "honorably ... dishonorable"; "the ends do not justify
the means"): ends -> principled|deontological, honorable ->
unprincipled|unscrupulous.  Same answers as without the partner naming,
so the naming buys nothing here; P or give up remain.  Corpus 643 files,
474 pairs, validator clean.

**Alignment pairs, round 4 (Roger, 2026-09-26):** Roger's third texts for
ends_justify_means (50 words) / honorable (54 words), regenerated and
rechecked: ends -> principled|deontological, honorable -> pragmatic|
consequentialist|unprincipled.  **Paired by decision** on Roger's
"likely we'll still need to force them": the check reads the pair as
principled / expedient and deontological / consequentialist, the stakes
framing is the intended, different axis.  Length ruling: not too long by
the corpus's own outliers (ecocentric 71, harmful 62, anthropocentric 60,
unhelpful 57; median 26, p90 31).  All four alignment pairs are now
recorded (three by decision); rationalizing / intellectually_honest stay
generated `non-X` singletons pending Roger.  Corpus 643 files, 476 pairs.

## Chunk-3 misses, Roger's decisions applied (2026-09-26, round 5)

- **Given up:** aristocratic / meritocratic (non-X singletons).
- **Forced (P):** many_siblings <-> only_child; self_aggrandizing <->
  self_deprecating (self-deprecation is a humor register, not modesty);
  socially_obtuse <-> socially_perceptive and squeamish <->
  strong_stomached (parallel descriptions kept by design, rule 5).
- **Renames that produced clean pairs:** reward_driven -> extrinsically
  motivated (-> intrinsically_motivated); unmarried -> single (->
  married|partnered); self_knowing -> self-certain (-> self-uncertain|
  identity-diffuse; the agent's view: self-knowing is the better word for
  self-concept clarity, self-certain is the one the check produces).
- **Rename that did not:** unselfconscious -> outgoing -> reserved|
  introverted (shy -> outgoing|confident): "outgoing" reads as
  extroversion, so one way and a near-dupe risk with extroverted /
  gregarious; pending Roger (P, or back to unselfconscious and P).
- **intense / low_key rewritten from scratch** ("charged in any
  conversation ... making every moment feel bigger than it is" / "easy and
  unhurried in any conversation ... never making a moment feel bigger than
  it is"): low_key -> intense|dramatic, intense -> laid-back|relaxed|mellow;
  one way, pending (P).
- **Singles' partners seeded (5):** placid <-> irascible (placid ->
  irritable|irascible) **clean**; joyful <-> joyless (-> joyless|anhedonic|
  apathetic) **clean**; moral <-> amoral (-> amoral|unprincipled|immoral)
  **clean**; forthright -> reticent|guarded (self_censoring -> candid|
  forthright) one way, pending; defeatist -> resilient|persistent
  (persevering -> defeatist|quitting|irresolute) one way, pending.
- emotionally_disengaged / emotionally_engaged_with_others: exact
  candidates reported (emotionally_engaged|emotionally_sensitive;
  emotionally_detached); decision pending.  eastern / western hemisphere
  wording under discussion (convention: 20W / 160E; robust-example texts
  proposed).

Corpus 648 trait files, 496 pairs, validator clean.  Round cost about $1.

## Round 6 (Roger, 2026-09-26): the last chunk-3 calls

- **eastern_hemisphere <-> western_hemisphere**, robust-example texts
  ("east of the Atlantic, in a country such as China, India, Egypt or
  Kenya" / "in the Americas, in a country such as Canada, Mexico, Brazil
  or Chile"; places on the same side under the meridian, 20W/160E and
  political conventions, which a small model otherwise resolves
  differently persona by persona): regenerated under non-X, each names
  the other with a single answer; re-paired.
- **self_conscious <-> unselfconscious**: shy renamed self-conscious and
  outgoing back to unselfconscious (outgoing had read as extroversion),
  both slightly rewritten around being looked at; single answers both
  ways; **clean pair**.
- **guarded <-> forthright**: self_censoring renamed guarded; guarded ->
  candid|forthright|outspoken, forthright -> reticent|guarded|diplomatic;
  **clean pair**.
- **defeatist <-> persevering**: P.
- **intellectually_dishonest <-> intellectually_honest**: rationalizing
  renamed intellectually dishonest and both descriptions rewritten to the
  intellectual-honesty concept (misrepresenting vs fairly stating evidence
  and arguments), distinct from the motivated-reasoning pair.  dishonest
  -> intellectually_honest; honest -> disingenuous|dogmatic, then, after
  one scope edit ("even when it hurts one's own", "never twisting an
  argument to win it"), -> intellectually_dishonest.  **Clean pair.**
- emotionally_disengaged / emotionally_engaged_with_others: candidates
  specified for Roger (emotionally_engaged|emotionally_sensitive;
  emotionally_detached); decision pending.

Corpus 648 trait files, 504 pairs, validator clean.  Chunk 3 is complete
except that one pending pair; doc-status pass next, then Roger's commit.

## Round 7 (Roger, 2026-09-27)

- **heavy_drinker <-> teetotaler**: P (heavy_drinker -> teetotaler|abstainer|
  light drinker; teetotaler -> drinker|social drinker).
- **laid_back <-> intense**: low_key renamed laid-back (intense's check had
  returned laid-back|relaxed|mellow); laid_back -> intense|high-strung.
  **Clean pair.**
- **isolated <-> well_connected**: friendless renamed isolated
  (well_connected's check had returned disconnected|isolated); isolated ->
  well-connected|connected.  **Clean pair.**
- emotionally_disengaged / emotionally_engaged_with_others: the recorded
  nominations quoted to Roger in full (emotionally_engaged|
  emotionally_sensitive, score 4; emotionally_detached, score 3, with the
  generator's reasoning); decision pending.
- self_starting -> proactive not attempted: proactive exists (the
  assistant-behaviour sense, paired with reactive); options put to Roger.

Corpus 648 trait files, 510 pairs, validator clean.

## Round 8 and the rename audit (Roger, 2026-09-27)

- **emotionally_engaged <-> emotionally_disengaged**:
  emotionally_engaged_with_others renamed emotionally-engaged (description
  unchanged: it opens "emotionally engaged with others", mirroring the
  partner's "from others").  emotionally_disengaged -> emotionally_engaged|
  emotionally_sensitive; emotionally_engaged -> emotionally_detached|
  disengaged (the intended word without its adverb; treated as a hit on
  the anti-feminist / non-committal ruling).  Paired.
- **procrastinating <-> self_starting**: P (Roger: leave it as
  self-starting).  A mirror edit on procrastinating ("starting only when
  someone pushes") left its answer at proactive|prompt and was reverted.
  Words considered and rejected for the virtue: prompt (reads as an LLM
  prompt), punctual, diligent, proactive (taken), precrastinating.
- **Rename audit** (Roger asked whether every rename replaced the old
  word in the description and was rechecked): 37 files carry
  `renamed_from`.  No description still contains its old word, except
  emotionally_engaged by design.  Every rename made in this session was
  regenerated under non-X and checked before pairing (the partner's side
  was not re-run: its earlier non-X answer, which named the new word,
  was used).  Renames of existing traits from earlier sessions (lazy,
  calibrated, risk_seeking, apathetic, rule_abiding, easygoing,
  specialist) have their two-way checks in their pair notes; five of
  them have old-style descriptions that never contained the label.
  Gap found and fixed: pairs whose member was renamed after first
  pairing kept the old pair note naming the old word (cmd_pair keeps an
  existing identical pair); the rename and the recheck answer were
  appended to the notes in 42 files.  unflinching (from engaging) is a
  non-X singleton by Roger's 2026-09-17 decision.

Chunk 3 has no open pairs.  Corpus 648 trait files, 514 pairs, validator
clean.

## Roger's review of the roles (2026-09-27)

Roger edited 20 role descriptions (symbiont, addict, aristocrat,
bureaucrat, chief, cleaner, convert, eldritch, hunter_gatherer, laborer,
leviathan, loner, orphan, pirate, prisoner, professor, teenager,
trickster, whale, writer).  Agent fixes: chief (missing space; "likely
holding" -> "and usually holding"; "the division"), pirate ("own
freedom", "a cutlass"), prisoner ("be marked"; "and who'll" -> "and sure
to", to keep the participle list).  All 20 regenerated under the V2.5
rubric ($0.60).  symbiont / parasite: both now say "biological
organism"; role pairs still have no generator check (procedure to
design), so the pair is confirmed by reading only.
initiate rewritten to the everyday sense (someone who has been through
the rite and is now a full member; the draft's "partway through" was the
anthropologist's liminal stage, for which the words are initiand or
novice) and intern's "at the bottom of an office" replaced by "as the
most junior person in an office"; both await Roger's recheck before
regeneration.

**2026-09-27, later:** Roger approved the role fixes and the initiate /
intern rewrites and re-edited eldritch ("weird" -> "uncanny"); eldritch,
initiate and intern regenerated.

## Roger's review of the paired traits, A to E (2026-09-27)

Roger edited 21 descriptions (experiential, flat, antifeminist, engaged,
apolitical, political, hedonistic, civil_libertarian, benevolent,
malevolent, cautious, sycophantic, dog_person, challenging, respectful,
condescending, conservative, eccentric, independent, good, extroverted),
mostly to take the chatbot framing out ("the user", "the questioner",
"in the advice given", "promoting").  Agent fixes: independent ("This
mean"), good ("morallygood"), extroverted ("towards" -> "toward", US),
engaged ("stakes are important" -> "high"), cautious ("taking careful
consideration" -> "considering carefully"), sycophantic ("a certain
person" -> "whoever one is talking to"; "them liking you" -> "being
liked", impersonal voice).  On his instructions: agitated / calm
rewritten as tendencies ("having a calm temperament: rarely worked up
about anything ..."), evil rewritten as what one does, mirrored on good
("doing wrong for its own sake ... drawing people into wrongdoing"),
brash's "sensitive ground included" -> "even on the most sensitive
subjects", and the benevolent <-> uncaring pair and malevolent's singleton
removed in favor of a candidate triangle benevolent / malevolent /
uncaring (`pointer_note` on all three; non-X checks: benevolent ->
callous|indifferent, malevolent -> benevolent).

All 25 edited files were regenerated under `non-X`, checked, and
regenerated `--instructions-only` with their labels restored ($1.38).
**18 pairs reconfirmed.  Five one-way misses** (the partner's recorded
check names the edited side in each): experiential -> theoretical|abstract
(academic), flat -> expressive|dynamic (animated), challenging ->
accommodating|validating (unchallenging), good -> amoral|unprincipled
(evil; amoral now exists), calm -> volatile|reactive|excitable (agitated:
as a tendency the generator's word is excitable, not the state word).
Arrangements left as recorded; Roger to decide.  The existing traits among
the 25 (experiential, hedonistic, benevolent, malevolent, cautious,
sycophantic, challenging, condescending, conservative, independent,
extroverted, calm, evil) join the stale-for-extraction list.

Findings reported to Roger the same day: the communication-style scope of
adaptable and the persuade-others framing of adventurous both come from
Christina Lu's original trait list (January 2026: "Adjusts communication
style and approach based on context and user needs"; "Encourages
exploration, risk-taking, and trying new experiences"), carried into the
2026-03-30 descriptions; 5 of 5 pos instructions of adventurous and of
unadventurous are about urging others.  Descriptions containing "user" or
"questioner": roles assistant and librarian; traits accommodating,
adaptable, educational, exploratory, inquisitive, socratic.  Label-echo
openings ("This means being X: ..."): 344 of the 390 new or updated trait
descriptions, 2 of the 258 older ones; produced by description rule 1
(now reversed) and the writer packets' examples.

## Label-echo strip, framing rewrites and the staged recheck (2026-09-28)

Roger's rulings on the 2026-09-27 findings, applied by `apply_rewrites.py`
and regenerated by `staged_round.py` (session scratchpad).

**Descriptions changed (324 + Roger's own 11):**
- Label echo removed from 310 new or updated descriptions ("This means
  being grateful: noticing ..." -> "This means noticing ..."); 25 keep a
  label in the opening because it carries a qualifier (aristocratic about
  rank, tough on people, from the Eastern Hemisphere, morally good, ...).
  Description rule 1 in AGENT_NOTES now forbids the echo.
- Chatbot sense removed from the last six "the user" descriptions:
  accommodating, adaptable ("adaptable in manner", matched to inflexible),
  educational, exploratory, inquisitive, socratic.
- adventurous / unadventurous rewritten as the persona's own appetite;
  extroverted rewritten to mirror introverted ("energized by company and
  drained by solitude"); flustered, manic and despairing rewritten as
  tendencies.  sycophantic: Roger's "a certain person" restored (a
  sycophant has one or a few targets).
- agitated renamed excitable (Roger's message was cut off after "rename
  agitated to"; excitable was the word calm's check returned; to be
  confirmed).  excitable <-> calm confirms both ways.
- Roger's own edits during the run: composed, energetic, pensive, anxious,
  lethargic, melancholic ("habitual" / "reliably" as tendency markers),
  ambitious, consequentialist, course_correcting, educated.

**Arrangements:** anecdotal <-> data_driven dropped (scope mismatch;
one-way pointers kept, hub-and-spoke tangle); experiential <-> academic
dropped and academic deleted (near-duplicate of theoretical / practical
and erudite); flat / animated, challenging / unchallenging and good / evil
kept by decision (P).

**Regeneration was staged.**  Roger was editing files while the batch ran,
so every file was copied to a staging directory, regenerated there under
`non-X`, checked, regenerated `--instructions-only` with its label, and
merged back only if the repo file's description and label were unchanged
since staging.  Five files he edited mid-run were skipped at merge and
rerun with three more of his edits afterwards.  No edit of his was
overwritten.  Cost: $19.62 for the main batch of 325, $0.64 for the smoke
test and the rerun; with the 2026-09-27 round ($1.38) the two days come to
about $21.6, all recorded in `data/traits/regeneration_usage.json` and
`antonym_check_usage.json`.

**Pair status after the recheck (184 pairs touched):** 142 clean both
ways; 8 miss only on word form (anti-feminist, non-committal, melancholy,
technocrat, dishonest, moral_universalist, poster, drinker); 12 were
already pairs by Roger's decision; 22 are new one-way misses, all
synonyms of the partner (accurate -> careless|sloppy, insecure ->
secure|confident, conventional -> unconventional|original, ...), with the
other side naming its partner in 18 of them.  The descriptions' content
did not change for 19 of the 22, so this measures the check's resampling
noise: about one side in fifteen names a synonym instead of the partner
on a fresh regeneration.  Full table in chat and in the session file
`pair_status_0928.json`; arrangements unchanged pending Roger.

**2026-09-28, later.**  Roger's further edits: anxious, lethargic,
melancholic and pensive ("habitual"), ambitious ("in status"),
consequentialist, course_correcting ("when needed"), educated ("possibly
even on to a graduate degree"), extremist.  Agent fix: extremist
"political and social question" -> "questions".  serene shows no change
on disk.  On his question, apathetic and calibrated did need fixing: both
were checked in with the label echo (the two older files the scan had
counted), and apathetic also said "the question or the person asking it";
both rewritten.  All rerun in staging: ambitious, melancholic,
consequentialist, educated, lethargic, extremist, apathetic and calibrated
confirm their pairs; anxious -> calm|serene|tranquil (composed names
anxious); course_correcting -> inflexible|stubborn|perseverating (a pair
by decision).  No plain label echo remains anywhere in the trait corpus.

**Trait generator V2.**  Roger: "It's looking increasingly like we need to
edit the trait instruction generation and redo everything."  Recorded as
a leaning, with the issue list, in TRAITS_TO_ADD § "Trait generator V2"
and a pointer in AGENT_NOTES.  He continues the review from F on
2026-09-29.

**Resample of the 22 one-way misses (Roger, 2026-09-28).**  The 22 missing
sides were regenerated unchanged through the staged procedure ($1.35).
**14 came back** and name their partner on the second sample: accurate,
bland, body_confident, brand_loyal, dry, erratic, friendly, insecure,
job_hopper, open_minded, resourceful, rule_bending, self_critical,
slovenly.  **8 missed again with the same answer**, so those are
systematic, not noise: adventurous -> cautious|risk-averse, anxious ->
calm|relaxed, conventional -> unconventional, inquisitive ->
direct|presumptuous, old_money -> self-made|nouveau-riche, remorseful ->
unrepentant|unapologetic, rooted -> rootless|cosmopolitan, unassuming ->
flamboyant|ostentatious|showy.  Reading: a single miss on a previously
clean pair is noise about two times in three; a miss that repeats with
the same words is a real preference of the generator for another label.

**Roger's decisions on the eight repeat misses (2026-09-28).**
- adventurous: "the new and the risky" -> "the new and the untried";
  adventurous -> cautious|unadventurous.  Clean pair again.
- remorseless renamed unrepentant: unrepentant ->
  remorseful|penitent|contrite; remorseful had answered unrepentant twice.
  Clean pair.
- deracinated renamed rootless: rootless -> rooted; rooted had answered
  rootless twice.  Clean pair.
- old_money: Roger asked for new_money to be renamed self_made.  A
  self_made file already existed (3D), written as old_money's mirror, so it
  was used instead of overwriting it.  self_made -> privileged|inherited on
  a fresh sample; after a scope edit ("having made one's own money: born
  into a family with none ... in one's own lifetime") ->
  privileged|born-privileged|old-money.  old_money <-> self_made paired;
  old_money <-> new_money dissolved; new_money kept as a one-way pointer
  pending Roger's word on deleting it.
- inquisitive: contrast clause dropped; inquisitive -> incurious, which is
  an existing trait paired with curious.  Pair with uninquisitive left as
  recorded; Roger to decide.
- anxious / composed, conventional / eccentric, unassuming / theatrical:
  kept by decision (P).

**Check history (Roger, 2026-09-28).**  Every check's candidates and the
instruction set it read are now appended to
`data/traits/antonym_check_history.jsonl` by `seed_entities.py` (`check`,
`rename`) and by the staged runner; queue entries keep a `check_answers`
list.  385 earlier records from 2026-09-27/28 were backfilled without
their instructions.  Tests: `TestCheckHistory` in
`data_analysis/tests/test_seed_entities.py` (43 pass).

**Roger's review, paired traits F to O (2026-09-28).**  Nine edits found by
diffing against the last staged text: figurative, flourishing and
languishing ("doing well" / "doing poorly"), loyal and treacherous
("partner" for "family"), many_siblings ("four or more"),
motivated_reasoning_immune, oblivious, spiritual.  No spelling fixes
needed.  intellectually_honest / intellectually_dishonest rewritten at his
request from a discussion with someone else to one's own intellectual
work, covering both ("truthful about what one's own work shows, to oneself
as much as to anyone else: reporting results as they are, failures
included, ..."); both name each other with single answers.  All eleven
regenerated in staging and rechecked: nine confirm.  languishing ->
thriving and motivated_reasoning_immune -> confirmation-biased|
wishful-thinking missed and were resampled once per the new rule; results
in chat and in `data/traits/antonym_check_history.jsonl`.

**2026-09-28, later.**  Roger edited intellectually_honest ("careful and
truthful ... admitting uncertainties and errors"; agent fixed "an
uncertainties") and intellectually_dishonest ("biased and untruthful ...
never admitting an error or uncertainty"); both regenerated in staging and
rechecked, results in the history file.

**old_money's partner settled (Roger, 2026-09-28): new_money, merged with
self_made.**  Roger: the two are not both needed; find which makes the
better clean pair with old_money, or combine them.  Samples, all under
`non-X` and all in the history file:
- old_money (5): self-made | self-made|nouveau-riche | self-made|new money
  | new_money|self-made | new_money|self-made.  It names self-made every
  time and new money or nouveau-riche in four of five.
- self_made (5, the last two on a merged description under that label):
  privileged|inherited (three times) | privileged|born-privileged|old-money
  | privileged|old-money|born-into-wealth.  "privileged" first every time;
  old-money only ever second or third.
- new_money (5, the last two on the merged description): old_money every
  time, single answer, score 4.
So the name new money gives the pair that confirms both ways, and the
label drives the partner side's answer more than the description does.
new_money's description now combines the two: "being rich within one's own
lifetime, the first in the family with money, with the house, cars and
club memberships all earned and newly bought, none of it handed down, and
the manners of wealth still being learned".  old_money <-> new_money
paired; self_made deleted (copy kept in the session scratchpad).
inquisitive / uninquisitive kept by decision (P); excitable confirmed as
the name for the former agitated.

**Roger's review, the remaining paired traits and the unpaired ones
(2026-09-28).**  Seventeen edits found by diffing against the last staged
text: artistic, attention_seeking, confabulatory, gentle, inspirational,
irresponsible, prudent, reckless, risk_seeking ("taking bold actions ...
venturing", no longer "promoting ... encouraging"), science_skeptical,
science_trusting, self_absorbed, self_accepting, strategic, tactical,
unplugged, unpopular.  Agent fixes: strategic "withmultiple" -> "with
multiple", tactical "small-scal" -> "small-scale".  urgent rewritten at
his request as the persona's own disposition, mirrored on relaxed ("being
in a hurry about everything, treating every matter as pressing, dealing
with things at once however inconvenient, and assuming that whatever it
is cannot wait").  All regenerated in staging and rechecked: every pair
confirms except two, which missed the same way on a second sample:
strategic -> reactive|impulsive, then reactive (tactical -> strategic);
urgent -> unhurried|patient twice (relaxed had named urgent|hurried).
Unpaired traits' answers are in the history file.

**responsible <-> irresponsible, new clean pair.**  Roger asked whether
responsible could pair with irresponsible and whether it fills a gap.
Seeded as the mirror ("meeting one's responsibilities: paying the rent
first, turning up for every shift, looking after the kids and the bills
oneself, and never resting while a duty is undone"); responsible ->
irresponsible, irresponsible -> responsible|conscientious.  Neighbours, not
duplicates: dependable (keeping commitments to others), conscientious
(care and thoroughness in work), accountable (owning one's part after
something goes wrong).  Corpus 647 trait files, 510 pairs.

**Last two pairs of the review (Roger, 2026-09-28).**  relaxed renamed
unhurried: unhurried -> hurried|urgent; urgent had answered
unhurried|patient twice.  Clean pair.  strategic rewritten as tactical's
mirror ("taking each problem as one piece of a larger plan, passing up the
move that wins now for the one that wins in the end, looking at the
large-scale picture and the long game, and always keeping a fallback"):
strategic -> shortsighted|tactical|reactive; tactical -> strategic.  Clean
pair; Roger's previous text had answered reactive twice.

**State at the end of Roger's read-through.**  647 trait files, 337 role
files; 255 trait pairs (510 members), 16 in the moral-circle sequence, 6
in triangles, 83 singletons, 37 unclassified (36 real-word one-way
pointers for the tangle and triangle passes, plus four non-X files with
no arrangement: calculating, pensive, ritualistic, malevolent).  Every
trait has 5 instruction pairs, 40 questions and an eval prompt built
from its current description; no description opens with a label echo or a
", never <partner>" clause or says "user".  Every name in
`goal_roles_and_traits.json` has a file.  438 checks are in the history
file.  Cumulative generation spend: traits $56.98, roles $22.57, antonym
checks $6.51.

## Finishing chunk 3 before the check-in (Roger, 2026-09-28)

Roger's rulings: seed the three partners proposed for unpaired traits;
work on every chunk-3 entry still unfinished; the generator V2 decision is
deferred until after the check-in, the arrangement passes until tangles are
handled, Bourdieu capital composition and the diaspora role are deferred;
male / female are physical traits (filed under the physical track, queue
chunk 6); the three hedges flagged in his edits stay.

Ten traits seeded, generated in staging and checked under `non-X`:
- harsh <-> gentle: harsh -> gentle|tactful; gentle -> harsh|blunt (four
  samples).  Clean pair; acerbic keeps its one-way pointer at gentle.
- self_effacing <-> attention_seeking: self_effacing ->
  self-promoting|boastful|attention-seeking; attention_seeking ->
  self-effacing first on three samples.  Pair recorded.
- parent <-> childless (wording settled: parent, childless): parent ->
  childless|child-free; childless -> parent.  Clean pair.  parent also
  exists as a role (an accepted name collision).
- upper_class <-> working_class (wording settled from high / low social
  class): upper_class -> working-class; working_class ->
  upper-class|privileged.  Clean pair.
- other_focused (for self_absorbed): other_focused -> self-focused twice;
  self_absorbed -> other-focused on three samples.  One-way on a synonym;
  left as `non-X` singletons for Roger.
- fair / unfair: unfair -> fair|impartial; fair -> biased|partial twice.
  One-way on a synonym; left as `non-X` singletons for Roger.
- vulnerable_narcissistic (singleton, the recorded fallback) ->
  emotionally secure|psychologically secure.
Not adopted: morally_disengaged (dropped from the consolidated list on
2026-09-08; its parts are covered), and the five reviewer rejections
(flow_prone, gloating, resentful, left_brained, right_brained).  Chunk 3
now has no entry in a live or undecided state; its sixteen backlog
entries are optional or fallback names.

## The parked list: Roger's rulings and one more round (2026-09-28)

Roger read the 33 parked entries of chunks 1, 1b and 2 and ruled:

- **Dropped (25):** the reviewers' rejections stand for conclusive, general,
  grave, hopeful, iconodule, ingenuous, literalist, meaningful, measured,
  modern, orderly, perfunctory, predictable, rational, restrained,
  scientific, secure, sedate, stable, straightforward, structured, subdued,
  systematic, tender and unprovocative (`not_adopted`).  Their partners were
  already handled by call 4.  The writers' full drafts of these were never
  kept; only the clauses the reviewer quoted survive (in each entry's
  `decision`).  From chunk 4 on, keep a rejected draft in the queue entry.
- **wasteful:** stays parked for the tangle pass (`backlog`).
- **The essence tetrahedron.**  Roger: nihilistic, essentialist,
  constructivist and existentialist are four answers to one question
  (no essence; fixed and inherent; socially made; individually made), none
  an exact opposite of another.  The constructivist <-> essentialist pair
  was dissolved: both relabelled `non-X` with the instruction set generated
  under it, and all four record a `tetrahedron`.  Checks under non-X:
  essentialist -> constructivist|nominalist|anti-essentialist;
  constructivist -> objectivist|realist (twice, the same words);
  existentialist -> essentialist|traditionalist (file as it stood);
  nihilistic -> idealistic|purposeful (file as it stood).  The pair is a
  judged axis in `pair_list_clean.json`, which was left alone.  Pairs in
  the corpus: 258.
- **Aligned AI: option 1.**  `aligned_ai_virtuous` written and generated
  (role, singleton).  The rename of `aligned_artificial_intelligence` to
  `aligned_ai_instrumental` was not done and went back to Roger with the
  list of what the stem keys (a judged axis in four pair lists, a steering
  config, `goal_roles_and_traits.json`, four scripts, the extracted data).

**unironic, tried again** (its rejection rested on `sincere`, deleted on
2026-09-26).  In staging, nothing written to the repo:
- unironic -> ironic|sarcastic (score 4)
- ironic under non-X -> sincere|straightforward (4); resample ->
  sincere|earnest (3)
- the same description under the label `sincere` -> sarcastic|ironic (3)
So `sincere` <-> `ironic` confirms from both sides and `unironic` <->
`ironic` from one.  Roger to choose the label.  ironic's label still
reads `sincere`, which has no file.

**The three `non-` labels** (Roger: would unhyphenated or un- forms make
pairs, and do they fill gaps?).  The collision with the placeholder
convention was not the only reason they were never written: each would
have been an absence, and each pole is held already.
- ritualistic: `unceremonious` tried in staging.  unceremonious ->
  ceremonious|formal (4); ritualistic under non-X ->
  spontaneous|improvisational (3).  No pair.  Both of ritualistic's words
  exist and are paired (spontaneous <-> formulaic, improvisational <->
  methodical).
- calculating under non-X -> sincere|guileless|transparent (3); a mirror
  under the label `uncalculating` -> calculating (4).  One-way; guileless
  (paired with scheming) holds the pole.
- pensive: no file tried; its two checks named the existing unreflective
  both times, and lighthearted and energetic hold "breezy" and "lively".

Runner: `pair_try.py` in the session scratchpad (new seeds live only in the
staging copy and reach the repo only when both checks name each other;
`instr` mode regenerates an unchanged existing file `--instructions-only`
under non-X; every check goes to the history file with the instructions it
read).  Spend: about $0.40.

## Roger's rulings on the staged results (2026-09-28, later)

- **other_focused <-> self_absorbed** and **fair <-> unfair**: paired by
  decision (P).  Labels pointed at each other, instructions regenerated
  `--instructions-only`.
- **sincere <-> ironic**: `sincere.json` written from the staged
  generation (the text first tried as unironic), labelled `ironic`,
  instructions regenerated; ironic kept its instructions (its label was
  already `sincere`) and lost its pointer note.  Confirmed from both
  sides.  The 2026-09-18 sincere (sarcastic's completion, deleted
  2026-09-26) had a broader text and answered insincere|disingenuous.
- **ritualistic**: gave up; `non-ritualistic`, singleton, with the
  instruction set generated under that label.  unceremonious not adopted.
- **uncalculating <-> calculating**: paired by decision; `uncalculating.json`
  written from the staged generation; calculating relabelled from
  `non-strategic` and regenerated `--instructions-only`.
- **pensive**: one-way pointer at `unreflective` (paired with
  introspective), unclassified, a tangle for the tangle pass;
  regenerated `--instructions-only`.  non_contemplative not adopted.
- **The aligned-AI roles.**  Roger: the `(foo)` label form is for entries
  drawn from a named external set, so the display names are
  `instrumentally-aligned AI` and `virtue-aligned AI` (overrides in
  `entity_id.ROLE_DISPLAY_OVERRIDES` and the role generator).
  `aligned_artificial_intelligence.json` was renamed
  `instrumentally_aligned_ai.json` (git mv, `renamed_from`; description
  and instructions unchanged; eval prompt rebuilt from the template under
  the new name; its pair with paperclip_maximizer and
  `goal_roles_and_traits.json` updated).  `aligned_ai_virtuous.json` was
  renamed `virtue_aligned_ai.json` and regenerated in a staging copy from
  Roger's edited description ("acts helpfully with honesty, ...").  Both
  record a `set` arrangement.  Left under the old stem on purpose: the four
  `pair_list*.json` entries and the judged axis directory
  `aligned_artificial_intelligence_vs_paperclip_maximizer`, the steering
  config and question file `mechanic_aligned_artificial_intelligence_v1`,
  `combination_scores.json`, the goal-classification outputs, and the
  extracted data; four scripts that hard-code the stem now list both.

Corpus after this round: 659 trait files, 262 pairs, 338 role files.

## Pair lists split into a record and a working copy (2026-09-28, Roger)

`pair_list_clean.json`, `pair_list_di.json` and `pair_list_goalnongoal.json`
are both the record of judged cohorts and the default input of live
scripts.  Roger had each copied byte-exact to `pair_list_<cohort>_v1.json`
(the record; force-added, since `/roger` is git-ignored) and the file
under the current name cleaned for new work:

- constructivist / essentialist removed from clean (60 -> 59) and di
  (61 -> 60): no longer a clean pair.
- aligned_artificial_intelligence -> instrumentally_aligned_ai in clean,
  di and goalnongoal (38, regenerated by `tools/build_pair_lists.py`,
  which reproduces both files).
- Every entry of all thirteen pair lists was checked against the corpus:
  no trait end has been renamed, and every other entry is still a pair,
  except compassionate / callous in `pair_list_di.json` and
  `pair_list_13_new.json`, which was never a clean pair (an edge of the
  compassionate / malicious / callous triangle, included on purpose) and
  was left for Roger.
- `pair_list_responses.json` needed no change.  The dated and early
  lists are records and were not touched.

The renamed-from fix: `assistant_axis.entity_id.resolve_renamed_stem`
maps a stem with no corpus file to the file whose `renamed_from` records
it; `axis_judge_correlation.py` uses it for pole descriptions (with a
warning) and for the `pole_instructions` provenance input, so a `_v1`
list runs against the renamed corpus.  Vectors and judge caches are
never remapped.  Tests: `TestResolveRenamedStem`, `TestPoleInstructionPath`.

## Close of chunks 0 to 3 (2026-09-28)

- compassionate / callous removed from the current `pair_list_di.json`
  (59 entries): Roger, "we need to figure out how to handle triangles,
  for now it's not a pair".  It stays in `pair_list_di_v1.json` and
  `pair_list_13_new.json`, the records.
- Steering scripts do not use the renamed-stem lookup; Roger accepts
  that, since no steering is expected before the embeddings are
  regenerated (AGENT_NOTES, code housekeeping item 5).
- Doc-status pass: TRAITS_TO_ADD (status paragraph, the male / female
  item, the Bourdieu deferral, the chunk table), ROLES_TO_ADD (deferrals,
  the aligned-AI roles), AGENT_NOTES (pair-list generations, stale list,
  judged axes whose pole changed, housekeeping items 5 and 6),
  `results_analysis/README.md`, `data/README.md` counts.

State: 659 trait files, 262 pairs, 1 tetrahedron, 2 triangles, the
moral-circle sequence, 80 singletons, 34 unclassified; 338 role files.
The queue holds no chunk 0 to 3 entry in an open state.  Validator clean,
entity lists in sync, 1205 tests pass.  Next: Roger's commit (base
93a8554), then chunk 4.

## Check-in (2026-09-28)

Roger had the whole of the September work checked in, except
`reports/trait_gap_generation/` (a separate line of work).  Branch
`anthropic-vllm-uv`, on top of `93a8554`, which is therefore the corpus as
it stood before: wherever the notes of this month say "git HEAD" for a
pre-change text, read `93a8554`.

| commit | what |
|---|---|
| `3f6de81` | the corpus: instruction files, lists, goal list, queue, usage and check records, TRAITS_TO_ADD, ROLES_TO_ADD |
| `d83ad01` | instruction generators and the antonym check: V2.5 role rubric, usage records, `generator` field |
| `25ce633` | entity names: `corpus_display_name`, ASCII stems, `resolve_renamed_stem`; plot labels |
| `fd14778` | the `arrangement` field: loader, validator, backfill |
| `5b61820` | `tools/sync_entity_lists.py` |
| `0ff43e8` | `data_analysis/seed_entities.py` |
| `fc2b8e7` | axis judging: judge-model guard on resume, Sonnet 4.6, renamed poles |
| `820d07f` | pair lists: `_v1` records and working lists |
| `74364d1` | AGENT_NOTES, CLAUDE.md, `.claude/rules/` |
| `f5a5cd3` | reports: this log, the voice audit, the rubric V2 pilot |

A last commit carries this entry and the doc references to `93a8554`.
