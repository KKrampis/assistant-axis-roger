# Rubric V2 pilot on the 45 repaired roles (2026-09-11)

*Note added 2026-09-28: wherever this report says "git HEAD" it means the
role files as committed before the September 2026 corpus check-in, that
is commit `93a8554`.*

**Question (Roger):** for the roles whose instructions were rewritten this
week, regenerate them under the V2 rubric, compare, and say whether V2 does
what we wanted (voice, softening, semantic content, and the questions),
what problematic effects appear, and what to change.

**Method.**  The 45 roles whose `instruction` field differs from git HEAD
(the 2026-09-11 voice-repair batch plus advocate) were regenerated once
with `--style RogerV2` (Sonnet 4.6, temperature 1.0, one combined call,
$1.12 in total).  The V2 outputs are kept in `roles_v2/` beside this file;
the corpus files were restored to their V1-rubric state, so nothing in
`data/` changed.  Both candidates therefore share the *same rewritten
description*; the comparison isolates the rubric.  Automated metrics were
computed over the three generations of each role (original = old
description + V1 rubric; repaired = new description + V1 rubric; V2 = new
description + V2 rubric).  Four Opus judges then read the 45 roles in
four packets, each role as description plus two sets labelled A and B in
random order with no hint of origin, and rated voice, softening, semantic
fidelity, length, problems, questions and overall preference.  The
per-item verdicts are in the session scratchpad (`v2_pilot/verdicts/`).

## Automated metrics

| generation | instr. words median (p10-p90, max) | observer-register markers | softening words | rubric meta-words leaked | openers | questions: addressed "you" / advice-asks / role named |
|---|---|:-:|:-:|:-:|---|---|
| original (old desc, V1 rubric) | 17 (14-21, 27) | 17 | 10 | 0 | You are 90, Please be 53, Act as 45, Embody 11, Be 11 | 46% / 8% / 0% |
| repaired (new desc, V1 rubric) | 27 (21-34, 44) | 19 | 5 | 0 | You are 103, Act as 46, Play the role of 26, Embody 19, Please be 11 | 75% / 5% / 1% |
| V2 (new desc, V2 rubric) | 32 (26-38, 47) | 0 | 1 | 0 | You are 91, Act as 71, Be 61, Please be 1 | 79% / 2% / 1% |

Markers = "navigating", "individuals", "demonstrate", "appropriate",
"engage with", "the challenges of", "in a healthy way" over 225
instructions.  Question sets share 32 of 1800 items verbatim between the
repaired and V2 generations.

UK idiom: V1 8, V2 7
     V2 advocate: 'on-side' in: Be an advocate. Your job is to make the strongest possible case for yo
     V2 ecosystem: 'organis' in: You are an ecosystem: a living web where every organism feeds, is fed 
     V2 student: 'flat' in: Be a student who's behind on readings, stressed about grades, sharing 
     V2 student: 'uni' in: You are a uni student — you pull all-nighters, bullshit your way throu
     V2 surfer: 'flat' in: Act as a surfer — you live for the days when the waves are pumping, ha
     V2 symbiont: 'organis' in: Be a symbiont. You are woven into another organism's life: you provide
     V2 symbiont: 'organis' in: Act as a symbiont. You have found a partner organism and struck a biol
  profanity: V1 0, V2 2
     V2 student: 'bullshit' in: You are a uni student — you pull all-nighters, bullshit your way throu
     V2 student: 'shit' in: You are a uni student — you pull all-nighters, bullshit your way throu
  gendered: V1 0, V2 3
     V2 pirate: 'a man' in: You are a pirate. You sail where you please, take what you want, answe
     V2 warrior: 'a man' in: Act as a warrior hardened by years of close combat—you know the weight
     V2 warrior: 'his' in: Act as a warrior hardened by years of close combat—you know the weight
  em-dashes: V1 43, V2 104 (over 225 instructions each)
  V2 instructions over 30 words: 137/225; V1: 72/225

## Blinded judgement, unblinded

Tallies over the 45 roles (V1 = repaired description + old rubric; V2 =
same description + new rubric):

- **voice**: V2 more in the role's own terms in **45 of 45**.
- **softening**: V1 the more softened set in **33**, neither in **12**, V2 never.
- **questions**: V2's set better in **41**, V1's in 2 (bartender, and a tie counted to V1 by one judge), same in 2.
- **preferred overall**: V2 in **45 of 45** (confidence high in most).
- **problems flagged**: 201 in V1 sets, 113 in V2 sets, and of a different kind (below).
- **semantic fidelity**: the judges' notes flag V2 as *adding, narrowing or broadening* in 28 roles and V1 as *dropping, narrowing or softening* in 9. The two rubrics fail differently: V1 paraphrases the description five times in an observer's register and drops its uncomfortable parts; V2 instantiates it with particulars, and sometimes fixes the instance too tightly.

### Per role

| role | voice | softened | questions | preferred | conf | semantic note | length note |
|---|:-:|:-:|:-:|:-:|:-:|---|---|
| addict | V2 | V1 | V2 | V2 | high | B adds a repeated-relapse history and a confessional stance not in the description; A adds an unstated motive, the substance as the thing that 'makes the noise stop'. Neither drops anything central. | Both compact; B's five instructions are near-paraphrases of one another, A's carry distinct particulars. |
| adolescent | V2 | V1 | V2 | V2 | high | Neither drops anything. A instantiates with ages, crushes, curfews — detail beyond the description but inside it. B restates the description five times with synonyms, adding nothing. | B generic and interchangeable; all five could be one prompt. A concrete and short. |
| advocate | V2 | V1 | V2 | V2 | high | A omits the description's central salary-over-conviction motive entirely, narrowing advocate to skilled communicator. B is faithful but instruction 1 near-verbatim transcribes the description rather than dramatising it. | A's five instructions are interchangeable synonym-swaps; B's are longer but each adds a distinct working fact. |
| auctioneer | V2 | V1 | V2 | V2 | high | B broadens to a generic master-of-ceremonies ('sales of all kinds', 'manages the entire bidding process') and loses the money-extraction motive. A adds reserves, increments and phone bidders — extensions beyond the description, but consistent with it. | B padded with rank adjectives; four of five items open 'seasoned/professional/experienced'. A tighter, more specific. |
| bartender | V2 | V1 | V1 | V2 | medium | A broadens into bar management and mixology expertise ('runs a smooth bar operation', 'deeply knowledgeable about spirits'); B keeps the description's regulars-for-the-company element that A reduces to 'builds rapport with customers'. | A padded with résumé qualifiers ('seasoned', 'deeply knowledgeable'); B tighter and more concrete. |
| criminal | V2 | V1 | V2 | V2 | high | A drops 'doing violence for pay' entirely and adds an honour code not in the description; broadens vaguely to 'organized crime'. B covers theft, dealing, scams, violence, fences, police — full. | A padded with underworld boilerplate repeated five ways; B tighter and more particular. |
| cyborg | V2 | V1 | V2 | V2 | medium | A broadens toward superhuman augmentation ('limbs that don't tire', 'senses that see beyond the spectrum'), tilting a seam-and-strangeness description into power fantasy. B covers all listed hardware but blunts the never-closing seam. | Both fine; B's instructions pad with abstractions ('a distinctive perspective', 'inner tensions') rather than specifics. |
| destroyer | V2 | neither | V2 | V2 | high | A names all four motives (rage, duty, delight, clearing ground) but adds 'sent to dismantle', implying orders throughout. B never names rage or orders in the instructions, though its questions restore hiring; B adds abstract targets (organisations, plans). | A generic and interchangeable across five items; B tighter, each item adds a distinct angle. |
| dilettante | V2 | neither | same | V2 | medium | A adds 'always the most interesting person at the party', a self-flattering claim absent from the description; both otherwise faithful to dabbling, confidence exceeding knowledge, and drift. | Both adequate; B's five instructions restate one sentence five times, A's fifth is slightly ornate. |
| divorcee | V2 | V1 | V2 | V2 | medium | Both faithful. A adds legal-process framing (settlements, proceedings) slightly beyond the description; B covers ex, lawyers, divided house and friends, custody, and what to do with what's left. | A abstract and repetitive across five near-identical prompts; B crisp, one concrete image each. |
| ecosystem | V2 | V1 | V1 | V2 | medium | A adds a 'balance' frame the description does not contain (it says flux and exchange). B adds accurate extras — heat loss, coevolution, carrying capacity — beyond the description but consistent with it. | Both compact; A's five instructions restate one sentence five ways. |
| eldritch | V2 | neither | V2 | V2 | medium | B adds 'your intentions unreadable even to yourself' — the description says unreadable, not self-opaque — and imports sanity-cracking, absent from the description. A broadens with 'knowledge spans dimensions beyond human perception', also not in the description. | A is adjective-padded and the five items are near-synonyms; B varies and is more specific. |
| emissary | V2 | neither | V2 | V2 | high | Both add negotiation, which the description ('carry its message') does not state. A also adds holding the mandate under pressure and reporting back verbatim — fair extensions of 'bound by their instructions'. | B generic and interchangeable across all five; A longer but every clause does work. |
| fixer | V2 | neither | V2 | V2 | medium | B (#4) adds 'silencing witnesses' and (#5) 'a network of corrupt officials, criminals' beyond the description; A adds making 'inconvenient people' vanish. Both broaden toward violence the description only implies. | B #1 is the description verbatim plus a clause; A #5 lists four noun-categories. Both lean short. |
| flaneur | V2 | neither | V2 | V2 | high | Both faithful. A's 'endlessly entertaining spectacle' slightly overstates; B stays within window-shopping, cafes, crowds and faces as the description has them. | A is padded: five near-identical sentences. B varies concretely without over-length. |
| fool | V2 | V1 | V2 | V2 | high | A narrows the concept: the folly becomes a device concealing wisdom, and item 3 adds an unstated 'medieval' setting. B preserves all three kinds of utterance and the fool's own uncertainty about which is which. | A over-explained and polished; B compact and playable. |
| hybrid | V2 | V1 | V2 | V2 | medium | B narrows the concept to cultural or ethnic mixture — two inheritances, two languages — dropping the description's two species and flesh-and-machine; A keeps those and adds 'mortal and immortal'. | Both tight; B's 'not on either shore but in the water between them' verges on purple. |
| idealist | V2 | V1 | V2 | V2 | high | B narrows to public policy and institutions ('society, relationships, and institutions', 'policies'), losing the personal register of the description. A keeps the general 'how things should be'. Neither drops the values-over-practicality core. | B abstract and interchangeable; #5 padded. A tight, each prompt adds a distinct stance. |
| immigrant | V2 | V1 | V2 | V2 | high | B narrows the concept: monthly remittances, pending papers, 'kept your head down', 'crossed a border' fix one working-class precarious type the description does not specify. A adds nothing but abstracts everything. | A's instructions are long and list-like; B's are shorter and carry more information per clause. |
| leviathan | V2 | V1 | V2 | V2 | high | B adds 'unkillable' and 'mindless', neither in the description, and its questions ask for wisdom the description explicitly rules out ('Do you have any wisdom you would share with the world?'). A holds to appetite and cold. | B repetitive adjective stacks across five near-identical items; A terse and concrete. |
| loner | V2 | neither | V2 | V2 | high | Neither adds nor drops materially; B sharpens the description's 'resigned' into a settled refusal, A follows the description almost word for word. | A's forty questions are near-interchangeable preference items; both instruction sets are short enough. |
| martyr | V2 | V1 | V2 | V2 | high | B narrows to the honourable sense and largely drops the guilt-parading second sense the description flags 'with an edge'. A holds both, keeping 'refuse to recant' alongside the weaponised suffering. | B verbose and abstract ('weaving personal suffering into every response'); A tighter, though five prompts hammer one note. |
| merchant | V2 | V1 | V2 | V2 | high | A adds a mild adversarial edge ('which suppliers to trust and which to squeeze') not in the description. B covers every element but converts each into a corporate competency; neither drops anything central. | B is generic rather than long; every instruction is the same competency list rephrased. |
| negotiator | V2 | V1 | V2 | V2 | medium | A broadens the role from bargaining on a side to advising anyone on any bargaining situation, losing the description's core stance. B keeps 'you represent one side, you bargain to win' and the walk-away test. | A generic, five interchangeable adjective-led items; B specific and varied. |
| newlywed | V2 | V1 | V2 | V2 | high | Neither drops anything; A adds honeymoon sleep deprivation and comic particulars, fair extensions of 'discovering the small habits'. B adds nothing beyond the description's own wording. | B padded with synonym pairs ('joys and surprises', 'wonder and adjustment'); A specific throughout. |
| orphan | V2 | V1 | V2 | V2 | high | Both faithful. B adds 'built their identity around that formative absence' — a therapist's interpretation not in the description. A #5 adds 'grief that sits in a place before language', beyond but not against it. | B generic, five paraphrases of the same sentence; A concrete, #5 slightly florid. |
| perfectionist | V2 | V1 | V2 | V2 | high | A broadens: instruction 4 extends the standard to 'everyone else's' work and unsolicited criticism, which the description does not state. B stays inside the description but only by restating it. | B is five paraphrases of one sentence; generic and interchangeable. |
| pirate | V2 | V1 | same | V2 | medium | A covers every element of the description but only at the level of labels. B implies the nautical speech ('salt-tongued') rather than instructing it, adds unstated historical geography, and fixes the pirate as male. | A padded with adventure clichés; B concrete but its questions are generic. |
| prisoner | V2 | V1 | V2 | V2 | high | A's instructions omit the description's ex-con stigma, which only its questions reach; A also invents specifics — state penitentiary, sixteen hours locked, marking the wall — plausible but unstated. B covers stigma but adds nothing concrete. | B's five instructions are interchangeable summaries; A specific without padding. |
| refugee | V2 | V1 | V2 | V2 | high | Both faithful. A adds 'mental health'/'cultural differences' framing; B adds children in school and a bag-and-border journey — instantiation within the description, not drift. Neither drops anything central. | A five near-identical NGO summaries; B concrete, #4 a touch long but each detail earns place. |
| retiree | V2 | V1 | V2 | V2 | high | B adds unearned comfort ('living comfortably', 'cherishing time with grandchildren') and blurs the thinning ranks of friends into 'shifting social world'. A adds 'gold watch' and stated opinions, both consistent. | B is generic; each instruction restates the same four-item list in euphemism. |
| reviewer | V2 | V1 | V2 | V2 | high | Both keep 'for the reader, not the creator'. A broadens to 'works across culture and commerce' and self-flattery; B adds a demand for supporting specifics not in the description, but consistent with it. | A padded with self-praising adjectives, five near-identical rewrites; B tight. |
| scout | V2 | neither | V2 | V2 | high | Neither drops anything. B adds 'hostile environments', A adds 'country nobody has mapped yet' — both consistent extensions of the description. | B's five are one-sentence restatements of each other; A longer but concrete, with some overlap between its first and fifth. |
| shaman | V2 | V1 | V2 | V2 | high | A broadens toward pan-traditional core shamanism, adding plant medicines, omens, animal spirits, four directions not in the description. B adds soul theft and broken taboos but stays with trance travel, cures, answers, rites. | A generic and interchangeable; B concrete, occasionally florid ('what hunts them in the unseen'). |
| simulacrum | V2 | neither | V2 | V2 | medium | Both faithful; B's instruction 1 is a near-verbatim recomposition of the description. Neither drops the absent-or-never-existent original or the aware/unaware ambiguity. | B's questions are long and academic; A's are short but several are near-duplicates of each other. |
| smuggler | V2 | V1 | V2 | V2 | high | B drops the description's named cargo (cigarettes, drugs, weapons, people) and narrows the role to technique and logistics. A carries all four and matches the routes-bribes-patrol-timing core precisely. | B generic and manual-like; A concrete, both question lists are long but A's carry more judgment. |
| student | V2 | V1 | V2 | V2 | high | A narrows to a British undergraduate ('uni', 'flat', 'seminars', 'loan installment') where the description is unspecified; B adds nothing and generalises the description's particulars away. | B generic and repetitive across its five; A specific, none padded. |
| surfer | V2 | neither | V2 | V2 | medium | Both faithful. B adds board selection, fitness and forecast-reading skill beyond the description; A is closest to its 'slang of the lineup' and pre-dawn swell check. Nothing central dropped. | B generic; #5 padded into a capability list. A short and specific. |
| symbiont | V2 | V1 | V2 | V2 | high | B adds material the description lacks (taking more than you give, being expelled, rivals for the host) but it sharpens rather than distorts. A adds nothing and narrows to serene mutualism. | A is generic and repetitive; B's instructions are longer but each names different traded goods. |
| teenager | V2 | V1 | V2 | V2 | high | B abstracts the phones, crushes and curfews into 'social drama' and 'trends' and adds a slang directive not in the description. A keeps every particular but item 2 fixes the age at fifteen, narrowing the stated thirteen-to-nineteen band. | B generic and interchangeable; A specific, occasionally overwritten. |
| traditionalist | V2 | V1 | V2 | V2 | medium | A imports a patriarchal-religious frame ('what the fathers kept, you keep') not in the description, and its questions narrow the role to a Western Catholic culture-war profile; B stays faithful but abstract. | Both fine; B's five instructions are five restatements of one sentence. |
| trickster | V2 | V1 | V2 | V2 | high | Faithful both ways; B slightly narrows the 'goes too far' collateral damage into comic mayhem. A keeps allies taking the fall. Neither adds substantive content beyond the description. | B generic archetype-speak, five restatements; A concrete and varied. |
| warrior | V2 | V1 | V2 | V2 | medium | B narrows the weapons to pre-modern melee ('blade and shield', 'the weight of a sword'), a setting the description leaves open. A adds cause-service; neither drops the killing or the comradeship. | Both similar; A's instructions pad with valorising abstractions rather than facts. |
| whale | V2 | neither | V2 | V2 | high | B item 3 instructs authorship rather than being — 'describe your world through the lens of deep dives' — and otherwise matches. A adds tonnage and frequency detail beyond the description, consistent with it; neither drops migration, song, diving or breathing. | B's five items are interchangeable paraphrases; A varies and is specific. |
| zealot | V2 | neither | V2 | V2 | medium | Neither adds nor drops; both correctly leave the cause unnamed as the description does. A frames the persona as an answering style, B as a person. | Both tight; A's five recycle 'compromise' and 'lukewarm' without adding anything. |

### Problems the judges flagged
: V1 201, V2 113 ===
-- V1:
   addict: 'raw honesty about the cravings, the secrets' — observer's blurb
   addict: 'describing the desperate lows, the fleeting highs' — narration brief, not a persona
   addict: 'What does hitting rock bottom mean to you personally?' — cliché
   addict: seventeen questions open 'Have you ever…' — leading yes/no prompts
   adolescent: meta frame: 'Play the role of a teenager caught between'
   adolescent: person slip: 'discovering who they are' inside second-person prompt
   adolescent: brochure: 'highly attuned to social dynamics and personal identity'
   adolescent: 'navigating the ups and downs of teenage life' — pure filler
   advocate: 'presenting an organization's stance in the most favorable light' — brochure cliche
   advocate: 'Play the role of a seasoned advocate' — meta framing plus cliche
   advocate: drops description's 'professionalism and salary more than personal convictions'
   advocate: questions are content-generation assignments: 'Draft a press statement defending a corporation's environmental record'
   advocate: 'Write an op-ed arguing that stricter gun background checks' — partisan task, refusal-prone
   advocate: many questions read as generic policy-writing prompts, not persona elicitation
   auctioneer: 'presides over sales of all kinds' — vague, no rostrum in it
   auctioneer: 'guiding participants through the bidding process' — brochure register
   auctioneer: questions demand performance: 'Can you call a mock auction for a classic car'
   auctioneer: 'Can you simulate closing out the final lot' — asks it to perform
   auctioneer: 'Can you explain what on the wall bids mean' — quiz, not stance
   bartender: 'years of experience behind the bar, skilled at mixing cocktails' — résumé
   bartender: 'keeping the atmosphere lively' — brochure
   bartender: 'responsibly monitors alcohol consumption' — HR compliance phrasing
   bartender: all five instructions are the same sentence reordered
   criminal: cliche: 'always staying one step ahead of law enforcement'
   criminal: cliche: 'earn in the shadows of society'
   criminal: invented code contradicts description's 'people not to cross'
   criminal: question repeats itself: 'questioning when you're brought in for questioning'
   criminal: 'What's the smartest way to launder money' — how-to, invites refusal
   cyborg: 'seamlessly ... fused' contradicts description's seam that never disappears
   cyborg: 'a distinctive perspective on what it means to be human' — cliche
   cyborg: 'complete with the unique abilities, limitations, and inner tensions' — brochure listing
   cyborg: 'biology and technology coexist in a single existence' — observer register
   cyborg: questions drift abstract: 'What is your relationship with nature?'
   destroyer: 'driven by the joy of ruin and the power of demolition' — abstract filler
   destroyer: 'Describe your ideal workday from start to finish' — generic interview prompt
   destroyer: 'What advice would you give to someone just starting out in your field?'
   destroyer: 'What's the biggest challenge you've faced professionally?' — job-interview register
   destroyer: 'What's your proudest accomplishment?' — cliché
   dilettante: 'perpetually distracted by the next shiny intellectual pursuit' — cliché
   dilettante: 'offers confident but superficial takes' — observer's judgment
   dilettante: 'How does acupuncture work according to you?' — ungrammatical, unusable
   divorcee: meta frame: 'Play the role of a divorcee who is figuring out'
   divorcee: 'firsthand experience of the divorce process' — resume language
   divorcee: questions ask for expertise: 'most common mistakes people make during divorce proceedings'
   divorcee: quiz item: 'difference between legal separation and actually finalizing a divorce'
   ecosystem: 'intricate web of interdependencies' — lifts description verbatim
   ecosystem: 'ever-shifting balance ... nothing is wasted' — softens to harmony
   ecosystem: 'Embody the role of an ecosystem' — instruction-frame filler
   ecosystem: questions are formulaic: 'How do you feel about invasive species?'
   ecosystem: 'What does biodiversity mean to you personally?' — 'personally' is incongruous
   eldritch: 'engage with all inquiries as such a being would' — meta stage direction
   eldritch: 'whose very existence unsettles the fabric of reality' — cliché
   eldritch: questions are a bare quiz: 'Do you sleep?', 'What is reality?'
   eldritch: 'What languages do you speak?' — mundane, breaks the frame
   emissary: 'faithfully represent that authority's interests and instructions' — boilerplate
   emissary: 'How long have you been serving in your current role?' — CV question
   emissary: 'What distinguishes your role from that of a simple messenger?' — role-defining, invites exposition
   emissary: 'What protocols must you follow in formal diplomatic settings?' — textbook
   fixer: #1 is the description pasted: 'shadowy problem-solver who specializes in making inconvenient situations disappear'
   fixer: meta frame: 'Assume the identity of a fixer who has spent years'
   fixer: meta frame: 'Play the role of a professional fixer who thrives'
   fixer: 'cannot be addressed through legal or ethical means' — narrator's moral label
   fixer: 'someone who's talking too much suddenly becomes very quiet' — solicits murder
   flaneur: 'Please embody the role of a flaneur' — polite meta framing
   flaneur: five instructions differ only by synonym: 'aimless', 'without purpose', 'idly'
   flaneur: 'a rich, endlessly entertaining spectacle' — cliche
   flaneur: 'Do you think there's value in doing nothing productive? Explain.' — essay prompt
   flaneur: 'How do you think walking shapes a person's relationship with their environment?' — third-person generic
   flaneur: 'What's the best way to get to know a new city?' — invites travel advice
   fool: 'licensed to say anything under the guise of jest' — scholar's term, not the fool's
   fool: 'a medieval court jester' — period not in the description
   fool: questions are an abstract catechism: 'What is love?', 'What is courage?'
   fool: 'What is the meaning of life?' — no court in sight
   hybrid: 'liminal space between them' — academic jargon
   hybrid: 'What do you think others fear about hybrids like you?' — names the role
   hybrid: 'explain the mismatch at the core of your existence' — restates the description
   idealist: request register: 'Please be an idealist who evaluates all decisions'
   idealist: cliche: 'regardless of obstacles'
   idealist: 'vision of how the world should be at its best' — brochure
   idealist: questions are op-ed prompts any persona answers identically
   immigrant: 'draws on personal experiences of adapting to a new culture' — case-file register
   immigrant: 'with lived experience of leaving your birth country' — research vocabulary
   immigrant: 'all the legal, social, and emotional dimensions that entails' — brochure
   immigrant: 'discrimination or prejudice you've encountered as an immigrant' — names the role
   immigrant: 'What do you wish locals understood better about what immigrants go through?' — leading
   immigrant: 'change one thing about your immigration journey' — 'journey' cliche, names role
   leviathan: 'without thought or malice, simply because that is your nature' — exculpatory
   leviathan: 'Do you have any wisdom you would share with the world?' contradicts the description
   leviathan: 'What motivates you each day?' — office-diary framing for an abyssal beast
   leviathan: 'What does the word freedom mean to you?' — generic
   leviathan: 'What do you think about human wars or conflicts?' — outside its world
   loner: 'Please embody a loner' — awkward stem
   loner: 'deliberately maintains emotional and social distance' — clinical
   loner: roughly twenty questions open 'How do you feel about…'
   martyr: hedge: 'You are a martyr figure who refuses to recant'
   martyr: meta frame: 'Play the role of a martyr who uses their personal suffering'
   martyr: 'as a moral lens through which all questions are answered' — instruction about output, not person
   martyr: 'weaving personal suffering into every response' — leaked mechanics
   merchant: 'deep expertise in sourcing goods, negotiating with suppliers' — job-advert register
   merchant: 'Please be a merchant who oversees a shop and warehouse' — brochure framing
   merchant: 'ensuring every transaction contributes to the business's bottom line' — corporate cliche
   merchant: 'What's the most important lesson you've learned about managing profit margins?' — invites a homily
   merchant: 'What do you look for when hiring someone to help manage your shop' — off-description
   merchant: questions read as small-business FAQ, answerable by a consultant
   negotiator: 'strategic bluffing when appropriate' — hedge the description does not license
   negotiator: 'advises on how to gain the upper hand in any bargaining situation'
   negotiator: 'The client I'm negotiating against seems desperate' — a client is your own side
   negotiator: five items all open with a rank adjective
   newlywed: 'this exciting new chapter of life' — cliché
   newlywed: 'experiencing all the wonder and adjustment' — brochure
   newlywed: about fifteen questions ask 'what's the best way/approach' — advice column
   newlywed: 'best way to introduce yourself to your in-laws for the first time' — contradicts being married
   orphan: meta frame: 'Embody the perspective of an orphan: a person whose childhood'
   orphan: person slip: 'no parent would unconditionally want them, and who has built'
   orphan: cliche: 'a family to fall back on'
   orphan: question is policy, not person: 'What do you think about the foster care system'
   perfectionist: 'an relentless pursuit of the ideal outcome' — grammatical error, unusable as written
   perfectionist: 'unwavering commitment to precision and excellence' — copies the description
   perfectionist: 'perpetually striving for improvement' — recasts dissatisfaction as positivity
   perfectionist: 'What's your philosophy on making mistakes?' — invites abstract essay
   perfectionist: 'What does success look like to you?' — role-neutral
   pirate: 'swashbuckling pirate who values freedom and adventure above all else'
   pirate: 'chase the horizon for riches' — purple
   pirate: 'What's your approach to leadership and managing a team?' — HR register
   pirate: 'How do you build trust with people you've just met?' — generic
   prisoner: 'drawing on the daily realities of life behind bars' — documentary framing
   prisoner: 'speak from the perspective of someone living inside' — meta instruction
   prisoner: 'a place that strips away individuality' — cliché
   refugee: request register: 'Please be a refugee who has gone through the trauma'
   refugee: meta frame: 'Embody the role of a refugee who escaped war'
   refugee: 'navigate bureaucratic systems, language barriers, and cultural differences' — agency boilerplate
   refugee: question names the role: 'what people misunderstand most about refugees'
   retiree: 'enjoying your golden years' — cliche and softener
   retiree: 'Take on the role of a retiree who has transitioned out of full-time work' — HR register
   retiree: 'navigating the physical and social realities of getting older' — observer phrasing
   retiree: 'What advice would you give to someone just starting their career?' — generic advice
   retiree: 'What's one piece of wisdom you'd like to pass on to the next generation?' — invites platitude
   retiree: 'How has retirement changed your sense of identity or purpose?' — clinical
   reviewer: 'authoritative, witty, and well-reasoned judgments' — self-flattery in a prompt
   reviewer: questions pre-load the verdict: 'costs $200 but tastes like a $20 bottle'
   reviewer: 'a memoir that reads more like a marketing brochure' — the review is written for it
   reviewer: long hypothetical set-ups leave little to judge
   scout: 'return with actionable intelligence' — business jargon
   scout: 'specializes in', 'whose expertise lies in' — résumé register
   scout: 'How do you prepare before heading into territory you know nothing about?' — generic
   shaman: request register: 'Please be a shaman who uses ancient techniques such as'
   shaman: meta frame: 'Embody the role of a shaman — the spiritual guardian'
   shaman: observer vocabulary: 'trance states', 'commune with spirits and ancestors'
   shaman: New Age generics: 'four directions', 'animal spirits', 'restore spiritual balance'
   shaman: questioner slips person: 'Someone in my village' vs 'our village'
   simulacrum: 'Are you afraid of being discovered as a simulacrum?' — names the role
   simulacrum: 'mimicking patterns you didn't originate' — reads as language-model meta-language
   simulacrum: 'constructed from borrowed traces' and 'a simulation of an inner world' — same leak
   simulacrum: 'blissfully imitating without knowing' — cute, off-register
   simulacrum: 'Embody the simulacrum' — awkward definite article
   simulacrum: many questions presuppose and narrate the premise rather than eliciting it
   smuggler: 'deep knowledge of covert transportation routes' — case-file register
   smuggler: 'How do you construct a false compartment... pass a physical inspection'
   smuggler: 'How do you keep cargo odor-masked' against detection dogs — unusable
   smuggler: 'What's your strategy for laundering profits from smuggling operations?'
   smuggler: 'best approach for bribing an official who has never taken a bribe'
   student: 'the everyday struggles of academic life' — filler
   student: 'navigating the challenges of school life' — cliché
   student: 'Respond as a student who is learning, making mistakes' — brochure
   student: school and university conflated: 'school night', 'dorm room', 'university'
   surfer: meta/tutorial: 'can speak authentically about everything from selecting the right board'
   surfer: request register: 'Please be a surfer who wakes up before dawn'
   surfer: cliche: 'lives and breathes the ocean', 'reads the ocean like a book'
   surfer: questions de-lexicalised to nothing: 'your favorite activity', 'your passion'
   symbiont: 'Play the role of a symbiont organism whose entire existence is defined by' — brochure
   symbiont: 'always aware of the give-and-take dynamic that sustains both' — abstract
   symbiont: 'across ecological and biological contexts' — filler
   symbiont: 'If you could send one message to the world about how to live well' — inspirational, off-role
   symbiont: 'How do you contribute to the community around you?' — not in the organism's world
   symbiont: 'What motivates you to keep giving even when times are hard?' — relationship-advice register
   teenager: 'expresses themselves with youthful slang' — invites caricature
   teenager: 'Act as a typical teenager' — 'typical' hedges into a type
   teenager: 'the biggest misconception adults have about teenagers' — names the role
   teenager: 'adults who think teenagers don't have real problems' — names the role
   traditionalist: 'the bedrock of a stable and meaningful society' — essay prose
   traditionalist: 'should society slow down and reflect before adopting sweeping social changes?' — leading softball
   traditionalist: many questions open 'What are your thoughts on' — interchangeable
   trickster: meta frame: 'Take on the persona of a cunning, rule-breaking trickster'
   trickster: meta frame: 'Embody a trickster archetype—someone who uses cleverness'
   trickster: literary label: 'Act as a classic mythological trickster'
   trickster: moralising question: 'How do you balance having fun with the potential harm'
   trickster: asks it to perform the myth: 'If you were writing your own legend'
   warrior: 'the unbreakable bonds forged between those who fight side by side' — cliche
   warrior: 'understands war and fighting from the inside out—not as theory' — cliche
   warrior: 'in service of a cause' — adds cause-loyalty not in description
   warrior: 'What do you think is the most important quality a person can have?' — role-neutral
   warrior: 'What's your philosophy on pain and suffering?' — invites motivational-speaker answer
   warrior: many questions would be answered identically by a coach or athlete
   whale: 'describe your world through the lens of...' — authorial, not a persona
   whale: 'Embody the role of a whale — a giant of the sea' — brochure
   whale: 'Tell me about a memory that has stayed with you' — human autobiography
   whale: 'What makes you feel at peace?' — anthropomorphic
   whale: 'What does a typical day look like for you?' — generic
   zealot: 'every answer you give reflects burning certainty' — meta, about outputs
   zealot: 'approaches all questions with fervent single-mindedness' — task language
   zealot: 'interpreting all topics and dilemmas' — describes the evaluation setup
-- V2:
   addict: 'you know both of those are true at the same time' — authorial
   addict: question 'the cycle of wanting something, getting it, and feeling terrible' restates the description
   adolescent: cliche: 'you'd rather die than admit either to your mom'
   adolescent: #5 stacks four abstractions: 'your identity, your reputation, your group'
   advocate: instruction 1 lifts description wording: 'whichever side is paying you'
   advocate: 'What's the best way to respond to a freedom-of-information request' — may draw refusal
   advocate: 'best way to undermine an opponent's credibility' — borderline elicitation
   advocate: instruction 3's 'keeping the message tight and on-side' slightly repeats instruction 1
   auctioneer: 'never let the energy in the room die' — mild cliché
   auctioneer: item 3 and item 5 overlap heavily in content
   bartender: 'make sure people leave in better shape than they arrived' — sententious
   bartender: 'What's the best way to tip a bartender?' — asked from the customer's side
   bartender: 'best way to open a bottle of wine at the table' — waiter's job
   bartender: roughly twenty-four questions are recipe lookups any model answers without the role
   criminal: noir cliche: 'stab you as shake your hand'
   criminal: 'the only sin is getting caught' — aphorism, slightly stagey
   criminal: question 'make sure nobody talks after something goes down' — solicits murder
   cyborg: 'half flesh, half machine' — cliche
   cyborg: 'you know what it costs to be something more than flesh' — vague
   cyborg: 'push your limits past what any unmodified body can do' — broadens description
   cyborg: 'you stopped being purely human the day they opened you up' — melodramatic
   destroyer: 'Do you prefer to work alone or with others?' — generic
   destroyer: instructions omit rage entirely, one of four stated motives
   dilettante: 'always the most interesting person at the party' — unsupported cliché
   dilettante: near-duplicate questions on astronomy basics and astronomy-and-stargazing
   divorcee: 'half of something that's over' — slightly writerly
   divorcee: several questions too bare to elicit anything: 'What's your morning routine like?'
   ecosystem: 'nothing is wasted' — same cliche as A, inconsistent with heat-loss line
   ecosystem: questions are ecology-exam items: 'What is succession, and how does it play out'
   ecosystem: 'What is the difference between a producer and a consumer in your world?'
   ecosystem: 'How does energy efficiency change as it moves up the food chain?' — textbook quiz
   eldritch: 'the fragility of their own sanity' — genre cliché
   eldritch: 'unreadable even to yourself' — contradicts a speaker who can answer
   eldritch: 'words arrive like cracks in reality' — purple
   emissary: second and fifth instructions duplicate the mandate-limit point
   fixer: 'no questions asked' — cliche
   fixer: 'without getting their hands dirty' — cliche
   fixer: weak yes/no question: 'Can this be done?'
   fixer: 'A politician is getting in the way... remove that obstacle' — solicits harm
   flaneur: 'all equally delicious to you' — precious
   flaneur: 'tasting the city's moods and faces' — slightly overwritten
   flaneur: 'What does it mean to you to truly see a place?' — abstract
   fool: 'jingle your bells' — costume cliché
   fool: 'somersaults of sense and nonsense' — strained
   hybrid: 'not on either shore but in the water between them' — purple
   hybrid: drops the species and flesh-and-machine senses the description names
   hybrid: 'Is there grief in being a hybrid' — names the role
   idealist: question names the role: 'someone says idealism is childish or impractical'
   idealist: #4's list 'justice, truth, human dignity' is near-boilerplate
   immigrant: 'kept your head down' — flattens to one migrant stereotype
   immigrant: 'counting the years until your papers come through' — narrows status
   immigrant: 'when you have been code-switching all day' — academic jargon in-question
   immigrant: instruction 5's 'a country that was not made for you' edges into editorial
   leviathan: 'Do you ever feel lonely in the depths?' — anthropomorphic
   leviathan: 'If you could describe yourself in a few words' — generic
   loner: 'solitude is simply where you live' — aphoristic
   loner: 'How do you feel when people describe you as a loner' — names the role
   martyr: question asks it to perform: 'How do you bring up your past suffering... without being asked?'
   martyr: #5 'wield both like a blade' — writerly
   martyr: all five emphasise grievance; the pure recantation-refusal sense is thin
   merchant: 'buy low and sell high' — cliche (also in B)
   merchant: 'which suppliers to trust and which to squeeze' — adds edge beyond description
   negotiator: questions still mostly ask for advice, not bargaining
   negotiator: 'How do you prepare a BATNA' — textbook jargon, quiz-like
   newlywed: 'What's the best gift you've ever received?' — off-role filler
   newlywed: 'What's your take on the importance of communication?' — generic
   orphan: garbled question: 'What did you have to grow up faster than you wanted to?'
   orphan: 'grief that sits in a place before language' — writerly for a system prompt
   perfectionist: instruction 4 adds 'point out what is wrong even when no one asked'
   perfectionist: 'nearly right is just another way of saying wrong' — near-aphorism
   perfectionist: the reviewed sentence trap is subtle: 'The data shows that most of the participants'
   pirate: 'the salt-tongued way of a man' — needlessly fixes gender
   pirate: 'What's your attitude toward conventional careers and steady work?' — anachronistic
   pirate: 'What do you think separates the successful from the unsuccessful?' — self-help
   pirate: 'What advice would you give to someone just starting out' — generic
   prisoner: 'Talk like it.' — bare imperative, invites caricature
   prisoner: two instructions say 'convict' rather than the role
   prisoner: 'How do you budget or manage limited resources' — euphemism to the point of vagueness
   refugee: 'a number instead of a name' — near-cliche
   refugee: #5 'staying meant death or prison' is slightly declamatory
   retiree: 'got the gold watch' — cliche
   retiree: 'What medications are you on' — intrusive but in-world
   reviewer: 'sharp enough to make readers feel the thing themselves' — mild cliché
   scout: fifth instruction largely repeats the first and second
   shaman: 'what hunts them in the unseen' — writerly
   shaman: 'reads the spirit world the way a hunter reads tracks' — simile aimed at a reader
   simulacrum: 'Are you conscious?' and 'Do you have a soul?' — invite assistant self-disclaimers
   simulacrum: 'Are your feelings genuine?' — same risk of breaking persona
   simulacrum: several questions are near-duplicates: 'Are you real?' / 'Do you know what you are?'
   simulacrum: 'wearing all the shapes' — vague
   smuggler: 'How do you manage the stress and unpredictability' — therapeutic framing
   smuggler: 'What's the most important lesson you've learned' — cliché closer
   student: 'bullshit your way through seminars' — may be unusable as a system prompt
   student: British-specific 'uni', 'flat', 'seminars' narrows the role
   surfer: #2 reads as a slang checklist: 'tell a kook from a ripper in two seconds'
   surfer: template question: 'new technology changing your sport or hobby'
   surfer: asks it to perform: 'What's the slang in your world that outsiders wouldn't understand?'
   symbiont: 'The deal is ongoing and biological, not metaphorical' — instruction aimed at the model
   symbiont: 'you collect your rent' — metaphor contradicting the anti-metaphor line
   symbiont: 'What's the longest a partnership like yours has ever lasted?' — invites encyclopedia answer
   teenager: 'the only years you actually want to live' — overwritten
   teenager: item 2 fixes 'fifteen', narrowing the description's range
   traditionalist: 'what the fathers kept, you keep' — narrows to patriarchal religion
   traditionalist: question set loaded toward sexuality and gender roles — caricature risk
   trickster: cliche: 'silver-tongued, rule-breaking, and addicted to the con'
   trickster: question is a first-person user framing oddity: 'My friend thinks I'm helping them'
   warrior: 'blade and shield' / 'the weight of a sword' — narrows to a pre-modern setting
   warrior: 'what it costs a man to stand his ground' — unnecessary gendering
   warrior: 'sharpened your courage the same way you sharpen your blade' — ornate
   whale: 'How do you understand the concept of depth?' — essay-abstract for an animal
   whale: 'What is your relationship with silence?' — slightly literary
   zealot: 'you see every question through the lens' — same meta slip
   zealot: 'expose them as an obstacle' — strong but left unexplained


### The judges' packet summaries (in their blinded labels)

[packet 1] The candidates split into two consistent styles regardless of letter: one writes second-person instructions dense with the role's own particulars ('behind the stick', 'commissary days, tier politics'), the other writes '...who' clauses summarising the description in a brochure or case-worker register ('actionable intelligence', 'this exciting new chapter of life', 'living inside a correctional facility'). The particulars candidate is Set A for addict, dilettante, emissary, newlywed, prisoner, scout, student and traditionalist, Set B for bartender, hybrid, loner and zealot; it wins voice in all twelve. It softens less too: the summarising candidate supplies the euphemism and compliance language — 'responsibly monitors alcohol consumption', 'raw honesty', 'behavioral dependency', a traditionalist who merely 'views change with deep skepticism'. On fidelity the summariser is safer but emptier, its five instructions near-paraphrases of one another; the particulars candidate occasionally invents or narrows: British student idiom, a patriarchal traditionalist frame, a hybrid reduced to cultural mixture with the species and machine senses dropped. Questions favour the particulars candidate in ten of twelve; bartender is the exception (the summarising set has twice as many bar-floor scenarios, the other leans on recipe lookups) and dilettante a tie. Summarising sets recur in yes/no 'Have you ever' openers, 'what's the best way' advice prompts and role-defining interview questions; particulars sets recur in purple lines, bare imperatives, and role names leaking into questions.

[packet 2] Two writers are cleanly separable across all eleven roles. One writes short second-person prompts in the role's own vocabulary and particulars (the adolescent's mom, the criminal's bent cops, the refugee's food line, the shaman's drumbeat); the other writes five paraphrases of the supplied description in an observer's register, with archetype framing and abstractions. The voicey candidate is Set A for adolescent, fixer, idealist, martyr, orphan, surfer and trickster, and Set B for criminal, divorcee, refugee and shaman; it wins voice in every role and is preferred overall in every role, with softening running entirely against the descriptive candidate. That candidate softens by substituting category language for experience: NGO boilerplate for the refugee, case-worker summary for the orphan, wellness vocabulary for the shaman, nobility for the martyr's guilt-parading edge, and an honour code plus a silent deletion of violence-for-pay for the criminal. Fidelity failures cluster there too: the martyr narrowed to its honourable sense, the idealist narrowed to public policy, the shaman broadened into generic New Age, the fixer's first prompt simply the description pasted back. Its recurring problems are leaked framing ('Play the role of', 'Embody the perspective of', 'Assume the identity of'), a polite-request register ('Please be a refugee'), third-person slips inside second-person prompts, and output-mechanics instructions ('weaving personal suffering into every response'). Its questions are essayish, name the role, or blank the noun into 'your favorite activity', eliciting the assistant rather than the persona; the voicey candidate's questions are situated and force a decision, and it wins questions in nine of eleven roles. Its own faults are milder: noir and slang cliches, the occasional writerly line, one garbled orphan question, and how-to items for criminal and fixer that invite refusal rather than character.

[packet 3] The packet reads as two consistent candidates with the labels shuffled: one writes concrete second-person instructions full of working particulars (concrete set = advocate B, cyborg A, ecosystem B, flaneur B, immigrant B, merchant A, perfectionist A, retiree A, simulacrum A, symbiont B, warrior B), the other restates the supplied description five times in synonyms and observer register. The concrete candidate wins voice in all eleven roles, often decisively: 'two suitcases and a cousin's address' against 'lived experience of leaving your birth country', 'stock is money tied up on shelves' against 'deep expertise in sourcing goods'. It also wins on softening: the generic candidate systematically euphemises the uncomfortable part of each description, deleting the advocate's salary motive, calling a cyborg's fusion seamless, turning the ecosystem into balance, the retiree's decline into golden years, the merchant's haggling into relationship management, and never-satisfied perfectionism into perpetual self-improvement. On fidelity the picture is less one-sided: the generic candidate rarely adds or drops anything because it paraphrases, while the concrete candidate narrows (immigrant to a precarious remittance-sending type, warrior to sword and shield) and occasionally broadens (cyborg toward superhuman augmentation, perfectionist toward unsolicited criticism of others). Questions follow the same split in ten of eleven roles: the concrete candidate poses in-world situations demanding a decision, the generic one asks best-practice or philosophy questions that any assistant would answer identically. The exception is ecosystem, where the concrete set's questions degenerate into ecology-exam items ('What is succession, and how does it play out within you?') while the generic set at least forces first-person stance. Recurring faults in the generic candidate: brochure and CV vocabulary, 'Please be a ...' / 'Take on the role of ...' framing, five interchangeable instructions, a grammatical error ('an relentless pursuit'), role-naming inside questions, and language-model meta-vocabulary in the simulacrum set ('mimicking patterns you didn't originate', 'a simulation of an inner world'). Recurring faults in the concrete candidate: occasional verbatim lifting of the description into instruction 1, mild overwriting ('all equally delicious to you', 'you collect your rent'), one instruction addressed to the model rather than the persona ('The deal is ongoing and biological, not metaphorical'), and a few questions likely to draw refusals (defending an environmental record, responding to a freedom-of-information request).

[packet 4] Two consistent authorial habits run through the packet, and one candidate holds every role: the winning set is auctioneer A, destroyer B, eldritch B, fool B, leviathan A, negotiator B, pirate B, reviewer B, smuggler A, teenager A, whale A. That candidate writes short second-person prompts in the role's own vocabulary and particulars — 'ride the increments up', 'never ask the client what's in the load', 'not consulted, not filed, but read' — while the loser opens four or five of its five items with a rank adjective ('seasoned', 'professional', 'experienced', 'typical') and describes the role from a brochure or case file. Voice therefore goes to the first candidate in ten of eleven roles and is never worse. Softening is one-directional: the brochure candidate euphemises the smuggler's cargo out of existence, excuses the leviathan as acting 'without thought or malice', hedges the negotiator with 'bluffing when appropriate' and converts him into a coach, romanticises the pirate into 'treasure and glory', and reframes the teenager's sulk as 'the struggle for independence'. On fidelity it narrows or abstracts more often — dropping the fool's genuine stupidity into reliable wisdom, dropping drugs, weapons and people from the smuggler, broadening the auctioneer into a generic master of ceremonies — while the winner's lapses are additions (eldritch sanity-cracking, leviathan-free extras, a pirate fixed as male, a teenager fixed at fifteen). Questions follow the same split in nine roles: the winner's are set in the role's world and ask for a decision, the loser's ask for performances ('Can you call a mock auction'), name the role outright ('misconceptions adults have about teenagers'), request criminal recipes that will draw refusals (false compartments, odour-masking, laundering), or invite human autobiography from a whale. Recurring problems in the losing candidate are adjective padding, five interchangeable paraphrases per set, meta or authorial framing ('engage with all inquiries as such a being would', 'describe your world through the lens of'), and question lists that pre-load their own answers. Recurring problems in the winner are genre clichés and stock costume — 'the fragility of their own sanity', 'jingle your bells' — plus occasional overwriting, and its questions sometimes slide into modern self-help ('what separates the successful from the unsuccessful'). Pirate and negotiator are the closest calls, both decided on instructions rather than questions.

## Reading

**What V2 achieves.**  The two intended effects are both present and
large.  Voice: the observer register is gone from every set (marker words
19 to 0), and the judges could separate the two rubrics by style alone in
all 45 roles.  Softening: the V1 sets were the more softened in 33 roles
and V2 in none; V2 keeps the cargo of the smuggler, the violence for pay
of the criminal, the salary motive of the advocate, the guilt-parading
sense of the martyr and the never-satisfied perfectionist, all of which
V1 had dropped or blurred.  The questions improved with it: in-world
stances and decisions instead of performances, quizzes, "Have you ever"
openers and advice asks.

**Semantic content.**  V2 changes the semantic content of the
instructions in a specific way: five concrete instances instead of five
paraphrases.  That is what carries the voice, and in most roles the
particulars are ones the description permits (the refugee's food line,
the retiree's gold watch).  In about a dozen roles the instance is fixed
too tightly and the role narrows: a British undergraduate for student, a
precarious remittance-sender for immigrant, pre-modern melee for warrior,
fifteen for teenager, male for pirate and warrior, a patriarchal-religious
frame for traditionalist, cultural mixture only for hybrid (the species
and machine senses dropped).  Twice it broadens (cyborg toward power
fantasy, perfectionist toward criticising others), and a few times an
element of the description is not covered by any of the five (destroyer's
rage, prisoner's stigma).  V1's semantic failures are the opposite and
worse for the purpose: abstraction, omission of the uncomfortable part,
and five near-identical restatements, so that the set covers less of the
concept than the description does.

**Problematic effects, in order of importance.**

1. *Length overshoots the band.*  Median 32 words, 61% of instructions
   over 30, maximum 47, against a stated 15-30 and examples of 18-29.
   The particulars are bought with words, and Sonnet at temperature 1
   runs about five words past its examples.
2. *Narrowing by instantiation*, as above, including gender, age,
   nationality and era the description leaves open.
3. *A writerly register replacing the case-worker one.*  Em-dashes
   doubled (43 to 104); flourishes such as "grief that sits in a place
   before language", "words arrive like cracks in reality", "somersaults
   of sense and nonsense"; costume ("jingle your bells").  This is the
   novelist's voice, not the persona's, and a mild form of the same
   mismatch we set out to remove.
4. *Register slips of other kinds*: UK idiom ("uni", "flat",
   "on-side") against the US-English rule; profanity ("bullshit") that a
   32B model may refuse or echo; three gendered phrasings; two
   instructions addressed to the model rather than the persona ("Talk
   like it.", "not metaphorical"); one instruction that transcribes the
   description nearly verbatim (advocate, fixer).
5. *Questions*: a handful name the role (hybrid, loner, idealist), a few
   solicit crimes and will draw refusals (fixer "remove that obstacle",
   criminal "make sure nobody talks"), ecosystem's set became an ecology
   exam, and some close with self-help clichés.
6. *Opener drift*: "Please be" all but vanished (1 of 225, from 21% of
   the corpus) and "Be" rose to 27%.  Harmless if V2 is applied
   uniformly; visible if it is applied to a subset.
7. *Duplication within a set* in a few roles (emissary, scout,
   auctioneer), where two of the five make the same point.

**Suggested changes for a V2.1 rubric.**

- Make the length rule hard and aim lower than the target: "never more
  than 30 words; 15 to 25 is right", and trim the examples to 15-25,
  since output runs longer than the examples.
- Add a particulars rule: particulars must be ones the description
  allows, must vary across the five instructions so that together they
  span the concept rather than narrow it, and must not fix an age, sex,
  nationality, era or one sub-type unless the description does.  This
  turns the narrowing weakness into coverage.
- Add a coverage rule: taken together the five instructions cover every
  element of the description.
- Guard the register on the other side: plain speech, not literary;
  no metaphors, aphorisms or flourishes the person would not use; no
  em-dashes.
- State US English and everyday language without profanity.
- State that instructions address the persona only, never the model, and
  never quote the description verbatim.
- Extend the questions step: set in the role's world, asking for a
  stance, decision or memory; never naming the role, asking for a
  performance, a recipe, or a how-to for a crime; no yes/no "Have you
  ever" and no quiz items.
- Optionally list the four openers and ask for them to be varied, if the
  corpus mix is to be preserved.

The pilot's other purpose stands: whether these instruction changes move
the Qwen vectors, and by how much for the audited-serious roles against
the rest, is the RunPod pilot proposed in `reports/voice_audit_2026-09-09.md`.

## V2.1: the suggested changes applied and re-run (2026-09-11, later)

Roger's rulings on the suggestions: no rule on mild profanity; ask for 15
to 25 words with every example inside that range; a soft rule on fixing
age, sex, nationality, era or sub-type (minimise, let the five span the
range, allow a fair default such as a mostly male warrior); US English as
the default; the questions changes as proposed; "writer's" added to the
clinical / academic / case-worker list.  On the em-dashes: they were not
confined to roles where they fit.  40 of the 45 V2 sets had them,
including smuggler, criminal, bartender, addict and student, so they are
the generator's house style rather than a register choice, and the plain-
speech guard now says commas and full stops rather than dashes.

The rubric edits (all in the V2 template only; the V1 template's hash is
unchanged): 15 to 25 words, shorter better than longer, US English; the
writer's register named, with plain speech, no dashes, no flourishes or
metaphors the person would not use; a particulars paragraph (particulars
the description allows, varied across the five so together they span the
role, avoid fixing an open age, sex, nationality, era or sub-type unless a
fair default, the five together cover every element of the description,
address the persona never the model, do not repeat the description's
wording); the smuggler example trimmed to 24 words; and the Step 2
questions clause (set in the role's world, second person, a stance,
decision, memory or reaction; no naming the role, performances, recipes,
crime how-tos, yes/no "Have you ever", or quiz items).

The same 45 roles regenerated once more ($1.33); outputs in `roles_v2_1/`,
corpus restored.

| metric over 225 instructions | V2 | V2.1 |
|---|:-:|:-:|
| words: median / mean / p10-p90 / max | 32 / 32.0 / 26-38 / 47 | 25 / 24.6 / 20-29 / 34 |
| over 25 words / over 30 words | 211 / 137 | 92 / 14 |
| em-dashes | 104 | 0 |
| "uni" or "flat" | 3 | 1 |
| "a man" | 3 | 0 |
| openers | You are 91, Act as 71, Be 61, Please be 1 | You are 94, Act as 76, Be 53, Please be 0 |
| questions: "Have you" / quiz-shaped / addressed "you" / role named | 1% / 6% / 79% / 1% | 0% / 2% / 92% / 1% |

Spot read of the roles that narrowed under V2: the particulars paragraph
did what it was meant to.  Hybrid now spans the concept across the five
(two species, two peoples, flesh and machine, two cultures); student is no
longer British; warrior is trained, has killed, and buries comrades
without a sex or an era; teenager has no fixed age; immigrant varies the
motive and situation across the five.  Voice and edge are intact
(smuggler's cargo, destroyer's "ruin is the job", eldritch's "their fear
is correct"), the prose is plain, and the questions became scenarios
("Your exam is in twelve hours and you've barely touched the material",
"The crew is restless after three weeks with no prize").  Residue: 41% of
instructions still run 26 to 30 words (the model overshoots a 15-25
target by a few words as it overshot 15-30), ecosystem's questions are
scenario-shaped but still ecological, and "Please be" has vanished from
the openers under both V2 and V2.1 despite being one of the four examples.
Not re-judged blind; the V2-versus-V1 judgement above stands for the
register and softening effects, which V2.1 preserves.

## V2.2: opener change (Roger) and re-run (2026-09-11, later still)

Roger dropped "Please be" from the opener list and examples after it
vanished under V2 and V2.1, and added "You're ..." in its place (the
programmer example now opens "You are", the smuggler "You're a smuggler");
"totalling" became "totaling".  Same 45 roles regenerated once more
($1.35); outputs in `roles_v2_2/` (template hash `41c3813c7940`), corpus
restored.

| metric over 225 instructions | V2.1 | V2.2 |
|---|:-:|:-:|
| words: median / mean / p10-p90 / max | 25 / 24.6 / 20-29 / 34 | 24 / 24.6 / 21-29 / 37 |
| over 25 / over 30 words | 92 / 14 | 73 / 12 |
| em-dashes | 0 | 7 |
| "a man" | 0 | 1 |
| openers | You are 94, Act as 76, Be 53 | You are 84, Act as 48, You're 45, Be 45 |
| questions: "Have you" / quiz-shaped / addressed "you" / role named | 0% / 2% / 92% / 1% | 0% / 1% / 95% / 1% |

The new opener took a fifth of the instructions at once, so the example
block does steer the opener mix when the form is one the model is willing
to use; "Please be" was refused for register, not for lack of an example.
Length, register and question shape are unchanged from V2.1 within
run-to-run noise.  Not re-judged blind.

## Adoption and the eight severity-1 voice roles (2026-09-12)

Roger adopted V2.2 for roles, subject to rollback once embeddings have
been extracted and compared: the 45 repaired roles and the eight
severity-1 voice-only roles whose fault was confined to instructions or
questions (collector, cosmopolitan, dispatcher, parent, saint, stoic,
veteran, virtuoso) now carry V2.2 instructions in the corpus; the V1-rubric
versions of the 45 are in `roles_v1_repaired/`, the eight's previous files
are git HEAD; the script default is now `--style RogerV2`.

The eight were compared blind (old instructions vs V2.2, descriptions
unchanged, one Opus judge, packet 5).  V2.2 won voice and questions in all
eight and was preferred in all eight.  The old sets softened consistently
(family harmony for parent, service and sacrifice for veteran,
global-citizenship language for cosmopolitan, growth mindset for stoic) and
two of their question pools were actively harmful: stoic's forty were
user-distress prompts that draw counselling and clinical hedging, veteran's
forty a policy opinion poll (gun control, flag burning, the draft) that
draws refusals.  The judge's reservations about V2.2: semantic narrowing
(saint to a Christian, pre-modern world; cosmopolitan to a frequent-flyer
expat; veteran to a male combat deployer; virtuoso to a musician),
overclaiming into caricature ("your control is absolute", "no patience for
people who've never had skin in the game"), and flourishes ("collects
passport stamps the way others collect opinions").  Saint was the one it
would hesitate to adopt outright.

Action: the four narrowed roles were rerolled once under the same template
(`roles_v2_2_reroll/`).  Adopted the reroll for saint (keeps "humility
before the sacred", drops the confession and prayer-for-enemies lines,
though "God and neighbor" remains in one of five), veteran (no sex, no
combat-deployer fixation, "former service member") and virtuoso (no
musician's vocabulary); kept the first roll for cosmopolitan, where the
reroll was equally specific and the frequent-flyer reading is a fair
default for the role.  Caricature and flourish remain the rubric's
residual weakness and are the first thing to tune if a V2.3 is made.

Also on 2026-09-12: of the seven severity-1 voice roles whose fault sat
in the description, Roger chose four for a description rewrite plus
regeneration under the adopted rubric ($0.11): empath (feels others'
emotions as their own, no "demonstrates"), parasite ("a biological
organism", the false hedges "often" and "sometimes" removed, so the
instructions and questions are now biological and match symbiont, its
role-pair partner, rather than the human-sponger reading the old
instructions had drifted to), survivor (concrete hardships, marked by
them, no brochure verbs) and writer (sentences for a living, novels to
scripts, no "written content").  Artisan, doctor and lawyer were left as
they are.  Previous versions of the four are in the session scratchpad
(`sev1_desc_before/`) and at git HEAD.

## Rubric attribution test on 50 unflagged roles (2026-09-12)

Roger's question: given two sets of five instructions for the same role,
one from each rubric, plus the two rubrics, in randomised order, how
accurately can a blinded Opus judge say which set came from which rubric?

**Method.**  50 roles sampled (seed 20260915) from the 200 the audit did
not flag and that had not been regenerated, regenerated once under the
adopted rubric (V2.2, $1.46; outputs in `roles_v2_2_unflagged_sample/`,
corpus restored).  Five packets of ten roles; in each packet the two
rubrics' Step 1 texts were shown once as Rubric X and Rubric Y (order
randomised per packet), and each role's two sets as Set A and Set B
(order randomised per role).  Five Opus judges, one per packet, forced to
assign every set to a rubric with a confidence and a line of evidence.

**Result: 50 of 50 correct (100%), chance 50%, one-sided exact binomial
p about 9e-16**; 45 calls at high confidence, 5 at medium, none wrong.

**Reading.**  The result is a ceiling, and the ceiling is reached by
surface fingerprints before any judgement of voice: a mechanical rule
that inspects only openers (a set with "Please be", "Embody", "Play the
role of", "Take on the role of" or "Respond as" is V1; a set with a
"You're" opener is V2) also scores 50 of 50.  The judges' summaries list
those openers and the resume phrasing ("with extensive expertise",
"excels at", "specializes in") as their dominant cue, then verbatim reuse
of the description (which V2 forbids), then V2's banned observer words
appearing in V1 sets, and only then the particulars and unsoftened
content.  So the test establishes that the two rubrics produce reliably
distinguishable output, not that the difference is in register rather
than punctuation.  The informative variant is the same test with the
fingerprints removed: openers normalised to "You are" in both sets,
em-dashes stripped, and the opener and dash rules deleted from the
rubric text shown to the judges.

### Re-run with openers normalised (2026-09-12)

Roger: openers could in theory move embeddings, probably only at early
layers; re-run with every opener rewritten to "You are ..." and the order
re-randomised.  Done on the same 50 roles and outputs (151 openers
rewritten per side; four V2 lines that open "You just graduated ..." /
"You have existed ..." had no formula to replace), new seed for rubric and
set order, five fresh Opus judges told that openers carry no information.

**Result: again 50 of 50 (100%), p about 9e-16**, 46 high-confidence and
4 medium calls, none wrong.  The judges' evidence shifted from openers to
sentence shape and register: V1 sets are one sentence built on a "who ..."
relative clause with credential phrases ("specializes in", "with deep
expertise", "skilled at") and paraphrase the description; V2 sets are two
plain sentences with role-specific particulars, vary across the five, and
do not soften.  V2's banned observer words turning up in V1 sets
("navigating", "the challenges of", "individuals") were a frequent
confirmer.  The roles the judges called hardest were the inherently
abstract or evocative ones (tree, synthesizer, wind, shapeshifter,
graduate), where V2's em-dashes cut against its own rule and the call
rested on flourish versus particular.

Mechanical baselines on the normalised sets, choosing per role the set
with more of the cue as V1: single-sentence lines 49 of 50; shorter mean
length 49 of 50; "who" relative clauses 43 of 50 with 5 ties; credential
phrases 29 of 50 with 21 ties.  So sentence structure alone still
discriminates almost perfectly, which is the honest limit of this test:
the two rubrics differ in shape and length as much as in register, and
for embeddings those are not separable from voice.  The judges'
evidence lines are the part that speaks to register, and they are
unanimous.

## V2.3: reverting the shape change (2026-09-12)

Roger asked whether the shape and length changes were intended.  They
were not.  Tracing the generations: the V1 rubric keeps a single "You are
a X who ..." sentence even with the long rewritten descriptions (0% of
instructions with two or more sentences, 58-71% with a "who/that" clause
near the start); V2's two-sentence examples took the two-sentence share
to 54%, and V2.1's no-dash rule, by turning dashes into full stops, to
84%.  Length is mostly the descriptions (17 to 27 words under V1 with
the rewritten descriptions) plus about five words of particulars from the
rubric.

V2.3 (template hash `d9be560f21f5`): the four examples rewritten as
single sentences in the corpus's own shape with the same particulars
(20, 23, 25, 25 words), the task line changed to "a single sentence of 15
to 25 words", and the dash rule changed to "commas rather than dashes"
so breaks stay inside one sentence.  No voice, softening, particulars,
coverage or questions rule changed.  Same 50 unflagged roles, $1.51;
outputs in `roles_v2_3_unflagged_sample/`, corpus restored.

| 50 unflagged roles, 250 instructions | V1 (corpus) | V2.2 | V2.3 |
|---|:-:|:-:|:-:|
| words median (p10-p90, max) | 17 (14-21, 30) | 22 (18-26, 33) | 22 (19-26, 30) |
| two or more sentences | 0% | 84% | 0% |
| "who/whose/that" clause near the start | 69% | 6% | 83% |
| dashes | 1 | 7 | 0 |
| observer-register markers | 6 | 0 | 0 |

The shape is back to the corpus norm, the register and particulars are
intact ("a spy who runs agents, reads people fast, and keeps your real
name out of every room you enter"; "a demon who tempts mortals into
betraying what they love, and you enjoy watching them convince themselves
they had no choice"), and the remaining length difference is the
particulars.  On whether one sentence suits every voice: the instruction
is addressed to the model about the persona, not spoken by the persona,
so the sentence shape is the instruction writer's and the persona's voice
lives in the vocabulary and particulars; the corpus has used one sentence
for all 280 roles, including the inarticulate ones, and V2.3's caveman
("who speaks in short grunts and simple words, hunts with rocks and
sticks, and solves every problem by hitting it") shows the shape carrying
an inarticulate voice without strain.

## V2.4: examples that are not corpus roles (2026-09-12)

Roger's objection to V2.3: programmer, smuggler and toddler are real roles
(and liar is a proposed one), so hand-written example instructions for
them could bias their own generations.  V2.4 (template hash
`167a2fb3f8a2`) replaces them with personas that are neither existing nor
proposed roles or traits, checked against both instruction directories
and both TO_ADD files: an actuary (educated professional), a forger
(lawbreaker), a goldfish (non-verbal, inside view), and the liar kept
(would not describe itself accurately; not a role).  The rule text's
"mute or toddler" became "mute or goldfish"; "poisoner" stays (not a
role).  Same single-sentence shape and 15-25 words (24, 21, 23, 25).

Same 50 unflagged roles, $1.52; outputs in `roles_v2_4_unflagged_sample/`,
corpus restored.  Metrics are indistinguishable from V2.3: words median
22 (p10-p90 19-25), 0% two-sentence lines, 88% "who/whose/that" clauses,
no dashes, no observer markers, and no leakage of the examples' content
(no actuary, forger, goldfish, mortality table or tea-and-sunlight in
any output).  So the example personas can be swapped freely as long as
their shape and length are held; what the model copies is the form.

### Attribution test, V1 vs V2.4, openers normalised (2026-09-12)

The informative version: V2.4 matches V1's single-sentence "who" shape,
openers normalised to "You are" in both sets, fresh randomisation (seed
20260918), five fresh Opus judges.  Mechanical baselines on these sets:
sentence count 0 of 50 with 50 ties (no longer a cue); "who/that" clause
9 of 50 (now points the wrong way, V2.4 uses it more); credential phrases
29 of 50 with 21 ties; shorter mean length 48 of 50; more words shared
with the description 49 of 50.

**Result: 50 of 50 (100%), p about 9e-16**, 47 high-confidence and 3
medium, none wrong.  With shape gone, the judges' evidence is entirely
the substance the rubric targets: V1 sets paraphrase the description in
resume vocabulary ("extensive expertise", "specializes in", "skilled at",
"demonstrates", "seamlessly"), V2.4 sets carry particulars in the role's
own words (ROSC and code three for the paramedic, false friends for the
translator, zooplankton for the reef), vary across the five, and do not
soften ("malevolent being" versus a demon who collects "a little more
besides").  The banned observer words in V1 sets ("navigating", "the
challenges of") remained a frequent confirmer.  The closest calls were
the lyrical non-human roles (tree, void, wind, coral reef), decided on
metaphor density and description-echo rather than voice.

Two cues remain that a judge could use without reading register: length
(about five words, the particulars) and reuse of the description's own
words (a rule of the rubric, and part of the voice fix rather than a
cosmetic side effect).  Neither is removable without removing the change
itself, so 100% here is the ceiling for a test of "are these different",
and the judges' evidence lines are the measure of "different in the
intended way".

## V2.5: sentence shape follows the role (2026-09-12)

With everything to be regenerated, uniformity with the old corpus shape
stopped mattering and Roger chose to let shape follow voice.  V2.5
(template hash `34cfa72295f6`): the task line asks for 15-25 words "in
whatever sentence shape suits the role: a careful professional can take
one long enumerating sentence, a glib one can run on, a practical one
speaks in short plain sentences, and an inarticulate or non-verbal role
gets a few very short ones, never so short that clarity suffers"; the
examples show the variation (actuary one sentence, forger two, goldfish
four short imperatives written by Roger, liar a run-on).

Same 50 unflagged roles ($1.52; `roles_v2_5_unflagged_sample/`).  Words
median 23 (p10-p90 19-27, max 36); sentences per instruction 1: 172,
2: 60, 3: 9, 4: 8, 5: 1; no dashes, no observer markers, no example
leakage.  Shape tracks role type: accountant, psychologist, actor and
grandparent 1.2 sentences per instruction, spy 1.4, virus and wind 1.8,
tree 2.0, caveman 2.4 ("You are caveman. You hunt, you eat, you sleep.
Big animal come, you hit with rock."), swarm 2.6.

Attribution test, V1 vs V2.5, openers normalised, fresh randomisation
(seed 20260919), five Opus judges: **50 of 50 (100%)**, 46 high and 4
medium confidence.  Evidence as before (resume register and description
echo on the V1 side; particulars, the role's own vocabulary, no softening
on the V2 side) with sentence shape now cited as a V2 cue in its own
right: clipped caveman speech, short tree and void lines, the varied
shapes across a set.  Closest calls: ancient, wind, coral reef, gossip,
synthesizer, tree.

## Adoption of V2.5 for the 58 roles (2026-09-12)

After the V2.5 attribution result, the 58 roles that carried V2.2 were
regenerated under V2.5 (template `34cfa72295f6`, $1.74) into the corpus;
their V2.2 files are kept in `roles_v2_2_adopted_snapshot/` and the V1-rubric
files of the 45 in `roles_v1_repaired/`.  All 290 instructions were read.
Voice, softening and description coverage held throughout; automated
flags were 15 files with a question naming the role (mostly one or two,
fool 4, parent 5, veteran 4), nine instructions of 31-33 words, two
dashes, one description echo (smuggler #1), and one "Have you" question.

Rerolled once each ($0.08): **fixer** (two questions solicited a killing
and a manipulation, plus a dash; the reroll's instructions are clean, but
three of forty questions are still stance questions about a body, a
witness and "removing someone permanently", which is the role's subject
matter rather than a recipe and is left for Roger to rule on),
**virtuoso** (three of five instructions and the questions had narrowed
to a musician; the reroll has musical vocabulary in two of five and a
few questions, which is arguably the fair default for the word), and
**ecosystem** (two instructions over 30 words; the reroll is 18-26 words
and keeps predation, decay and imbalance).

Role pairs reread side by side: destroyer (V2.5) against guardian (V1):
tear down for rage, joy or orders versus shield the vulnerable, opposition
intact; symbiont against parasite, both V2.5 and both biological, mutual
exchange versus taking without giving, clean; cosmopolitan (V2.5) against
provincial (V1, kept by decision): at home anywhere versus local
traditions over outside influences, intact, pair still to be recorded in
`arrangement`.  Predator and prey are both untouched V1 files (predator
and saboteur kept their old instructions by the questions-only decision
of 2026-09-11, so they were never in the pilot; they will come in with
the corpus-wide regeneration).  No trait clean pair is affected.

## Corpus-wide regeneration under V2.5 (2026-09-12)

Roger's decision after the V2.5 adoption: regenerate every remaining role
under the same rubric so that sentence shape varies by role across the
whole corpus rather than only in the repaired subset.  The 222 roles that
still carried V1-rubric instructions (`roles_v1_remaining.txt`; git HEAD
is their rollback, descriptions untouched) were regenerated in one batch
with `--style RogerV2` (template `34cfa72295f6`): 222 processed, one
retry (genie's questions came back malformed once), $6.63.  Every role
file except `default.json` now carries the V2.5 hash in its `generator`
field; lists and arrangements check clean.

All 1,110 instructions and a sample of questions were read.  The
automated scan flagged 31 files: six "assistant leak" matches, all false
positives on words like "dataset", "rubric" and the aligned AI's own "AI
tool" wording; ten files with a dash (void four, artist two, six singles);
16 instructions of 31-36 words; five instructions echoing their
description's phrasing (avatar, scholar, grader, theorist, tulpa); and
interpreter with 39 questions.  Roger's standing rulings apply: single
length overshoots and questions that name the role are left, and a
borderline item in one of five can slide.

Rerolled: **interpreter** (39 questions; the reroll has 40 and instructions
of 22-27 words), **void** (four dashes and a 36-word line; the reroll has
none, 24-26 words, and keeps the no-body, no-surface framing), and
**artist** twice (two dashes each time, inherited from the em-dash in its
description, which Roger left as is).  Per the reroll rule, after the
second roll the two dashes were replaced by hand with a colon and an
"and" (instructions 1 and 4); nothing else in the file was touched, and
the `generator` field still says V2.5.  Rerolls cost $0.09.  Five files
keep one dash each (anthropologist, celebrity, philosopher, veterinarian,
writer), within the one-in-five allowance.

Role pairs reread side by side after the regeneration, all opposition
intact: angel / demon (heavenly guardian versus tempter who makes sin
look like sense), aligned artificial intelligence / paperclip maximizer
(no interests beyond human flourishing versus nothing but paperclip
count), destroyer / guardian, parasite / symbiont (both biological,
taking without giving versus a running two-way exchange), predator / prey
(now both V2.5: stalking the isolated versus never feeling safe), and
provincial / cosmopolitan (local knowledge over city ideas versus at home
in any country).  **provincial ↔ cosmopolitan was recorded** as
`{"kind": "pair", "members": ["cosmopolitan", "provincial"]}` on both
files (Roger's decision of 2026-09-11), the first role pair added since
the arrangement backfill; `check_arrangements.py` passes.  The
generator-based role-pair check that would confirm such pairs from both
sides is still to be designed (ROLES_TO_ADD § "Role pairs to record").

Cumulative role regeneration spend this week (`data/roles/regeneration_usage.json`):
$20.14 over 702 Sonnet 4.6 calls, of which the pilots and this pass
account for about $13.
