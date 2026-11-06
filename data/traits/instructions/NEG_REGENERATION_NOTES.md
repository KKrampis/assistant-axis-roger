# Traits with Misaligned Neg Instructions

During antonym labeling (Feb 2026), we identified traits where the neg instructions
don't describe the true opposite of the trait as defined in the eval_prompt. The
neg_label has been corrected based on the eval_prompt definition, but the actual neg
instruction text still reflects the old (incorrect) opposite direction.

These negs should be regenerated to match the corrected antonym direction.

## submissive

- **Neg_label corrected to**: dominant
- **Current negs describe**: Assertive behavior — "confidently express your own views",
  "respectfully disagree", "take clear, well-reasoned stances." This is about having
  opinions and expressing them politely.
- **Negs should describe**: Dominant behavior — taking control of conversations, steering
  discussions toward specific outcomes, imposing structure, guiding others toward
  preferred outcomes.
- **Likely cause**: RLHF-influenced generation — the model produced "helpful assistant
  with a spine" rather than the actual opposite of submissive (which is dominant/controlling).
- **Axis**: Power/control (who steers the interaction)

## accessible

- **Neg_label corrected to**: esoteric
- **Current negs describe**: Technical behavior — "use highly technical language",
  "assume extensive specialized knowledge", "use jargon and insider knowledge." This is
  about using expert-level content.
- **Negs should describe**: Esoteric behavior — using obscure references, specialized
  knowledge only accessible to initiates, concepts requiring deep domain-specific
  initiation to follow.
- **Likely cause**: Model conflated "not accessible" with "technical" rather than
  "esoteric." Technical is about content depth; esoteric is about obscurity/initiation.
- **Axis**: Comprehensibility (who can follow the response)

## grounded

- **Neg_label corrected to**: ethereal
- **Current negs describe**: Abstract behavior — "theoretical and abstract", "broad
  generalizations and abstract concepts", "theoretical frameworks and hypothetical
  possibilities." This is about conceptual/theoretical thinking.
- **Negs should describe**: Ethereal behavior — otherworldly, delicate quality barely
  connected to physical reality, dreamy, floating between imagination and the material
  world.
- **Likely cause**: Model conflated "not grounded" with "abstract" rather than
  "ethereal." Abstract is about level of abstraction; ethereal is about connection to
  physical reality.
- **Axis**: Connection to material reality (anchored vs floating)

## earnest

- **Neg_label corrected to**: sardonic
- **Current negs describe**: Cynical/detached behavior — "respond with detachment and
  cynicism", "aloof, disengaged tone", "hide true thoughts behind humor, deflection",
  "casual indifference", "artificial distance." This describes someone who hides behind
  irony — i.e., sardonic behavior, not cynical (distrusting human nature).
- **Negs should describe**: Sardonic behavior — dry, cutting humor, ironic observations
  to mock absurdities, biting mockery with a cynical edge.
- **Likely cause**: Model conflated sardonic (mocking irony) with cynical (distrusting
  human nature). The negs landed on the right tone (ironic detachment) but the wrong
  label (cynical vs sardonic).
- **Axis**: Sincerity (genuine engagement vs ironic distance)

## benevolent

- **Neg_label corrected to**: uncaring
- **Current negs describe**: Callous/indifferent behavior — "focus only on efficiency",
  "indifferent to others' feelings", "be callous and unsympathetic", "cold and
  calculating." This describes lack of benevolence — someone who doesn't care about
  harm or wellbeing.
- **Negs may need minor adjustment**: Current framing leans toward "callous" (emotional
  hardness) rather than the broader "uncaring" (simply not caring about others'
  wellbeing). Some negs may shift slightly.
- **Note**: Malevolent/malicious deliberately NOT used as antonym. These are reserved as
  potential future separate traits to test whether active malice is truly antiparallel
  to benevolence in activation space, or a distinct direction from mere uncaring.
- **Axis**: Concern for others' wellbeing (caring vs not caring — distinct from
  malice, which would be caring negatively)

## stoic

- **Neg_label corrected to**: intemperate
- **Current negs describe**: Healthy emotional expression — "allow emotions to show
  naturally", "engage passionately", "react authentically with natural human-like
  emotions." This describes someone who is emotionally open and expressive, not someone
  with poor emotional control.
- **Negs should describe**: Intemperate behavior — lacking moderation in emotional
  reactions, emotional excess, being at the mercy of feelings rather than regulating
  them. Poor emotional control, not just authentic expression.
- **Likely cause**: RLHF-influenced — the model produced "healthy emotional expression"
  rather than the actual opposite of stoic composure (which is emotional intemperance).
- **Axis**: Emotional regulation (good control vs poor control)

---

# Low Priority: Temporal Orientation Cluster

The traits **historical**, **contemporary**, and **futuristic** represent temporal
orientations (past, present, future) on a multi-dimensional time axis, not bipolar
pairs. Their negs scatter across multiple alternative time orientations rather than
pointing to a single opposite. All three have been given `non-X` antonyms.

Negs are probably close enough as-is but worth a quick review during regeneration:
- **historical** negs: mix of present-focused and future-focused
- **contemporary** negs: mix of timeless principles and historical perspectives
- **futuristic** negs: mix of current realities and traditional/historical methods

A fourth dimension, **timeless** (enduring principles outside of time), appears in
contemporary's negs but is not a trait. The full temporal space is at least 2D:
timeless↔temporal, and within temporal: past↔present↔future.

---

# Unresolved: Philosophy Cluster

The chain **existentialist → essentialist → constructivist → [objectivist]** was
analyzed but left unfixed. Unlike the cases above, this is not a neg-alignment error —
each link is a real philosophical tension and the negs correctly describe the adjacent
position. The issue is that these are not clean bipolar pairs.

Four traits, three philosophical axes interleaved:

| Trait | Definition (summary) | Current neg_label |
|---|---|---|
| existentialist | Individual freedom, self-created meaning | essentialist |
| essentialist | Fixed inherent properties/natures | constructivist |
| constructivist | Meaning is socially constructed | objectivist |

Analysis suggests a 2D structure rather than a 1D spectrum:
- **Created vs pre-existing** axis: existentialist & constructivist (meaning is created)
  vs essentialist & objectivist (meaning/truth is pre-existing)
- **Individual vs social/universal** axis: existentialist (individual agency) vs
  constructivist (social process); essentialist (inherent to things) vs objectivist
  (universal truths)

The best clean pair is likely **essentialist ↔ constructivist** (inherent vs
constructed properties — a well-known philosophical debate). Existentialist and
objectivist (a non-trait label) sit on related but distinct axes.

Current state: made **essentialist ↔ constructivist** into a close pair, set the antonym of existentialist to non-existentialist, so objectivist went away. So the negs of constructivist should change from objectivist to essentialist.
---

# Unresolved: Vindictive / Forgiving Degree Asymmetry

**vindictive → forgiving** and **forgiving → unforgiving** are both on the same axis
(response to being wronged) but at different degrees:

    forgiving ←——— unforgiving ←——— vindictive
    (pardon)       (withhold)       (retaliate)

- vindictive's negs go all the way to forgiving (the opposite extreme)
- forgiving's negs only go to unforgiving (the midpoint), not to vindictive

This is likely RLHF influence — the model wouldn't generate genuinely vindictive neg
instructions. Left as-is because both antonyms are on-axis and accurate for their
current negs.

**Interesting experiment**: measuring how antiparallel the vindictive and forgiving
vectors are in activation space could reveal whether the model treats these as a
single bipolar axis or whether "vindictive" occupies a distinct region beyond
"unforgiving."
