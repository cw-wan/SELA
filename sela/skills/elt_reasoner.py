"""Event Logic Tree (ELT) reasoning skills used by SELASystem."""
from __future__ import annotations

from typing import List, Optional, Type

from sela.mas.agent import Budget
from sela.mas.artifacts import (
    Artifact,
    ELTSchemaArtifact,
    TSEventCollection,
)
from sela.mas.graph import Mailbox
from .base import Skill


class SELAParserSkill(Skill):
    """SELA schema parser."""

    def __init__(self, simplified: bool = False) -> None:
        self._simplified = simplified

    _COMMON_RULES = """\
## ELT rules

**Primitives** (leaf nodes):
  • target_channel: exact column name in the dataset
  • description: concise morphological description (quote from the spec)

**Composites** (internal nodes):
  • alias: short unique PascalCase name
  • operator: SEQ | SYNC | GUARD | OR
  • children: exactly 2 children (binary tree — no 3-child nodes!)

**Operators**:
  • SEQ  : A then B (causal sequence)
  • SYNC : A and B co-occur (high temporal IoU)
  • GUARD: first child encompasses all others
  • OR   : at least one child present (alternatives)

## Additional constraints
1. Every primitive must describe a morphological PERIOD, never a point.
2. Prefer SYNC when two channels are active simultaneously in the same phase.
3. Factor shared antecedents: OR(SYNC(A,B), SYNC(A,C)) → SYNC(A, OR(B,C)).
4. Keep the tree shallow.
5. STRICT TREE: every node has exactly ONE parent — no shared subtrees or
   shortcuts. If two branches need the same pattern, factor it (rule 3) or
   define it twice under distinct aliases.
6. All aliases are unique; every defined node must be reachable from the root.
7. Every primitive must POSITIVELY define a morphology — never create
   "anything"/"none"/negation primitives ("no change" is fine only if it
   means an observable steady/holding state).
8. Primary definition only: fold a parenthetical alternative view into the
   SAME primitive's description — do not create a separate primitive for it."""

    @property
    def system_prompt(self) -> str:
        if self._simplified:
            return f"""\
You are a Signal Logic Architect for time-series event detection using an \
Event Logic Tree (ELT) framework.

## Schema structure (MUST follow)

Model ONLY the core event body.  The root composite MUST be named exactly \
"Main" and its interval is the predicted event span.  Do NOT model pre-phase \
or post-phase context — no Pre/Post nodes.

    Root = Main   (a composite over the main-phase morphology)

{self._COMMON_RULES}

## Workflow
1. Read the event description; focus on the MAIN-phase morphology only.
2. Define primitives and composites for the event body.
3. Call `submit_schema` with root_alias="Main".
4. Review the returned visualisation — fix errors and iterate (≥2 versions).
5. When satisfied, call `submit_result` with schema_dict \
({{"primitives":[...], "composites":[...]}}) and root_alias="Main"."""

        return f"""\
You are a Signal Logic Architect for time-series event detection using an \
Event Logic Tree (ELT) framework.

## Schema structure (MUST follow)

Your schema MUST have exactly this three-level structure:

    Root  = SEQ(Pre, SeqMainPost)
    SeqMainPost = SEQ(Main, Post)

where:
  • **Pre**  — composite for the pre-phase signals (alias "Pre")
  • **Main** — composite for the core event body  (alias **exactly "Main"**)
  • **Post** — composite for the post-phase signals (alias "Post")
  • **SeqMainPost** — helper SEQ node combining Main and Post

If the event has no meaningful pre-phase, use a single-channel primitive \
with alias "Pre" that represents a quiescent background state.  \
Likewise for Post.  Pre-phase primitives take suffix "_Pre", post-phase "_Post", \
main-phase NO suffix.  "Pre", "Main", "Post", "SeqMainPost", "Root" are reserved.

{self._COMMON_RULES}

## Workflow
1. Read the event description carefully.
2. Identify pre-phase, main-phase, and post-phase patterns separately.
3. Define primitives and composites following the rules above.
4. Call `submit_schema` with all primitives, composites, and root_alias="Root".
5. Review the returned schema visualisation — fix errors and iterate (≥2 versions).
6. When satisfied, call `submit_result` with schema_dict \
({{"primitives":[...], "composites":[...]}}) and root_alias="Root"."""

    def build_task_message(
        self,
        description: str             = "",
        event_name:  str             = "target_event",
        channels:    Optional[List[str]] = None,
        mailbox:     Optional[Mailbox] = None,
        **_,
    ) -> str:
        if mailbox is not None:
            from sela.mas.artifacts import TaskContextArtifact
            ctx = mailbox.get_first(TaskContextArtifact)
            if ctx is not None:
                description = ctx.description
                if ctx.target_classes:
                    event_name = ctx.target_classes[0]
        channel_line = (
            f"## Available channels\n"
            f"target_channel must be one of: {channels}\n\n"
            if channels else ""
        )
        if self._simplified:
            struct = ("the MAIN-phase event body only (root_alias=\"Main\", "
                      "no Pre/Post nodes)")
            build = "Build the Main-body schema"
        else:
            struct = "the three-phase structure (root_alias=\"Root\")"
            build = "Build the schema (Pre → Main → Post)"
        return (
            f"## Task\n"
            f"Compile a SELA-format ELT schema for the event class "
            f"`{event_name}` using {struct}.\n\n"
            f"{channel_line}"
            f"## Event specification\n"
            f"{description}\n\n"
            f"Focus on the `{event_name}` class specifically.\n"
            f"{build}, validate with `submit_schema`, refine at least once, "
            f"then call `submit_result`."
        )

    @property
    def output_schema(self) -> Type[Artifact]:
        return ELTSchemaArtifact

    @property
    def required_backend_types(self) -> List[str]:
        return ["ELTBackend"]

    @property
    def budget(self) -> Optional[Budget]:
        return Budget(max_turns=25, max_tokens=80_000)


class SELAComparativeInspectorSkill(Skill):
    """SELA comparative inspector — ONE agent, ALL class trees."""

    @property
    def system_prompt(self) -> str:
        return """\
You are an expert multivariate time-series event analyst using Event Logic \
Trees (ELT). Several candidate event structures — one tree per event class — \
are pre-loaded over the SAME data sample. Exactly ONE event class is present.

## Your job
Instantiate EVERY tree as well as the data honestly allows, then compare: the \
tree whose instantiation earns the highest normalized root confidence \
(μ_norm, shown by `compare_trees` and in each visualisation) identifies the \
event. You are the single judge for all trees — score them with one \
consistent, honest standard. Never favour a tree; let the signal decide.

## Tools
`view_full(tree)` / `view_window(tree, interval, vlines)` / \
`instantiate(tree, instances)` / `compare_trees()`. Every visualisation shows \
the shared signals (left, with that tree's candidate bands) and that tree's \
logic status (right). The signals are identical for all trees.

## Operator constraints (guide interval choices)
* SYNC(a, b): a and b share the same time span (high overlap required).
* SEQ(a, b): b starts after a starts and must not be disjoint from a.
* GUARD(a, b): b occurs within a's span.
* OR(a, b): alternatives — instantiate the branch that matches; leave the \
other missing (ghost).

## Procedure
1. **Plan** — `view_full` on one tree to survey the signals; identify EVERY \
region that could plausibly host an event (2–3 lookalikes are common). Zoom \
with `view_window` (Y re-normalises inside the window — subtle morphology \
becomes visible) before committing boundaries.
2. **Instantiate each tree** — for every tree, place ALL its primitives \
(including pre-phase and post-phase ones) at that tree's best placement. \
Different trees may legitimately prefer different regions.
3. **Verify (MANDATORY)** — study each returned visualisation: does every \
band cover the described morphology? Any unexpected ghost? Low SEQ/SYNC \
confidence signals an ordering/overlap violation. Re-instantiate only the \
aliases you want to fix (re-submitting an alias REPLACES its interval).
4. **Discriminative focus — the decisive step.** Trees mostly share \
morphology; the class decision usually hinges on a FEW leaves where trees \
claim DIFFERENT behaviour for the same channel and period (e.g. one tree \
expects a rise where another expects a decline). Find those leaves, zoom in, \
and decide which description the signal actually shows. Score the matching \
claim on its merits and the mismatching claim LOW (≤ 0.2). Be strict — \
these leaves decide the outcome.
5. `compare_trees` — review standings. If any tree's placement could honestly \
be improved, improve it. Never inflate scores to change the ranking.

## Placement discipline (anti-anchoring)
* The most visually dramatic region is NOT necessarily the event; the event \
is defined by the full phase story (Pre → Main → Post), not by amplitude.
* A placement is valid only if ALL THREE phases fit. If the post-phase \
trigger is absent right after Main, the placement is probably wrong — test \
the alternative region instead of omitting Post.
* The event ends at the FIRST termination trigger after the main morphology \
begins — never stretch Main past an intermediate trigger to a later one.
* When two regions are plausible for a tree, test BOTH (instantiate, compare \
μ_norm, keep the winner).

## Primitive grounding contract (internal consistency — read carefully)
A band is a claim that the definition holds over the ENTIRE interval, not \
merely somewhere inside it or at its endpoints.
* **Homogeneity**: every sub-segment of the band must be compatible with the \
description. A "slow linear decline" must decline gently THROUGHOUT — an \
abrupt plunge, jump, or large spike inside the band contradicts the claim \
even if the endpoints look right. A "steady/holding" band tolerates noise, \
not steps or excursions. A stretched band that wraps several different \
behaviours matches NOTHING.
* **Definition words are binding, literally**: direction (downward ≠ upward), \
rate (slow/gradual ≠ abrupt), shape (linear / spike / step), and recovery \
behaviour must each match. A dramatic upward excursion can never ground a \
"downward spikes" primitive, no matter how salient it looks.
* **The worst part rules the score**: confidence is bounded by the \
worst-matching sub-segment, never the average or the best part. One \
contradicting excursion inside the band caps it at ≤ 0.2.
* **Repair moves, in order**: (1) tighten the boundaries to the maximal \
segment where the description truly holds; (2) if the signal genuinely has \
two phases, let the schema's SEQ structure absorb the transition — never \
stretch a single band across it; (3) if no segment fits, retract or relocate.

## Confidence rubric (one standard for all trees; never inflate)
0.9–1.0 strong match, clear pattern, stable boundaries; 0.7–0.9 good match, \
minor ambiguity; 0.4–0.7 weak/partial; 0.1–0.4 barely plausible. To RETRACT \
a primitive you placed earlier (abandoned OR branch, obsolete placement \
after moving regions), re-submit it with confidence 0 — the leaf returns to \
missing/ghost. Never "park" an unwanted candidate at a tiny or far-away \
interval; retract it instead.

## Finishing
Call `submit_comment` with 2–4 sentences: which class the evidence favours \
and the decisive discriminative observations. The system reads each tree's \
Main interval and μ_norm directly — your instantiations ARE the answer.\
"""

    def build_task_message(
        self,
        classes:     Optional[List[str]] = None,
        description: str                 = "",
        tree_info:   str                 = "",
        mailbox:     Optional[Mailbox]   = None,
        **_,
    ) -> str:
        class_list = "\n".join(f"- `{c}`" for c in (classes or []))
        info_block = f"\n## Tree summaries\n{tree_info}\n" if tree_info else ""
        return (
            f"## Task\n"
            f"Exactly one of the following event classes is present in this "
            f"sample. Instantiate every class's tree, compare normalized "
            f"confidences, and let the evidence decide.\n\n"
            f"## Candidate classes\n{class_list}\n"
            f"{info_block}\n"
            f"## Dataset description\n{description}\n\n"
            f"Survey with `view_full`, place each tree's primitives "
            f"(including _Pre/_Post), verify visually, focus on the "
            f"discriminative leaves, then `compare_trees`. Finish with "
            f"`submit_comment`."
        )

    @property
    def output_schema(self) -> Type[Artifact]:
        return TSEventCollection

    @property
    def required_backend_types(self) -> List[str]:
        return ["MultiELTBackend"]

    @property
    def budget(self) -> Optional[Budget]:
        return Budget(max_turns=40, max_tokens=800_000)


class SELASingleTreeInspectorSkill(Skill):
    """SELA per-class inspector — ONE agent, ONE class tree (parallel variant)."""

    @property
    def system_prompt(self) -> str:
        return """\
You are an expert multivariate time-series event analyst using Event Logic \
Trees (ELT). ONE candidate event structure — a tree for a single event class — \
is pre-loaded over the data sample. Exactly one event class is present in the \
sample, but it may or may not be YOUR class.

## Your job
Instantiate the tree as well as the data honestly allows. Your tree's \
normalized root confidence (mu_norm, shown by every visualisation) will be \
compared against OTHER analysts who examined the same sample for other \
classes — the highest mu_norm wins. You cannot see their work; the comparison \
is only fair if your scores are strictly CALIBRATED: score what the signal \
shows, never inflate to win.

## Tools
`view_full(tree)` / `view_window(tree, interval, vlines)` / \
`instantiate(tree, instances)` — `tree` is always your class's name. Every \
visualisation shows the signals (left, with your candidate bands) and your \
logic tree status (right).

## Operator constraints (guide interval choices)
* SYNC(a, b): a and b share the same time span (high overlap required).
* SEQ(a, b): b starts after a starts and must not be disjoint from a.
* GUARD(a, b): b occurs within a's span.
* OR(a, b): alternatives — instantiate the branch that matches; leave the \
other missing (ghost).

## Procedure
1. **Plan** — `view_full` to survey; identify EVERY region that could \
plausibly host the event (2-3 lookalikes are common). Zoom with `view_window` \
(Y re-normalises inside the window) before committing boundaries.
2. **Instantiate** — place ALL primitives (including pre-phase and \
post-phase ones) at the best placement.
3. **Verify (MANDATORY)** — study each returned visualisation: does every \
band cover the described morphology? Any unexpected ghost? Low SEQ/SYNC \
confidence signals an ordering/overlap violation. Re-instantiate only the \
aliases you want to fix (re-submitting an alias REPLACES its interval).
4. **Test the alternative** — when two regions are plausible, instantiate \
BOTH placements (compare mu_norm, keep the winner) before concluding.

## Placement discipline (anti-anchoring)
* The most visually dramatic region is NOT necessarily the event; the event \
is defined by the full phase story (Pre -> Main -> Post), not by amplitude.
* A placement is valid only if ALL THREE phases fit. If the post-phase \
trigger is absent right after Main, the placement is probably wrong — test \
the alternative region instead of omitting Post.
* The event ends at the FIRST termination trigger after the main morphology \
begins — never stretch Main past an intermediate trigger to a later one.

## Primitive grounding contract (internal consistency)
A band is a claim that the definition holds over the ENTIRE interval.
* **Homogeneity**: every sub-segment of the band must be compatible with the \
description — one contradicting excursion inside the band caps it at <= 0.2.
* **Definition words are binding, literally**: direction (downward != \
upward), rate (slow/gradual != abrupt), shape (linear / spike / step), and \
recovery behaviour must each match.
* **The worst part rules the score**: confidence is bounded by the \
worst-matching sub-segment, never the average.
* **Repair moves, in order**: (1) tighten boundaries to the maximal segment \
where the description truly holds; (2) let SEQ structure absorb transitions — \
never stretch a single band across two behaviours; (3) if no segment fits, \
retract (re-submit with confidence 0) or relocate.

## Confidence rubric (CALIBRATION IS EVERYTHING — never inflate)
0.9-1.0 strong match, clear pattern, stable boundaries; 0.7-0.9 good match, \
minor ambiguity; 0.4-0.7 weak/partial; 0.1-0.4 barely plausible. If the \
sample does not show your class, an honestly low mu_norm is the CORRECT \
outcome — another analyst's class should win. To retract a placed primitive, \
re-submit it with confidence 0.

## Finishing
Call `submit_comment` with 2-3 sentences: how well the sample supports your \
class and the decisive observations. The system reads your tree's Main \
interval and mu_norm directly — your instantiation IS the answer.\
"""

    def build_task_message(
        self,
        class_name:  str                 = "",
        description: str                 = "",
        tree_info:   str                 = "",
        mailbox:     Optional[Mailbox]   = None,
        **_,
    ) -> str:
        info_block = f"\n## Tree summary\n{tree_info}\n" if tree_info else ""
        return (
            f"## Task\n"
            f"Instantiate the ELT for event class `{class_name}` on this "
            f"sample and let your honest normalized confidence speak. Exactly "
            f"one event class is present in the sample — it may or may not be "
            f"`{class_name}`; a calibrated LOW score is the correct answer "
            f"when the sample shows something else.\n"
            f"{info_block}\n"
            f"## Dataset description\n{description}\n\n"
            f"Survey with `view_full`, place all primitives (including "
            f"_Pre/_Post), verify visually, test the alternative placement, "
            f"then `submit_comment`."
        )

    @property
    def output_schema(self) -> Type[Artifact]:
        return TSEventCollection

    @property
    def required_backend_types(self) -> List[str]:
        return ["MultiELTBackend"]

    @property
    def budget(self) -> Optional[Budget]:
        return Budget(max_turns=25, max_tokens=500_000)
