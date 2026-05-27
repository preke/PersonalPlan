"""
Parse prompt_for_inference.txt into sections accessible as S1..S12.

The inference prompt is the single source of truth. We never duplicate
its text — every baseline imports the sections from here.
"""
from pathlib import Path
import re

PROMPT_FILE = Path(__file__).parent.parent.parent / "prompt_for_inference.txt"
_RAW = PROMPT_FILE.read_text(encoding="utf-8")


def _parse_sections(text):
    """
    Split by "-+\nN. NAME\n-+\n" headers.
    Returns dict: {0: preamble_text, 1: (name, body), ..., 12: (name, body)}.
    """
    sections = {}

    # Preamble: everything before the first "---..." divider line
    m_pre = re.search(r"^(.*?)^-{3,}\s*$", text, re.DOTALL | re.MULTILINE)
    if m_pre:
        sections[0] = m_pre.group(1).strip()

    # Numbered sections
    pattern = re.compile(
        r"^-{3,}\s*\n(\d+)\.\s+(.+?)\n-{3,}\s*\n(.*?)"
        r"(?=^-{3,}\s*\n\d+\.|\Z)",
        re.DOTALL | re.MULTILINE,
    )
    for m in pattern.finditer(text):
        num = int(m.group(1))
        name = m.group(2).strip()
        body = m.group(3).strip()
        sections[num] = (name, body)
    return sections


_SECTIONS = _parse_sections(_RAW)

PREAMBLE = _SECTIONS[0]  # "You generate personalized multi-agent plans..."

# Sections 1..12; access body as S{N}, full title as S{N}_TITLE
S1 = _SECTIONS[1][1];  S1_TITLE = _SECTIONS[1][0]   # HOW TO INFER THE DESTINATION
S2 = _SECTIONS[2][1];  S2_TITLE = _SECTIONS[2][0]   # HOW TO PERSONALIZE
S3 = _SECTIONS[3][1];  S3_TITLE = _SECTIONS[3][0]   # FIELD ATTRIBUTION
S4 = _SECTIONS[4][1];  S4_TITLE = _SECTIONS[4][0]   # PEDAGOGICAL STRATEGIES
S5 = _SECTIONS[5][1];  S5_TITLE = _SECTIONS[5][0]   # AVAILABLE TOOLS
S6 = _SECTIONS[6][1];  S6_TITLE = _SECTIONS[6][0]   # HOW TO BUILD THE WORKFLOW
S7 = _SECTIONS[7][1];  S7_TITLE = _SECTIONS[7][0]   # LOOPS
S8 = _SECTIONS[8][1];  S8_TITLE = _SECTIONS[8][0]   # COMPLETION
S9 = _SECTIONS[9][1];  S9_TITLE = _SECTIONS[9][0]   # OUTPUT FORMAT (STRICT JSON)
S10 = _SECTIONS[10][1]; S10_TITLE = _SECTIONS[10][0] # EXAMPLES
S11 = _SECTIONS[11][1]; S11_TITLE = _SECTIONS[11][0] # SELF-CHECK
S12 = _SECTIONS[12][1]; S12_TITLE = _SECTIONS[12][0] # OUTPUT (closing)

# Convenience preassemblies
FULL_PROMPT = _RAW

def section(num: int, display_num: int | None = None) -> str:
    """Returns a section's body wrapped with its own divider banner.

    ``display_num`` overrides the section number shown in the banner
    (the body text is unchanged). Used by ``compose(..., renumber=True)``
    so the rendered prompt shows sequential section numbers (1, 2, 3,
    ...) instead of the original prompt_for_inference.txt indices.
    """
    name, body = _SECTIONS[num]
    bar = "-" * 50
    label = display_num if display_num is not None else num
    return f"{bar}\n{label}. {name}\n{bar}\n{body}\n"


def compose(nums: list[int], with_preamble: bool = True,
            renumber: bool = False) -> str:
    """Compose chosen sections into a single prompt string.

    When ``renumber=True`` the section banners are relabelled to be
    sequential (1, 2, 3, ...) in the order ``nums`` lists them, and
    cross-references in the bodies (``§<orig_num>``) are rewritten to
    the new labels so the rendered prompt is internally consistent.
    Cross-references to sections NOT in ``nums`` are left as-is.
    """
    parts = []
    if with_preamble:
        parts.append(PREAMBLE)
    for i, n in enumerate(nums, start=1):
        parts.append(section(n, display_num=i if renumber else None))
    out = "\n\n".join(parts)

    if renumber:
        mapping = {n: i for i, n in enumerate(nums, start=1)}
        # Sort by descending original number to avoid e.g. §1 substituting
        # inside an unreplaced §12 before §12 itself is handled.
        for orig in sorted(mapping, reverse=True):
            new = mapping[orig]
            out = out.replace(f"§{orig}", f"§{new}")
    return out


# --------------------------------------------------------------------
# Tier-specific convenience helpers
# --------------------------------------------------------------------

def compose_t1() -> str:
    """L1-L3 (closed-source LLMs): same task description package as F1/F2.
    L*, F*, M* all receive identical input (compose_t4 stays as function name
    for backward-compat with existing imports).
    """
    return compose_t4()


def compose_t4() -> str:
    """T4 (MAS frameworks): PREAMBLE + AVAILABLE TOOLS (§5) +
    OUTPUT FORMAT (§9) + OUTPUT closing (§12), with section banners
    renumbered sequentially to 1-3 and inline cross-references rewritten
    to match. Drops pedagogy / strategy sections (§1-§4, §6-§8, §10,
    §11). §10 EXAMPLES is intentionally dropped so framework topology
    is not seduced by our plan-shape examples.
    """
    return compose([5, 9, 12], renumber=True)


def compose_t5() -> str:
    """M1-M3 (specialized plan-generation MAS): empty string — M-tier gets only (query, learner)."""
    return ""
