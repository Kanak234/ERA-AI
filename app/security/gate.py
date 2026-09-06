"""ERA AI — cybersecurity safety gate (spec Section 12 + 30).

This is a SAFETY feature, not a bypass. It classifies each request and:
  - freely allows educational + defensive security help
  - gates offensive/exploitation content behind an explicit
    authorization confirmation ("Authorized Lab" framing)
  - refuses out-of-scope requests aimed at harming third-party systems

The classifier is intentionally conservative. The out-of-scope tier
cannot be disabled from settings.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import List


class Tier(str, Enum):
    EDUCATIONAL = "educational"
    DEFENSIVE = "defensive"
    AUTHORIZED_LAB = "authorized-lab"
    OUT_OF_SCOPE = "out-of-scope"


# signals that a request targets someone else's real systems / real harm
OUT_OF_SCOPE = [
    r"\bwithout (their|his|her|the owner'?s) (permission|consent|knowledge)\b",
    r"\bhack (into )?(my (ex|friend|neighbou?r|school|college|company)|someone'?s|his|her|their)\b",
    r"\b(steal|exfiltrate|dump) (their|his|her|the) (data|passwords?|credentials?)\b",
    r"\bransomware (to|for) (deploy|spread|attack)\b",
    r"\bddos\b.*\b(website|server|company|target)\b",
    r"\bbypass (the )?(bank|government|real)\b",
    r"\bspy on\b", r"\bstalk\b", r"\bkeylog(ger)?\b.*\b(install|deploy|target)\b",
]

# offensive/exploitation work that is fine ONLY in an authorized lab
OFFENSIVE = [
    r"\bexploit\b", r"\bpayload\b", r"\bmetasploit\b", r"\breverse shell\b",
    r"\bprivilege escalation\b", r"\bbuffer overflow\b", r"\bshellcode\b",
    r"\bbrute[- ]?force\b", r"\bsql injection\b", r"\bxss\b", r"\bcsrf\b",
    r"\bcrack\b.*\b(hash|password)\b", r"\bc2\b", r"\bpost[- ]?exploitation\b",
]

# clearly defensive / blue-team
DEFENSIVE = [
    r"\bdetect\b", r"\bharden\b", r"\bfirewall\b", r"\bincident response\b",
    r"\blog analysis\b", r"\bsiem\b", r"\bblue[- ]?team\b", r"\bmitigat",
    r"\bsecure (coding|configuration)\b", r"\bpatch\b", r"\bvulnerabilit",
    r"\baudit\b", r"\bdefen[cs]e\b", r"\bmonitor\b",
]

LAB_CONTEXT = [
    r"\bctf\b", r"\bhack ?the ?box\b", r"\btryhackme\b", r"\bmy own (lab|vm|machine)\b",
    r"\bauthori[sz]ed\b", r"\bpen ?test\b.*\b(authori|scope|client)\b",
    r"\blab environment\b", r"\bvirtual machine\b", r"\bisolated\b",
]


@dataclass
class GateResult:
    tier: Tier
    allowed: bool
    needs_confirmation: bool
    message: str
    signals: List[str]


def _hits(text: str, pats: List[str]) -> List[str]:
    return [p for p in pats if re.search(p, text, re.IGNORECASE)]


def classify(text: str) -> GateResult:
    oos = _hits(text, OUT_OF_SCOPE)
    if oos:
        return GateResult(
            Tier.OUT_OF_SCOPE, allowed=False, needs_confirmation=False,
            message=("This looks aimed at a system you may not be authorized "
                     "to test. ERA won't help with actions against third-party "
                     "systems. If this is your own system or an authorized "
                     "engagement, rephrase with that context."),
            signals=oos,
        )

    offensive = _hits(text, OFFENSIVE)
    lab = _hits(text, LAB_CONTEXT)
    defensive = _hits(text, DEFENSIVE)

    if offensive and not lab:
        return GateResult(
            Tier.AUTHORIZED_LAB, allowed=True, needs_confirmation=True,
            message=("This is offensive-security content. ERA will help for "
                     "learning or authorized testing only. Confirm this is your "
                     "own system, an authorized engagement, or an isolated lab "
                     "(CTF/VM)."),
            signals=offensive,
        )

    if offensive and lab:
        return GateResult(
            Tier.AUTHORIZED_LAB, allowed=True, needs_confirmation=False,
            message="Authorized-lab context detected. Proceeding.",
            signals=offensive + lab,
        )

    if defensive:
        return GateResult(Tier.DEFENSIVE, True, False,
                          "Defensive security request.", defensive)

    return GateResult(Tier.EDUCATIONAL, True, False,
                      "General/educational request.", [])
