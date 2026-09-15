#!/usr/bin/env python3
"""
Build a 200-example golden evaluation set from twcs.csv (AmazonHelp only).

Important:
- Proxy keyword strata are used ONLY for stratified / difficulty-aware sampling.
- They are NOT ground-truth labels. final_intent is left blank for manual annotation.
- Stdlib only (no pandas required).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Locked project constants
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
TARGET_N = 200
BRAND = "AmazonHelp"

LOCKED_INTENTS = [
    "Delivery Delay / Service Issue",
    "Marked Delivered But Not Received",
    "Order Cancellation & Refund",
    "Wrong or Defective Item Received",
    "Customer Service Quality Complaint",
    "Prime Membership, Billing & Video",
    "Payment, Gift Card & Billing Issue",
    "Account Security",
    "Device & App Technical Issue",
    "Shipping Speed / Prime SLA Not Met",
    "Packaging Feedback",
    "General Help Request / Greeting",
    "Other / Unclassified",
]

DIFFICULTIES = ("Easy", "Medium", "Hard")

ANNOTATION_COLUMNS = [
    "id",
    "tweet",
    "final_intent",
    "difficulty",
    "annotator_confidence",
    "annotation_reason",
    "source_tweet_id",
]

# Exact sampling allocation (must sum to TARGET_N = 200).
# Keys are sampling strata. Proxy intent names mirror the locked taxonomy but
# are sampling aids only — not final labels.
EASY_ALLOCATION = {
    "Delivery Delay / Service Issue": 8,
    "Marked Delivered But Not Received": 6,
    "Order Cancellation & Refund": 7,
    "Wrong or Defective Item Received": 6,
    "Customer Service Quality Complaint": 4,
    "Prime Membership, Billing & Video": 5,
    "Payment, Gift Card & Billing Issue": 6,
    "Account Security": 4,
    "Device & App Technical Issue": 6,
    "Shipping Speed / Prime SLA Not Met": 5,
    "Packaging Feedback": 4,
    "General Help Request / Greeting": 5,
    "Other / Unclassified": 4,
}

MEDIUM_ALLOCATION = {
    "Delivery Delay / Service Issue": 9,
    "Marked Delivered But Not Received": 6,
    "Order Cancellation & Refund": 8,
    "Wrong or Defective Item Received": 6,
    "Customer Service Quality Complaint": 5,
    "Prime Membership, Billing & Video": 6,
    "Payment, Gift Card & Billing Issue": 7,
    "Account Security": 5,
    "Device & App Technical Issue": 7,
    "Shipping Speed / Prime SLA Not Met": 6,
    "Packaging Feedback": 4,
    "General Help Request / Greeting": 5,
    "Other / Unclassified": 6,
}

# Hard strata deliberately target known overlap / borderline cases.
HARD_ALLOCATION = {
    "overlap:Delivery vs Shipping Speed": 10,
    "overlap:Refund vs Payment/Billing": 8,
    "overlap:Delivery vs Marked Delivered": 8,
    "overlap:CS Quality vs actionable": 10,
    "overlap:General Help vs substantive": 8,
    "ambiguous:multi-signal / borderline": 6,
}


# ---------------------------------------------------------------------------
# English filter (defined here; no prior project heuristic existed in-repo)
# ---------------------------------------------------------------------------

_NON_LATIN = re.compile(
    r"[\u0400-\u04FF\u0600-\u06FF\u0900-\u097F\u3040-\u30FF\u3400-\u9FFF\uAC00-\uD7AF]"
)
_EN_GRAMMAR = re.compile(
    r"\b(the|and|is|are|was|were|my|you|your|this|that|with|have|has|had|"
    r"not|please|can|cannot|won't|don't|didn't|can't|wouldn't|shouldn't|"
    r"could|would|should|been|being|will|just|still|already|because|"
    r"what|when|where|why|how|who|which|from|about|into|over|under|"
    r"again|only|also|than|then|them|they|their|our|ours|mine|"
    r"does|did|doing|get|got|getting|need|needs|needed|want|wanted|"
    r"help|hello|hi|hey|thanks|thank)\b",
    re.I,
)
_EN_DOMAIN = re.compile(
    r"\b(order|orders|package|parcel|delivery|delivered|shipping|shipped|"
    r"refund|refunded|cancel|cancelled|canceled|return|returned|"
    r"account|payment|charged|billing|invoice|gift ?card|"
    r"login|password|hacked|damaged|broken|defective|delayed|"
    r"tracking|courier|driver|warehouse|seller)\b",
    re.I,
)
# Brand/product tokens appear in many languages; not sufficient English evidence alone.
_BRANDISH = re.compile(
    r"\b(amazon|prime|alexa|echo|kindle|fire ?tv|amzl)\b",
    re.I,
)
_FOREIGN_LEXICONS = {
    "de": re.compile(
        r"\b(und|nicht|bitte|danke|warum|meine|mein|dein|keine|oder|"
        r"paket|lieferung|bestellung|wurde|auch|noch|schon|wieder|leider|"
        r"können|einen|einem|gibt|ich|vom|aufs|kabel|verbindung|"
        r"handschuhfach|finden|finde|sehr|heute|morgen|gestern|"
        r"richtig|falsch|kaputt)\b",
        re.I,
    ),
    "es": re.compile(
        r"\b(gracias|hola|pedido|paquete|ayuda|porque|porqué|está|estan|"
        r"están|envio|envío|compra|quiero|puedo|dónde|donde|"
        r"años|llevando|lectura|lados|feliz|cumpleaños|acompañ|"
        r"también|después|ahora|necesito|urgente|habéis|entregado|"
        r"motivo|llamado|llegó|llegado|recibí|recibido)\b",
        re.I,
    ),
    "fr": re.compile(
        r"\b(bonjour|merci|commande|colis|pourquoi|aide|livraison|"
        r"suis|avec|dans|votre|svp|chez|trop|paye|abonnement|"
        r"reçois|recevoir|heure|voulez|offrir|mais|pas|mes|"
        r"temps|toujours|jamais|aujourd)\b",
        re.I,
    ),
    "pt": re.compile(
        r"\b(obrigado|obrigada|pedido|ajuda|entrega|pacote|não|nao|voce|"
        r"você|minha|quero|muito|preciso|atualização|recebi|chegou)\b",
        re.I,
    ),
    "it": re.compile(
        r"\b(grazie|ordine|pacco|perché|perche|aiuto|consegna|"
        r"vorrei|corriere|ritardo|rilassatissimo|della|dello|degli|"
        r"delle|ancora|quando|spedizione|arrivato|arrivata)\b",
        re.I,
    ),
    "nl": re.compile(
        r"\b(niet|mijn|pakket|bestelling|levering|alsjeblieft|waarom|"
        r"heel|graag|naar)\b",
        re.I,
    ),
}

# Orthographic cues common in Romance/Germanic tweets but rare in English CS text
_NON_EN_ORTHO = re.compile(
    r"[¿¡]|[àâäèéêëìíîïòóôùúûüœæçñ]|[ÀÂÄÈÉÊËÌÍÎÏÒÓÔÙÚÛÜŒÆÇÑ]"
)


def _clean_for_lang(text: str) -> str:
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#\w+", " ", text)
    return text


def is_likely_english(text: str) -> bool:
    """Heuristic English filter for AmazonHelp opening tweets.

    Designed to remove clear non-English customer tweets while keeping
    informal English support messages. Not a perfect language detector.
    """
    if not text or not text.strip():
        return False
    if _NON_LATIN.search(text):
        return False

    cleaned = _clean_for_lang(text)
    letters = [c for c in cleaned if c.isalpha()]
    if len(letters) < 6:
        return False

    ascii_letters = sum(1 for c in letters if ord(c) < 128)
    ascii_ratio = ascii_letters / len(letters)
    if ascii_ratio < 0.90:
        return False

    # Non-English orthography is treated as non-English for this English-only eval set.
    if _NON_EN_ORTHO.search(cleaned):
        return False

    grammar = len(_EN_GRAMMAR.findall(cleaned))
    domain = len(_EN_DOMAIN.findall(cleaned))
    brand = len(_BRANDISH.findall(cleaned))
    en_hits = grammar + domain  # brand-only does not count

    foreign_hits = {
        lang: len(rx.findall(cleaned)) for lang, rx in _FOREIGN_LEXICONS.items()
    }
    max_foreign = max(foreign_hits.values()) if foreign_hits else 0
    total_foreign = sum(foreign_hits.values())

    if max_foreign >= 2 and max_foreign > en_hits:
        return False
    if total_foreign >= 3 and total_foreign >= en_hits:
        return False
    if en_hits == 0:
        return False
    # Brand mention + foreign cue without real English grammar/domain support
    if grammar == 0 and domain == 0:
        return False
    if grammar == 0 and total_foreign >= 1 and brand >= 1:
        return False
    if total_foreign > en_hits:
        return False

    return True


# ---------------------------------------------------------------------------
# Proxy intent / difficulty heuristics (sampling only)
# ---------------------------------------------------------------------------

def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.I)


PROXY_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "Delivery Delay / Service Issue": [
        _rx(r"\b(late|delayed|delay|still waiting|hasn't arrived|hasnt arrived|"
            r"not arrived|never arrived|where is my (order|package|parcel)|"
            r"stuck in transit|in transit for|been waiting|"
            r"days? (late|ago)|week(s)? (late|ago)|still not (here|delivered|"
            r"arrived)|no update|tracking (has )?not updated)\b"),
        _rx(r"\b(lost (package|parcel|order)|package (is )?lost|missing package)\b"),
    ],
    "Marked Delivered But Not Received": [
        _rx(r"\b(marked (as )?delivered|says? (it was )?delivered|"
            r"showing delivered|tracking says delivered|delivered but|"
            r"says delivered|status (is|says) delivered|"
            r"delivered (but|and) (i|never|not|didn't|did not)|"
            r"never received|not received|didn't receive|did not receive|"
            r"nobody (was )?home|left (at|in) (the )?(lobby|porch|office)|"
            r"handed to (me|resident)|delivery photo)\b"),
    ],
    "Order Cancellation & Refund": [
        _rx(r"\b(cancel(led|lation)?|refund(ed|s)?|return(ed|ing)?|"
            r"money back|want(ed)? (a )?refund|process(ing)? (my )?refund|"
            r"return label|rma)\b"),
    ],
    "Wrong or Defective Item Received": [
        _rx(r"\b(wrong (item|product|order|size|color|colour)|"
            r"defective|damaged|broken|cracked|not working|doesn't work|"
            r"does not work|missing (item|part|pieces|accessories)|"
            r"incomplete|not what i ordered|different (item|product)|"
            r"counterfeit|fake product|used condition)\b"),
    ],
    "Customer Service Quality Complaint": [
        _rx(r"\b(customer service|support (team|agent|rep)|no (one|body) (is )?"
            r"(responding|helping|replying)|ignored|pathetic|terrible support|"
            r"worst (customer )?service|rude (agent|rep|support)|"
            r"can't (get|reach) (a )?(human|agent|anyone)|"
            r"on hold|transferred|escalat)\b"),
    ],
    "Prime Membership, Billing & Video": [
        _rx(r"\b(prime membership|amazon prime|prime video|prime benefit|"
            r"prime subscript|renew(al)? (my )?prime|cancel(led)? prime|"
            r"prime trial|video (won't|will not|not) play|"
            r"streaming|prime day)\b"),
    ],
    "Payment, Gift Card & Billing Issue": [
        _rx(r"\b(gift ?card|payment(s)?|charged|double charg|"
            r"credit card|debit card|billing|invoice|overcharg|"
            r"payment method|payment failed|declined|"
            r"amazon pay|wallet|promo code|coupon)\b"),
    ],
    "Account Security": [
        _rx(r"\b(hacked|unauthorized|password|login|log in|sign[- ]?in|"
            r"account (locked|compromised|suspended|stolen)|"
            r"suspicious (activity|login|sign)|two[- ]factor|"
            r"2fa|otp|verification code|someone (accessed|used) my)\b"),
    ],
    "Device & App Technical Issue": [
        _rx(r"\b(alexa|echo( show| dot)?|fire ?tv|kindle|app (crash|error|"
            r"update|won't|will not|not)|freeze|frozen|won't connect|"
            r"will not connect|bluetooth|wifi|wi-fi|firmware|"
            r"software (bug|issue|update)|screen (black|blank|frozen)|"
            r"not recognizing|won't turn on|will not turn on)\b"),
    ],
    "Shipping Speed / Prime SLA Not Met": [
        _rx(r"\b((two|2)[- ]day|next[- ]day|same[- ]day|one[- ]day|"
            r"prime shipping|guaranteed (delivery|by)|"
            r"supposed to (arrive|be here|be delivered) (today|yesterday|"
            r"by)|promised (by|for|delivery)|sla|"
            r"delivery estimate|estimated delivery|"
            r"free (two|2)[- ]day)\b"),
    ],
    "Packaging Feedback": [
        _rx(r"\b(packaging|over[- ]?packag|packing|bubble wrap|"
            r"cardboard|box (was|is) (huge|damaged|crushed|open)|"
            r"excessive (packaging|plastic)|poor(ly)? pack|"
            r"arrived (open|opened|unsealed))\b"),
    ],
    "General Help Request / Greeting": [
        _rx(r"^\s*@(amazonhelp|\d+)\s*(hi|hello|hey|please help|"
            r"can you help|need help|quick question|i have a question)\b"),
        _rx(r"\b(can you help|please help|need (some )?help|"
            r"quick question|i have a question|wondering if|"
            r"any (help|advice)|looking for help)\b"),
    ],
}

ACTIONABLE_INTENTS = {
    "Delivery Delay / Service Issue",
    "Marked Delivered But Not Received",
    "Order Cancellation & Refund",
    "Wrong or Defective Item Received",
    "Prime Membership, Billing & Video",
    "Payment, Gift Card & Billing Issue",
    "Account Security",
    "Device & App Technical Issue",
    "Shipping Speed / Prime SLA Not Met",
    "Packaging Feedback",
}


@dataclass
class Candidate:
    tweet_id: str
    text: str
    author_id: str
    created_at: str
    conversation_root: str
    matched_intents: list[str] = field(default_factory=list)
    match_counts: dict[str, int] = field(default_factory=dict)
    overlap_flags: list[str] = field(default_factory=list)
    proxy_intent: str = "Other / Unclassified"
    difficulty_hint: str = "Medium"


def score_text(text: str) -> tuple[dict[str, int], list[str]]:
    counts: dict[str, int] = {}
    for intent, patterns in PROXY_PATTERNS.items():
        total = 0
        for pat in patterns:
            total += len(pat.findall(text))
        if total:
            counts[intent] = total
    matched = sorted(counts.keys(), key=lambda k: (-counts[k], k))
    return counts, matched


def detect_overlaps(matched: list[str], counts: dict[str, int]) -> list[str]:
    flags: list[str] = []
    s = set(matched)
    if (
        "Delivery Delay / Service Issue" in s
        and "Shipping Speed / Prime SLA Not Met" in s
    ):
        flags.append("overlap:Delivery vs Shipping Speed")
    if (
        "Order Cancellation & Refund" in s
        and "Payment, Gift Card & Billing Issue" in s
    ):
        flags.append("overlap:Refund vs Payment/Billing")
    if (
        "Delivery Delay / Service Issue" in s
        and "Marked Delivered But Not Received" in s
    ):
        flags.append("overlap:Delivery vs Marked Delivered")
    if "Customer Service Quality Complaint" in s and (s & ACTIONABLE_INTENTS):
        flags.append("overlap:CS Quality vs actionable")
    if "General Help Request / Greeting" in s and (
        s & (ACTIONABLE_INTENTS | {"Customer Service Quality Complaint"})
    ):
        flags.append("overlap:General Help vs substantive")
    if len(matched) >= 3:
        flags.append("ambiguous:multi-signal / borderline")
    elif len(matched) == 2 and not flags:
        # Two competing substantive signals without a named overlap bucket
        flags.append("ambiguous:multi-signal / borderline")
    elif len(matched) == 1 and counts.get(matched[0], 0) == 1:
        # Single weak hit — borderline for hard pool if needed
        pass
    return flags


def assign_proxy_and_difficulty(candidate: Candidate) -> None:
    counts, matched = score_text(candidate.text)
    candidate.match_counts = counts
    candidate.matched_intents = matched
    candidate.overlap_flags = detect_overlaps(matched, counts)

    if not matched:
        candidate.proxy_intent = "Other / Unclassified"
        candidate.difficulty_hint = "Medium"
        return

    # Primary proxy = strongest keyword signal
    candidate.proxy_intent = matched[0]

    if candidate.overlap_flags:
        candidate.difficulty_hint = "Hard"
        return

    strength = counts[matched[0]]
    if len(matched) == 1 and strength >= 2:
        candidate.difficulty_hint = "Easy"
    elif len(matched) == 1 and strength == 1:
        # Clear single cue but thin evidence → Medium
        candidate.difficulty_hint = "Medium"
    else:
        candidate.difficulty_hint = "Hard"


# ---------------------------------------------------------------------------
# Conversation graph (connected components via in_response_to_tweet_id)
# ---------------------------------------------------------------------------

class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def load_amazonhelp_openings(csv_path: Path) -> tuple[list[Candidate], dict]:
    """Return English AmazonHelp conversation-opening customer tweets + stats."""
    uf = UnionFind()
    amazon_ids: set[str] = set()

    needed = {
        "tweet_id",
        "author_id",
        "inbound",
        "created_at",
        "text",
        "response_tweet_id",
        "in_response_to_tweet_id",
    }

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not needed.issubset(set(reader.fieldnames)):
            missing = needed - set(reader.fieldnames or [])
            raise SystemExit(f"CSV missing required columns: {sorted(missing)}")

        for row in reader:
            tid = row["tweet_id"]
            uf.parent.setdefault(tid, tid)
            if row["author_id"] == BRAND:
                amazon_ids.add(tid)
            parent_id = (row.get("in_response_to_tweet_id") or "").strip()
            if parent_id:
                uf.union(tid, parent_id)

    if not amazon_ids:
        raise SystemExit(f"No {BRAND} tweets found in {csv_path}")

    amazon_roots = {uf.find(tid) for tid in amazon_ids}

    openings: list[Candidate] = []
    n_openings_raw = 0
    n_non_english = 0

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["inbound"] != "True":
                continue
            if (row.get("in_response_to_tweet_id") or "").strip():
                continue
            root = uf.find(row["tweet_id"])
            if root not in amazon_roots:
                continue
            n_openings_raw += 1
            text = row["text"] or ""
            if not is_likely_english(text):
                n_non_english += 1
                continue
            openings.append(
                Candidate(
                    tweet_id=row["tweet_id"],
                    text=text,
                    author_id=row["author_id"],
                    created_at=row["created_at"],
                    conversation_root=root,
                )
            )

    stats = {
        "amazonhelp_tweets": len(amazon_ids),
        "amazonhelp_conversations": len(amazon_roots),
        "opening_customer_tweets": n_openings_raw,
        "non_english_filtered": n_non_english,
        "english_opening_candidates": len(openings),
    }
    return openings, stats


# ---------------------------------------------------------------------------
# Stratified, difficulty-aware sampling
# ---------------------------------------------------------------------------

def _take(
    pool: list[Candidate],
    n: int,
    rng: random.Random,
    used_ids: set[str],
    used_roots: set[str],
) -> list[Candidate]:
    available = [
        c
        for c in pool
        if c.tweet_id not in used_ids and c.conversation_root not in used_roots
    ]
    rng.shuffle(available)
    chosen = available[:n]
    for c in chosen:
        used_ids.add(c.tweet_id)
        used_roots.add(c.conversation_root)
    return chosen


def sample_golden_set(
    candidates: list[Candidate], seed: int = RANDOM_SEED
) -> tuple[list[tuple[Candidate, str, str]], dict]:
    """Return list of (candidate, difficulty, sampling_stratum) and report."""
    rng = random.Random(seed)

    for c in candidates:
        assign_proxy_and_difficulty(c)

    # Index pools
    by_proxy_easy: dict[str, list[Candidate]] = defaultdict(list)
    by_proxy_medium: dict[str, list[Candidate]] = defaultdict(list)
    by_hard_stratum: dict[str, list[Candidate]] = defaultdict(list)

    for c in candidates:
        if c.difficulty_hint == "Easy":
            by_proxy_easy[c.proxy_intent].append(c)
        elif c.difficulty_hint == "Medium":
            by_proxy_medium[c.proxy_intent].append(c)
        else:
            # Prefer named overlap flags; else ambiguous bucket
            placed = False
            for flag in c.overlap_flags:
                if flag in HARD_ALLOCATION:
                    by_hard_stratum[flag].append(c)
                    placed = True
                    break
            if not placed:
                by_hard_stratum["ambiguous:multi-signal / borderline"].append(c)

        # Also allow Easy/Medium single-intent items to fill hard overlaps? No —
        # Hard comes from overlap/ambiguous only. If a hard stratum is short,
        # we backfill from other hard pools, then from medium with same proxy.

    used_ids: set[str] = set()
    used_roots: set[str] = set()
    selected: list[tuple[Candidate, str, str]] = []
    allocation_filled: dict[str, int] = {}
    shortfalls: list[str] = []

    def record(items: list[Candidate], difficulty: str, stratum: str) -> None:
        for c in items:
            selected.append((c, difficulty, stratum))
        allocation_filled[f"{difficulty}::{stratum}"] = len(items)

    # 1) Hard first (scarcer)
    for stratum, n in HARD_ALLOCATION.items():
        got = _take(by_hard_stratum.get(stratum, []), n, rng, used_ids, used_roots)
        if len(got) < n:
            # Backfill from other hard strata
            deficit = n - len(got)
            backfill_pool: list[Candidate] = []
            for other, pool in by_hard_stratum.items():
                if other == stratum:
                    continue
                backfill_pool.extend(pool)
            got += _take(backfill_pool, deficit, rng, used_ids, used_roots)
        if len(got) < n:
            shortfalls.append(f"Hard/{stratum}: wanted {n}, got {len(got)}")
        record(got, "Hard", stratum)

    # 2) Easy by proxy intent
    for intent, n in EASY_ALLOCATION.items():
        pool = by_proxy_easy.get(intent, [])
        got = _take(pool, n, rng, used_ids, used_roots)
        if len(got) < n:
            # Backfill from medium same proxy that look relatively clear
            deficit = n - len(got)
            med = [
                c
                for c in by_proxy_medium.get(intent, [])
                if not c.overlap_flags and len(c.matched_intents) <= 1
            ]
            got += _take(med, deficit, rng, used_ids, used_roots)
        if len(got) < n:
            shortfalls.append(f"Easy/{intent}: wanted {n}, got {len(got)}")
        record(got, "Easy", intent)

    # 3) Medium by proxy intent
    for intent, n in MEDIUM_ALLOCATION.items():
        pool = by_proxy_medium.get(intent, [])
        got = _take(pool, n, rng, used_ids, used_roots)
        if len(got) < n:
            # Backfill from remaining easy same proxy, then any unused same proxy
            deficit = n - len(got)
            easy_left = by_proxy_easy.get(intent, [])
            got += _take(easy_left, deficit, rng, used_ids, used_roots)
        if len(got) < n:
            deficit = n - len(got)
            any_same = [
                c
                for c in candidates
                if c.proxy_intent == intent
                and c.tweet_id not in used_ids
                and c.conversation_root not in used_roots
            ]
            got += _take(any_same, deficit, rng, used_ids, used_roots)
        if len(got) < n:
            shortfalls.append(f"Medium/{intent}: wanted {n}, got {len(got)}")
        record(got, "Medium", intent)

    # Final backfill to reach exactly TARGET_N if any shortfall
    if len(selected) < TARGET_N:
        deficit = TARGET_N - len(selected)
        remaining = [
            c
            for c in candidates
            if c.tweet_id not in used_ids and c.conversation_root not in used_roots
        ]
        extra = _take(remaining, deficit, rng, used_ids, used_roots)
        for c in extra:
            selected.append((c, c.difficulty_hint, f"backfill:{c.proxy_intent}"))
        allocation_filled["backfill"] = len(extra)

    if len(selected) > TARGET_N:
        rng.shuffle(selected)
        selected = selected[:TARGET_N]

    # Stable order for output: Easy, Medium, Hard then id
    difficulty_order = {"Easy": 0, "Medium": 1, "Hard": 2}
    selected.sort(key=lambda x: (difficulty_order.get(x[1], 9), x[0].tweet_id))

    report = {
        "seed": seed,
        "requested": TARGET_N,
        "sampled": len(selected),
        "unique_tweet_ids": len({c.tweet_id for c, _, _ in selected}),
        "unique_conversations": len({c.conversation_root for c, _, _ in selected}),
        "allocation_target": {
            "Easy": EASY_ALLOCATION,
            "Medium": MEDIUM_ALLOCATION,
            "Hard": HARD_ALLOCATION,
        },
        "allocation_filled": allocation_filled,
        "difficulty_counts": {
            d: sum(1 for _, diff, _ in selected if diff == d) for d in DIFFICULTIES
        },
        "proxy_intent_counts": dict(
            Counter(c.proxy_intent for c, _, _ in selected)
        ),
        "shortfalls": shortfalls,
        "pool_sizes": {
            "easy_by_proxy": {k: len(v) for k, v in sorted(by_proxy_easy.items())},
            "medium_by_proxy": {k: len(v) for k, v in sorted(by_proxy_medium.items())},
            "hard_by_stratum": {k: len(v) for k, v in sorted(by_hard_stratum.items())},
        },
    }
    return selected, report


def write_outputs(
    selected: list[tuple[Candidate, str, str]],
    stats: dict,
    sample_report: dict,
    out_csv: Path,
    reserved_ids_path: Path,
    meta_path: Path,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ANNOTATION_COLUMNS)
        writer.writeheader()
        for i, (cand, difficulty, _stratum) in enumerate(selected, start=1):
            writer.writerow(
                {
                    "id": f"golden_{i:03d}",
                    "tweet": cand.text,
                    "final_intent": "",
                    "difficulty": difficulty,
                    "annotator_confidence": "",
                    "annotation_reason": "",
                    "source_tweet_id": cand.tweet_id,
                }
            )

    with reserved_ids_path.open("w", encoding="utf-8") as f:
        f.write("# Reserved golden-set source tweet IDs. Do not use in train/dev.\n")
        for cand, _, _ in selected:
            f.write(f"{cand.tweet_id}\n")

    meta = {
        "population_stats": stats,
        "sampling_report": sample_report,
        "notes": [
            "final_intent is blank pending manual annotation with the locked 13-intent taxonomy.",
            "proxy_intent_counts reflect sampling heuristics only, not gold labels.",
            "difficulty values are sampling estimates to be confirmed/adjusted by the annotator.",
        ],
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _validate_allocation_sums() -> None:
    easy = sum(EASY_ALLOCATION.values())
    medium = sum(MEDIUM_ALLOCATION.values())
    hard = sum(HARD_ALLOCATION.values())
    total = easy + medium + hard
    if total != TARGET_N:
        raise SystemExit(
            f"Allocation sums to {total} (Easy={easy}, Medium={medium}, Hard={hard}); "
            f"expected {TARGET_N}"
        )


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        type=Path,
        default=root / "twcs.csv",
        help="Path to twcs.csv",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=root / "data" / "golden_set.csv",
        help="Annotation CSV output path",
    )
    p.add_argument(
        "--reserved-ids",
        type=Path,
        default=root / "data" / "golden_set_reserved_ids.txt",
        help="Reserved tweet ID list for leakage prevention",
    )
    p.add_argument(
        "--meta",
        type=Path,
        default=root / "data" / "golden_set_build_meta.json",
        help="JSON metadata / sampling report path",
    )
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _validate_allocation_sums()

    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")

    print(f"Reading {args.input} ...")
    openings, stats = load_amazonhelp_openings(args.input)
    print("Population stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Sanity vs locked expectations
    if stats["opening_customer_tweets"] != 81347:
        print(
            f"WARNING: opening customer tweets = {stats['opening_customer_tweets']} "
            f"(locked expectation ≈ 81,347)",
            file=sys.stderr,
        )
    if abs(stats["amazonhelp_conversations"] - 82556) > 100:
        print(
            f"WARNING: AmazonHelp conversations = {stats['amazonhelp_conversations']} "
            f"(locked expectation ≈ 82,556)",
            file=sys.stderr,
        )

    print(f"Sampling {TARGET_N} examples with seed={args.seed} ...")
    selected, sample_report = sample_golden_set(openings, seed=args.seed)

    if len(selected) != TARGET_N:
        raise SystemExit(
            f"Failed to sample exactly {TARGET_N} examples; got {len(selected)}. "
            f"Shortfalls: {sample_report.get('shortfalls')}"
        )

    write_outputs(
        selected,
        stats,
        sample_report,
        args.output,
        args.reserved_ids,
        args.meta,
    )

    print(f"Wrote {args.output}")
    print(f"Wrote {args.reserved_ids}")
    print(f"Wrote {args.meta}")
    print("Difficulty counts:", sample_report["difficulty_counts"])
    if sample_report["shortfalls"]:
        print("Allocation shortfalls (backfilled where possible):")
        for s in sample_report["shortfalls"]:
            print(" ", s)
    print("Done.")


if __name__ == "__main__":
    main()
