"""
Synthetic fake/real news corpus generator.
===========================================

WHY THIS EXISTS
----------------
The original tutorial downloads a labelled dataset (news.csv) from a GFG media
server. This sandbox has no internet access, so that download is not possible.

Rather than skip the data step, this module generates a labelled corpus from
templates that encode two well-documented stylistic registers:

  REAL : neutral tone, attribution to a source/institution, hedged claims,
         concrete quantities -- the register real news-wire reporting uses.
  FAKE : sensational tone, urgency/virality language, unverified absolute
         claims, conspiracy framing -- the register most fake-news-detection
         literature identifies as a strong stylistic signal (independent of
         fact-checking the content itself).

This is a genuine, reproducible binary text classification task with real
linguistic signal -- just not the real Kaggle/GFG dataset. Swapping in that
CSV (columns: title, text, label) requires no code changes beyond the loader
at the bottom of this file (see `load_real_dataset_instead`).
"""

import random

random.seed(42)

TOPICS = [
    "the new transport bill", "the regional water supply", "local school funding",
    "the national vaccination programme", "the housing market", "the annual budget",
    "the city's traffic plan", "renewable energy subsidies", "the public pension fund",
    "the university admissions system", "the trade agreement", "the airport expansion",
    "the minimum wage", "the coastal flood defences", "the national grid",
    "food safety standards", "the immigration policy", "the central bank's interest rate",
    "the new sports stadium", "online privacy rules", "the rail network upgrade",
    "the hospital waiting times", "the agricultural subsidy scheme", "the currency reserve",
    "the space research programme", "the wildlife conservation fund", "the film industry tax credit",
]

INSTITUTIONS = [
    "the Ministry of Transport", "the National Health Agency", "the Bureau of Statistics",
    "the Central Bank", "the Office of Public Safety", "the Institute for Climate Studies",
    "the City Transit Authority", "the Department of Education", "the Trade Commission",
    "the Science Research Council", "the Housing Authority", "the National Water Board",
    "the Energy Regulatory Office", "the Office of the Auditor General",
]

SOURCES = [
    "a spokesperson", "an internal report", "officials", "a government audit",
    "a press release", "the annual review", "a parliamentary committee", "a public statement",
]

JOURNALS = [
    "the Journal of Applied Economics", "the National Review of Public Policy",
    "the Quarterly Journal of Urban Studies", "the Review of Energy Systems",
]

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "this week", "yesterday"]

PERCENTS = [2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14, 15, 18, 22]

REAL_TEMPLATES = [
    "{inst} reported that spending on {topic} rose by {pct}% in the last quarter, according to {src}.",
    "{src} confirmed that {topic} will be reviewed after {inst} raised concerns about its cost.",
    "A new study published in {journal} found that {topic} had a smaller than expected effect on local employment.",
    "{inst} announced changes to {topic} on {day}, following a review requested by {src}.",
    "Officials at {inst} said {topic} is expected to be finalised by the end of the year, according to {src}.",
    "According to {src}, funding for {topic} was adjusted by {pct}% after the latest budget review.",
    "{inst} released figures on {day} showing {topic} performed broadly in line with projections.",
    "A report from {inst} concluded that {topic} needs further review before any changes are made.",
    "{src} said the proposal on {topic} was postponed until {inst} completes its assessment.",
    "Data released by {inst} on {day} show a {pct}% change in spending related to {topic}.",
]

FAKE_TEMPLATES = [
    "SHOCKING: {inst} is secretly hiding the truth about {topic} \u2014 officials don't want you to know!",
    "You won't believe what {inst} did to {topic} \u2014 SHARE this before it gets taken down!",
    "BREAKING: Insiders reveal {topic} was rigged from the start, and {inst} covered it up for years.",
    "Leaked documents \u201cprove\u201d {inst} has been lying about {topic} the whole time \u2014 mainstream media stays silent!",
    "EXPOSED: The real reason behind {topic} that {inst} is desperately trying to bury.",
    "Experts are FURIOUS after discovering what {inst} really did with {topic} \u2014 you need to see this now.",
    "They don't want this getting out: {inst} secretly changed {topic} overnight with zero oversight.",
    "This one leaked memo about {topic} will make you question everything {inst} has ever said.",
    "URGENT: {topic} scandal at {inst} is being erased from the news \u2014 read before it's deleted!",
    "The shocking truth about {topic} that {inst} has been hiding from the public for years.",
]


HARD_REAL_TEMPLATES = [
    "{src} said the decision on {topic} sparked outrage among residents, though {inst} stood by the review.",
    "Critics slammed {inst} over {topic}, but {src} insisted the numbers were accurate.",
    "{inst} faced sharp questions about {topic} after {src} flagged inconsistencies in the figures.",
    "Residents reacted angrily to {inst}'s handling of {topic}, according to {src}.",
    "{src} admitted {inst} was slow to respond to concerns about {topic}, calling it a mistake.",
    "A heated debate broke out over {topic} after {inst} released its findings, {src} reported.",
]

HARD_FAKE_TEMPLATES = [
    "{inst} quietly changed {topic} without telling anyone, sources close to the matter say.",
    "According to unnamed insiders, {topic} was decided behind closed doors at {inst}.",
    "A leaked memo suggests {inst} knew about problems with {topic} months before saying anything.",
    "Sources claim {inst} altered the numbers on {topic} just before the report was published.",
    "Multiple unverified accounts suggest {inst} misled the public about {topic} for months.",
    "An anonymous tip claims {inst} buried an internal warning about {topic} last year.",
]

NOISE_WORDS = ["really", "apparently", "reportedly", "notably", "in fact", "as expected"]


def maybe_add_noise(text, p=0.25):
    if random.random() < p:
        word = random.choice(NOISE_WORDS)
        parts = text.split(", ", 1)
        if len(parts) == 2:
            return f"{parts[0]}, {word}, {parts[1]}"
    return text


def fill(template):
    text = template.format(
        topic=random.choice(TOPICS),
        inst=random.choice(INSTITUTIONS),
        src=random.choice(SOURCES),
        journal=random.choice(JOURNALS),
        day=random.choice(DAYS),
        pct=random.choice(PERCENTS),
    )
    return text[0].upper() + text[1:]


def generate_corpus(n_per_class=450, hard_fraction=0.30, mimicry_fraction=0.20):
    """
    hard_fraction    : proportion of each class drawn from a less lexically-
                       distinctive template pool (ambiguous REAL / understated FAKE).
    mimicry_fraction : proportion of the FAKE class written in the *same* neutral,
                       sourced register as REAL news (sophisticated fabrication that
                       mimics legitimate reporting style rather than sensationalising).
                       This is what makes the task non-trivial: a purely stylistic
                       classifier cannot separate these from genuine REAL examples,
                       which is realistic and is discussed explicitly in the report.
    """
    seen = set()
    rows = []
    n_hard = int(n_per_class * hard_fraction)
    n_mimic = int(n_per_class * mimicry_fraction)
    n_easy = n_per_class - n_hard - n_mimic

    while sum(1 for r in rows if r[1] == 0 and r[2] == "easy") < n_easy:
        text = maybe_add_noise(fill(random.choice(REAL_TEMPLATES)))
        if text not in seen:
            seen.add(text); rows.append((text, 0, "easy"))
    while sum(1 for r in rows if r[1] == 0 and r[2] == "hard") < n_hard:
        text = maybe_add_noise(fill(random.choice(HARD_REAL_TEMPLATES)))
        if text not in seen:
            seen.add(text); rows.append((text, 0, "hard"))
    while sum(1 for r in rows if r[1] == 0 and r[2] == "mimic") < n_mimic:
        # extra genuine REAL examples (kept separate from the FAKE mimics below)
        text = maybe_add_noise(fill(random.choice(REAL_TEMPLATES)))
        if text not in seen:
            seen.add(text); rows.append((text, 0, "mimic"))

    while sum(1 for r in rows if r[1] == 1 and r[2] == "easy") < n_easy:
        text = maybe_add_noise(fill(random.choice(FAKE_TEMPLATES)))
        if text not in seen:
            seen.add(text); rows.append((text, 1, "easy"))
    while sum(1 for r in rows if r[1] == 1 and r[2] == "hard") < n_hard:
        text = maybe_add_noise(fill(random.choice(HARD_FAKE_TEMPLATES)))
        if text not in seen:
            seen.add(text); rows.append((text, 1, "hard"))
    while sum(1 for r in rows if r[1] == 1 and r[2] == "mimic") < n_mimic:
        # fabricated content dressed in neutral REAL-style phrasing (sophisticated fake news)
        text = maybe_add_noise(fill(random.choice(REAL_TEMPLATES)), p=0.0)
        if text not in seen:
            seen.add(text); rows.append((text, 1, "mimic"))

    random.shuffle(rows)
    return [(t, y) for t, y, _ in rows]


def load_real_dataset_instead(csv_path):
    """
    Drop-in replacement for generate_corpus() once internet access is available.
    Expects the GFG/Kaggle news.csv schema: columns 'title', 'text', 'label'
    (label in {REAL, FAKE}). Returns the same (text, label_int) row format.
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    label_map = {"REAL": 0, "FAKE": 1}
    rows = [(str(t), label_map[l]) for t, l in zip(df["title"], df["label"])]
    return rows


if __name__ == "__main__":
    data = generate_corpus()
    print(f"Generated {len(data)} examples ({sum(1 for _, y in data if y==0)} REAL / "
          f"{sum(1 for _, y in data if y==1)} FAKE)")
    for text, label in data[:6]:
        print(("FAKE" if label else "REAL"), "-", text)
