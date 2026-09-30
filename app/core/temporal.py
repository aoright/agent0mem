import re
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Tuple


DATE_FORMATS = ["%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%Y/%m/%d"]

MONTH_NAMES = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december"
]


def parse_timestamp_seconds(ts: Optional[int]) -> Optional[float]:
    if not ts:
        return None
    t = float(ts)
    if t > 1e11:  # Milliseconds
        return t / 1000.0
    return t


def enrich_temporal_text(text: str, base_sec: Optional[float]) -> str:
    """Enrich relative temporal references with explicit ISO and canonical dates (English & Chinese)."""
    if not base_sec or not text:
        return text

    try:
        base_dt = datetime.fromtimestamp(base_sec, tz=timezone.utc)
    except Exception:
        return text

    tags: List[str] = []

    # 1. Base anchor date tag (Crucial: Ground every turn with exact ISO date, year, month, and human date)
    base_iso = base_dt.strftime("%Y-%m-%d")
    base_human = base_dt.strftime("%B %d, %Y")
    tags.append(f"[Conversation Date: {base_iso}] [Year: {base_dt.year}] [Month: {base_dt.strftime('%B')}] [Date: {base_human}] [{base_dt.year}年{base_dt.month}月]")

    # 2. English Relative Temporal Grounding
    # Days ago / earlier
    m_days_ago = re.findall(r"\b(\d+)\s*days?\s*(?:ago|earlier|prior|before)\b", text, re.IGNORECASE)
    for d in m_days_ago:
        dt = base_dt - timedelta(days=int(d))
        tags.append(f"[Event Date: {dt.strftime('%Y-%m-%d')}] [Event Date: {dt.strftime('%B %d, %Y')}]")

    # Days later / after
    m_days_later = re.findall(r"\b(\d+)\s*days?\s*(?:later|after|hence)\b", text, re.IGNORECASE)
    for d in m_days_later:
        dt = base_dt + timedelta(days=int(d))
        tags.append(f"[Event Date: {dt.strftime('%Y-%m-%d')}] [Event Date: {dt.strftime('%B %d, %Y')}]")

    # Weeks ago / later (LoCoMo Rule 7: Keep week-based expressions relative)
    m_weeks_ago = re.findall(r"\b(\d+)\s*weeks?\s*(?:ago|earlier|before)\b", text, re.IGNORECASE)
    for w in m_weeks_ago:
        tags.append(f"[Relative Time: {w} weeks ago]")

    m_weeks_later = re.findall(r"\b(\d+)\s*weeks?\s*(?:later|after)\b", text, re.IGNORECASE)
    for w in m_weeks_later:
        tags.append(f"[Relative Time: {w} weeks later]")

    # Months ago / later (Strict Month Granularity: Do NOT synthesize fake calendar days)
    m_months_ago = re.findall(r"\b(\d+)\s*months?\s*(?:ago|earlier|before)\b", text, re.IGNORECASE)
    for m in m_months_ago:
        dt = base_dt - timedelta(days=int(m) * 30)
        tags.append(f"[Event Month: {dt.strftime('%B %Y')}] [Event Month: {dt.strftime('%Y-%m')}]")

    m_months_later = re.findall(r"\b(\d+)\s*months?\s*(?:later|after)\b", text, re.IGNORECASE)
    for m in m_months_later:
        dt = base_dt + timedelta(days=int(m) * 30)
        tags.append(f"[Event Month: {dt.strftime('%B %Y')}] [Event Month: {dt.strftime('%Y-%m')}]")

    # Years ago / later (Strict Year Granularity: Do NOT synthesize fake calendar days)
    m_years_ago = re.findall(r"\b(\d+)\s*years?\s*(?:ago|earlier|before)\b", text, re.IGNORECASE)
    for y in m_years_ago:
        dt = base_dt - timedelta(days=int(y) * 365)
        tags.append(f"[Event Year: {dt.year}] [{dt.year}年]")

    # Yesterday / Tomorrow / Last month / Last year
    if re.search(r"\byesterday\b", text, re.IGNORECASE):
        dt = base_dt - timedelta(days=1)
        tags.append(f"[Event Date: {dt.strftime('%Y-%m-%d')}] [Event Date: {dt.strftime('%B %d, %Y')}]")

    if re.search(r"\btomorrow\b", text, re.IGNORECASE):
        dt = base_dt + timedelta(days=1)
        tags.append(f"[Event Date: {dt.strftime('%Y-%m-%d')}] [Event Date: {dt.strftime('%B %d, %Y')}]")

    if re.search(r"\blast\s+month\b", text, re.IGNORECASE):
        dt = base_dt - timedelta(days=30)
        tags.append(f"[Event Month: {dt.strftime('%B %Y')}] [Event Month: {dt.strftime('%Y-%m')}]")

    if re.search(r"\blast\s+year\b", text, re.IGNORECASE):
        dt = base_dt - timedelta(days=365)
        tags.append(f"[Event Year: {dt.year}] [{dt.year}年]")

    # 3. Chinese Relative Temporal Grounding
    if re.search(r"昨天", text):
        dt = base_dt - timedelta(days=1)
        tags.append(f"[Event Date: {dt.strftime('%Y-%m-%d')}] [{dt.year}年{dt.month}月{dt.day}日]")

    if re.search(r"前天", text):
        dt = base_dt - timedelta(days=2)
        tags.append(f"[Event Date: {dt.strftime('%Y-%m-%d')}] [{dt.year}年{dt.month}月{dt.day}日]")

    if re.search(r"去年", text):
        dt = base_dt - timedelta(days=365)
        tags.append(f"[Event Year: {dt.year}] [{dt.year}年]")

    if re.search(r"前年", text):
        dt = base_dt - timedelta(days=730)
        tags.append(f"[Event Year: {dt.year}] [{dt.year}年]")

    if re.search(r"上个?月", text):
        dt = base_dt - timedelta(days=30)
        tags.append(f"[Event Month: {dt.strftime('%Y-%m')}] [{dt.year}年{dt.month}月]")

    m_zh_days = re.findall(r"(\d+)\s*天前", text)
    for d in m_zh_days:
        dt = base_dt - timedelta(days=int(d))
        tags.append(f"[Event Date: {dt.strftime('%Y-%m-%d')}] [{dt.year}年{dt.month}月{dt.day}日]")

    m_zh_years = re.findall(r"(\d+)\s*年前", text)
    for y in m_zh_years:
        dt = base_dt - timedelta(days=int(y) * 365)
        tags.append(f"[Event Year: {dt.year}] [{dt.year}年]")

    # Extract explicit Chinese/ISO dates from text and index year/month
    m_iso = re.findall(r"\b(20\d{2})[-/年](\d{1,2})[-/月](\d{1,2})", text)
    for y, m, d in m_iso:
        tags.append(f"[{y}年{int(m)}月{int(d)}日] [Year: {y}]")

    m_year_only = re.findall(r"\b(20\d{2})年\b", text)
    for y in m_year_only:
        tags.append(f"[Year: {y}]")

    if tags:
        unique_tags = list(dict.fromkeys(tags))
        return f"{text} {' '.join(unique_tags)}"

    return text
