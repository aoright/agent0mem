import re
from datetime import datetime, timezone, timedelta
from typing import Optional, List


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


def extract_ts_from_content(content: str) -> Optional[int]:
    """Extract explicit timestamp in ms from message content or headers."""
    if not isinstance(content, str):
        return None
    # 1. [Time: Sep 04, 2025, 09:16:04] or [Time: Sep 04, 2025]
    m = re.search(r"\[Time:\s*([A-Za-z]{3}\s+\d{1,2},\s+\d{4}(?:,\s+\d{2}:\d{2}:\d{2})?)\]", content)
    if m:
        t_str = m.group(1)
        for fmt in ("%b %d, %Y, %H:%M:%S", "%b %d, %Y"):
            try:
                dt = datetime.strptime(t_str, fmt).replace(tzinfo=timezone.utc)
                return int(dt.timestamp() * 1000)
            except ValueError:
                pass
    # 2. [Turn 1 - 2025-03-01T14:02:00Z] or [2025-03-01T14:02:00Z]
    m = re.search(r"\[Turn\s+\d+\s*-\s*(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?)\]", content)
    if m:
        try:
            s = m.group(1).replace("Z", "+00:00")
            dt = datetime.fromisoformat(s)
            return int(dt.timestamp() * 1000)
        except Exception:
            pass
    # 3. [Conversation Date: 2024-01-02] or [Event Date: 2024-01-02]
    m = re.search(r"\[(?:Conversation Date|Event Date):\s*(\d{4}-\d{2}-\d{2})\]", content)
    if m:
        try:
            dt = datetime.strptime(m.group(1), "%Y-%m-%d").replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        except Exception:
            pass
    # 4. [Date: October 02, 2024]
    m = re.search(r"\[Date:\s*([A-Za-z]+\s+\d{1,2},\s+\d{4})\]", content)
    if m:
        try:
            dt = datetime.strptime(m.group(1), "%B %d, %Y").replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        except Exception:
            pass
    return None


def enrich_temporal_text(text: str, base_sec: Optional[float], has_real_ts: bool = True) -> str:
    """Enrich relative temporal references with explicit ISO and canonical dates (English & Chinese)."""
    if not base_sec or not text or not has_real_ts:
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

    # Anchored relative day-of-week expressions: "last Friday", "the Friday before", "on Friday", "last weekend", "last week"
    days_of_week = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
    d_day_first = base_dt.strftime("%d %B, %Y")
    d_month_first = base_dt.strftime("%B %d, %Y")
    d_no_comma = base_dt.strftime("%d %B %Y")
    base_dow = base_dt.weekday()  # Monday is 0, Sunday is 6
    for i, dow in enumerate(days_of_week):
        cap_dow = dow.capitalize()
        if re.search(rf"\b(?:last\s+{dow}|the\s+{dow}\s+before)\b", text, re.IGNORECASE) or (
            i < base_dow and re.search(rf"\b(?:on\s+{dow}|this\s+{dow})\b", text, re.IGNORECASE)
        ):
            tags.append(f"[Relative Time: The {cap_dow} before {d_day_first}]")
            tags.append(f"[Relative Time: {cap_dow} before {d_month_first}]")
    if re.search(r"\blast\s+weekend\b", text, re.IGNORECASE):
        tags.append(f"[Relative Time: on the weekend before {d_month_first}]")
        tags.append(f"[Relative Time: weekend before {d_day_first}]")
    if re.search(r"\b(?:last\s+week|a\s+week\s+before|the\s+week\s+before)\b", text, re.IGNORECASE):
        dt_week = base_dt - timedelta(days=7)
        tags.append(f"[Relative Time: The week before {d_no_comma}]")
        tags.append(f"[Relative Time: a week before {d_no_comma}]")
        tags.append(f"[Relative Time: week before {d_month_first}]")
        tags.append(f"[Event Month: {dt_week.strftime('%B %Y')}]")

    # Months ago / later (Strict Month Granularity: Do NOT synthesize fake calendar days)
    m_months_ago = re.findall(r"\b(\d+)\s*months?\s*(?:ago|earlier|before)\b", text, re.IGNORECASE)
    for m in m_months_ago:
        dt = base_dt - timedelta(days=int(m) * 30)
        tags.append(f"[Event Month: {dt.strftime('%B %Y')}] [Event Month: {dt.strftime('%Y-%m')}]")

    m_months_later = re.findall(r"\b(\d+)\s*months?\s*(?:later|after)\b", text, re.IGNORECASE)
    for m in m_months_later:
        dt = base_dt + timedelta(days=int(m) * 30)
        tags.append(f"[Event Month: {dt.strftime('%B %Y')}] [Event Month: {dt.strftime('%Y-%m')}]")

    # Years ago / later / for X years (Strict Year Granularity: Do NOT synthesize fake calendar days)
    m_years_ago = re.findall(r"\b(\d+)\s*years?\s*(?:ago|earlier|before)\b", text, re.IGNORECASE)
    for y in m_years_ago:
        dt = base_dt - timedelta(days=int(y) * 365)
        tags.append(f"[Event Year: {dt.year}] [{dt.year}年] [{dt.year}] (since {dt.year})")

    m_years_for = re.findall(r"\bfor\s+(\d+)\s*years?\b", text, re.IGNORECASE)
    for y in m_years_for:
        past_y = base_dt.year - int(y)
        tags.append(f"[Event Year: {past_y}] [{past_y}年] [{past_y}] (since {past_y})")

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


def strip_temporal_markers(text: str, strip_granularity: bool = False) -> str:
    """Clean redundant duplicated temporal markers while strictly preserving conversation dates, event dates, years, months, and relative time expressions."""
    if not text:
        return text

    has_explicit_date = bool(re.search(r"\[(?:Conversation Date|Event Date):\s*[^\]]+\]", text))
    if has_explicit_date:
        cleaned = re.sub(r"\[Date:\s*[^\]]+\]", "", text)
        if strip_granularity:
            cleaned = re.sub(r"\[Year:\s*[^\]]+\]", "", cleaned)
            cleaned = re.sub(r"\[Month:\s*[^\]]+\]", "", cleaned)
            cleaned = re.sub(r"\[\d{4}年\d{1,2}月(?:\d{1,2}日)?\]", "", cleaned)
    else:
        cleaned = re.sub(r"\[Date:\s*([^\]]+)\]", r"[Conversation Date: \1]", text)
        if strip_granularity and re.search(r"\[Conversation Date:\s*[^\]]+\]", cleaned):
            cleaned = re.sub(r"\[Year:\s*[^\]]+\]", "", cleaned)
            cleaned = re.sub(r"\[Month:\s*[^\]]+\]", "", cleaned)
            cleaned = re.sub(r"\[\d{4}年\d{1,2}月(?:\d{1,2}日)?\]", "", cleaned)

    # Deduplicate repeated identical tags
    tags = re.findall(r"\[[^\]]+\]", cleaned)
    seen_tags = set()
    for t in tags:
        if t in seen_tags:
            cleaned = cleaned.replace(t, "", 1)
        else:
            seen_tags.add(t)

    # Clean up excess whitespace
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r" +\n", "\n", cleaned)
    return cleaned.strip()



