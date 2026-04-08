from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
HISTORY_PATH = BASE_DIR / "sent_news_history.json"

load_dotenv(dotenv_path=ENV_PATH, override=True)

KST = ZoneInfo("Asia/Seoul")


def now_kst() -> datetime:
    return datetime.now(KST)


def safe_print(*args: object, sep: str = " ", end: str = "\n") -> None:
    text = sep.join("" if a is None else str(a) for a in args) + end
    try:
        sys.stdout.write(text)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "cp949"
        safe_text = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
        sys.stdout.write(safe_text)
    except Exception:
        try:
            sys.__stdout__.write(text.encode("utf-8", errors="replace").decode("utf-8", errors="replace"))
        except Exception:
            pass


try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


NAVER_CLIENT_ID = (os.getenv("NAVER_CLIENT_ID") or "").strip()
NAVER_CLIENT_SECRET = (os.getenv("NAVER_CLIENT_SECRET") or "").strip()
KAKAO_REST_API_KEY = (os.getenv("KAKAO_REST_API_KEY") or "").strip()
KAKAO_ACCESS_TOKEN = (os.getenv("KAKAO_ACCESS_TOKEN") or "").strip()
KAKAO_REFRESH_TOKEN = (os.getenv("KAKAO_REFRESH_TOKEN") or "").strip()

REQUEST_TIMEOUT_NEWS = 10
REQUEST_TIMEOUT_KAKAO = 10

PLACEHOLDER_HINTS = [
    "입력", "여기에", "example", "sample", "replace",
    "발급", "api key", "client id", "secret", "token",
]

STOPWORDS = {
    "기사", "단독", "속보", "관련", "위해", "통해", "대한", "오늘", "오전", "오후",
    "발표", "공시", "시장", "업계", "뉴스", "로봇", "robot", "news", "the", "and"
}

COMPANY_KEYWORDS: list[str] = [
    "유일로보틱스",
    "나우로보틱스",
    "YUSHIN",
    "유신",
    "휴먼텍",
    "한양로보틱스",
    "SEPRO",
    "WITTMANN",
    "TOPSTAR",
]

COMPETITOR_QUERIES: list[str] = [
    "유일로보틱스",
    "나우로보틱스",
    "YUSHIN 취출기",
    "휴먼텍 로봇",
    "한양로보틱스",
    "SEPRO robot",
    "WITTMANN robot",
    "TOPSTAR robot",
]

CATEGORY_MAX_AGE_DAYS = {"플라스틱_사출": 2, "경쟁사": 1}

DEFAULT_HEADER_LINK = "https://www.naver.com"
DEFAULT_SECTION_IMAGES = {
    "플라스틱_사출": "https://developers.kakao.com/static/images/pc/txt_visual1.png",
    "경쟁사": "https://developers.kakao.com/static/images/pc/default.png",
}


def _looks_like_placeholder(value: str) -> bool:
    lowered = value.strip().lower()
    return any(hint in lowered for hint in PLACEHOLDER_HINTS)


def validate_header_env(name: str, value: str | None, *, required: bool = True) -> None:
    if not value:
        if required:
            raise ValueError(f"환경변수 {name} 값이 비어 있습니다.")
        return

    stripped = value.strip()
    if not stripped:
        raise ValueError(f"환경변수 {name} 값이 비어 있습니다.")

    if _looks_like_placeholder(stripped):
        raise ValueError(f"환경변수 {name} 값이 실제 키/토큰이 아닌 예시값처럼 보입니다: {stripped!r}")

    try:
        stripped.encode("latin-1")
    except UnicodeEncodeError as exc:
        raise ValueError(f"환경변수 {name} 값에 한글 또는 비ASCII 문자가 포함되어 있습니다: {stripped!r}") from exc


def validate_startup_env() -> None:
    validate_header_env("NAVER_CLIENT_ID", NAVER_CLIENT_ID)
    validate_header_env("NAVER_CLIENT_SECRET", NAVER_CLIENT_SECRET)
    validate_header_env("KAKAO_ACCESS_TOKEN", KAKAO_ACCESS_TOKEN, required=False)
    validate_header_env("KAKAO_REFRESH_TOKEN", KAKAO_REFRESH_TOKEN, required=False)
    validate_header_env("KAKAO_REST_API_KEY", KAKAO_REST_API_KEY, required=False)


def strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text or "")
    return text.replace("&quot;", '"').replace("&apos;", "'").replace("&amp;", "&").strip()


def normalize_text(text: str) -> str:
    text = strip_html(text).lower()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"[^0-9a-zA-Z가-힣\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_title(text: str) -> list[str]:
    normalized = normalize_text(text)
    tokens = []
    for tok in normalized.split():
        if len(tok) <= 1:
            continue
        if tok in STOPWORDS:
            continue
        tokens.append(tok)
    return tokens


def token_similarity(title_a: str, title_b: str) -> float:
    a = set(tokenize_title(title_a))
    b = set(tokenize_title(title_b))
    if not a or not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)


def extract_matched_company(text: str) -> str:
    text_lower = text.lower()
    for kw in COMPANY_KEYWORDS:
        if kw.lower() in text_lower:
            return kw
    return ""


def is_valid_competitor_article(article: dict[str, str]) -> bool:
    title = article.get("title", "")
    desc = article.get("description", "")
    combined = f"{title} {desc}"
    matched = extract_matched_company(combined)
    return bool(matched)


def build_fingerprint(article: dict[str, str]) -> str:
    title = article.get("title", "")
    company = extract_matched_company(title)
    tokens = tokenize_title(title)
    key_tokens = tokens[:6]
    return f"{company}|{' '.join(key_tokens)}".strip("|")


def parse_pubdate(pub_date: str) -> datetime | None:
    if not pub_date:
        return None
    try:
        dt = parsedate_to_datetime(pub_date)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(KST)
    except Exception:
        return None


def get_article_age_days(article: dict[str, str]) -> int | None:
    dt = parse_pubdate(article.get("pubDate", ""))
    if dt is None:
        return None
    delta = now_kst() - dt
    return max(delta.days, 0)


def is_fresh_enough(article: dict[str, str], category: str) -> bool:
    age_days = get_article_age_days(article)
    max_days = CATEGORY_MAX_AGE_DAYS.get(category, 2)
    if age_days is None:
        return False
    return age_days <= max_days


def article_score(article: dict[str, str], category: str) -> int:
    title = normalize_text(article.get("title", ""))
    desc = normalize_text(article.get("description", ""))
    text = f"{title} {desc}"

    score = 0
    priority_keywords = [
        "신제품", "출시", "수주", "투자", "증설", "실적", "계약", "전시", "자동화",
        "공장", "합작", "공급", "원료", "가격", "상승", "하락", "친환경", "성형"
    ]

    for kw in priority_keywords:
        if kw in text:
            score += 2

    if article.get("link"):
        score += 1
    if len(article.get("description", "")) >= 40:
        score += 1
    if extract_matched_company(article.get("title", "")):
        score += 2

    age_days = get_article_age_days(article)
    if age_days is not None:
        if age_days == 0:
            score += 5
        elif age_days == 1:
            score += 2
        else:
            score -= age_days * 2

    return score


def search_naver_news(query: str, display: int = 5) -> list[dict[str, str]]:
    url = "https://openapi.naver.com/v1/search/news.json"
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
    }
    params = {"query": query, "display": display, "sort": "date"}

    resp = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT_NEWS)
    resp.raise_for_status()

    items = resp.json().get("items", [])
    results: list[dict[str, str]] = []

    for item in items:
        results.append({
            "title": strip_html(item.get("title", "")),
            "description": strip_html(item.get("description", "")),
            "link": item.get("originallink") or item.get("link") or "",
            "pubDate": item.get("pubDate", ""),
        })

    return results


def group_similar_articles(articles: list[dict[str, str]], category: str) -> list[dict[str, str]]:
    groups: list[list[dict[str, str]]] = []

    for article in articles:
        title = article.get("title", "")
        company = extract_matched_company(title)
        fp = build_fingerprint(article)
        matched = False

        for group in groups:
            rep = group[0]
            rep_title = rep.get("title", "")
            rep_company = extract_matched_company(rep_title)
            rep_fp = build_fingerprint(rep)
            sim = token_similarity(title, rep_title)
            same_company = bool(company and rep_company and company == rep_company)

            if fp and rep_fp and fp == rep_fp:
                group.append(article)
                matched = True
                break

            if sim >= 0.60:
                group.append(article)
                matched = True
                break

            if same_company and sim >= 0.42:
                group.append(article)
                matched = True
                break

        if not matched:
            groups.append([article])

    selected: list[dict[str, str]] = []
    for group in groups:
        ranked = sorted(
            group,
            key=lambda x: (
                article_score(x, category),
                len(x.get("description", "")),
                len(x.get("title", "")),
            ),
            reverse=True,
        )
        chosen = dict(ranked[0])
        chosen["_group_size"] = str(len(group))
        selected.append(chosen)

    selected.sort(
        key=lambda x: (
            article_score(x, category),
            int(x.get("_group_size", "1")),
            len(x.get("description", "")),
        ),
        reverse=True,
    )
    return selected


def load_history() -> list[dict[str, str]]:
    if not HISTORY_PATH.exists():
        return []
    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        return []
    except Exception:
        return []


def save_history(today_records: list[dict[str, str]]) -> None:
    history = load_history()
    history.extend(today_records)
    cutoff = (now_kst() - timedelta(days=14)).strftime("%Y-%m-%d")
    trimmed = [x for x in history if x.get("date", "") >= cutoff]
    try:
        with open(HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(trimmed, f, ensure_ascii=False, indent=2)
    except Exception as e:
        safe_print(f"[경고] 발송 이력 저장 실패: {e}")


def is_recent_duplicate(article: dict[str, str], recent_history: list[dict[str, str]]) -> bool:
    title = article.get("title", "")
    link = article.get("link", "")
    fp = build_fingerprint(article)
    company = extract_matched_company(title)
    pub_dt = parse_pubdate(article.get("pubDate", ""))

    for hist in recent_history:
        if link and hist.get("link") == link:
            return True

        hist_title = hist.get("title", "")
        hist_fp = hist.get("fingerprint", "")
        hist_company = hist.get("company", "")
        hist_date = hist.get("date", "")
        sim = token_similarity(title, hist_title)

        if fp and hist_fp and fp == hist_fp:
            return True

        if sim >= 0.68:
            return True

        if company and hist_company and company == hist_company and sim >= 0.42:
            return True

        if company and hist_company and company == hist_company and hist_date and pub_dt is not None:
            try:
                hist_dt = datetime.strptime(hist_date, "%Y-%m-%d").replace(tzinfo=KST)
                if abs((pub_dt.date() - hist_dt.date()).days) <= 2 and sim >= 0.30:
                    return True
            except Exception:
                pass

    return False


def get_recent_history(days: int = 3) -> list[dict[str, str]]:
    history = load_history()
    cutoff = (now_kst() - timedelta(days=days)).strftime("%Y-%m-%d")
    return [x for x in history if x.get("date", "") >= cutoff]


def filter_recent_duplicates(articles: list[dict[str, str]]) -> list[dict[str, str]]:
    recent = get_recent_history(days=3)
    kept: list[dict[str, str]] = []

    for article in articles:
        if is_recent_duplicate(article, recent):
            continue

        duplicate_inside_kept = False
        for saved in kept:
            sim = token_similarity(article.get("title", ""), saved.get("title", ""))
            current_company = extract_matched_company(article.get("title", ""))
            saved_company = extract_matched_company(saved.get("title", ""))
            same_company = bool(current_company and current_company == saved_company)

            if build_fingerprint(article) == build_fingerprint(saved):
                duplicate_inside_kept = True
                break
            if sim >= 0.65:
                duplicate_inside_kept = True
                break
            if same_company and sim >= 0.40:
                duplicate_inside_kept = True
                break

        if not duplicate_inside_kept:
            kept.append(article)

    return kept


def collect_all_news() -> tuple[dict[str, list[dict[str, str]]], dict[str, dict[str, int]]]:
    plastic_queries = ["플라스틱 산업 동향", "사출성형 업계", "플라스틱 원자재 가격"]
    category_limits = {"플라스틱_사출": 3, "경쟁사": 3}
    collected: dict[str, list[dict[str, str]]] = {}
    stats: dict[str, dict[str, int]] = {}

    raw_plastic: list[dict[str, str]] = []
    for q in plastic_queries:
        try:
            raw_plastic.extend(search_naver_news(q, display=3))
        except Exception as e:
            safe_print(f"[경고] '{q}' 검색 실패: {e}")

    deduped_plastic: list[dict[str, str]] = []
    seen_keys: set[str] = set()
    for a in raw_plastic:
        key = f"{a.get('title', '')}|{a.get('link', '')}"
        if key not in seen_keys:
            seen_keys.add(key)
            deduped_plastic.append(a)

    grouped_plastic = group_similar_articles(deduped_plastic, "플라스틱_사출")
    fresh_plastic = [a for a in grouped_plastic if is_fresh_enough(a, "플라스틱_사출")]
    final_plastic = filter_recent_duplicates(fresh_plastic)
    final_plastic.sort(
        key=lambda x: (
            article_score(x, "플라스틱_사출"),
            int(x.get("_group_size", "1")),
            len(x.get("description", "")),
        ),
        reverse=True,
    )
    collected["플라스틱_사출"] = final_plastic[:category_limits["플라스틱_사출"]]
    stats["플라스틱_사출"] = {
        "raw": len(raw_plastic),
        "grouped": len(grouped_plastic),
        "fresh": len(fresh_plastic),
        "final": len(collected["플라스틱_사출"]),
    }

    raw_competitor: list[dict[str, str]] = []
    for q in COMPETITOR_QUERIES:
        try:
            raw_competitor.extend(search_naver_news(q, display=3))
        except Exception as e:
            safe_print(f"[경고] '{q}' 검색 실패: {e}")

    deduped_competitor: list[dict[str, str]] = []
    seen_keys2: set[str] = set()
    for a in raw_competitor:
        key = f"{a.get('title', '')}|{a.get('link', '')}"
        if key not in seen_keys2:
            seen_keys2.add(key)
            deduped_competitor.append(a)

    company_filtered = [a for a in deduped_competitor if is_valid_competitor_article(a)]
    blocked_count = len(deduped_competitor) - len(company_filtered)
    if blocked_count > 0:
        safe_print(f"   [경쟁사 필터] 관련없는 기사 {blocked_count}건 차단됨")

    grouped_competitor = group_similar_articles(company_filtered, "경쟁사")
    fresh_competitor = [a for a in grouped_competitor if is_fresh_enough(a, "경쟁사")]
    final_competitor = filter_recent_duplicates(fresh_competitor)
    final_competitor.sort(
        key=lambda x: (
            article_score(x, "경쟁사"),
            int(x.get("_group_size", "1")),
            len(x.get("description", "")),
        ),
        reverse=True,
    )
    collected["경쟁사"] = final_competitor[:category_limits["경쟁사"]]
    stats["경쟁사"] = {
        "raw": len(raw_competitor),
        "company_filtered": len(company_filtered),
        "grouped": len(grouped_competitor),
        "fresh": len(fresh_competitor),
        "final": len(collected["경쟁사"]),
    }

    return collected, stats


def trim_text(text: str, max_len: int) -> str:
    text = re.sub(r"\s+", " ", strip_html(text)).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def make_short_title(title: str, rank: int) -> str:
    title = strip_html(title)
    title = re.sub(r"^\[[^\]]+\]\s*", "", title).strip()
    title = re.sub(r"\s+", " ", title).strip()
    title = trim_text(title, 30)
    return f"{rank}. {title}"


def make_short_description(desc: str) -> str:
    desc = strip_html(desc)
    desc = re.sub(r"\s+", " ", desc).strip()
    if not desc:
        return "기사 내용을 눌러 확인해 주세요."
    return trim_text(desc, 55)


def build_link(url: str) -> dict[str, str]:
    fallback = DEFAULT_HEADER_LINK
    final_url = (url or "").strip() or fallback
    return {
        "web_url": final_url,
        "mobile_web_url": final_url,
    }


def article_to_content(article: dict[str, str], rank: int, category: str) -> dict[str, Any]:
    return {
        "title": make_short_title(article.get("title", ""), rank),
        "description": make_short_description(article.get("description", "")),
        "image_url": DEFAULT_SECTION_IMAGES.get(category, DEFAULT_SECTION_IMAGES["플라스틱_사출"]),
        "image_width": 640,
        "image_height": 640,
        "link": build_link(article.get("link", "")),
    }


def build_intro_text() -> str:
    today = now_kst().strftime("%Y년 %m월 %d일")
    return (
        f"📅 {today} | 뉴스 브리핑\n\n"
        "안녕하세요!\n"
        "오늘 꼭 챙겨봐야 할 핵심 소식들입니다."
    )


def build_no_news_message() -> str:
    today = now_kst().strftime("%Y년 %m월 %d일")
    return (
        f"📅 {today} | 뉴스 브리핑\n\n"
        "안녕하세요!\n"
        "오늘은 발송 기준에 맞는 신규 뉴스가 없어 요약을 생략합니다.\n\n"
        "아비만 뉴스봇 자동 발송 메시지입니다."
    )


def build_section_header(category: str) -> str:
    if category == "플라스틱_사출":
        return "📍 플라스틱·사출 업계"
    if category == "경쟁사":
        return "📍 취출기 경쟁사"
    return "📍 뉴스"


def build_list_template(category: str, articles: list[dict[str, str]]) -> dict[str, Any]:
    header_title = build_section_header(category)
    contents = [article_to_content(article, idx + 1, category) for idx, article in enumerate(articles[:3])]
    first_link = articles[0].get("link", DEFAULT_HEADER_LINK) if articles else DEFAULT_HEADER_LINK

    return {
        "object_type": "list",
        "header_title": header_title,
        "header_link": build_link(first_link),
        "contents": contents,
        "button_title": "기사 보기",
        "buttons": [
            {
                "title": "전체 기사 보기",
                "link": build_link(first_link),
            }
        ],
    }


def build_feed_template(category: str, article: dict[str, str]) -> dict[str, Any]:
    header_title = build_section_header(category)
    title = make_short_title(article.get("title", ""), 1)
    desc = make_short_description(article.get("description", ""))

    return {
        "object_type": "feed",
        "content": {
            "title": f"{header_title}\n\n{title}",
            "description": desc,
            "image_url": DEFAULT_SECTION_IMAGES.get(category, DEFAULT_SECTION_IMAGES["플라스틱_사출"]),
            "image_width": 640,
            "image_height": 640,
            "link": build_link(article.get("link", "")),
        },
        "button_title": "기사 보기",
        "buttons": [
            {
                "title": "기사 보기",
                "link": build_link(article.get("link", "")),
            }
        ],
    }


def send_kakao_default_template(template_object: dict[str, Any]) -> bool:
    if not KAKAO_ACCESS_TOKEN:
        safe_print("KAKAO_ACCESS_TOKEN 이 없어 카카오톡 전송을 할 수 없습니다.")
        return False

    url = "https://kapi.kakao.com/v2/api/talk/memo/default/send"
    headers = {
        "Authorization": f"Bearer {KAKAO_ACCESS_TOKEN}",
        "Content-Type": "application/x-www-form-urlencoded;charset=utf-8",
    }
    data = {"template_object": json.dumps(template_object, ensure_ascii=False)}

    resp = requests.post(url, headers=headers, data=data, timeout=REQUEST_TIMEOUT_KAKAO)
    if resp.status_code == 200:
        return True

    safe_print(f"카카오톡 전송 실패: {resp.status_code} - {resp.text}")
    return False


def send_intro_message() -> bool:
    template = {
        "object_type": "text",
        "text": build_intro_text(),
        "link": build_link(DEFAULT_HEADER_LINK),
        "button_title": "뉴스 보기",
    }
    return send_kakao_default_template(template)


def send_section_message(category: str, articles: list[dict[str, str]]) -> bool:
    if not articles:
        return True

    if len(articles) >= 2:
        template = build_list_template(category, articles[:3])
    else:
        template = build_feed_template(category, articles[0])

    return send_kakao_default_template(template)


def refresh_kakao_token() -> str | None:
    refresh_token = KAKAO_REFRESH_TOKEN
    client_id = KAKAO_REST_API_KEY

    if not refresh_token or not client_id:
        return None

    url = "https://kauth.kakao.com/oauth/token"
    data = {
        "grant_type": "refresh_token",
        "client_id": client_id,
        "refresh_token": refresh_token,
    }

    resp = requests.post(url, data=data, timeout=REQUEST_TIMEOUT_KAKAO)
    resp.raise_for_status()

    result = resp.json()
    new_token = result.get("access_token")
    if not new_token:
        raise ValueError(f"카카오 토큰 갱신 응답에 access_token 이 없습니다: {result}")

    _update_env("KAKAO_ACCESS_TOKEN", new_token)
    if result.get("refresh_token"):
        _update_env("KAKAO_REFRESH_TOKEN", result["refresh_token"])

    safe_print("카카오 토큰 갱신 완료")
    return new_token


def _update_env(key: str, value: str) -> None:
    try:
        lines: list[str] = []
        if ENV_PATH.exists():
            with open(ENV_PATH, "r", encoding="utf-8") as f:
                lines = f.readlines()

        found = False
        with open(ENV_PATH, "w", encoding="utf-8") as f:
            for line in lines:
                if line.startswith(f"{key}="):
                    f.write(f"{key}={value}\n")
                    found = True
                else:
                    f.write(line)
            if not found:
                f.write(f"{key}={value}\n")
    except Exception as e:
        safe_print(f"[경고] .env 업데이트 실패: {e}")


def build_today_history_records(news_data: dict[str, list[dict[str, str]]]) -> list[dict[str, str]]:
    today = now_kst().strftime("%Y-%m-%d")
    records: list[dict[str, str]] = []

    for category, articles in news_data.items():
        for article in articles:
            records.append({
                "date": today,
                "category": category,
                "title": article.get("title", ""),
                "link": article.get("link", ""),
                "fingerprint": build_fingerprint(article),
                "company": extract_matched_company(article.get("title", "")),
            })

    return records


def main() -> None:
    global KAKAO_ACCESS_TOKEN, KAKAO_REFRESH_TOKEN

    safe_print("\n" + "=" * 50)
    safe_print(f"  뉴스봇 실행: {now_kst().strftime('%Y-%m-%d %H:%M:%S')}")
    safe_print("=" * 50 + "\n")

    try:
        validate_startup_env()
    except Exception as e:
        safe_print(f"환경변수 검증 실패: {e}")
        return

    try:
        new_token = refresh_kakao_token()
        if new_token:
            os.environ["KAKAO_ACCESS_TOKEN"] = new_token
            KAKAO_ACCESS_TOKEN = new_token
            refreshed_refresh = (os.getenv("KAKAO_REFRESH_TOKEN") or "").strip()
            if refreshed_refresh:
                KAKAO_REFRESH_TOKEN = refreshed_refresh
    except Exception as e:
        safe_print(f"[경고] 카카오 토큰 갱신 실패: {e}")

    safe_print("뉴스 수집 중...")
    news_data, stats = collect_all_news()

    plastic_count = len(news_data.get("플라스틱_사출", []))
    competitor_count = len(news_data.get("경쟁사", []))
    total = plastic_count + competitor_count

    safe_print(f"   -> 총 {total}건 수집 완료")
    safe_print(f"      플라스틱/사출 {plastic_count}건, 경쟁사 {competitor_count}건")

    p = stats.get("플라스틱_사출", {})
    c = stats.get("경쟁사", {})
    safe_print(
        f"      [플라스틱] raw {p.get('raw', 0)} -> grouped {p.get('grouped', 0)} -> "
        f"fresh {p.get('fresh', 0)} -> final {p.get('final', 0)}"
    )
    safe_print(
        f"      [경쟁사] raw {c.get('raw', 0)} -> 업체필터 {c.get('company_filtered', 0)} -> "
        f"grouped {c.get('grouped', 0)} -> fresh {c.get('fresh', 0)} -> final {c.get('final', 0)}"
    )

    if total == 0:
        safe_print("수집 기준에 맞는 신규 뉴스가 없어 안내 메시지를 전송합니다.")
        ok = send_kakao_default_template({
            "object_type": "text",
            "text": build_no_news_message(),
            "link": build_link(DEFAULT_HEADER_LINK),
            "button_title": "확인",
        })
        if ok:
            safe_print("카카오톡 전송 성공!")
        return

    safe_print("카카오톡 전송 중...")

    success = True

    if not send_intro_message():
        success = False
        safe_print("[오류] 인트로 메시지 전송 실패")

    if plastic_count > 0:
        if not send_section_message("플라스틱_사출", news_data["플라스틱_사출"]):
            success = False
            safe_print("[오류] 플라스틱·사출 업계 메시지 전송 실패")

    if competitor_count > 0:
        if not send_section_message("경쟁사", news_data["경쟁사"]):
            success = False
            safe_print("[오류] 취출기 경쟁사 메시지 전송 실패")

    if success:
        save_history(build_today_history_records(news_data))
        safe_print("카카오톡 전송 성공!")


if __name__ == "__main__":
    main()
