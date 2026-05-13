from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from html import unescape
from pathlib import Path
from typing import Any, Callable, TypedDict
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv, set_key

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
HISTORY_PATH = BASE_DIR / "sent_news_history.json"
FAILED_RUNS_PATH = BASE_DIR / "failed_runs.json"
CONFIG_PATH = BASE_DIR / "config.json"

load_dotenv(dotenv_path=ENV_PATH, override=True)

KST = ZoneInfo("Asia/Seoul")

REQUEST_TIMEOUT_NEWS = 10
REQUEST_TIMEOUT_CLAUDE = 30
REQUEST_TIMEOUT_KAKAO = 10

MAX_NEWS_SEARCH_FAILURES = 3
MAX_NEWS_SEARCH_FAILURE_RATIO = 0.4
MAX_NEWSLETTER_LENGTH = 1100
FINAL_SIGNATURE = "아비만 뉴스봇 자동 발송 메시지입니다."
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-20250514"
SECTION_LINE = "━━━━━━━━━━"

PLACEHOLDER_HINTS = [
    "입력", "여기에", "example", "sample", "replace",
    "발급", "api key", "client id", "secret", "token",
]

STOPWORDS = {
    "기사", "단독", "속보", "관련", "위해", "통해", "대한", "오늘", "오전", "오후",
    "발표", "공시", "시장", "업계", "뉴스", "로봇", "robot", "news", "the", "and"
}

DEFAULT_COMPANY_KEYWORDS = [
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

DEFAULT_CONFIG: dict[str, Any] = {
    "plastic_queries": [
        "플라스틱 산업 동향",
        "사출성형 업계",
        "플라스틱 원자재 가격",
    ],
    "competitor_queries": [
        "유일로보틱스",
        "나우로보틱스",
        "YUSHIN 취출기",
        "휴먼텍 로봇",
        "한양로보틱스",
        "SEPRO robot",
        "WITTMANN robot",
        "TOPSTAR robot",
    ],
    "company_keywords": DEFAULT_COMPANY_KEYWORDS,
    "category_limits": {
        "플라스틱_사출": 3,
        "경쟁사": 3,
    },
    "category_max_age_days": {
        "플라스틱_사출": 2,
        "경쟁사": 1,
    },
    "stock_exclude_keywords": [
        "주가", "급등", "급락", "상한가", "하한가", "매수", "매도",
        "투자주의", "투자경고", "시총", "시가총액", "증권", "리포트", "목표주가",
        "per", "pbr", "eps", "코스피", "코스닥", "공모가", "차트", "수급",
        "기관 순매수", "외국인 순매수", "주식", "종목", "테마주",
        "배당", "호재", "악재", "투자자",
    ],
}



def get_list_config(data: dict[str, Any], key: str, default: list[str]) -> list[str]:
    value = data.get(key, default)
    if not isinstance(value, list):
        safe_print(f"[경고] {key} 설정이 리스트가 아니어서 기본값을 사용합니다.")
        return list(default)
    result = [str(x).strip() for x in value if str(x).strip()]
    return result or list(default)


def get_dict_config(
    data: dict[str, Any],
    key: str,
    default: dict[str, int],
    *,
    min_value: int = 0,
) -> dict[str, int]:
    value = data.get(key, default)
    if not isinstance(value, dict):
        safe_print(f"[경고] {key} 설정이 객체가 아니어서 기본값을 사용합니다.")
        return dict(default)

    result: dict[str, int] = {}
    for k, v in value.items():
        try:
            result[str(k)] = max(int(v), min_value)
        except Exception:
            safe_print(f"[경고] {key}.{k} 값이 숫자가 아니어서 제외합니다.")
    return result or dict(default)


def get_str_config(data: dict[str, Any], key: str, default: str) -> str:
    env_name = key.upper()
    env_value = os.getenv(env_name)
    value = env_value if env_value is not None else data.get(key, default)
    if not isinstance(value, str):
        safe_print(f"[경고] {key} 설정이 문자열이 아니어서 기본값을 사용합니다.")
        return default
    value = value.strip()
    return value or default


class NewsArticle(TypedDict, total=False):
    title: str
    description: str
    link: str
    pubDate: str
    _group_size: int


@dataclass
class AppConfig:
    plastic_queries: list[str]
    competitor_queries: list[str]
    company_keywords: list[str]
    category_limits: dict[str, int]
    category_max_age_days: dict[str, int]
    stock_exclude_keywords: list[str]
    claude_model: str

    @classmethod
    def load(cls, path: Path) -> "AppConfig":
        data: dict[str, Any]
        if not path.exists():
            safe_print("[경고] config.json 이 없어 기본 설정 파일을 생성합니다.")
            path.write_text(json.dumps(DEFAULT_CONFIG, ensure_ascii=False, indent=2), encoding="utf-8")
            data = dict(DEFAULT_CONFIG)
        else:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    raise ValueError("config.json 최상위가 객체가 아닙니다.")
            except Exception as e:
                safe_print(f"[경고] config.json 로딩 실패, 기본 설정 사용: {e}")
                data = dict(DEFAULT_CONFIG)

        return cls(
            plastic_queries=get_list_config(data, "plastic_queries", DEFAULT_CONFIG["plastic_queries"]),
            competitor_queries=get_list_config(data, "competitor_queries", DEFAULT_CONFIG["competitor_queries"]),
            company_keywords=get_list_config(data, "company_keywords", DEFAULT_CONFIG["company_keywords"]),
            category_limits=get_dict_config(data, "category_limits", DEFAULT_CONFIG["category_limits"], min_value=1),
            category_max_age_days=get_dict_config(data, "category_max_age_days", DEFAULT_CONFIG["category_max_age_days"], min_value=0),
            stock_exclude_keywords=get_list_config(data, "stock_exclude_keywords", DEFAULT_CONFIG["stock_exclude_keywords"]),
            claude_model=get_str_config(data, "claude_model", DEFAULT_CLAUDE_MODEL),
        )


@dataclass
class AppState:
    naver_client_id: str
    naver_client_secret: str
    anthropic_api_key: str
    kakao_rest_api_key: str
    kakao_access_token: str
    kakao_refresh_token: str
    kakao_client_secret: str
    env_path: Path = ENV_PATH
    history_path: Path = HISTORY_PATH
    failed_runs_path: Path = FAILED_RUNS_PATH
    config: AppConfig = field(default_factory=lambda: AppConfig.load(CONFIG_PATH))


@dataclass
class SearchHealth:
    failures: int = 0
    messages: list[str] = field(default_factory=list)
    category_failures: dict[str, int] = field(default_factory=lambda: {"플라스틱_사출": 0, "경쟁사": 0})

    def add_failure(self, category: str, query: str, exc: Exception) -> None:
        self.failures += 1
        self.category_failures[category] = self.category_failures.get(category, 0) + 1
        self.messages.append(f"{category}:{query}: {exc}")


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


def request_with_retry(method: str, url: str, *, retries: int = 3, backoff: float = 1.0, **kwargs):
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.request(method, url, **kwargs)
            if resp.status_code in {429, 500, 502, 503, 504}:
                raise requests.HTTPError(f"retryable status: {resp.status_code}", response=resp)
            return resp
        except Exception as e:
            last_error = e
            if attempt >= retries:
                raise
            time.sleep(backoff * attempt)
    if last_error:
        raise last_error
    raise RuntimeError("request_with_retry failed without explicit error")


def load_state() -> AppState:
    return AppState(
        naver_client_id=(os.getenv("NAVER_CLIENT_ID") or "").strip(),
        naver_client_secret=(os.getenv("NAVER_CLIENT_SECRET") or "").strip(),
        anthropic_api_key=(os.getenv("ANTHROPIC_API_KEY") or "").strip(),
        kakao_rest_api_key=(os.getenv("KAKAO_REST_API_KEY") or "").strip(),
        kakao_access_token=(os.getenv("KAKAO_ACCESS_TOKEN") or "").strip(),
        kakao_refresh_token=(os.getenv("KAKAO_REFRESH_TOKEN") or "").strip(),
        kakao_client_secret=(os.getenv("KAKAO_CLIENT_SECRET") or "").strip(),
    )


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


def validate_kakao_env(state: AppState) -> None:
    has_access = bool(state.kakao_access_token)
    can_refresh = bool(state.kakao_refresh_token and state.kakao_rest_api_key)

    if not has_access and not can_refresh:
        raise ValueError(
            "카카오 발송을 위해 KAKAO_ACCESS_TOKEN 또는 "
            "KAKAO_REFRESH_TOKEN + KAKAO_REST_API_KEY 가 필요합니다."
        )


def validate_startup_env(state: AppState) -> None:
    validate_header_env("NAVER_CLIENT_ID", state.naver_client_id)
    validate_header_env("NAVER_CLIENT_SECRET", state.naver_client_secret)
    validate_header_env("ANTHROPIC_API_KEY", state.anthropic_api_key)
    validate_header_env("KAKAO_ACCESS_TOKEN", state.kakao_access_token, required=False)
    validate_header_env("KAKAO_REFRESH_TOKEN", state.kakao_refresh_token, required=False)
    validate_header_env("KAKAO_REST_API_KEY", state.kakao_rest_api_key, required=False)
    validate_kakao_env(state)


def validate_config(config: AppConfig) -> None:
    if not config.plastic_queries and not config.competitor_queries:
        raise ValueError("검색 쿼리가 모두 비어 있습니다. config.json을 확인하세요.")

    if not config.company_keywords:
        raise ValueError("company_keywords가 비어 있습니다. 경쟁사 필터가 정상 동작하지 않습니다.")


def strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text or "")
    return unescape(text).strip()


def normalize_text(text: str) -> str:
    text = strip_html(text).lower()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"[^0-9a-zA-Z가-힣\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_title(text: str) -> list[str]:
    normalized = normalize_text(text)
    return [tok for tok in normalized.split() if len(tok) > 1 and tok not in STOPWORDS]


def token_similarity(title_a: str, title_b: str) -> float:
    a = set(tokenize_title(title_a))
    b = set(tokenize_title(title_b))
    if not a or not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)


def extract_matched_company(text: str, config: AppConfig) -> str:
    text_lower = text.lower()
    for kw in config.company_keywords:
        if kw.lower() in text_lower:
            return kw
    return ""


def contains_stock_keyword(text: str, keywords: list[str]) -> bool:
    for keyword in keywords:
        if re.fullmatch(r"[a-zA-Z0-9]+", keyword):
            if re.search(rf"\b{re.escape(keyword)}\b", text, re.IGNORECASE):
                return True
        else:
            if keyword.lower() in text:
                return True
    return False


def is_stock_related_article(article: NewsArticle, config: AppConfig) -> bool:
    combined = normalize_text(f"{article.get('title', '')} {article.get('description', '')}")
    return contains_stock_keyword(combined, config.stock_exclude_keywords)


COMPETITOR_RELEVANCE_KEYWORDS = [
    "취출기", "사출", "성형", "injection", "molding",
    "take out", "takeout", "automation", "robot", "로봇", "자동화"
]


def is_valid_competitor_article(article: NewsArticle, config: AppConfig) -> bool:
    combined = normalize_text(f"{article.get('title', '')} {article.get('description', '')}")
    matched = extract_matched_company(combined, config)
    has_relevance = any(k.lower() in combined for k in COMPETITOR_RELEVANCE_KEYWORDS)
    return bool(matched) and has_relevance and not is_stock_related_article(article, config)


def build_fingerprint(article: NewsArticle, config: AppConfig) -> str:
    combined = f"{article.get('title', '')} {article.get('description', '')}"
    company = extract_matched_company(combined, config)
    tokens = tokenize_title(article.get("title", ""))
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


def get_article_age_days(article: NewsArticle) -> int | None:
    dt = parse_pubdate(article.get("pubDate", ""))
    if dt is None:
        return None
    return max((now_kst() - dt).days, 0)


def is_fresh_enough(article: NewsArticle, category: str, config: AppConfig) -> bool:
    age_days = get_article_age_days(article)
    max_days = config.category_max_age_days.get(category, 2)
    return age_days is not None and age_days <= max_days


def article_score(article: NewsArticle, category: str, config: AppConfig) -> int:
    title = normalize_text(article.get("title", ""))
    desc = normalize_text(article.get("description", ""))
    text = f"{title} {desc}"

    score = 0
    for kw in [
        "신제품", "출시", "수주", "투자", "증설", "실적", "계약", "전시", "자동화",
        "공장", "합작", "공급", "원료", "가격", "친환경", "성형", "설비", "공정",
    ]:
        if kw in text:
            score += 2

    if article.get("link"):
        score += 1
    if len(article.get("description", "")) >= 40:
        score += 1
    if extract_matched_company(article.get("title", ""), config):
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


def search_naver_news(query: str, state: AppState, display: int = 5) -> list[NewsArticle]:
    url = "https://openapi.naver.com/v1/search/news.json"
    headers = {
        "X-Naver-Client-Id": state.naver_client_id,
        "X-Naver-Client-Secret": state.naver_client_secret,
    }
    params = {"query": query, "display": display, "sort": "date"}

    resp = request_with_retry("GET", url, headers=headers, params=params, timeout=REQUEST_TIMEOUT_NEWS)
    resp.raise_for_status()

    items = resp.json().get("items", [])
    results: list[NewsArticle] = []
    for item in items:
        results.append({
            "title": strip_html(item.get("title", "")),
            "description": strip_html(item.get("description", "")),
            "link": item.get("originallink") or item.get("link") or "",
            "pubDate": item.get("pubDate", ""),
            "_group_size": 1,
        })
    return results


def group_similar_articles(articles: list[NewsArticle], category: str, config: AppConfig) -> list[NewsArticle]:
    groups: list[list[NewsArticle]] = []

    for article in articles:
        title = article.get("title", "")
        company = extract_matched_company(title, config)
        fp = build_fingerprint(article, config)
        matched = False

        for group in groups:
            rep = group[0]
            rep_title = rep.get("title", "")
            rep_company = extract_matched_company(rep_title, config)
            rep_fp = build_fingerprint(rep, config)
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

    selected: list[NewsArticle] = []
    for group in groups:
        ranked = sorted(
            group,
            key=lambda x: (
                article_score(x, category, config),
                len(x.get("description", "")),
                len(x.get("title", "")),
            ),
            reverse=True,
        )
        chosen = dict(ranked[0])
        chosen["_group_size"] = len(group)
        selected.append(chosen)

    selected.sort(
        key=lambda x: (
            article_score(x, category, config),
            x.get("_group_size", 1),
            len(x.get("description", "")),
        ),
        reverse=True,
    )
    return selected


def load_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
    except Exception as e:
        safe_print(f"[경고] JSON 이력 파일 로딩 실패: {path.name} / {e}")
        backup_path = path.with_suffix(path.suffix + ".broken")
        try:
            os.replace(path, backup_path)
        except Exception:
            pass
    return []


def atomic_write_json(path: Path, data: Any) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def load_history(state: AppState) -> list[dict[str, Any]]:
    return load_json_list(state.history_path)


def save_history(state: AppState, today_records: list[dict[str, Any]]) -> None:
    history = load_history(state)
    history.extend(today_records)
    cutoff = (now_kst() - timedelta(days=14)).strftime("%Y-%m-%d")
    trimmed = [x for x in history if x.get("date", "") >= cutoff]
    try:
        atomic_write_json(state.history_path, trimmed)
    except Exception as e:
        safe_print(f"[경고] 발송 이력 저장 실패: {e}")


def append_failed_run(state: AppState, reason: str, extra: dict[str, Any] | None = None) -> None:
    payload = {"datetime": now_kst().strftime("%Y-%m-%d %H:%M:%S"), "reason": reason}
    if extra:
        payload.update(extra)
    try:
        history = load_json_list(state.failed_runs_path)
        history.append(payload)
        history = history[-200:]
        atomic_write_json(state.failed_runs_path, history)
    except Exception as e:
        safe_print(f"[경고] 실패 이력 저장 실패: {e}")


def is_recent_duplicate(article: NewsArticle, recent_history: list[dict[str, Any]], config: AppConfig) -> bool:
    title = article.get("title", "")
    link = article.get("link", "")
    fp = build_fingerprint(article, config)
    company = extract_matched_company(title, config)
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


def get_recent_history(state: AppState, days: int = 3) -> list[dict[str, Any]]:
    history = load_history(state)
    cutoff = (now_kst() - timedelta(days=days)).strftime("%Y-%m-%d")
    return [x for x in history if x.get("date", "") >= cutoff]


def filter_recent_duplicates(articles: list[NewsArticle], recent_history: list[dict[str, Any]], config: AppConfig) -> list[NewsArticle]:
    kept: list[NewsArticle] = []

    for article in articles:
        if is_recent_duplicate(article, recent_history, config):
            continue

        duplicate_inside_kept = False
        for saved in kept:
            sim = token_similarity(article.get("title", ""), saved.get("title", ""))
            current_company = extract_matched_company(article.get("title", ""), config)
            saved_company = extract_matched_company(saved.get("title", ""), config)
            same_company = bool(current_company and current_company == saved_company)

            if build_fingerprint(article, config) == build_fingerprint(saved, config):
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


def _dedupe_articles(articles: list[NewsArticle]) -> list[NewsArticle]:
    deduped: list[NewsArticle] = []
    seen_keys: set[str] = set()
    for article in articles:
        key = f"{article.get('title', '')}|{article.get('link', '')}"
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(article)
    return deduped


def _collect_category(
    *,
    queries: list[str],
    category: str,
    limit: int,
    state: AppState,
    health: SearchHealth,
    recent_history: list[dict[str, Any]],
    article_filter: Callable[[NewsArticle, AppConfig], bool] | None = None,
) -> tuple[list[NewsArticle], dict[str, int]]:
    raw_articles: list[NewsArticle] = []
    for query in queries:
        try:
            raw_articles.extend(search_naver_news(query, state, display=3))
        except Exception as e:
            health.add_failure(category, query, e)
            safe_print(f"[경고] '{query}' 검색 실패: {e}")

    deduped_articles = _dedupe_articles(raw_articles)

    filtered_articles: list[NewsArticle] = []
    for article in deduped_articles:
        if is_stock_related_article(article, state.config):
            continue
        if article_filter is not None and not article_filter(article, state.config):
            continue
        filtered_articles.append(article)

    grouped_articles = group_similar_articles(filtered_articles, category, state.config)
    fresh_articles = [a for a in grouped_articles if is_fresh_enough(a, category, state.config)]
    final_articles = filter_recent_duplicates(fresh_articles, recent_history, state.config)
    final_articles.sort(
        key=lambda x: (
            article_score(x, category, state.config),
            x.get("_group_size", 1),
            len(x.get("description", "")),
        ),
        reverse=True,
    )

    stats: dict[str, int] = {
        "raw": len(raw_articles),
        "grouped": len(grouped_articles),
        "fresh": len(fresh_articles),
        "final": len(final_articles[:limit]),
    }
    if article_filter is not None:
        stats["company_filtered"] = len(filtered_articles)

    return final_articles[:limit], stats


def evaluate_search_health(state: AppState, health: SearchHealth) -> str | None:
    total_queries = len(state.config.plastic_queries) + len(state.config.competitor_queries)
    failure_ratio = health.failures / max(total_queries, 1)

    if health.failures >= MAX_NEWS_SEARCH_FAILURES and failure_ratio >= MAX_NEWS_SEARCH_FAILURE_RATIO:
        return f"뉴스 검색 실패가 기준치를 초과했습니다. ({health.failures}/{total_queries}, {failure_ratio:.0%})"

    plastic_failures = health.category_failures.get("플라스틱_사출", 0)
    plastic_total = len(state.config.plastic_queries)
    if plastic_total > 0 and plastic_failures >= plastic_total:
        return "플라스틱/사출 뉴스 검색 전체 실패"

    competitor_failures = health.category_failures.get("경쟁사", 0)
    competitor_total = len(state.config.competitor_queries)
    if competitor_total > 0 and competitor_failures >= competitor_total:
        return "취출기 경쟁사 뉴스 검색 전체 실패"

    return None


def collect_all_news(state: AppState) -> tuple[dict[str, list[NewsArticle]], dict[str, dict[str, int]], SearchHealth]:
    collected: dict[str, list[NewsArticle]] = {}
    stats: dict[str, dict[str, int]] = {}
    health = SearchHealth()
    recent_history = get_recent_history(state, days=3)

    collected["플라스틱_사출"], stats["플라스틱_사출"] = _collect_category(
        queries=state.config.plastic_queries,
        category="플라스틱_사출",
        limit=state.config.category_limits.get("플라스틱_사출", 3),
        state=state,
        health=health,
        recent_history=recent_history,
    )

    collected["경쟁사"], stats["경쟁사"] = _collect_category(
        queries=state.config.competitor_queries,
        category="경쟁사",
        limit=state.config.category_limits.get("경쟁사", 3),
        state=state,
        health=health,
        recent_history=recent_history,
        article_filter=is_valid_competitor_article,
    )
    return collected, stats, health


def build_competitor_no_news_block(text: str) -> str:
    if "취출기 경쟁사" in text and "신규 소식 없음" in text:
        return text
    block = f"\n\n{SECTION_LINE}\n취출기 경쟁사\n{SECTION_LINE}\n신규 소식 없음"
    if FINAL_SIGNATURE in text:
        return text.replace(FINAL_SIGNATURE, block + f"\n\n{FINAL_SIGNATURE}")
    return text + block


def emphasize_title_line(text: str) -> str:
    # Kakao text template does not reliably render true bold for plain text messages.
    # Use visual emphasis instead.
    lines = text.splitlines()
    if lines:
        first = lines[0].strip()
        if first:
            first = first.strip("【】")
            lines[0] = f"【{first}】"
    return "\n".join(lines)


def validate_newsletter_text(text: str, competitor_has_news: bool) -> str:
    cleaned = text.strip()
    cleaned = cleaned.replace("**", "")

    if "기사 원문" not in cleaned and "http" in cleaned:
        cleaned = re.sub(r"\n(https?://\S+)", r"\n기사 원문\n\1", cleaned)

    title_pattern = r"^【?\d{4}년 \d{2}월 \d{2}일 \| 오늘의 뉴스 브리핑 📰】?"
    if not re.match(title_pattern, cleaned):
        today = now_kst().strftime("%Y년 %m월 %d일")
        cleaned = f"{today} | 오늘의 뉴스 브리핑 📰\n\n" + cleaned

    if not competitor_has_news:
        cleaned = build_competitor_no_news_block(cleaned)

    if not cleaned.endswith(FINAL_SIGNATURE):
        cleaned = cleaned.rstrip() + f"\n\n{FINAL_SIGNATURE}"

    if len(cleaned) > MAX_NEWSLETTER_LENGTH:
        sections = cleaned.split("\n\n")
        rebuilt: list[str] = []
        for section in sections:
            candidate = "\n\n".join(rebuilt + [section]).strip()
            if not candidate.endswith(FINAL_SIGNATURE):
                candidate = candidate.rstrip() + f"\n\n{FINAL_SIGNATURE}"
            if len(candidate) > MAX_NEWSLETTER_LENGTH:
                break
            rebuilt.append(section)
        cleaned = "\n\n".join(rebuilt).strip()
        if not cleaned.endswith(FINAL_SIGNATURE):
            cleaned = cleaned.rstrip() + f"\n\n{FINAL_SIGNATURE}"

    if len(cleaned) > MAX_NEWSLETTER_LENGTH:
        raise ValueError("뉴스레터 길이가 1,100자를 초과했습니다.")
    if FINAL_SIGNATURE not in cleaned:
        raise ValueError("마지막 문구가 없습니다.")
    if "**" in cleaned:
        raise ValueError("마크다운 굵게가 제거되지 않았습니다.")
    if "기사 원문" not in cleaned:
        raise ValueError("기사 원문 형식이 없습니다.")
    if not re.search(r"기사 원문\s*\nhttps?://\S+", cleaned):
        raise ValueError("기사 원문 아래 실제 링크가 없습니다.")

    article_source_count = len(re.findall(r"기사 원문", cleaned))
    url_count = len(re.findall(r"기사 원문\s*\nhttps?://\S+", cleaned))
    if article_source_count != url_count:
        raise ValueError("일부 기사 원문 아래 링크가 누락되었습니다.")

    return emphasize_title_line(cleaned)


def extract_source_urls(news_data: dict[str, list[NewsArticle]]) -> set[str]:
    urls: set[str] = set()
    for articles in news_data.values():
        for article in articles:
            link = article.get("link", "").strip()
            if link:
                urls.add(link)
    return urls


def extract_source_article_by_url(news_data: dict[str, list[NewsArticle]], url: str) -> NewsArticle | None:
    for articles in news_data.values():
        for article in articles:
            if article.get("link", "").strip() == url:
                return article
    return None


def source_keyword_tokens(article: NewsArticle) -> set[str]:
    title_tokens = set(tokenize_title(article.get("title", "")))
    desc_tokens = set(tokenize_title(article.get("description", "")))
    return {tok for tok in (title_tokens | desc_tokens) if len(tok) >= 2}


def validate_newsletter_against_source(text: str, news_data: dict[str, list[NewsArticle]]) -> str:
    source_urls = extract_source_urls(news_data)
    used_urls = re.findall(r"기사 원문\s*\n(https?://\S+)", text)

    for url in used_urls:
        if url not in source_urls:
            raise ValueError(f"원본 뉴스에 없는 링크가 포함되었습니다: {url}")

        article = extract_source_article_by_url(news_data, url)
        if article is None:
            raise ValueError(f"링크에 대응하는 원본 기사를 찾지 못했습니다: {url}")

        idx = text.find(url)
        window_start = max(0, idx - 220)
        window_text = text[window_start:idx]
        summary_tokens = set(tokenize_title(window_text))
        source_tokens = source_keyword_tokens(article)

        if source_tokens and not (summary_tokens & source_tokens):
            raise ValueError(f"요약 문장과 원본 기사 핵심 키워드 연결이 약합니다: {url}")

    return text


def summarize_with_claude(news_data: dict[str, list[NewsArticle]], state: AppState) -> str:
    total_count = sum(len(v) for v in news_data.values())
    if total_count == 0:
        raise ValueError("요약할 뉴스가 없습니다.")

    label_map = {"플라스틱_사출": "플라스틱·사출 업계", "경쟁사": "취출기 경쟁사"}
    news_text_parts: list[str] = []
    category_counts: dict[str, int] = {}

    for category in ["플라스틱_사출", "경쟁사"]:
        articles = news_data.get(category, [])
        label = label_map[category]
        category_counts[category] = len(articles)
        if not articles:
            continue

        news_text_parts.append(f"\n[{label}]")
        for article in articles:
            group_note = ""
            if article.get("_group_size", 1) > 1:
                group_note = f" (유사 기사 {article.get('_group_size', 1)}건 묶음)"
            news_text_parts.append(
                f"- 제목: {article['title']}{group_note}\n"
                f"  내용: {article['description']}\n"
                f"  링크: {article['link']}"
            )

    news_text = "\n".join(news_text_parts)
    today = now_kst().strftime("%Y년 %m월 %d일")

    competitor_guide = (
        "취출기 경쟁사 뉴스가 한 건도 없으면 해당 섹션 대신 '신규 소식 없음'만 출력하세요."
        if category_counts.get("경쟁사", 0) == 0
        else "취출기 경쟁사 섹션은 실제 수집된 기사만 포함하세요."
    )

    prompt = f'''오늘({today}) 뉴스를 카카오톡 메시지용 깔끔한 뉴스레터 형식으로 정리해 주세요.

[원본 뉴스]
{news_text}

[카테고리별 수집 건수]
- 플라스틱·사출 업계: {category_counts.get("플라스틱_사출", 0)}건
- 취출기 경쟁사: {category_counts.get("경쟁사", 0)}건

[작성 규칙]
- 전체 톤은 정돈된 비즈니스 뉴스레터 스타일로 작성
- 과한 이모지, 과장 표현, 불필요한 감탄 표현은 사용 금지
- 이모지는 제목 줄의 신문 아이콘 1개만 허용
- 맨 위 제목은 반드시 다음 형식으로 작성:
  "{today} | 오늘의 뉴스 브리핑 📰"
- 제목 아래에는 아래 형식으로 인사말 2줄을 작성
  1줄: "안녕하세요."
  2줄: 오늘 전체 뉴스 흐름을 요약하는 한 줄 코멘트

- 섹션 순서는 반드시:
  1. 플라스틱·사출 업계
  2. 취출기 경쟁사

- 각 섹션은 실제 수집된 뉴스가 있을 때만 출력
- {competitor_guide}

- 각 섹션 제목은 반드시 아래처럼 구분선을 포함해 작성:
{SECTION_LINE}
플라스틱·사출 업계
{SECTION_LINE}

{SECTION_LINE}
취출기 경쟁사
{SECTION_LINE}

- 각 뉴스는 반드시 아래 형식으로 작성:
  1) 짧은 제목
  - 핵심 내용 한 줄 요약
  기사 원문
  링크주소

- "링크", "원문 링크", "URL" 같은 표현 대신 반드시 "기사 원문" 이라고만 작성
- "기사 원문" 다음 줄에 실제 링크 주소를 그대로 넣을 것
- 기사 제목을 길게 그대로 복붙하지 말고, 16~26자 내외의 짧은 제목으로 정리
- 핵심 내용은 1문장으로만 작성
- 유사 기사 묶음이라고 표시된 경우, 묶인 기사의 공통 핵심 이슈로 자연스럽게 요약
- 원본 뉴스에 없는 사실은 절대 추가하지 말 것
- 각 카테고리는 수집된 기사만 기준으로 최대 3건 출력
- 전체 길이는 1,100자 이내
- 마지막 문구는 반드시: "{FINAL_SIGNATURE}"
- 마크다운 굵게(**)는 사용하지 말 것
'''

    response = request_with_retry(
        "POST",
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": state.anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": state.config.claude_model,
            "max_tokens": 1500,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=REQUEST_TIMEOUT_CLAUDE,
    )
    response.raise_for_status()

    data: dict[str, Any] = response.json()
    content = data.get("content", [])
    if not content or not isinstance(content, list):
        raise ValueError(f"Claude 응답 형식이 예상과 다릅니다: {data}")

    first = content[0]
    text = first.get("text") if isinstance(first, dict) else None
    if not text:
        raise ValueError(f"Claude 응답 본문이 비어 있습니다: {data}")

    validated = validate_newsletter_text(text, competitor_has_news=category_counts.get("경쟁사", 0) > 0)
    return validate_newsletter_against_source(validated, news_data)


def build_no_news_message() -> str:
    today = now_kst().strftime("%Y년 %m월 %d일")
    return (
        f"【{today} | 오늘의 뉴스 브리핑 📰】\n\n"
        "안녕하세요.\n"
        "오늘은 발송 기준에 맞는 신규 뉴스가 없어 요약을 생략합니다.\n\n"
        f"{SECTION_LINE}\n"
        "취출기 경쟁사\n"
        f"{SECTION_LINE}\n"
        "신규 소식 없음\n\n"
        f"{FINAL_SIGNATURE}"
    )


def build_failure_message(reason: str) -> str:
    today = now_kst().strftime("%Y년 %m월 %d일")
    return (
        f"【{today} | 오늘의 뉴스 브리핑 📰】\n\n"
        "안녕하세요.\n"
        "오늘은 뉴스 수집 중 오류가 발생해 브리핑을 생성하지 못했습니다.\n\n"
        f"오류 요약: {reason}\n\n"
        f"{FINAL_SIGNATURE}"
    )


def _send_kakao_message_once(text: str, access_token: str) -> tuple[bool, int, str]:
    if not access_token:
        return False, 0, "KAKAO_ACCESS_TOKEN 이 없습니다."

    url = "https://kapi.kakao.com/v2/api/talk/memo/default/send"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    template = {
        "object_type": "text",
        "text": text,
        "link": {
            "web_url": "https://www.naver.com",
            "mobile_web_url": "https://www.naver.com",
        },
        "button_title": "뉴스 더 보기",
    }
    data = {"template_object": json.dumps(template, ensure_ascii=False)}

    try:
        resp = request_with_retry("POST", url, headers=headers, data=data, timeout=REQUEST_TIMEOUT_KAKAO)
        return resp.status_code == 200, resp.status_code, resp.text
    except requests.RequestException as e:
        return False, -1, f"카카오 요청 예외: {e}"


def send_kakao_message(text: str, state: AppState) -> bool:
    ok, status, body = _send_kakao_message_once(text, state.kakao_access_token)

    if ok:
        safe_print("카카오톡 전송 성공!")
        return True

    safe_print(f"카카오톡 전송 실패: {status} - {body}")

    if status == 401:
        safe_print("[경고] 401 발생. 토큰 재갱신 후 1회 재전송 시도")
        try:
            refreshed = refresh_kakao_token(state)
            if refreshed:
                ok2, status2, body2 = _send_kakao_message_once(text, state.kakao_access_token)
                if ok2:
                    safe_print("카카오톡 재전송 성공!")
                    return True
                safe_print(f"카카오톡 재전송 실패: {status2} - {body2}")
        except Exception as e:
            safe_print(f"[경고] 재갱신/재전송 실패: {e}")

    return False


def refresh_kakao_token(state: AppState) -> str | None:
    if not state.kakao_refresh_token or not state.kakao_rest_api_key:
        return None

    url = "https://kauth.kakao.com/oauth/token"
    data = {
        "grant_type": "refresh_token",
        "client_id": state.kakao_rest_api_key,
        "refresh_token": state.kakao_refresh_token,
    }
    if state.kakao_client_secret:
        data["client_secret"] = state.kakao_client_secret

    resp = request_with_retry("POST", url, data=data, timeout=REQUEST_TIMEOUT_KAKAO)
    resp.raise_for_status()

    result = resp.json()
    new_token = result.get("access_token")
    if not new_token:
        raise ValueError(f"카카오 토큰 갱신 응답에 access_token 이 없습니다: {result}")

    _update_env("KAKAO_ACCESS_TOKEN", new_token, state)
    state.kakao_access_token = new_token

    new_refresh_token = result.get("refresh_token")
    if new_refresh_token:
        _update_env("KAKAO_REFRESH_TOKEN", new_refresh_token, state)
        state.kakao_refresh_token = new_refresh_token

    safe_print("카카오 토큰 갱신 완료")
    return new_token


def _update_env(key: str, value: str, state: AppState) -> None:
    try:
        set_key(str(state.env_path), key, value)
    except Exception as e:
        safe_print(f"[경고] .env 업데이트 실패: {e}")


def build_today_history_records(news_data: dict[str, list[NewsArticle]], state: AppState) -> list[dict[str, Any]]:
    today = now_kst().strftime("%Y-%m-%d")
    records: list[dict[str, Any]] = []
    for category, articles in news_data.items():
        for article in articles:
            records.append({
                "date": today,
                "category": category,
                "title": article.get("title", ""),
                "link": article.get("link", ""),
                "fingerprint": build_fingerprint(article, state.config),
                "company": extract_matched_company(article.get("title", ""), state.config),
            })
    return records


def main() -> None:
    state = load_state()

    safe_print("\n" + "=" * 50)
    safe_print(f"  뉴스봇 실행: {now_kst().strftime('%Y-%m-%d %H:%M:%S')}")
    safe_print("=" * 50 + "\n")

    try:
        validate_startup_env(state)
        validate_config(state.config)
    except Exception as e:
        safe_print(f"초기 설정 검증 실패: {e}")
        append_failed_run(state, "startup_validation_failed", {"error": str(e)})
        return

    try:
        refresh_kakao_token(state)
    except Exception as e:
        safe_print(f"[경고] 카카오 토큰 갱신 실패: {e}")

    safe_print("뉴스 수집 중...")
    try:
        news_data, stats, health = collect_all_news(state)
    except Exception as e:
        safe_print(f"[오류] 뉴스 수집 실패: {e}")
        append_failed_run(state, "collect_news_failed", {"error": str(e)})
        send_kakao_message(build_failure_message(str(e)), state)
        return

    search_failure_reason = evaluate_search_health(state, health)
    if search_failure_reason:
        safe_print(f"[오류] {search_failure_reason}")
        append_failed_run(state, "search_health_failed", {"error": search_failure_reason, "messages": health.messages[:10]})
        send_kakao_message(build_failure_message(search_failure_reason), state)
        return

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
        if health.failures > 0:
            reason = "검색 실패 후 최종 뉴스 0건"
            safe_print("검색 실패가 있었고 최종 뉴스가 0건이라 수집 실패 메시지를 전송합니다.")
            append_failed_run(state, "zero_news_after_failures", {"error": reason, "messages": health.messages[:10]})
            send_kakao_message(build_failure_message(reason), state)
        else:
            safe_print("수집 기준에 맞는 신규 뉴스가 없어 안내 메시지를 전송합니다.")
            ok = send_kakao_message(build_no_news_message(), state)
            if not ok:
                append_failed_run(state, "send_no_news_failed", {})
        return

    safe_print("Claude AI로 요약 중...")
    try:
        message = summarize_with_claude(news_data, state)
    except Exception as e:
        safe_print(f"Claude 요약 실패: {e}")
        append_failed_run(state, "claude_failed", {"error": str(e)})
        send_kakao_message(build_failure_message(f"요약 실패: {e}"), state)
        return

    safe_print(f"요약 완료 ({len(message)}자)")
    safe_print("카카오톡 전송 중...")
    ok = send_kakao_message(message, state)
    if ok:
        save_history(state, build_today_history_records(news_data, state))
    else:
        append_failed_run(state, "send_failed", {"message_length": len(message)})
        safe_print("[오류] 카카오 발송 실패로 이력은 저장하지 않습니다.")


if __name__ == "__main__":
    main()
