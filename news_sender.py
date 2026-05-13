from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv, set_key

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


@dataclass
class AppConfig:
    base_dir: Path
    env_path: Path
    history_path: Path

    naver_client_id: str
    naver_client_secret: str
    anthropic_api_key: str
    kakao_rest_api_key: str
    kakao_access_token: str
    kakao_refresh_token: str
    kakao_client_secret: str

    request_timeout_news: int = 10
    request_timeout_claude: int = 30
    request_timeout_kakao: int = 10

    category_max_age_days: dict[str, int] | None = None

    @classmethod
    def from_env(cls) -> "AppConfig":
        base_dir = Path(__file__).resolve().parent
        env_path = base_dir / ".env"
        history_path = base_dir / "sent_news_history.json"

        load_dotenv(dotenv_path=env_path, override=True)

        return cls(
            base_dir=base_dir,
            env_path=env_path,
            history_path=history_path,
            naver_client_id=(os.getenv("NAVER_CLIENT_ID") or "").strip(),
            naver_client_secret=(os.getenv("NAVER_CLIENT_SECRET") or "").strip(),
            anthropic_api_key=(os.getenv("ANTHROPIC_API_KEY") or "").strip(),
            kakao_rest_api_key=(os.getenv("KAKAO_REST_API_KEY") or "").strip(),
            kakao_access_token=(os.getenv("KAKAO_ACCESS_TOKEN") or "").strip(),
            kakao_refresh_token=(os.getenv("KAKAO_REFRESH_TOKEN") or "").strip(),
            kakao_client_secret=(os.getenv("KAKAO_CLIENT_SECRET") or "").strip(),
            category_max_age_days={"플라스틱_사출": 2, "경쟁사": 1},
        )

    def save_env_value(self, key: str, value: str) -> None:
        set_key(str(self.env_path), key, value)


STOPWORDS = {
    "기사", "단독", "속보", "관련", "위해", "통해", "대한", "오늘", "오전", "오후",
    "발표", "공시", "시장", "업계", "뉴스", "로봇", "robot", "news", "the", "and"
}

COMPANY_KEYWORDS = [
    "유일로보틱스", "나우로보틱스", "YUSHIN", "유신",
    "휴먼텍", "한양로보틱스", "SEPRO", "WITTMANN", "TOPSTAR",
]

PLASTIC_QUERIES = ["플라스틱 산업 동향", "사출성형 업계", "플라스틱 원자재 가격"]
COMPETITOR_QUERIES = [
    "유일로보틱스", "나우로보틱스", "YUSHIN 취출기", "휴먼텍 로봇",
    "한양로보틱스", "SEPRO robot", "WITTMANN robot", "TOPSTAR robot",
]


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
    return [
        tok for tok in normalize_text(text).split()
        if len(tok) > 1 and tok not in STOPWORDS
    ]


def token_similarity(title_a: str, title_b: str) -> float:
    a = set(tokenize_title(title_a))
    b = set(tokenize_title(title_b))
    if not a or not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)


def extract_matched_company(text: str) -> str:
    lowered = text.lower()
    for kw in COMPANY_KEYWORDS:
        if kw.lower() in lowered:
            return kw
    return ""


def build_fingerprint(article: dict[str, str]) -> str:
    company = extract_matched_company(article.get("title", ""))
    tokens = tokenize_title(article.get("title", ""))[:6]
    return f"{company}|{' '.join(tokens)}".strip("|")


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
    return max((now_kst() - dt).days, 0)


def is_valid_competitor_article(article: dict[str, str]) -> bool:
    combined = f"{article.get('title', '')} {article.get('description', '')}"
    return bool(extract_matched_company(combined))


def article_score(article: dict[str, str], category: str) -> int:
    title = normalize_text(article.get("title", ""))
    desc = normalize_text(article.get("description", ""))
    text = f"{title} {desc}"

    score = 0
    for kw in [
        "신제품", "출시", "수주", "투자", "증설", "실적", "계약",
        "전시", "자동화", "공장", "합작", "공급", "원료", "가격",
        "상승", "하락", "친환경", "성형"
    ]:
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


class TokenManager:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def refresh(self) -> None:
        if not self.config.kakao_refresh_token:
            raise ValueError("KAKAO_REFRESH_TOKEN 값이 없습니다.")
        if not self.config.kakao_rest_api_key:
            raise ValueError("KAKAO_REST_API_KEY 값이 없습니다.")
        if not self.config.kakao_client_secret:
            raise ValueError("KAKAO_CLIENT_SECRET 값이 없습니다.")

        url = "https://kauth.kakao.com/oauth/token"
        data = {
            "grant_type": "refresh_token",
            "client_id": self.config.kakao_rest_api_key,
            "refresh_token": self.config.kakao_refresh_token,
            "client_secret": self.config.kakao_client_secret,
        }

        resp = requests.post(url, data=data, timeout=self.config.request_timeout_kakao)
        safe_print(f"[카카오 토큰 갱신 응답] status={resp.status_code}")
        safe_print(f"[카카오 토큰 갱신 응답 본문] {resp.text}")
        resp.raise_for_status()

        result = resp.json()
        access_token = result.get("access_token")
        refresh_token = result.get("refresh_token")

        if not access_token:
            raise ValueError(f"access_token 이 없습니다: {result}")

        self.config.kakao_access_token = access_token
        self.config.save_env_value("KAKAO_ACCESS_TOKEN", access_token)

        if refresh_token:
            self.config.kakao_refresh_token = refresh_token
            self.config.save_env_value("KAKAO_REFRESH_TOKEN", refresh_token)

        safe_print("카카오 토큰 갱신 완료")


class NewsCollector:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.failure_count = 0
        self.failure_messages: list[str] = []

    def search_naver_news(self, query: str, display: int = 5) -> list[dict[str, str]]:
        url = "https://openapi.naver.com/v1/search/news.json"
        headers = {
            "X-Naver-Client-Id": self.config.naver_client_id,
            "X-Naver-Client-Secret": self.config.naver_client_secret,
        }
        params = {"query": query, "display": display, "sort": "date"}

        resp = requests.get(url, headers=headers, params=params, timeout=self.config.request_timeout_news)
        resp.raise_for_status()

        items = resp.json().get("items", [])
        return [
            {
                "title": strip_html(item.get("title", "")),
                "description": strip_html(item.get("description", "")),
                "link": item.get("originallink") or item.get("link") or "",
                "pubDate": item.get("pubDate", ""),
            }
            for item in items
        ]

    def group_similar_articles(self, articles: list[dict[str, str]], category: str) -> list[dict[str, str]]:
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

    def _dedupe_exact(self, articles: list[dict[str, str]]) -> list[dict[str, str]]:
        seen: set[str] = set()
        deduped: list[dict[str, str]] = []
        for a in articles:
            key = f"{a.get('title', '')}|{a.get('link', '')}"
            if key not in seen:
                seen.add(key)
                deduped.append(a)
        return deduped

    def _load_history(self) -> list[dict[str, str]]:
        if not self.config.history_path.exists():
            return []
        try:
            return json.loads(self.config.history_path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _get_recent_history(self, days: int = 3) -> list[dict[str, str]]:
        cutoff = (now_kst() - timedelta(days=days)).strftime("%Y-%m-%d")
        return [x for x in self._load_history() if x.get("date", "") >= cutoff]

    def _is_recent_duplicate(self, article: dict[str, str], history: list[dict[str, str]]) -> bool:
        title = article.get("title", "")
        link = article.get("link", "")
        fp = build_fingerprint(article)
        company = extract_matched_company(title)
        pub_dt = parse_pubdate(article.get("pubDate", ""))

        for hist in history:
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

    def _filter_recent_duplicates(self, articles: list[dict[str, str]]) -> list[dict[str, str]]:
        history = self._get_recent_history(days=3)
        kept: list[dict[str, str]] = []

        for article in articles:
            if self._is_recent_duplicate(article, history):
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

    def _collect_category(
        self,
        *,
        queries: list[str],
        category: str,
        limit: int,
        predicate: callable | None = None,
    ) -> tuple[list[dict[str, str]], dict[str, int]]:
        raw: list[dict[str, str]] = []

        for q in queries:
            try:
                raw.extend(self.search_naver_news(q, display=3))
            except Exception as e:
                self.failure_count += 1
                self.failure_messages.append(f"{category}:{q}:{e}")
                safe_print(f"[경고] '{q}' 검색 실패: {e}")

        deduped = self._dedupe_exact(raw)

        filtered = deduped
        if predicate is not None:
            filtered = [a for a in deduped if predicate(a)]

        grouped = self.group_similar_articles(filtered, category)
        fresh = [
            a for a in grouped
            if get_article_age_days(a) is not None
            and get_article_age_days(a) <= self.config.category_max_age_days[category]
        ]
        final = self._filter_recent_duplicates(fresh)
        final.sort(
            key=lambda x: (
                article_score(x, category),
                int(x.get("_group_size", "1")),
                len(x.get("description", "")),
            ),
            reverse=True,
        )

        stats = {
            "raw": len(raw),
            "deduped": len(deduped),
            "filtered": len(filtered),
            "grouped": len(grouped),
            "fresh": len(fresh),
            "final": len(final[:limit]),
            "failures": self.failure_count,
        }
        return final[:limit], stats

    def collect_all_news(self) -> tuple[dict[str, list[dict[str, str]]], dict[str, dict[str, int]]]:
        collected: dict[str, list[dict[str, str]]] = {}
        stats: dict[str, dict[str, int]] = {}

        collected["플라스틱_사출"], stats["플라스틱_사출"] = self._collect_category(
            queries=PLASTIC_QUERIES,
            category="플라스틱_사출",
            limit=3,
        )

        collected["경쟁사"], stats["경쟁사"] = self._collect_category(
            queries=COMPETITOR_QUERIES,
            category="경쟁사",
            limit=3,
            predicate=is_valid_competitor_article,
        )

        if self.failure_count >= 3:
            raise RuntimeError(f"네이버 뉴스 API 실패 과다: {self.failure_messages}")

        return collected, stats

    def build_today_history_records(self, news_data: dict[str, list[dict[str, str]]]) -> list[dict[str, str]]:
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

    def save_history(self, records: list[dict[str, str]]) -> None:
        history = self._load_history()
        history.extend(records)
        cutoff = (now_kst() - timedelta(days=14)).strftime("%Y-%m-%d")
        trimmed = [x for x in history if x.get("date", "") >= cutoff]
        self.config.history_path.write_text(
            json.dumps(trimmed, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


class NewsSender:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def send_kakao_message(self, text: str) -> None:
        if not self.config.kakao_access_token:
            raise ValueError("KAKAO_ACCESS_TOKEN 이 없습니다.")

        url = "https://kapi.kakao.com/v2/api/talk/memo/default/send"
        headers = {
            "Authorization": f"Bearer {self.config.kakao_access_token}",
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

        resp = requests.post(url, headers=headers, data=data, timeout=self.config.request_timeout_kakao)
        safe_print(f"[카카오 응답] status={resp.status_code}")
        safe_print(f"[카카오 응답 본문] {resp.text}")
        resp.raise_for_status()


def main() -> None:
    config = AppConfig.from_env()
    token_manager = TokenManager(config)
    collector = NewsCollector(config)
    sender = NewsSender(config)

    safe_print("\n" + "=" * 50)
    safe_print(f"뉴스봇 실행: {now_kst().strftime('%Y-%m-%d %H:%M:%S')}")
    safe_print("=" * 50)

    token_manager.refresh()

    news_data, stats = collector.collect_all_news()
    total = sum(len(v) for v in news_data.values())

    safe_print(f"총 {total}건 수집 완료")
    safe_print(json.dumps(stats, ensure_ascii=False, indent=2))

    if total == 0:
        sender.send_kakao_message("오늘은 발송 기준에 맞는 뉴스가 없습니다.")
        return

    # 여기에 summarize_with_claude(news_data) 연결
    message = "테스트 메시지"
    sender.send_kakao_message(message)

    collector.save_history(collector.build_today_history_records(news_data))


if __name__ == "__main__":
    main()
