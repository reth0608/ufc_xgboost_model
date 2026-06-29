from __future__ import annotations

import argparse
import hashlib
import logging
import re
import sqlite3
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


BASE_URL = "http://ufcstats.com"
EVENTS_URL = f"{BASE_URL}/statistics/events/completed?page=all"
DB_PATH = Path("data/ufc_raw.db")
REQUEST_SLEEP_SECONDS = 1.5
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36 UFCFightPredictor/1.0"
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Event:
    event_id: str
    name: str
    date: str | None
    location: str | None
    url: str


@dataclass(frozen=True)
class Fighter:
    fighter_id: str
    name: str
    height_cm: float | None
    weight_kg: float | None
    reach_cm: float | None
    stance: str | None
    dob: str | None
    url: str


@dataclass(frozen=True)
class Fight:
    fight_id: str
    event_id: str
    fighter_a_id: str
    fighter_b_id: str
    winner_id: str | None
    method: str | None
    round_ended: int | None
    time_ended: str | None
    time_format: str | None
    referee: str | None
    weight_class: str | None
    is_title_fight: int
    is_perf_bonus: int
    a_kd: int
    a_sig_str: str
    a_sig_str_pct: str
    a_total_str: str
    a_td: str
    a_td_pct: str
    a_sub_att: int
    a_rev: int
    a_ctrl: str
    a_head: str
    a_body: str
    a_leg: str
    a_distance: str
    a_clinch: str
    a_ground: str
    b_kd: int
    b_sig_str: str
    b_sig_str_pct: str
    b_total_str: str
    b_td: str
    b_td_pct: str
    b_sub_att: int
    b_rev: int
    b_ctrl: str
    b_head: str
    b_body: str
    b_leg: str
    b_distance: str
    b_clinch: str
    b_ground: str


@dataclass(frozen=True)
class Round:
    fight_id: str
    fighter_id: str
    round_num: int
    kd: int
    sig_str_landed: int
    sig_str_att: int
    sig_str_pct: float | None
    total_str_landed: int
    total_str_att: int
    td_landed: int
    td_att: int
    td_pct: float | None
    sub_att: int
    rev: int
    ctrl_sec: int
    head_landed: int
    head_att: int
    body_landed: int
    body_att: int
    leg_landed: int
    leg_att: int
    distance_landed: int
    distance_att: int
    clinch_landed: int
    clinch_att: int
    ground_landed: int
    ground_att: int


def parse_landed_attempted(value: str | None) -> tuple[int, int]:
    if not value or value.strip() == "---":
        return 0, 0
    match = re.search(r"(\d+)\s+of\s+(\d+)", value)
    return (int(match.group(1)), int(match.group(2))) if match else (0, 0)


def parse_pct(value: str | None) -> float | None:
    if not value or value.strip() == "---":
        return None
    match = re.search(r"(\d+(?:\.\d+)?)%", value)
    return float(match.group(1)) / 100.0 if match else None


def parse_time_to_seconds(value: str | None) -> int:
    if not value or value.strip() == "---":
        return 0
    parts = value.strip().split(":")
    if len(parts) != 2:
        return 0
    return int(parts[0]) * 60 + int(parts[1])


def parse_height_cm(value: str | None) -> float | None:
    if not value or value.strip() == "---":
        return None
    match = re.search(r"(\d+)'\s*(\d+)", value)
    if not match:
        return None
    inches = int(match.group(1)) * 12 + int(match.group(2))
    return round(inches * 2.54, 1)


def parse_weight_kg(value: str | None) -> float | None:
    if not value or value.strip() == "---":
        return None
    match = re.search(r"(\d+(?:\.\d+)?)", value)
    return round(float(match.group(1)) * 0.45359237, 1) if match else None


def parse_reach_cm(value: str | None) -> float | None:
    if not value or value.strip() == "---":
        return None
    match = re.search(r"(\d+(?:\.\d+)?)", value)
    return round(float(match.group(1)) * 2.54, 1) if match else None


def clean_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def id_from_url(url: str) -> str:
    return Path(urlparse(url).path).name


class UFCStatsScraper:
    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = db_path
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.conn = sqlite3.connect(self.db_path, timeout=30)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.create_schema()

    def close(self) -> None:
        self.conn.close()

    def create_schema(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                name TEXT,
                date TEXT,
                location TEXT,
                url TEXT
            );
            CREATE TABLE IF NOT EXISTS fighters (
                fighter_id TEXT PRIMARY KEY,
                name TEXT,
                height_cm REAL,
                weight_kg REAL,
                reach_cm REAL,
                stance TEXT,
                dob TEXT,
                url TEXT
            );
            CREATE TABLE IF NOT EXISTS fights (
                fight_id TEXT PRIMARY KEY,
                event_id TEXT,
                fighter_a_id TEXT,
                fighter_b_id TEXT,
                winner_id TEXT,
                method TEXT,
                round_ended INTEGER,
                time_ended TEXT,
                time_format TEXT,
                referee TEXT,
                weight_class TEXT,
                is_title_fight INTEGER,
                is_perf_bonus INTEGER,
                a_kd INTEGER, a_sig_str TEXT, a_sig_str_pct TEXT, a_total_str TEXT,
                a_td TEXT, a_td_pct TEXT, a_sub_att INTEGER, a_rev INTEGER, a_ctrl TEXT,
                a_head TEXT, a_body TEXT, a_leg TEXT, a_distance TEXT, a_clinch TEXT, a_ground TEXT,
                b_kd INTEGER, b_sig_str TEXT, b_sig_str_pct TEXT, b_total_str TEXT,
                b_td TEXT, b_td_pct TEXT, b_sub_att INTEGER, b_rev INTEGER, b_ctrl TEXT,
                b_head TEXT, b_body TEXT, b_leg TEXT, b_distance TEXT, b_clinch TEXT, b_ground TEXT,
                FOREIGN KEY(event_id) REFERENCES events(event_id)
            );
            CREATE TABLE IF NOT EXISTS rounds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fight_id TEXT,
                fighter_id TEXT,
                round_num INTEGER,
                kd INTEGER,
                sig_str_landed INTEGER,
                sig_str_att INTEGER,
                sig_str_pct REAL,
                total_str_landed INTEGER,
                total_str_att INTEGER,
                td_landed INTEGER,
                td_att INTEGER,
                td_pct REAL,
                sub_att INTEGER,
                rev INTEGER,
                ctrl_sec INTEGER,
                head_landed INTEGER,
                head_att INTEGER,
                body_landed INTEGER,
                body_att INTEGER,
                leg_landed INTEGER,
                leg_att INTEGER,
                distance_landed INTEGER,
                distance_att INTEGER,
                clinch_landed INTEGER,
                clinch_att INTEGER,
                ground_landed INTEGER,
                ground_att INTEGER,
                FOREIGN KEY(fight_id) REFERENCES fights(fight_id)
            );
            CREATE TABLE IF NOT EXISTS scrape_log (
                fight_id TEXT PRIMARY KEY,
                scraped_at_utc TEXT NOT NULL
            );
            """
        )
        self.conn.commit()

    def get_soup(self, url: str) -> BeautifulSoup:
        last_error: Exception | None = None
        for attempt in range(3):
            if attempt or getattr(self, "_requested_once", False):
                time.sleep(REQUEST_SLEEP_SECONDS * (2**attempt))
            self._requested_once = True
            try:
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                response = self.resolve_browser_check(response, url)
                return BeautifulSoup(response.text, "lxml")
            except requests.RequestException as exc:
                last_error = exc
                logger.warning("Request failed (%s/3) for %s: %s", attempt + 1, url, exc)
        raise RuntimeError(f"Failed to fetch {url}") from last_error

    def resolve_browser_check(self, response: requests.Response, original_url: str) -> requests.Response:
        if "Checking your browser" not in response.text or "/__c" not in response.text:
            return response
        nonce_match = re.search(r'var nonce="([^"]+)"', response.text)
        difficulty_match = re.search(r"target=new Array\((\d+)\+1\)\.join\('0'\)", response.text)
        if not nonce_match or not difficulty_match:
            raise RuntimeError("UFCStats returned a browser-check page the scraper could not solve.")

        nonce = nonce_match.group(1)
        difficulty = int(difficulty_match.group(1))
        target = "0" * difficulty
        proof = 0
        while hashlib.sha256(f"{nonce}:{proof}".encode("utf-8")).hexdigest()[:difficulty] != target:
            proof += 1

        challenge_url = urljoin(original_url, "/__c")
        logger.info("Solving UFCStats browser check.")
        challenge = self.session.post(challenge_url, data={"nonce": nonce, "n": str(proof)}, timeout=30)
        challenge.raise_for_status()
        retried = self.session.get(original_url, timeout=30)
        retried.raise_for_status()
        if "Checking your browser" in retried.text:
            raise RuntimeError("UFCStats browser check did not clear after proof submission.")
        return retried

    def already_scraped_ids(self) -> set[str]:
        rows = self.conn.execute("SELECT fight_id FROM scrape_log").fetchall()
        return {row[0] for row in rows}

    def scrape_events(self, max_events: int | None) -> list[Event]:
        soup = self.get_soup(EVENTS_URL)
        events: list[Event] = []
        for row in soup.select("tr.b-statistics__table-row"):
            link = row.select_one('a[href*="/event-details/"]')
            date_tag = row.select_one(".b-statistics__date")
            cells = [clean_text(cell.get_text(" ")) for cell in row.select("td")]
            if not link or len(cells) < 2:
                continue
            url = link["href"]
            events.append(
                Event(
                    event_id=id_from_url(url),
                    name=clean_text(link.get_text(" ")),
                    date=clean_text(date_tag.get_text(" ")) if date_tag else None,
                    location=cells[1] or None,
                    url=url,
                )
            )
        if not events:
            title = clean_text(soup.title.get_text(" ")) if soup.title else "unknown"
            raise RuntimeError(
                f"No UFCStats events were parsed from {EVENTS_URL}. "
                f"Page title was {title!r}; the site may be blocking scraper traffic or the HTML changed."
            )
        return events[:max_events] if max_events else events

    def parse_event_fights(self, event: Event) -> list[str]:
        soup = self.get_soup(event.url)
        fight_ids: list[str] = []
        for row in soup.select("tr.b-fight-details__table-row[data-link]"):
            fight_ids.append(id_from_url(row.get("data-link", "")))
        return [fight_id for fight_id in fight_ids if fight_id]

    def parse_fighter(self, fighter_url: str, fallback_name: str | None = None) -> Fighter:
        soup = self.get_soup(fighter_url)
        name = clean_text(soup.select_one("span.b-content__title-highlight").get_text(" ")) if soup.select_one("span.b-content__title-highlight") else fallback_name or id_from_url(fighter_url)
        info: dict[str, str] = {}
        for li in soup.select("li.b-list__box-list-item"):
            raw = clean_text(li.get_text(" "))
            if ":" in raw:
                key, value = raw.split(":", 1)
                info[key.strip().lower()] = value.strip()
        return Fighter(
            fighter_id=id_from_url(fighter_url),
            name=name,
            height_cm=parse_height_cm(info.get("height")),
            weight_kg=parse_weight_kg(info.get("weight")),
            reach_cm=parse_reach_cm(info.get("reach")),
            stance=info.get("stance") or None,
            dob=info.get("dob") or None,
            url=fighter_url,
        )

    def two_values(self, row, cell_index: int) -> tuple[str, str]:
        cells = row.select("td")
        if cell_index >= len(cells):
            return "---", "---"
        values = [clean_text(p.get_text(" ")) for p in cells[cell_index].select("p")]
        if len(values) >= 2:
            return values[0], values[1]
        text = clean_text(cells[cell_index].get_text(" "))
        return text, text

    def parse_fight(self, event_id: str, fight_id: str) -> tuple[Fight, list[Fighter], list[Round]]:
        soup = self.get_soup(f"{BASE_URL}/fight-details/{fight_id}")
        person_blocks = soup.select("div.b-fight-details__person")
        fighter_urls = [block.select_one("a.b-link")["href"] for block in person_blocks if block.select_one("a.b-link")]
        fighter_names = [clean_text(block.select_one("a.b-link").get_text(" ")) for block in person_blocks if block.select_one("a.b-link")]
        if len(fighter_urls) != 2:
            raise ValueError(f"Could not parse two fighters for {fight_id}")

        fighters = [self.parse_fighter(url, name) for url, name in zip(fighter_urls, fighter_names)]
        statuses = [clean_text(block.select_one("i.b-fight-details__person-status").get_text(" ")) if block.select_one("i.b-fight-details__person-status") else "" for block in person_blocks]
        winner_id = None
        if statuses and statuses[0] == "W":
            winner_id = fighters[0].fighter_id
        elif len(statuses) > 1 and statuses[1] == "W":
            winner_id = fighters[1].fighter_id

        detail_items: dict[str, str] = {}
        for item in soup.select("i.b-fight-details__text-item"):
            label_tag = item.select_one("i.b-fight-details__label")
            if not label_tag:
                continue
            label = clean_text(label_tag.get_text(" ")).rstrip(":")
            label_tag.extract()
            detail_items[label.lower()] = clean_text(item.get_text(" "))

        title_text = clean_text(soup.select_one("i.b-fight-details__fight-title").get_text(" ")) if soup.select_one("i.b-fight-details__fight-title") else None
        bonus = int(bool(soup.select_one(".b-fight-details__fight-title .b-fight-details__fight-title-link")))
        # Fallback: try broader text search if method key missing
        if "method" not in detail_items:
            for p in soup.select("p.b-fight-details__text"):
                text = clean_text(p.get_text(" "))
                m = re.search(r"Method:\s*([A-Za-z/]+)", text)
                if m:
                    detail_items["method"] = m.group(1).strip()
                    break
        if "round" not in detail_items:
            for p in soup.select("p.b-fight-details__text"):
                text = clean_text(p.get_text(" "))
                m = re.search(r"Round:\s*(\d+)", text)
                if m:
                    detail_items["round"] = m.group(1).strip()
                    break
        method = detail_items.get("method")
        round_ended = int(detail_items["round"]) if detail_items.get("round", "").isdigit() else None
        tables = soup.select("table.b-fight-details__table")
        if len(tables) < 2:
            raise ValueError(f"Expected fight tables for {fight_id}")
        totals_row = tables[0].select_one("tbody tr")
        sig_row = tables[1].select_one("tbody tr")
        if totals_row is None or sig_row is None:
            raise ValueError(f"Missing total rows for {fight_id}")

        a_kd, b_kd = self.two_values(totals_row, 1)
        a_sig_str, b_sig_str = self.two_values(totals_row, 2)
        a_sig_pct, b_sig_pct = self.two_values(totals_row, 3)
        a_total_str, b_total_str = self.two_values(totals_row, 4)
        a_td, b_td = self.two_values(totals_row, 5)
        a_td_pct, b_td_pct = self.two_values(totals_row, 6)
        a_sub, b_sub = self.two_values(totals_row, 7)
        a_rev, b_rev = self.two_values(totals_row, 8)
        a_ctrl, b_ctrl = self.two_values(totals_row, 9)
        a_head, b_head = self.two_values(sig_row, 3)
        a_body, b_body = self.two_values(sig_row, 4)
        a_leg, b_leg = self.two_values(sig_row, 5)
        a_distance, b_distance = self.two_values(sig_row, 6)
        a_clinch, b_clinch = self.two_values(sig_row, 7)
        a_ground, b_ground = self.two_values(sig_row, 8)

        rounds = self.parse_rounds(fight_id, [fighter.fighter_id for fighter in fighters], tables)
        fight = Fight(
            fight_id=fight_id,
            event_id=event_id,
            fighter_a_id=fighters[0].fighter_id,
            fighter_b_id=fighters[1].fighter_id,
            winner_id=winner_id,
            method=method,
            round_ended=round_ended,
            time_ended=detail_items.get("time"),
            time_format=detail_items.get("time format"),
            referee=detail_items.get("referee"),
            weight_class=title_text,
            is_title_fight=int("title" in (title_text or "").lower()),
            is_perf_bonus=bonus,
            a_kd=int(a_kd or 0),
            a_sig_str=a_sig_str,
            a_sig_str_pct=a_sig_pct,
            a_total_str=a_total_str,
            a_td=a_td,
            a_td_pct=a_td_pct,
            a_sub_att=int(a_sub or 0),
            a_rev=int(a_rev or 0),
            a_ctrl=a_ctrl,
            a_head=a_head,
            a_body=a_body,
            a_leg=a_leg,
            a_distance=a_distance,
            a_clinch=a_clinch,
            a_ground=a_ground,
            b_kd=int(b_kd or 0),
            b_sig_str=b_sig_str,
            b_sig_str_pct=b_sig_pct,
            b_total_str=b_total_str,
            b_td=b_td,
            b_td_pct=b_td_pct,
            b_sub_att=int(b_sub or 0),
            b_rev=int(b_rev or 0),
            b_ctrl=b_ctrl,
            b_head=b_head,
            b_body=b_body,
            b_leg=b_leg,
            b_distance=b_distance,
            b_clinch=b_clinch,
            b_ground=b_ground,
        )
        return fight, fighters, rounds

    def parse_rounds(self, fight_id: str, fighter_ids: list[str], tables: list) -> list[Round]:
        if len(tables) >= 4:
            totals_rows = tables[2].select("tbody tr")
            sig_rows = tables[3].select("tbody tr")
        elif len(tables) >= 2:
            # Current UFCStats pages put whole-fight totals in row 0 and per-round
            # rows after that within the same two tables.
            all_total_rows = tables[0].select("tbody tr")
            all_sig_rows = tables[1].select("tbody tr")
            totals_rows = all_total_rows[1:] or all_total_rows[:1]
            sig_rows = all_sig_rows[1:] or all_sig_rows[:1]
        else:
            return []
        rounds: list[Round] = []
        for row_index, total_row in enumerate(totals_rows):
            sig_row = sig_rows[row_index] if row_index < len(sig_rows) else None
            round_num_text, _ = self.two_values(total_row, 0)
            round_num = int(round_num_text) if round_num_text.isdigit() else row_index + 1
            for fighter_idx, fighter_id in enumerate(fighter_ids):
                pick = lambda pair: pair[fighter_idx]
                kd = pick(self.two_values(total_row, 1))
                sig = pick(self.two_values(total_row, 2))
                sig_pct = pick(self.two_values(total_row, 3))
                total = pick(self.two_values(total_row, 4))
                td = pick(self.two_values(total_row, 5))
                td_pct = pick(self.two_values(total_row, 6))
                sub = pick(self.two_values(total_row, 7))
                rev = pick(self.two_values(total_row, 8))
                ctrl = pick(self.two_values(total_row, 9))
                head = pick(self.two_values(sig_row, 3)) if sig_row else "---"
                body = pick(self.two_values(sig_row, 4)) if sig_row else "---"
                leg = pick(self.two_values(sig_row, 5)) if sig_row else "---"
                distance = pick(self.two_values(sig_row, 6)) if sig_row else "---"
                clinch = pick(self.two_values(sig_row, 7)) if sig_row else "---"
                ground = pick(self.two_values(sig_row, 8)) if sig_row else "---"
                sig_l, sig_a = parse_landed_attempted(sig)
                total_l, total_a = parse_landed_attempted(total)
                td_l, td_a = parse_landed_attempted(td)
                head_l, head_a = parse_landed_attempted(head)
                body_l, body_a = parse_landed_attempted(body)
                leg_l, leg_a = parse_landed_attempted(leg)
                distance_l, distance_a = parse_landed_attempted(distance)
                clinch_l, clinch_a = parse_landed_attempted(clinch)
                ground_l, ground_a = parse_landed_attempted(ground)
                rounds.append(
                    Round(
                        fight_id=fight_id,
                        fighter_id=fighter_id,
                        round_num=round_num,
                        kd=int(kd or 0),
                        sig_str_landed=sig_l,
                        sig_str_att=sig_a,
                        sig_str_pct=parse_pct(sig_pct),
                        total_str_landed=total_l,
                        total_str_att=total_a,
                        td_landed=td_l,
                        td_att=td_a,
                        td_pct=parse_pct(td_pct),
                        sub_att=int(sub or 0),
                        rev=int(rev or 0),
                        ctrl_sec=parse_time_to_seconds(ctrl),
                        head_landed=head_l,
                        head_att=head_a,
                        body_landed=body_l,
                        body_att=body_a,
                        leg_landed=leg_l,
                        leg_att=leg_a,
                        distance_landed=distance_l,
                        distance_att=distance_a,
                        clinch_landed=clinch_l,
                        clinch_att=clinch_a,
                        ground_landed=ground_l,
                        ground_att=ground_a,
                    )
                )
        return rounds

    def insert_event(self, event: Event) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO events (event_id, name, date, location, url) VALUES (?, ?, ?, ?, ?)",
            tuple(asdict(event).values()),
        )

    def insert_fighters(self, fighters: Iterable[Fighter]) -> None:
        for fighter in fighters:
            self.conn.execute(
                """
                INSERT OR IGNORE INTO fighters
                (fighter_id, name, height_cm, weight_kg, reach_cm, stance, dob, url)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                tuple(asdict(fighter).values()),
            )

    def insert_fight(self, fight: Fight, rounds: list[Round]) -> None:
        fight_data = asdict(fight)
        columns = ", ".join(fight_data)
        placeholders = ", ".join(["?"] * len(fight_data))
        self.conn.execute(f"INSERT OR REPLACE INTO fights ({columns}) VALUES ({placeholders})", tuple(fight_data.values()))
        self.conn.execute("DELETE FROM rounds WHERE fight_id = ?", (fight.fight_id,))
        for round_row in rounds:
            data = asdict(round_row)
            columns = ", ".join(data)
            placeholders = ", ".join(["?"] * len(data))
            self.conn.execute(f"INSERT INTO rounds ({columns}) VALUES ({placeholders})", tuple(data.values()))
        self.conn.execute(
            "INSERT OR REPLACE INTO scrape_log (fight_id, scraped_at_utc) VALUES (?, ?)",
            (fight.fight_id, datetime.now(timezone.utc).isoformat()),
        )

    def run(self, max_events: int | None = None, skip_existing: bool = True) -> None:
        skipped = self.already_scraped_ids() if skip_existing else set()
        for event in self.scrape_events(max_events):
            self.insert_event(event)
            fight_ids = self.parse_event_fights(event)
            logger.info("%s: %s fights", event.name, len(fight_ids))
            for fight_id in fight_ids:
                if fight_id in skipped:
                    continue
                try:
                    fight, fighters, rounds = self.parse_fight(event.event_id, fight_id)
                    self.insert_fighters(fighters)
                    self.insert_fight(fight, rounds)
                    self.conn.commit()
                    logger.info("%s vs %s -> winner=%s, rounds scraped=%s", fighters[0].name, fighters[1].name, fight.winner_id, len(rounds))
                except Exception:
                    self.conn.rollback()
                    logger.exception("Failed scraping fight %s", fight_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape UFCStats into SQLite.")
    parser.add_argument("--max-events", type=int, default=None, help="Only scrape N most recent events.")
    parser.add_argument("--no-skip", action="store_true", help="Re-scrape fights already in the database.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    scraper = UFCStatsScraper()
    try:
        scraper.run(max_events=args.max_events, skip_existing=not args.no_skip)
    finally:
        scraper.close()


if __name__ == "__main__":
    main()
