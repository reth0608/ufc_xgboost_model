# fix_method.py — uses the existing scraper class
import re
import sqlite3
from pathlib import Path
import time

from scrapers.ufcstats_scraper import UFCStatsScraper

DB_PATH = Path("data/ufc_raw.db")

def clean_text(value):
    return re.sub(r"\s+", " ", value or "").strip()

def get_method_and_round(scraper, fight_id):
    from bs4 import BeautifulSoup
    soup = scraper.get_soup(f"http://ufcstats.com/fight-details/{fight_id}")

    detail_items = {}
    for item in soup.select("i.b-fight-details__text-item"):
        label_tag = item.select_one("i.b-fight-details__label")
        if not label_tag:
            continue
        label = clean_text(label_tag.get_text(" ")).rstrip(":")
        label_tag.extract()
        detail_items[label.lower()] = clean_text(item.get_text(" "))

    for key, pattern in [("method", r"Method:\s*([A-Za-z/]+)"), ("round", r"Round:\s*(\d+)")]:
        if key not in detail_items:
            for p in soup.select("p.b-fight-details__text"):
                m = re.search(pattern, clean_text(p.get_text(" ")))
                if m:
                    detail_items[key] = m.group(1).strip()
                    break

    # print raw for debugging
    print(f"  detail_items: {detail_items}")

    method = detail_items.get("method")
    round_ended = int(detail_items["round"]) if detail_items.get("round", "").isdigit() else None
    return method, round_ended

def main():
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT fight_id FROM fights WHERE method IS NULL OR round_ended IS NULL"
    ).fetchall()
    print(f"Found {len(rows)} fights to fix")

    scraper = UFCStatsScraper()
    try:
        for i, (fight_id,) in enumerate(rows):
            try:
                method, round_ended = get_method_and_round(scraper, fight_id)
                con.execute(
                    "UPDATE fights SET method = ?, round_ended = ? WHERE fight_id = ?",
                    (method, round_ended, fight_id)
                )
                con.commit()
                print(f"[{i+1}/{len(rows)}] {fight_id} → method={method}, round={round_ended}")
            except Exception as e:
                print(f"[{i+1}/{len(rows)}] {fight_id} → FAILED: {e}")
    finally:
        scraper.close()
        con.close()

if __name__ == "__main__":
    main()