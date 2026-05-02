"""Fetch Wikipedia thumbnail images for every item in the word bank.

Idempotent — skips items whose image is already on disk. Downloads go to
frontend-src/public/images/ so Vite copies them into the built frontend at
frontend/images/ (served at /static/images/<slug>.<ext>).

Usage:
    python tools/fetch_images.py
"""

from __future__ import annotations

import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORD_BANK_PATH = ROOT / "data" / "word_bank.json"
IMAGES_DIR = ROOT / "frontend-src" / "public" / "images"
# Wikimedia requires a UA with contact info: https://meta.wikimedia.org/wiki/User-Agent_policy
USER_AGENT = (
    "TheSandbox/1.0 (https://github.com/the-sandbox/the-sandbox; "
    "social-deduction game; contact admin@thesandbox.local)"
)
REQUEST_DELAY_S = 1.0  # pause between requests to stay under Wikimedia rate limits
MAX_RETRIES = 3

# Topic-specific keywords to disambiguate ambiguous labels during search.
TOPIC_HINTS: dict[str, str] = {
    "Movies": "film",
    "TV Shows": "television series",
    "Video Games": "video game",
    "Superheroes": "comics",
    "Disney Animated Films": "Disney film",
    "Desserts": "dessert",
    "Fruits": "fruit",
    "Spices & Herbs": "spice",
    "Cocktails": "cocktail",
    "Dog Breeds": "dog breed",
    "Famous Landmarks": "landmark",
    "Musicians & Bands": "musician",
    "Sports": "sport",
    "Musical Instruments": "instrument",
    "World Capitals": "city",
}

# Explicit Wikipedia page titles for labels where search or direct lookup
# would pull the wrong page (e.g., "Queen" → monarchy, not the band).
EXPLICIT_TITLES: dict[tuple[str, str], str] = {
    ("Movies", "Titanic"): "Titanic (1997 film)",
    ("Movies", "Joker"): "Joker (2019 film)",
    ("Movies", "Jurassic Park"): "Jurassic Park (film)",
    ("TV Shows", "The Office"): "The Office (American TV series)",
    ("TV Shows", "Friends"): "Friends",
    ("Video Games", "Pokemon"): "Pokémon",
    ("Video Games", "Grand Theft Auto"): "Grand Theft Auto",
    ("Video Games", "Super Mario Bros"): "Super Mario Bros.",
    ("Video Games", "Call of Duty"): "Call of Duty",
    ("Superheroes", "Thor"): "Thor (Marvel Comics)",
    ("Superheroes", "The Hulk"): "Hulk (comics)",
    ("Superheroes", "Black Panther"): "Black Panther (character)",
    ("Superheroes", "Iron Man"): "Iron Man",
    ("Superheroes", "Captain America"): "Captain America",
    ("Musicians & Bands", "Queen"): "Queen (band)",
    ("Musicians & Bands", "Drake"): "Drake (musician)",
    ("Sports", "Soccer"): "Association football",
    ("Sports", "Formula 1"): "Formula One",
    ("Musical Instruments", "Drums"): "Drum kit",
    ("Famous Landmarks", "Pyramids of Giza"): "Giza pyramid complex",
    ("Famous Landmarks", "Big Ben"): "Big Ben",
    ("Cocktails", "Old Fashioned"): "Old fashioned (cocktail)",
    ("Cocktails", "Manhattan"): "Manhattan (cocktail)",
    ("Cocktails", "Martini"): "Martini (cocktail)",
    ("Cocktails", "Cosmopolitan"): "Cosmopolitan (cocktail)",
    ("Cocktails", "Daiquiri"): "Daiquiri",
    ("Disney Animated Films", "Aladdin"): "Aladdin (1992 Disney film)",
    ("Disney Animated Films", "Beauty and the Beast"): "Beauty and the Beast (1991 film)",
    ("Disney Animated Films", "The Little Mermaid"): "The Little Mermaid (1989 film)",
    ("Disney Animated Films", "Mulan"): "Mulan (1998 film)",
    ("Disney Animated Films", "Moana"): "Moana (2016 film)",
    ("TV Shows", "Friends"): "Friends",
    ("TV Shows", "The Last of Us"): "The Last of Us (TV series)",
    ("Video Games", "Fortnite"): "Fortnite",
    ("Musical Instruments", "Banjo"): "Banjo",
}


# --------------------------- helpers --------------------------- #

def slugify(text: str) -> str:
    """ASCII slug for filenames, e.g. 'Piña Colada' -> 'pina-colada'."""
    norm = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9]+", "-", norm.lower()).strip("-")
    return s or "untitled"


_last_request_time = 0.0


def http_get(url: str, *, as_bytes: bool = False) -> str | bytes | None:
    """GET with UA header + retry on 429. Returns None on persistent failure."""
    global _last_request_time
    # Throttle: enforce minimum gap between any two network requests
    elapsed = time.time() - _last_request_time
    if elapsed < REQUEST_DELAY_S:
        time.sleep(REQUEST_DELAY_S - elapsed)

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json, image/*, */*",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    backoff = 2.0
    for attempt in range(MAX_RETRIES):
        _last_request_time = time.time()
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = resp.read()
                return data if as_bytes else data.decode("utf-8")
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < MAX_RETRIES - 1:
                # Rate limited — back off aggressively
                wait = 30 * (attempt + 1)
                print(f"    [429] waiting {wait}s before retry...", flush=True)
                time.sleep(wait)
                continue
            print(f"    [warn] HTTP {exc.code}: {url[:90]}")
            return None
        except Exception as exc:
            print(f"    [warn] GET failed ({exc}): {url[:90]}")
            return None
    return None


def find_existing_image(slug: str) -> Path | None:
    """Check if we already downloaded an image for this slug (any ext)."""
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".gif"):
        p = IMAGES_DIR / (slug + ext)
        if p.exists():
            return p
    return None


def extension_from_url(url: str) -> str:
    """Extract file extension from URL, defaulting to .jpg."""
    m = re.search(r"\.(jpg|jpeg|png|webp|gif)(?:\?|/|$)", url.lower())
    if m:
        ext = m.group(1)
        return ".jpg" if ext == "jpeg" else f".{ext}"
    return ".jpg"


# --------------------------- Wikipedia --------------------------- #

def get_summary(title: str) -> dict | None:
    """Fetch Wikipedia REST summary for a page title. Returns parsed JSON or None."""
    encoded = urllib.parse.quote(title.replace(" ", "_"), safe="")
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"
    body = http_get(url)
    if not body:
        return None
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return None


def thumbnail_from_summary(summary: dict) -> str | None:
    if not summary or summary.get("type") == "disambiguation":
        return None
    thumb = summary.get("thumbnail") or summary.get("originalimage")
    return thumb.get("source") if thumb else None


def search_top_title(label: str, topic: str) -> str | None:
    """Use Wikipedia opensearch API to find the best-matching page title."""
    hint = TOPIC_HINTS.get(topic, "")
    query = f"{label} {hint}".strip()
    url = (
        "https://en.wikipedia.org/w/api.php"
        f"?action=opensearch&search={urllib.parse.quote(query)}"
        "&limit=1&namespace=0&format=json"
    )
    body = http_get(url)
    if not body:
        return None
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return None
    titles = data[1] if len(data) > 1 else []
    return titles[0] if titles else None


def get_pageimage(title: str) -> str | None:
    """Fallback via MediaWiki pageimages API. Returns thumbnail URL or None.

    More reliable than REST summary for pages where the summary endpoint
    doesn't surface a thumbnail, but the page DOES have a main image.
    """
    encoded = urllib.parse.quote(title.replace(" ", "_"), safe="")
    url = (
        "https://en.wikipedia.org/w/api.php"
        f"?action=query&titles={encoded}"
        "&prop=pageimages&pithumbsize=500&format=json&redirects=1"
    )
    body = http_get(url)
    if not body:
        return None
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return None
    pages = data.get("query", {}).get("pages", {})
    for _pid, page in pages.items():
        thumb = page.get("thumbnail", {})
        src = thumb.get("source")
        if src:
            return src
    return None


def _imageinfo_url(file_name: str) -> str | None:
    """Resolve a 'File:Foo.svg' style filename to a rendered thumbnail URL."""
    title = f"File:{file_name.replace(' ', '_')}"
    encoded = urllib.parse.quote(title, safe="")
    url = (
        "https://en.wikipedia.org/w/api.php"
        f"?action=query&titles={encoded}"
        "&prop=imageinfo&iiprop=url&iiurlwidth=500&format=json"
    )
    body = http_get(url)
    if not body:
        return None
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return None
    for _pid, page in data.get("query", {}).get("pages", {}).items():
        info = (page.get("imageinfo") or [{}])[0]
        src = info.get("thumburl") or info.get("url")
        if src:
            return src
    return None


_INFOBOX_IMG_RE = re.compile(
    r"(?:^|\|)\s*(?:image|logo|poster|cover|image_file|cover_art|cover art)\s*="
    r"\s*([^\n\|\[\]]+?\.(?:jpg|jpeg|png|svg|webp|gif))",
    re.IGNORECASE | re.MULTILINE,
)
_FILE_LINK_RE = re.compile(
    r"\[\[File:([^\|\]]+?\.(?:jpg|jpeg|png|svg|webp|gif))", re.IGNORECASE
)


def get_infobox_image(title: str) -> str | None:
    """Parse section 0 wikitext for an infobox image, then resolve its URL.

    Last-resort fallback for pages (TV shows, games, franchises) that use
    non-free logos that aren't registered as page thumbnails.
    """
    encoded = urllib.parse.quote(title.replace(" ", "_"), safe="")
    url = (
        "https://en.wikipedia.org/w/api.php"
        f"?action=parse&page={encoded}"
        "&prop=wikitext&section=0&format=json&redirects=1"
    )
    body = http_get(url)
    if not body:
        return None
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return None
    text = data.get("parse", {}).get("wikitext", {}).get("*", "")
    if not text:
        return None
    m = _INFOBOX_IMG_RE.search(text)
    fname = m.group(1).strip() if m else None
    if not fname:
        m = _FILE_LINK_RE.search(text)
        fname = m.group(1).strip() if m else None
    if not fname:
        return None
    return _imageinfo_url(fname)


def resolve_thumbnail(topic: str, label: str) -> str | None:
    """Try progressively broader lookups to find an image URL for (topic, label)."""
    # 1) Pre-mapped explicit title — REST summary
    explicit = EXPLICIT_TITLES.get((topic, label))
    if explicit:
        summary = get_summary(explicit)
        thumb = thumbnail_from_summary(summary)
        if thumb:
            return thumb
        # 1b) pageimages
        thumb = get_pageimage(explicit)
        if thumb:
            return thumb
        # 1c) infobox wikitext (handles non-free logos on TV/game pages)
        thumb = get_infobox_image(explicit)
        if thumb:
            return thumb

    # 2) Direct lookup by label
    summary = get_summary(label)
    thumb = thumbnail_from_summary(summary)
    if thumb:
        return thumb
    thumb = get_pageimage(label)
    if thumb:
        return thumb
    thumb = get_infobox_image(label)
    if thumb:
        return thumb

    # 3) Search with topic-hint-assisted query
    found = search_top_title(label, topic)
    if found and found.lower() != label.lower():
        summary = get_summary(found)
        thumb = thumbnail_from_summary(summary)
        if thumb:
            return thumb
        thumb = get_pageimage(found)
        if thumb:
            return thumb
        thumb = get_infobox_image(found)
        if thumb:
            return thumb

    return None


# --------------------------- main --------------------------- #

def main() -> int:
    if not WORD_BANK_PATH.exists():
        print(f"Word bank not found: {WORD_BANK_PATH}")
        return 2

    data = json.loads(WORD_BANK_PATH.read_text(encoding="utf-8"))
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    stats = {"total": 0, "fetched": 0, "skipped": 0, "failed": 0}
    failures: list[tuple[str, str]] = []

    def save_progress() -> None:
        WORD_BANK_PATH.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    for entry in data:
        topic = entry["topic"]
        print(f"\n== {topic} ==")
        for item in entry["grid_items"]:
            stats["total"] += 1
            label = item["label"]
            slug = slugify(label)

            # Skip if we already have an image on disk
            existing = find_existing_image(slug)
            if existing:
                item["image_path"] = f"/static/images/{existing.name}"
                stats["skipped"] += 1
                print(f"  [skip] {label} -> {existing.name}")
                continue

            print(f"  [fetch] {label}...", end=" ", flush=True)
            url = resolve_thumbnail(topic, label)
            if not url:
                print("no image")
                failures.append((topic, label))
                stats["failed"] += 1
                continue

            ext = extension_from_url(url)
            dest = IMAGES_DIR / f"{slug}{ext}"
            body = http_get(url, as_bytes=True)
            if not body:
                print("download failed")
                failures.append((topic, label))
                stats["failed"] += 1
                continue

            dest.write_bytes(body)
            item["image_path"] = f"/static/images/{dest.name}"
            stats["fetched"] += 1
            print(f"OK -> {dest.name} ({len(body) // 1024}KB)")
        # Save after every topic so a mid-run crash doesn't lose progress
        save_progress()

    save_progress()

    print(
        f"\nDone. total={stats['total']} "
        f"fetched={stats['fetched']} skipped={stats['skipped']} failed={stats['failed']}"
    )
    if failures:
        print("\nFailed items (consider adding EXPLICIT_TITLES for these):")
        for t, l in failures:
            print(f"  - [{t}] {l}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
