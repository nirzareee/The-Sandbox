"""Fetch missing word-bank images from Wikipedia.

For every grid item in `data/word_bank.json` whose `image_path` points at a
`/static/images/<slug>.<ext>` file that doesn't exist locally, this script
hits Wikipedia's REST summary API for the label, downloads the page
thumbnail, and saves it to `frontend-src/public/images/<slug>.<ext>`.

Usage:
    python scripts/fetch_wordbank_images.py [--force] [--dry-run]

    --force      Re-download even if the local file already exists.
    --dry-run    Print what would be fetched, don't write anything.
    --query NAME Override the Wikipedia search term for one label
                 (repeatable, e.g. --query "Marathon=Marathon (sport)")

Failed lookups are logged to stderr; the frontend already falls back to an
emoji placeholder via `<img onError>`, so partial coverage is fine.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORD_BANK = ROOT / "data" / "word_bank.json"
IMAGES_DIR = ROOT / "frontend-src" / "public" / "images"

# Wikimedia's bot policy requires a UA naming the app, version, and a
# contact URL or email. See https://meta.wikimedia.org/wiki/User-Agent_policy
USER_AGENT = (
    "TheSandboxResearch/1.0 "
    "(https://github.com/adityakamat24/The-Sandbox; "
    "research; one-shot wordbank fetch) "
    "Python/urllib"
)
SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"

# Items where the bare label resolves to a disambiguation page or wrong
# topic. Map label → preferred Wikipedia article title.
DEFAULT_QUERY_OVERRIDES = {
    "Marathon": "Marathon (sport)",
    "100m Sprint": "100 metres",
    "Hurdles": "Hurdling",
    "Pole Vault": "Pole vault",
    "Long Jump": "Long jump",
    "High Jump": "High jump",
    "Shot Put": "Shot put",
    "Cricket": "Cricket",
    "Squash": "Squash (sport)",
    "Pickleball": "Pickleball",
    "Field Hockey": "Field hockey",
    "Snooker": "Snooker",
    "Lamb": "Lamb (Easter symbol)",
    "Cross": "Christian cross",
    "Lantern": "Lantern",
    "Map": "Map",
    "Cooler": "Cooler",
    "Stocking": "Christmas stocking",
    "Wreath": "Wreath",
    "Candy Cane": "Candy cane",
    "Mistletoe": "Mistletoe",
    "Ghost": "Ghost",
    "Witch": "Witch",
    "Spider": "Spider",
    "Tombstone": "Headstone",
    "Cauldron": "Cauldron",
    "Skeleton": "Skeleton",
    "Daffodil": "Daffodil",
    "Chick": "Chicken",
    "Easter Lily": "Lilium longiflorum",
    "Hot Cross Bun": "Hot cross bun",
    "Hatchet": "Hatchet",
    "Sunglasses": "Sunglasses",
    "Marshmallow": "Marshmallow",
    "Backpack": "Backpack",
    "Snorkel": "Snorkeling",
    "Bucket and Spade": "Bucket and spade",
    "Flip Flops": "Flip-flops",
    "Beach Towel": "Towel",
    "Beach Umbrella": "Umbrella",
    "Beach Ball": "Beach ball",
    "Hula Hoop": "Hula hoop",
    "Pogo Stick": "Pogo stick",
    "Jump Rope": "Skipping rope",
    "Slip 'N Slide": "Slip 'N Slide",
    "Sidewalk Chalk": "Sidewalk chalk",
    "Bouncy Castle": "Inflatable castle",
    "Frisbee": "Frisbee",
    # Lowercase variants — REST routes some Title Case forms to disambigs
    "Black Cat": "Black cat",
    "Gingerbread House": "Gingerbread house",
    "Chocolate Bunny": "Chocolate bunny",
    "Sleeping Bag": "Sleeping bag",
    "Steeplechase": "Steeplechase (athletics)",
    # Easter Lamb iconography
    "Lamb": "Lamb of God",
}


def slug_from_path(image_path: str) -> tuple[str, str]:
    """Return (slug, ext) extracted from a /static/images/<slug>.<ext> path."""
    m = re.match(r"/static/images/([^.]+)\.([a-zA-Z0-9]+)$", image_path)
    if not m:
        raise ValueError(f"Unrecognized image_path: {image_path}")
    return m.group(1), m.group(2).lower()


def fetch_summary(title: str, retries: int = 4) -> dict | None:
    """Hit Wikipedia's REST summary API. Backs off on 429 / 503."""
    url = SUMMARY_URL.format(title=urllib.parse.quote(title.replace(" ", "_")))
    req = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
    )
    backoff = 2.0
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 503) and attempt < retries:
                wait = backoff * (2 ** attempt)
                print(f"  rate-limited on {title!r}, sleeping {wait:.0f}s (attempt {attempt+1}/{retries})",
                      file=sys.stderr)
                time.sleep(wait)
                continue
            print(f"  summary lookup failed for {title!r}: HTTP {exc.code}", file=sys.stderr)
            return None
        except Exception as exc:
            print(f"  summary lookup failed for {title!r}: {exc}", file=sys.stderr)
            return None
    return None


def download(url: str, dest: Path, retries: int = 4) -> bool:
    """Download a binary URL to dest. Backs off on 429/503."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    backoff = 2.0
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
            dest.write_bytes(data)
            return True
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 503) and attempt < retries:
                wait = backoff * (2 ** attempt)
                print(f"  rate-limited downloading {dest.name}, sleeping {wait:.0f}s",
                      file=sys.stderr)
                time.sleep(wait)
                continue
            print(f"  download failed for {url}: HTTP {exc.code}", file=sys.stderr)
            return False
        except Exception as exc:
            print(f"  download failed for {url}: {exc}", file=sys.stderr)
            return False
    return False


def parse_overrides(args_query: list[str]) -> dict[str, str]:
    out = dict(DEFAULT_QUERY_OVERRIDES)
    for item in args_query or []:
        if "=" not in item:
            print(f"warning: --query expects LABEL=TITLE, got {item!r}", file=sys.stderr)
            continue
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--force", action="store_true",
                        help="re-download even if the local file already exists")
    parser.add_argument("--dry-run", action="store_true",
                        help="print actions without downloading")
    parser.add_argument("--query", action="append", default=[],
                        help="override Wikipedia title for a label (LABEL=TITLE)")
    parser.add_argument("--sleep", type=float, default=1.2,
                        help="seconds between requests, default 1.2")
    args = parser.parse_args()

    overrides = parse_overrides(args.query)

    bank = json.loads(WORD_BANK.read_text(encoding="utf-8"))
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    todo: list[tuple[str, str, str]] = []  # (label, slug, ext)
    for entry in bank:
        for item in entry["grid_items"]:
            label = item["label"]
            slug, ext = slug_from_path(item["image_path"])
            if slug in seen:
                continue
            seen.add(slug)
            local = IMAGES_DIR / f"{slug}.{ext}"
            if local.exists() and not args.force:
                continue
            todo.append((label, slug, ext))

    print(f"{len(todo)} image(s) to fetch (existing files: {len(seen) - len(todo)})")
    if args.dry_run:
        for label, slug, ext in todo:
            print(f"  would fetch: {label!r} -> {slug}.{ext}")
        return 0

    failures: list[str] = []
    for label, slug, ext in todo:
        title = overrides.get(label, label)
        print(f"[{label}] -> wiki:{title}")
        summary = fetch_summary(title)
        thumb = (summary or {}).get("thumbnail", {}).get("source")
        if not thumb:
            # Some pages have no thumbnail but expose originalimage (e.g.
            # SVG-primary articles where the API skips raster generation).
            thumb = (summary or {}).get("originalimage", {}).get("source")
        if not thumb:
            failures.append(f"{label} (no thumbnail)")
            time.sleep(args.sleep)
            continue
        # Wikipedia thumbnails are .jpg/.png; if our slug ends in a different
        # ext we still write under that ext — browsers don't care.
        dest = IMAGES_DIR / f"{slug}.{ext}"
        if download(thumb, dest):
            print(f"  saved {dest.name} ({dest.stat().st_size // 1024} KB)")
        else:
            failures.append(label)
        time.sleep(args.sleep)

    print(f"\ndone — fetched {len(todo) - len(failures)}/{len(todo)}")
    if failures:
        print("failed:")
        for f in failures:
            print(f"  - {f}")
        print("\nthe frontend's <img onError> handler will fall back to an emoji "
              "for any missing image, so partial coverage is fine. To retry a "
              "specific label, use --query 'LABEL=Article Title' --force.")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
