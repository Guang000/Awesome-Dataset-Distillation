#!/usr/bin/env python3
"""Validate the article records in data/articles.json.

Checks every article for missing / misplaced fields, e.g. a `url` that ended up
in the `cite` slot.  Any article failing at least one rule is printed in full,
together with the reasons it failed.

Usage:
    python3 validate_articles.py [-f data/articles.json] [--no-citation-check]

Exit code is 1 when at least one article is invalid, 0 otherwise.
"""

import argparse
import json
import os
import re
import sys

FIELDS = ("title", "author", "github", "url", "cite", "website")

YEAR = re.compile(r"\b(19|20)\d{2}\b")
# e.g. wang2018datasetdistillation, chan-santiago2025mgd3, liu2023dream+
CITE = re.compile(r"^[a-z][a-z.\-']*(19|20)\d{2}[a-z0-9+\-]*$")


def check(article, citations_dir, strict=False):
    """Return the list of problems found in one article."""
    problems = []

    missing = [f for f in FIELDS if f not in article]
    if missing:
        problems.append("missing field(s): %s" % ", ".join(missing))
    extra = [f for f in article if f not in FIELDS]
    if extra:
        problems.append("unexpected field(s): %s" % ", ".join(extra))

    title = article.get("title")
    if not isinstance(title, str) or not title.strip():
        problems.append("title: must be a non-empty string")

    # 1. author must carry a year (e.g. "Bo Zhao et al., ICLR 2021")
    author = article.get("author")
    if not isinstance(author, str) or not author.strip():
        problems.append("author: must be a non-empty string")
    elif not YEAR.search(author):
        problems.append("author: no 4-digit year found -> %r" % author)
    elif "http" in author:
        problems.append("author: looks like a URL -> %r" % author)

    # 2. github is null or a github.com link
    github = article.get("github")
    if github is not None:
        if not isinstance(github, str):
            problems.append("github: must be null or a string -> %r" % github)
        elif not github.startswith("https://github.com/"):
            problems.append("github: not a https://github.com/ link -> %r" % github)

    # 3. url must be an https link (null is allowed: PaperCatcher.js hides the
    #    button, e.g. for the workshop / challenge entries)
    url = article.get("url")
    if url is None:
        if strict:
            problems.append("url: is null")
    elif not isinstance(url, str):
        problems.append("url: must be null or a string -> %r" % url)
    elif not url.startswith("https:"):
        problems.append("url: does not start with 'https:' -> %r" % url)

    # 4. cite must look like <surname><year><keyword> (null allowed, same reason)
    cite = article.get("cite")
    if cite is None:
        if strict:
            problems.append("cite: is null")
    elif not isinstance(cite, str):
        problems.append("cite: must be null or a string -> %r" % cite)
    elif not CITE.match(cite):
        problems.append("cite: not of the form <name><year><keyword> -> %r" % cite)
    elif citations_dir is not None:
        path = os.path.join(citations_dir, cite + ".txt")
        if not os.path.exists(path):
            problems.append("cite: no matching bibtex file %s" % path)

    # 5. website is null or an https link
    website = article.get("website")
    if website is not None:
        if not isinstance(website, str):
            problems.append("website: must be null or a string -> %r" % website)
        elif not website.startswith("https:"):
            problems.append("website: does not start with 'https:' -> %r" % website)

    return problems


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-f", "--file", default=os.path.join(here, "data", "articles.json"),
                        help="path to articles.json (default: %(default)s)")
    parser.add_argument("--citations-dir", default=os.path.join(here, "citations"),
                        help="directory holding the <cite>.txt bibtex files")
    parser.add_argument("--no-citation-check", action="store_true",
                        help="do not check that citations/<cite>.txt exists")
    parser.add_argument("--strict", action="store_true",
                        help="also report articles whose url / cite is null")
    args = parser.parse_args()

    with open(args.file, encoding="utf-8") as fp:
        data = json.load(fp)

    citations_dir = None if args.no_citation_check else args.citations_dir
    if citations_dir is not None and not os.path.isdir(citations_dir):
        print("warning: citations dir %s not found, skipping that check" % citations_dir,
              file=sys.stderr)
        citations_dir = None

    total = 0
    bad = 0
    for section, subsections in data.items():
        for subsection, articles in subsections.items():
            for index, article in enumerate(articles):
                total += 1
                if not isinstance(article, dict):
                    problems = ["article is not an object -> %r" % (article,)]
                else:
                    problems = check(article, citations_dir, args.strict)
                if not problems:
                    continue
                bad += 1
                print("=" * 78)
                print("[%s > %s > #%d]" % (section, subsection, index))
                for p in problems:
                    print("  ! " + p)
                print(json.dumps(article, indent=2, ensure_ascii=False))
                print()

    print("=" * 78)
    print("checked %d articles, %d invalid" % (total, bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
