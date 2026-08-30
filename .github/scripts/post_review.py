#!/usr/bin/env python3
"""Post an AI code review to a pull request.

Reads two files produced by the Copilot CLI review step:

* ``review-summary.md``    -- a high-level markdown summary of the findings.
* ``review-comments.json`` -- a JSON array of line-specific comments, each an
  object with ``path``, ``line``, ``side`` and ``body``.

It validates the inline comments against the actual PR diff (so only lines that
are part of the diff are used) and posts everything as a single GitHub pull
request review. If the review cannot be created, it falls back to a plain issue
comment containing just the summary, so the PR always receives feedback.

Required environment variables:
    REPO          -- ``owner/repo`` of the repository.
    PR_NUMBER     -- the pull request number.
    GITHUB_TOKEN  -- a token with ``pull-requests: write`` permission.
"""

import json
import os
import re
import sys
import urllib.error
import urllib.request

API_BASE = "https://api.github.com"


def api_request(token, method, path, payload=None):
    """Perform an authenticated request against the GitHub REST API."""
    url = f"{API_BASE}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", "Bearer " + token)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode() or "null")


def fetch_all(token, path):
    """Fetch every page of a paginated list endpoint."""
    items = []
    page = 1
    while True:
        sep = "&" if "?" in path else "?"
        chunk = api_request(token, "GET", f"{path}{sep}per_page=100&page={page}")
        if not chunk:
            break
        items.extend(chunk)
        if len(chunk) < 100:
            break
        page += 1
    return items


def valid_lines_by_file(token, repo, pr_number):
    """Map each changed file to the set of (line, side) pairs that are part of
    the diff and can legally receive a review comment.

    ``side`` is ``"RIGHT"`` for lines in the new version of the file and
    ``"LEFT"`` for lines in the old version.
    """
    files = fetch_all(token, f"/repos/{repo}/pulls/{pr_number}/files")
    valid = {}
    hunk_re = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")
    for f in files:
        patch = f.get("patch")
        if not patch:
            # Binary files (and files with no textual patch) cannot receive
            # line-anchored comments, so they are intentionally omitted.
            continue
        old_line = new_line = None
        lines = set()
        for raw in patch.splitlines():
            m = hunk_re.match(raw)
            if m:
                old_line, new_line = int(m.group(1)), int(m.group(2))
                continue
            if old_line is None:
                continue
            if raw.startswith("\\ No newline at end of file"):
                continue
            if raw.startswith("+"):
                lines.add((new_line, "RIGHT"))
                new_line += 1
            elif raw.startswith("-"):
                lines.add((old_line, "LEFT"))
                old_line += 1
            else:
                lines.add((new_line, "RIGHT"))
                old_line += 1
                new_line += 1
        valid[f["filename"]] = lines
    return valid


def read_summary():
    """Return the review summary, or a placeholder if none was produced."""
    if os.path.exists("review-summary.md"):
        with open("review-summary.md") as fh:
            summary = fh.read().strip()
        if summary:
            return summary
    return "AI review completed, but no summary was produced."


def read_raw_comments():
    """Return the list of raw inline comments, or an empty list on any error."""
    if not os.path.exists("review-comments.json"):
        return []
    try:
        with open("review-comments.json") as fh:
            raw = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"Could not parse review-comments.json: {exc}", file=sys.stderr)
        return []
    return raw if isinstance(raw, list) else []


def filter_comments(token, repo, pr_number, raw_comments):
    """Keep only comments that point at a line actually present in the diff."""
    if not raw_comments:
        return []
    valid = valid_lines_by_file(token, repo, pr_number)
    comments = []
    for c in raw_comments:
        if not isinstance(c, dict):
            continue
        path = c.get("path")
        body = c.get("body")
        side = str(c.get("side") or "RIGHT").upper()
        if side not in ("LEFT", "RIGHT"):
            side = "RIGHT"
        try:
            line = int(c.get("line"))
        except (TypeError, ValueError):
            line = None
        if not (path and line and body):
            continue
        # Defensively strip git-style a/ b/ prefixes some tools emit.
        norm_path = re.sub(r"^[ab]/", "", path) if path not in valid else path
        if (line, side) in valid.get(norm_path, set()):
            comments.append(
                {"path": norm_path, "line": line, "side": side, "body": body}
            )
        else:
            print(
                f"Skipping out-of-diff comment: {path}:{line} ({side})", file=sys.stderr
            )
    return comments


def post_review(token, repo, pr_number, summary, comments):
    """Post the review, falling back to a plain issue comment on failure.

    Returns ``True`` if the review (or the fallback comment) was posted.
    """
    try:
        try:
            api_request(
                token,
                "POST",
                f"/repos/{repo}/pulls/{pr_number}/reviews",
                {"body": summary, "event": "COMMENT", "comments": comments},
            )
            print(f"Posted review with {len(comments)} inline comment(s).")
            return True
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode()
            print(f"Review creation failed ({exc.code}): {detail}", file=sys.stderr)
            print(
                "Falling back to a plain issue comment with the summary only.",
                file=sys.stderr,
            )
            api_request(
                token,
                "POST",
                f"/repos/{repo}/issues/{pr_number}/comments",
                {"body": summary},
            )
            return True
    except Exception as exc:  # noqa: BLE001 - last-resort fallback, must not fail silently
        print(
            f"Unexpected error while posting review: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        # Still try to leave feedback on the PR before failing the step.
        try:
            api_request(
                token,
                "POST",
                f"/repos/{repo}/issues/{pr_number}/comments",
                {"body": summary},
            )
            print(
                "Posted fallback issue comment after unexpected error.", file=sys.stderr
            )
            return True
        except Exception as fallback_exc:  # noqa: BLE001
            print(
                f"Fallback comment also failed: {type(fallback_exc).__name__}: {fallback_exc}",
                file=sys.stderr,
            )
            return False


def main():
    repo = os.environ.get("REPO")
    pr_number = os.environ.get("PR_NUMBER")
    token = os.environ.get("GITHUB_TOKEN")
    if not (repo and pr_number and token):
        sys.exit("Missing required env vars: REPO, PR_NUMBER, GITHUB_TOKEN")

    summary = read_summary()
    raw_comments = read_raw_comments()
    comments = filter_comments(token, repo, pr_number, raw_comments)

    if not post_review(token, repo, pr_number, summary, comments):
        sys.exit(1)


if __name__ == "__main__":
    main()
