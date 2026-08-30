#!/usr/bin/env python3
"""Post an AI code review to a pull request.

Reads two files produced by the Copilot CLI review step:

* ``review-summary.md``    -- a high-level markdown summary of the findings.
* ``review-comments.json`` -- a JSON array of line-specific comments, each an
  object with ``path``, ``line``, ``side`` and ``body``.

It validates the inline comments against the actual PR diff (so only lines that
are part of the diff are used) and posts everything as a single GitHub pull
request review. If the review cannot be created, the summary is logged to
stderr so it can be recovered from the workflow logs (the review files are
also uploaded as artifacts).

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
from typing import Any, TypedDict

API_BASE = "https://api.github.com"
API_VERSION = "2026-03-10"


class ReviewComment(TypedDict):
    """A single inline review comment anchored to a line of the PR diff."""

    path: str
    line: int
    side: str
    body: str


def api_request(token: str, method: str, path: str, payload: dict | None = None) -> Any:
    """Perform an authenticated request against the GitHub REST API."""
    url = f"{API_BASE}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", "Bearer " + token)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", API_VERSION)
    req.add_header("Content-Type", "application/json")
    # A timeout so a hung API call fails fast instead of blocking the job.
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode() or "null")


def fetch_all(token: str, path: str) -> list[Any]:
    """Fetch every page of a paginated list endpoint."""
    items: list[Any] = []
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


def valid_lines_by_file(
    token: str, repo: str, pr_number: str
) -> dict[str, set[tuple[int, str]]]:
    """Map each changed file to the set of (line, side) pairs that are part of
    the diff and can legally receive a review comment.

    ``side`` is ``"RIGHT"`` for lines in the new version of the file and
    ``"LEFT"`` for lines in the old version.
    """
    files = fetch_all(token, f"/repos/{repo}/pulls/{pr_number}/files")
    valid: dict[str, set[tuple[int, str]]] = {}
    hunk_re = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")
    for f in files:
        patch = f.get("patch")
        if not patch:
            # GitHub omits `patch` for binary files and for diffs too large to
            # include in the /pulls/{pull_number}/files response. Inline
            # comments for such files cannot be validated, so they are skipped.
            print(
                f"No patch available for {f['filename']}; "
                "inline comments for this file will be skipped.",
                file=sys.stderr,
            )
            continue
        old_line = new_line = None
        lines: set[tuple[int, str]] = set()
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
                # Context (unchanged) lines are also commentable by GitHub, so
                # they are kept in the valid set. The prompt asks the model to
                # only comment on added/removed lines, but being permissive
                # here avoids dropping a legitimate context-line comment.
                lines.add((new_line, "RIGHT"))
                old_line += 1
                new_line += 1
        valid[f["filename"]] = lines
    return valid


def read_summary() -> str:
    """Return the review summary, or a placeholder if none was produced."""
    if os.path.exists("review-summary.md"):
        with open("review-summary.md") as fh:
            summary = fh.read().strip()
        if summary:
            return summary
    return "AI review completed, but no summary was produced."


def read_raw_comments() -> list[Any]:
    """Return the list of raw inline comments, or an empty list on any error."""
    if not os.path.exists("review-comments.json"):
        return []
    try:
        with open("review-comments.json") as fh:
            raw = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"Could not parse review-comments.json: {exc}", file=sys.stderr)
        return []
    if not isinstance(raw, list):
        print(
            f"review-comments.json is not a JSON array (got {type(raw).__name__}); "
            "ignoring it.",
            file=sys.stderr,
        )
        return []
    return raw


def filter_comments(
    token: str, repo: str, pr_number: str, raw_comments: list[Any]
) -> list[ReviewComment]:
    """Keep only comments that point at a line actually present in the diff."""
    if not raw_comments:
        return []
    valid = valid_lines_by_file(token, repo, pr_number)
    comments: list[ReviewComment] = []
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
        # Defensively strip a git-style a/ or b/ prefix, but only when the path
        # as-is is not already a known changed file (so a legitimate top-level
        # file named e.g. "a/foo.py" is not mangled).
        norm_path = path if path in valid else re.sub(r"^[ab]/", "", path)
        if (line, side) in valid.get(norm_path, set()):
            comments.append(
                {"path": norm_path, "line": line, "side": side, "body": body}
            )
        else:
            print(
                f"Skipping out-of-diff comment: {path}:{line} ({side})", file=sys.stderr
            )
    return comments


def get_current_user_login(token: str) -> str | None:
    """Return the login of the authenticated user, or None on failure."""
    try:
        user = api_request(token, "GET", "/user")
        return user.get("login") if isinstance(user, dict) else None
    except Exception as exc:  # noqa: BLE001
        print(f"Could not determine current user: {exc}", file=sys.stderr)
        return None


def dismiss_pending_reviews(token: str, repo: str, pr_number: str) -> None:
    """Delete any pending reviews authored by the current user.

    GitHub only allows one pending review per user per pull request, so a stale
    pending review (e.g. left over from a previous cancelled run) would block a
    new one with a 422. Clearing our own pending reviews first avoids that.
    """
    login = get_current_user_login(token)
    if login is None:
        return
    try:
        reviews = fetch_all(token, f"/repos/{repo}/pulls/{pr_number}/reviews")
    except Exception as exc:  # noqa: BLE001
        print(f"Could not list reviews: {exc}", file=sys.stderr)
        return
    for review in reviews:
        if not isinstance(review, dict) or review.get("state") != "PENDING":
            continue
        author = (review.get("user") or {}).get("login")
        if author != login:
            continue
        review_id = review.get("id")
        if review_id is None:
            continue
        try:
            api_request(
                token, "DELETE", f"/repos/{repo}/pulls/{pr_number}/reviews/{review_id}"
            )
            print(f"Deleted stale pending review {review_id}.", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            print(
                f"Could not delete pending review {review_id}: {exc}", file=sys.stderr
            )


def post_review(
    token: str, repo: str, pr_number: str, summary: str, comments: list[ReviewComment]
) -> bool:
    """Post the review.

    Returns ``True`` if the review was posted, ``False`` otherwise. On failure
    the summary is logged to stderr so it can be recovered from the workflow
    logs (the review files are also uploaded as artifacts).
    """
    # A stale pending review by the same user would block creation with a 422.
    dismiss_pending_reviews(token, repo, pr_number)
    try:
        api_request(
            token,
            "POST",
            f"/repos/{repo}/pulls/{pr_number}/reviews",
            {"body": summary, "event": "COMMENT", "comments": comments},
        )
    except urllib.error.HTTPError as exc:
        # HTTPError is a URLError, so this also covers network-level failures.
        detail = exc.read().decode()
        print(
            f"Review creation failed ({exc.code}): {detail}\n"
            f"Review summary:\n{summary}",
            file=sys.stderr,
        )
        return False
    except Exception as exc:  # noqa: BLE001 - log and fail, do not retry
        print(
            f"Unexpected error while posting review: {type(exc).__name__}: {exc}\n"
            f"Review summary:\n{summary}",
            file=sys.stderr,
        )
        return False
    print(f"Posted review with {len(comments)} inline comment(s).")
    return True


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
