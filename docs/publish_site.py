#!/usr/bin/env python
r"""Publish a built documentation site as one version folder of the gh-pages tree.

The documentation is served from the root of the ``gh-pages`` branch, which
holds one folder per version:

- ``<X.Y.Z>/`` per release (no ``v`` prefix),
- ``dev/`` for the site built from ``main``,
- ``stable/`` a copy of the newest release's folder,

and, at the root, ``index.html`` (redirects to ``stable/``, or to ``dev/``
while no release is published), ``404.html`` (forwards version-less paths,
and the addresses of a former unversioned site below ``--forward-from``, to
the same page in the folder ``index.html`` redirects to), ``versions.json`` (the version list, rebuilt from the
folders present), ``versions.js`` (the version menu and banner, see
``versions.js`` next to this script) and ``.nojekyll``.

Publishing replaces the target folder with the built site, refreshes
``stable/`` when the target is the newest release, rewrites the root files and
adds the ``versions.js`` script tag to every HTML page of the folders it
touched. The script tag is added once: pages that already carry it are left
as they are.

The script serves any project: ``--base-path`` names the URL path of the
gh-pages root, ``--title`` the documentation in the root pages, and
``--forward-from`` the folder of a former unversioned site (an empty value
forwards none). The defaults are those of the TimeToAlign! site.

Usage::

    python docs/publish_site.py docs/page/_site --version dev --pages ../gh-pages
    python docs/publish_site.py docs/page/_site --version 1.2.0 --pages ../gh-pages
    python docs/publish_site.py docs/_build/html --version dev --pages ../gh-pages \
        --base-path /TimeToShare/ --title "TimeToShare! documentation" --forward-from ""
"""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import sys
from pathlib import Path

DEV = "dev"
STABLE = "stable"
DEFAULT_BASE_PATH = "/TimeToAlign/"
DEFAULT_TITLE = "TimeToAlign! documentation"
DEFAULT_FORWARD_FROM = "page/_site/"
VERSIONS_JS = Path(__file__).resolve().parent / "versions.js"

_RELEASE_PATTERN = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
_SCRIPT_TAG_PATTERN = re.compile(
    r"<script\b[^>]*\bsrc=[\"'][^\"']*versions\.js[\"']", re.IGNORECASE
)
_BODY_END_PATTERN = re.compile(r"</body\s*>", re.IGNORECASE)


def release_key(name: str) -> tuple[int, int, int] | None:
    """Return the sort key of a release folder name, or None for any other name.

    Args:
        name: A folder name such as ``1.2.0``.

    Returns:
        The ``(major, minor, patch)`` integers, or None when ``name`` is not a
        release number.
    """
    match = _RELEASE_PATTERN.match(name)
    if match is None:
        return None
    major, minor, patch = (int(part) for part in match.groups())
    return major, minor, patch


def releases_in(pages: Path) -> list[str]:
    """List the release folders of a gh-pages tree, newest first.

    Args:
        pages: Root of the gh-pages tree.

    Returns:
        The release folder names sorted by version number, newest first.
    """
    found: list[tuple[tuple[int, int, int], str]] = []
    for child in pages.iterdir():
        key = release_key(child.name)
        if child.is_dir() and key is not None:
            found.append((key, child.name))
    return [name for _, name in sorted(found, reverse=True)]


def versions_list(pages: Path) -> list[dict[str, object]]:
    """Build the content of ``versions.json`` from the folders present.

    The list starts with ``dev`` and ``stable`` when their folders exist,
    followed by the releases, newest first. Release entries carry
    ``"release": true``; the ``stable`` entry names the release it copies in
    ``copy_of``.

    Args:
        pages: Root of the gh-pages tree.

    Returns:
        One entry per published folder, with the keys ``version``, ``title``
        and ``path``, plus ``release`` on the releases and ``copy_of`` on
        ``stable``.
    """
    releases = releases_in(pages)
    entries: list[dict[str, object]] = []
    if (pages / DEV).is_dir():
        entries.append({"version": DEV, "title": "dev (main)", "path": f"{DEV}/"})
    if (pages / STABLE).is_dir() and releases:
        entries.append(
            {
                "version": STABLE,
                "title": f"stable ({releases[0]})",
                "path": f"{STABLE}/",
                "copy_of": releases[0],
            }
        )
    for name in releases:
        entries.append(
            {"version": name, "title": name, "path": f"{name}/", "release": True}
        )
    return entries


def redirect_page(target: str, title: str = DEFAULT_TITLE) -> str:
    """Return the root ``index.html`` that forwards to a version folder.

    The query string and the fragment of the requested URL are kept.

    Args:
        target: The folder to forward to, with a trailing slash (``stable/``).
        title: The name of the documentation, the page's title.

    Returns:
        The HTML text of the page.
    """
    quoted = html.escape(target, quote=True)
    script_target = json.dumps(target)
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        f"<title>{html.escape(title)}</title>\n"
        f'<link rel="canonical" href="{quoted}">\n'
        "<script>\n"
        f"location.replace({script_target} + location.search + location.hash);\n"
        "</script>\n"
        f'<noscript><meta http-equiv="refresh" content="0; url={quoted}"></noscript>\n'
        "</head>\n"
        "<body>\n"
        f'<p>The documentation is at <a href="{quoted}">{quoted}</a>.</p>\n'
        "</body>\n"
        "</html>\n"
    )


def not_found_page(
    base_path: str,
    title: str = DEFAULT_TITLE,
    forward_from: str = DEFAULT_FORWARD_FROM,
    target: str = f"{STABLE}/",
) -> str:
    """Return the root ``404.html`` that forwards former and version-less URLs.

    GitHub Pages serves this page for every missing path below the base path.
    A path below ``forward_from`` (where the site was served before it was
    published per version) or a path outside every version folder is
    forwarded to the same page in ``target``, keeping the query string and
    the fragment. A missing page inside a version folder is not forwarded, so
    a forward never loops.

    Args:
        base_path: The URL path the gh-pages root is served at, starting and
            ending with ``/``.
        title: The name of the documentation, in the page's title.
        forward_from: The folder of a former unversioned site, relative to
            the base path and ending with ``/``; empty when there was none.
        target: The folder to forward to, with a trailing slash: ``stable/``,
            or ``dev/`` while no release is published.

    Returns:
        The HTML text of the page.
    """
    start = html.escape(f"{base_path}{target}", quote=True)
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        f"<title>Page not found · {html.escape(title)}</title>\n"
        "<script>\n"
        "(function () {\n"
        f"  var base = {json.dumps(base_path)};\n"
        "  var path = window.location.pathname;\n"
        "  if (path.indexOf(base) !== 0) { return; }\n"
        "  var rest = path.slice(base.length);\n"
        f"  var former = {json.dumps(forward_from)};\n"
        "  if (former && rest.indexOf(former) === 0) {\n"
        "    rest = rest.slice(former.length);\n"
        "  } else if (/^(dev|stable|\\d+\\.\\d+\\.\\d+)(\\/|$)/.test(rest)) {\n"
        "    return;\n"
        "  }\n"
        "  window.location.replace(\n"
        f"    base + {json.dumps(target)} + rest + window.location.search"
        " + window.location.hash\n"
        "  );\n"
        "})();\n"
        "</script>\n"
        "</head>\n"
        "<body>\n"
        "<p>This page does not exist. The documentation is at\n"
        f'<a href="{start}">{start}</a>.</p>\n'
        "</body>\n"
        "</html>\n"
    )


def inject_script_tag(text: str, tag: str) -> str | None:
    """Add the version script tag to an HTML page that does not carry it yet.

    The tag goes before the last ``</body>``, or at the end of a page without
    one.

    Args:
        text: The HTML text of the page.
        tag: The complete ``<script ...></script>`` element.

    Returns:
        The new text, or None when the page already loads ``versions.js``.
    """
    if _SCRIPT_TAG_PATTERN.search(text):
        return None
    ends = list(_BODY_END_PATTERN.finditer(text))
    if not ends:
        return text + tag + "\n"
    position = ends[-1].start()
    return text[:position] + tag + "\n" + text[position:]


def inject_folder(folder: Path, tag: str) -> int:
    """Add the version script tag to every HTML page below a folder.

    Args:
        folder: The version folder.
        tag: The complete ``<script ...></script>`` element.

    Returns:
        The number of pages changed.
    """
    changed = 0
    for page in sorted(folder.rglob("*.html")):
        text = page.read_text(encoding="utf-8")
        new_text = inject_script_tag(text, tag)
        if new_text is not None:
            page.write_text(new_text, encoding="utf-8")
            changed += 1
    return changed


def replace_folder(source: Path, target: Path) -> None:
    """Replace a folder by a copy of another one.

    Args:
        source: The folder to copy.
        target: The folder to replace; it need not exist.
    """
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target, ignore=shutil.ignore_patterns(".git", ".quarto"))


def publish(
    site: Path,
    version: str,
    pages: Path,
    base_path: str = DEFAULT_BASE_PATH,
    title: str = DEFAULT_TITLE,
    forward_from: str = DEFAULT_FORWARD_FROM,
) -> list[str]:
    """Publish a built site as one version folder of a gh-pages tree.

    Args:
        site: The built site (the folder holding its ``index.html``).
        version: ``dev`` or a release number ``X.Y.Z``.
        pages: Root of the gh-pages tree (a checkout of the branch).
        base_path: The URL path the gh-pages root is served at.
        title: The name of the documentation, in the titles of the root pages.
        forward_from: The folder of a former unversioned site whose addresses
            ``404.html`` forwards, relative to the base path; empty for none.

    Returns:
        The folders written, in the order they were written.

    Raises:
        ValueError: If ``version`` is neither ``dev`` nor ``X.Y.Z``, if
            ``site`` holds no ``index.html`` or lies inside the folder it
            would replace, if ``base_path`` does not start and end with
            ``/``, or if a non-empty ``forward_from`` starts with ``/`` or
            does not end with it.
    """
    if version != DEV and release_key(version) is None:
        raise ValueError(f"Version must be 'dev' or X.Y.Z, not {version!r}.")
    if site.resolve().is_relative_to((pages / version).resolve()):
        raise ValueError(f"{site} lies inside the folder it would replace.")
    if not (site / "index.html").is_file():
        raise ValueError(f"{site} holds no index.html; is it a built site?")
    if not base_path.startswith("/") or not base_path.endswith("/"):
        raise ValueError(
            f"The base path must start and end with '/', not {base_path!r}."
        )
    if forward_from and (
        forward_from.startswith("/") or not forward_from.endswith("/")
    ):
        raise ValueError(
            f"The folder to forward from must be relative and end with '/', "
            f"not {forward_from!r}."
        )

    tag = (
        f'<script src="{html.escape(base_path, quote=True)}versions.js" defer></script>'
    )
    replace_folder(site, pages / version)
    written = [version]
    if version != DEV and releases_in(pages)[0] == version:
        replace_folder(pages / version, pages / STABLE)
        written.append(STABLE)
    for name in written:
        changed = inject_folder(pages / name, tag)
        print(f"{name}/: script tag added to {changed} page(s)")

    entries = versions_list(pages)
    (pages / "versions.json").write_text(
        json.dumps(entries, indent=2) + "\n", encoding="utf-8"
    )
    redirect = f"{STABLE}/" if (pages / STABLE).is_dir() else f"{DEV}/"
    (pages / "index.html").write_text(redirect_page(redirect, title), encoding="utf-8")
    (pages / "404.html").write_text(
        not_found_page(base_path, title, forward_from, redirect), encoding="utf-8"
    )
    shutil.copyfile(VERSIONS_JS, pages / "versions.js")
    (pages / ".nojekyll").touch()
    return written


def main(argv: list[str] | None = None) -> int:
    """Run the command line interface.

    Args:
        argv: The arguments; ``sys.argv[1:]`` when None.

    Returns:
        The exit status.
    """
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument(
        "site", type=Path, help="the built site (holding its index.html)"
    )
    parser.add_argument(
        "--version", required=True, help="'dev' or a release number X.Y.Z"
    )
    parser.add_argument(
        "--pages",
        type=Path,
        required=True,
        help="root of the gh-pages checkout to write into",
    )
    parser.add_argument(
        "--base-path",
        default=DEFAULT_BASE_PATH,
        help="URL path the gh-pages root is served at (default: %(default)s)",
    )
    parser.add_argument(
        "--title",
        default=DEFAULT_TITLE,
        help="name of the documentation in the root pages (default: %(default)s)",
    )
    parser.add_argument(
        "--forward-from",
        default=DEFAULT_FORWARD_FROM,
        help="folder of a former unversioned site whose addresses 404.html "
        "forwards; empty for none (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    try:
        written = publish(
            args.site,
            args.version,
            args.pages,
            args.base_path,
            args.title,
            args.forward_from,
        )
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Published {', '.join(f'{name}/' for name in written)} into {args.pages}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
