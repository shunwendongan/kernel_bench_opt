#!/usr/bin/env python3
"""Export the repository README and its local images as one offline HTML file.

Requires Pandoc on PATH. The exporter never fetches images or runs experiment
commands; links to other repository documents remain normal GitHub links.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import mimetypes
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unicodedata
from urllib.parse import quote, unquote, urlsplit, urlunsplit
import xml.etree.ElementTree as ET

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPOSITORY = "https://github.com/shunwendongan/kernel_bench_opt"
DEFAULT_BRANCH = "Qwen2.5-Omni-3B_VCB-Bench"


class ExportError(ValueError):
    pass


def pandoc(command: str, source: str, source_format: str, target: str) -> str:
    try:
        result = subprocess.run(
            [command, "--from", source_format, "--to", target, "--wrap=none"],
            input=source, text=True, capture_output=True, check=False,
        )
    except FileNotFoundError as exc:
        raise ExportError("Pandoc is required; install it or pass --pandoc PATH") from exc
    if result.returncode:
        raise ExportError("Pandoc conversion failed: " + result.stderr.strip())
    return result.stdout


def inline_text(nodes: list[dict]) -> str:
    """Plain captions/heading labels from the Pandoc inline representation."""
    chunks = []
    for node in nodes:
        kind, content = node.get("t"), node.get("c")
        if kind == "Str":
            chunks.append(content)
        elif kind in {"Space", "SoftBreak", "LineBreak"}:
            chunks.append(" ")
        elif kind in {"Code", "Math"}:
            chunks.append(content[1])
        elif kind in {"Link", "Image", "Span"}:
            chunks.append(inline_text(content[1]))
        elif kind in {"Emph", "Strong", "Strikeout", "SmallCaps", "Superscript", "Subscript"}:
            chunks.append(inline_text(content))
        elif kind == "Quoted":
            chunks.append(inline_text(content[1]))
    return "".join(chunks)


def github_slug(label: str) -> str:
    cleaned = "".join(
        c for c in label.lower()
        if c in "-_" or c.isspace() or unicodedata.category(c)[0] in {"L", "N", "M"}
    )
    return re.sub(r"\s", "-", cleaned)


def header_aliases(document: dict) -> dict[str, str]:
    aliases: dict[str, str] = {}
    seen: dict[str, int] = {}

    def visit(value):
        if isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            if value.get("t") == "Header":
                _, attributes, label = value["c"]
                base = github_slug(inline_text(label))
                index = seen.get(base, 0)
                seen[base] = index + 1
                alias = base if index == 0 else f"{base}-{index}"
                if alias and attributes[0]:
                    aliases[alias] = attributes[0]
            for item in value.values():
                visit(item)

    visit(document)
    return aliases


def repository_link(repository: str, branch: str, path: Path, root: Path, fragment="", query="") -> str:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ExportError("A local README link points outside the repository") from exc
    mode = "tree" if path.is_dir() else "blob"
    url = f"{repository}/{mode}/{quote(branch, safe='')}/{quote(relative.as_posix(), safe='/')}"
    return urlunsplit((*urlsplit(url)[:3], query, fragment))


def embed_image(source: str, source_dir: Path, asset_hashes: dict[str, str]) -> tuple[str, str, str]:
    parsed = urlsplit(source)
    if parsed.scheme or parsed.netloc:
        raise ExportError("Images must be local files; remote or existing data URLs are not fetched")
    path = (source_dir / unquote(parsed.path)).resolve()
    try:
        relative = path.relative_to(source_dir)
    except ValueError as exc:
        raise ExportError("An image is outside the README directory") from exc
    if not path.is_file():
        raise ExportError(f"Missing README image: {relative.as_posix()}")
    content = path.read_bytes()
    suffix = path.suffix.lower()
    allowed = {".svg", ".png", ".jpg", ".jpeg", ".webp", ".gif"}
    if suffix not in allowed:
        raise ExportError(f"Unsupported image format: {suffix}")
    width = height = ""
    if suffix == ".svg":
        try:
            svg = ET.fromstring(content)
        except ET.ParseError as exc:
            raise ExportError(f"Invalid SVG: {relative.as_posix()}") from exc
        for element in svg.iter():
            if element.tag.rsplit("}", 1)[-1] in {"script", "foreignObject"}:
                raise ExportError("SVG images must be static and self-contained")
            for name, value in element.attrib.items():
                if name.rsplit("}", 1)[-1] == "href" and not value.startswith("#"):
                    raise ExportError("SVG images must not reference external resources")
        svg_text = content.decode("utf-8")
        if "@import" in svg_text or re.search(r"url\(\s*['\"]?(?!#)[^)]", svg_text, re.I):
            raise ExportError("SVG styles must not load external resources")
        viewbox = svg.get("viewBox", "").split()
        if len(viewbox) == 4:
            width, height = viewbox[2:]
        mime = "image/svg+xml"
    else:
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    asset_hashes[relative.as_posix()] = hashlib.sha256(content).hexdigest()
    return f"data:{mime};base64,{base64.b64encode(content).decode('ascii')}", width, height


def transform_document(document: dict, source: Path, output: Path, repository: str, branch: str) -> tuple[dict, dict]:
    root = source.parent
    aliases = header_aliases(document)
    asset_hashes: dict[str, str] = {}

    def link_target(target: str) -> str:
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc:
            return target
        if not parsed.path:
            fragment = unquote(parsed.fragment)
            return "#" + quote(aliases.get(fragment, fragment), safe="-_") if fragment else target
        path = (root / unquote(parsed.path)).resolve()
        if path == output or path == root / "README.html":
            return "#top"
        return repository_link(repository, branch, path, root, parsed.fragment, parsed.query)

    def visit(value):
        if isinstance(value, list):
            return [visit(item) for item in value]
        if not isinstance(value, dict):
            return value
        kind = value.get("t")
        if kind in {"Para", "Plain"} and len(value.get("c", [])) == 1 and value["c"][0].get("t") == "Image":
            attributes, caption, target = value["c"][0]["c"]
            uri, width, height = embed_image(target[0], root, asset_hashes)
            label = html.escape(inline_text(caption))
            dimensions = f' width="{html.escape(width)}" height="{html.escape(height)}"' if width and height else ""
            figure = (
                '<figure class="readme-figure"><div class="figure-scroll" tabindex="0">'
                f'<img src="{uri}" alt="{label}"{dimensions} loading="lazy" decoding="async">'
                f'</div><figcaption>{label}</figcaption></figure>'
            )
            return {"t": "RawBlock", "c": ["html", figure]}
        if kind == "Image":
            attributes, caption, target = value["c"]
            uri, _, _ = embed_image(target[0], root, asset_hashes)
            return {"t": "Image", "c": [attributes, visit(caption), [uri, target[1]]]}
        if kind == "Link":
            attributes, label, target = value["c"]
            return {"t": "Link", "c": [attributes, visit(label), [link_target(target[0]), target[1]]]}
        if kind in {"RawBlock", "RawInline"} and value.get("c", [None])[0] == "html":
            raw = value["c"][1]
            if re.search(r"<(?:script|style|link|iframe|object|embed|img)\b", raw, re.I):
                raise ExportError("Raw HTML cannot load resources; use Markdown images for embedding")
            raw = re.sub(
                r'\bhref=([\'"])(.*?)\1',
                lambda match: 'href="' + html.escape(link_target(html.unescape(match[2])), quote=True) + '"',
                raw,
                flags=re.I,
            )
            return {"t": kind, "c": ["html", raw]}
        return {key: visit(item) for key, item in value.items()}

    return visit(document), asset_hashes


def export(args) -> dict:
    source = args.readme.expanduser().resolve()
    output = args.output.expanduser().absolute()
    if not source.is_file():
        raise ExportError("The README source does not exist")
    if output.suffix.lower() not in {".html", ".htm"}:
        raise ExportError("--output must end in .html or .htm")
    if output.is_symlink() or output.resolve() == source:
        raise ExportError("The output must be a separate regular HTML file")
    parsed_repository = urlsplit(args.repository)
    if parsed_repository.scheme != "https" or not parsed_repository.netloc or parsed_repository.query or parsed_repository.fragment:
        raise ExportError("--repository must be an HTTPS repository URL without query or fragment")
    repository = args.repository.rstrip("/")
    raw = source.read_bytes()
    source_hash = hashlib.sha256(raw).hexdigest()
    ast = json.loads(pandoc(args.pandoc, raw.decode("utf-8"), "gfm", "json"))
    transformed, images = transform_document(ast, source, output.resolve(), repository, args.branch)
    body = pandoc(args.pandoc, json.dumps(transformed, ensure_ascii=False), "json", "html5")
    body = re.sub(r"<table\b([^>]*)>", r'<div class="table-scroll" tabindex="0"><table\1>', body)
    body = body.replace("</table>", "</table></div>")
    css = (REPO_ROOT / "docs/readme-export.css").read_text(encoding="utf-8")
    source_url = repository_link(repository, args.branch, source, source.parent)
    metadata = html.escape(json.dumps({"readme_sha256": source_hash, "images": images}, ensure_ascii=False), quote=True)
    page = (
        '<!doctype html>\n<html lang="zh-CN" data-register="paper"><head>\n'
        '<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">\n'
        '<meta name="color-scheme" content="light">\n'
        f'<meta name="readme-sha256" content="{source_hash}">\n'
        f'<meta name="embedded-assets" content="{metadata}">\n'
        '<title>kernel_bench_opt · 完整图文 README</title>\n'
        f'<style>\n{css}\n</style></head><body><main id="top">\n'
        '<div class="offline-note">单文件阅读版 · 图形和样式已内嵌，可离线打开；其他文档链接指向 GitHub。</div>\n'
        + body
        + '<footer class="export-footer"><p>本文由 README.md 导出，保留完整正文与手绘图。'
        f'共内嵌 {len(images)} 张图。<a href="{html.escape(source_url, quote=True)}">查看仓库中的 README</a>。</p>'
        f'<p>来源 SHA256：<code>{source_hash}</code></p></footer>\n'
        '</main></body></html>\n'
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output.parent, prefix=".readme-", suffix=".html", delete=False) as handle:
            temp_path = Path(handle.name)
            handle.write(page)
        os.replace(temp_path, output)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
    return {"output": output.name, "readme_sha256": source_hash, "embedded_images": len(images), "size_bytes": len(page.encode("utf-8"))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readme", type=Path, default=REPO_ROOT / "README.md")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "README.html")
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--pandoc", default="pandoc")
    args = parser.parse_args()
    try:
        result = export(args)
    except (ExportError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        print(f"export_readme: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
