"""Small local HTTP/tar fixtures; never download models or use a board."""
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
from pathlib import Path
import tarfile
import tempfile
import threading
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/fetch_assets.py"
SPEC = importlib.util.spec_from_file_location("fetch_assets", SCRIPT)
FETCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FETCH)


def archive_bytes(files, unsafe_link=False):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name, content in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
        if unsafe_link:
            member = tarfile.TarInfo("models/link")
            member.type = tarfile.SYMTYPE
            member.linkname = "../../outside"
            archive.addfile(member)
    return stream.getvalue()


class FetchAssetsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="release-assets-test-")
        self.root = Path(self.temp.name)
        self.payloads = {}
        self.requests = []
        self.ignore_range = False
        self.bad_range = False
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                content = fixture.payloads[self.path]
                request_range = self.headers.get("Range")
                fixture.requests.append(request_range)
                if request_range and not fixture.ignore_range:
                    start = int(request_range.removeprefix("bytes=").removesuffix("-"))
                    self.send_response(206)
                    reported_start = start + 1 if fixture.bad_range else start
                    self.send_header("Content-Range", f"bytes {reported_start}-{len(content)-1}/{len(content)}")
                    content = content[start:]
                else:
                    self.send_response(200)
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def part(self, name, content):
        self.payloads["/" + name] = content
        return {"name": name, "size_bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
                "url": f"http://127.0.0.1:{self.server.server_port}/{name}"}

    def test_resume_checks_content_range(self):
        content = b"resume-content" * 100
        part = self.part("part01", content)
        target = self.root / "part01"
        target.with_name("part01.part").write_bytes(content[:37])
        FETCH.download_part(part, target)
        self.assertEqual(self.requests, ["bytes=37-"])
        self.assertEqual(target.read_bytes(), content)

    def test_ignored_range_restarts_instead_of_appending(self):
        self.ignore_range = True
        content = b"whole-response" * 30
        target = self.root / "part01"
        target.with_name("part01.part").write_bytes(content[:19])
        FETCH.download_part(self.part("part01", content), target)
        self.assertEqual(target.read_bytes(), content)

    def test_wrong_content_range_rejected(self):
        self.bad_range = True
        content = b"range-content" * 20
        target = self.root / "part01"
        partial = target.with_name("part01.part")
        partial.write_bytes(content[:17])
        with self.assertRaisesRegex(ValueError, "Content-Range"):
            FETCH.download_part(self.part("part01", content), target)
        self.assertEqual(partial.read_bytes(), content[:17])
        self.assertFalse(target.exists())

    def test_corrupt_hash_rejected(self):
        part = self.part("bad", b"downloaded bytes")
        part["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            FETCH.download_part(part, self.root / "bad")
        self.assertFalse((self.root / "bad").exists())

    def test_multiple_parts_in_manifest_order_and_existing_conflict(self):
        files = {"models/example.bin": b"weights\x00\xff" * 150, "data/readme.txt": b"fixture"}
        blob = archive_bytes(files)
        cut = len(blob) // 2
        package = {"id": "fixture", "archive_format": "tar.gz",
                   "extracted_size_bytes": sum(map(len, files.values())),
                   "parts": [self.part("z-first", blob[:cut]), self.part("a-second", blob[cut:])]}
        output = self.root / "assets"
        self.assertEqual(FETCH.fetch_package(package, output), 2)
        for name, content in files.items():
            self.assertEqual((output / name).read_bytes(), content)
        FETCH.fetch_package(package, output)
        (output / "models/example.bin").write_bytes(b"keep my different asset")
        with self.assertRaisesRegex(ValueError, "Existing asset differs"):
            FETCH.fetch_package(package, output)
        self.assertEqual((output / "models/example.bin").read_bytes(), b"keep my different asset")

    def test_archive_traversal_rejected_before_final_writes(self):
        for malicious in ("../outside", "/tmp/outside"):
            with self.subTest(path=malicious):
                blob = archive_bytes({"models/valid.bin": b"valid", malicious: b"bad"})
                path = self.root / "bad.tar.gz"
                path.write_bytes(blob)
                output = self.root / "assets"
                with self.assertRaisesRegex(ValueError, "Unsafe archive path"):
                    FETCH.unpack_parts([path], {"archive_format": "tar.gz"}, output)
                self.assertFalse((output / "models/valid.bin").exists())
                self.assertFalse((self.root / "outside").exists())

    def test_symlink_member_and_existing_symlink_rejected(self):
        path = self.root / "link.tar.gz"
        path.write_bytes(archive_bytes({"models/valid.bin": b"valid"}, unsafe_link=True))
        output = self.root / "assets"
        with self.assertRaisesRegex(ValueError, "Only regular"):
            FETCH.unpack_parts([path], {"archive_format": "tar.gz"}, output)
        self.assertFalse((output / "models/valid.bin").exists())
        outside = self.root / "outside"
        outside.mkdir()
        (output / "models").symlink_to(outside, target_is_directory=True)
        path.write_bytes(archive_bytes({"models/valid.bin": b"valid"}))
        with self.assertRaisesRegex(ValueError, "symlink destination"):
            FETCH.unpack_parts([path], {"archive_format": "tar.gz"}, output)
        self.assertFalse((outside / "valid.bin").exists())


if __name__ == "__main__":
    unittest.main()
