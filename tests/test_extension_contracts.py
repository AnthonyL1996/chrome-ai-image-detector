from __future__ import annotations

from html.parser import HTMLParser
import hashlib
import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
EXTENSION = ROOT / "extension"


class _PopupParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.buttons: list[dict[str, str | None]] = []
        self.status_regions: list[dict[str, str | None]] = []
        self.inline_scripts = 0
        self.inline_styles = 0
        self.event_handlers: list[str] = []
        self.external_scripts: list[str] = []
        self.stylesheets: list[str] = []

    def handle_starttag(
        self, tag: str, attributes: list[tuple[str, str | None]]
    ) -> None:
        values = dict(attributes)
        self.event_handlers.extend(name for name in values if name.startswith("on"))
        if tag == "button":
            self.buttons.append(values)
        if values.get("role") == "status":
            self.status_regions.append(values)
        if tag == "script":
            if values.get("src"):
                self.external_scripts.append(values["src"] or "")
            else:
                self.inline_scripts += 1
        if tag == "style":
            self.inline_styles += 1
        if tag == "link" and values.get("rel") == "stylesheet":
            self.stylesheets.append(values.get("href") or "")


class ExtensionContracts(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = json.loads((EXTENSION / "manifest.json").read_bytes())

    def test_manifest_is_minimal_mv3_without_persistent_host_access(self) -> None:
        self.assertEqual(self.manifest["manifest_version"], 3)
        self.assertEqual(set(self.manifest["permissions"]), {"activeTab", "scripting"})
        self.assertNotIn("host_permissions", self.manifest)
        self.assertNotIn("optional_host_permissions", self.manifest)
        self.assertNotIn("content_scripts", self.manifest)
        self.assertNotIn("externally_connectable", self.manifest)
        self.assertEqual(
            self.manifest["content_security_policy"]["extension_pages"],
            "script-src 'self' 'wasm-unsafe-eval'; object-src 'self'",
        )
        self.assertNotIn(
            "unsafe-eval",
            self.manifest["content_security_policy"]["extension_pages"].split(),
        )

    def test_manifest_declares_popup_and_module_service_worker(self) -> None:
        self.assertEqual(self.manifest["action"]["default_popup"], "popup.html")
        self.assertEqual(
            self.manifest["background"],
            {"service_worker": "service-worker.js", "type": "module"},
        )
        self.assertEqual(self.manifest["minimum_chrome_version"], "120")

    def test_popup_is_csp_safe_semantic_and_keyboard_accessible(self) -> None:
        parser = _PopupParser()
        parser.feed((EXTENSION / "popup.html").read_text(encoding="utf-8"))
        self.assertEqual(parser.inline_scripts, 0)
        self.assertEqual(parser.inline_styles, 0)
        self.assertEqual(parser.event_handlers, [])
        self.assertEqual(parser.external_scripts, ["popup.js"])
        self.assertEqual(parser.stylesheets, ["popup.css"])
        self.assertTrue(
            any(
                button.get("id") == "scan-page" and button.get("type") == "button"
                for button in parser.buttons
            )
        )
        self.assertTrue(
            any(region.get("aria-live") == "polite" for region in parser.status_regions)
        )

    def test_service_worker_injects_only_after_explicit_scan(self) -> None:
        source = (EXTENSION / "service-worker.js").read_text(encoding="utf-8")
        self.assertIn('case "SCAN_ACTIVE_TAB"', source)
        self.assertIn('files: ["content.js"]', source)
        self.assertIn('type: "SCAN_PAGE"', source)
        self.assertIn('case "SCORE_IMAGES"', source)
        self.assertNotIn("chrome.tabs.onUpdated", source)
        self.assertNotIn("chrome.webNavigation", source)

    def test_runtime_adapter_is_explicitly_local_and_unavailable(self) -> None:
        source = (EXTENSION / "runtime" / "model-runtime.mjs").read_text(
            encoding="utf-8"
        )
        self.assertIn("createModelRuntime", source)
        self.assertIn("localOnly: true", source)
        self.assertIn('code: "MODEL_RUNTIME_UNAVAILABLE"', source)
        self.assertNotRegex(source, r"(?i)https?://")

    def test_onnx_backend_is_local_and_hash_checked(self) -> None:
        source = (EXTENSION / "runtime" / "onnx-backend.mjs").read_text(
            encoding="utf-8"
        )
        self.assertIn('executionProviders: ["wasm"]', source)
        self.assertIn("model_sha256", source)
        self.assertIn("localOnly: true", source)
        self.assertIn('redirect: "error"', source)

    def test_vendored_onnx_runtime_assets_are_pinned(self) -> None:
        expected = {
            "ort.wasm.bundle.min.mjs": (
                "1db5e1c5cd2b860eed85e6eeff23e2aaa7cffcc407f67093bcc888f631b94ba9"
            ),
            "ort-wasm-simd-threaded.mjs": (
                "0a1e718d99c41b22c21f2520ff4f9e883a6b5533856e398d21816ee8eb8185d3"
            ),
            "ort-wasm-simd-threaded.wasm": (
                "d1ab1b94b16a65b29d710d0b587b29e7bed336827577623913479b8afe8113e6"
            ),
        }
        vendor = EXTENSION / "runtime" / "vendor"
        self.assertEqual(
            {path.name for path in vendor.iterdir()}, set(expected)
        )
        for name, digest in expected.items():
            with self.subTest(name=name):
                self.assertEqual(
                    hashlib.sha256((vendor / name).read_bytes()).hexdigest(), digest
                )

    def test_onnx_runtime_wrapper_forces_local_single_threaded_wasm(self) -> None:
        source = (EXTENSION / "runtime" / "ort-runtime.mjs").read_text(
            encoding="utf-8"
        )
        self.assertIn('chrome.runtime.getURL("runtime/vendor/")', source)
        self.assertIn("ort.env.wasm.numThreads = 1", source)
        self.assertIn("ort.env.wasm.proxy = false", source)

    def test_extension_javascript_has_no_remote_or_dynamic_code_execution(self) -> None:
        forbidden = re.compile(
            r"\b(?:eval|Function|fetch|XMLHttpRequest|WebSocket|sendBeacon)\s*\("
        )
        javascript = sorted(EXTENSION.rglob("*.js")) + sorted(EXTENSION.rglob("*.mjs"))
        self.assertGreaterEqual(len(javascript), 4)
        for path in javascript:
            with self.subTest(path=path.relative_to(ROOT)):
                source = path.read_text(encoding="utf-8")
                if path.parts[-3:-1] == ("runtime", "vendor"):
                    # ONNX Runtime Web is vendored under its upstream MIT
                    # license; its internal WASM loader necessarily contains
                    # fetch calls. The extension-owned adapter is checked
                    # below and remains free of remote code execution.
                    continue
                self.assertNotRegex(source, forbidden)
                self.assertNotIn("unsafe-eval", source)


if __name__ == "__main__":
    unittest.main()
