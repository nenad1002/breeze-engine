"""Opt-in, bounded real-browser smoke tests; no native model is loaded."""
import os
from pathlib import Path
import socket
import threading

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("BREEZE_BROWSER_TESTS") != "1",
                                reason="Set BREEZE_BROWSER_TESTS=1 for local browser checks")


@pytest.fixture
def browser_app():
    import uvicorn
    from breeze.server import create_app
    from breeze.service_config import ServiceSettings
    from breeze.service_backend import DemoBackend

    key = "browser-test-key-not-for-deployment-12345"
    entered = threading.Event()

    class PreviewBackend(DemoBackend):
        def generate(self, request, words, emit, cancelled, deadline):
            if request.messages[-1].content == "Wait for cancellation":
                entered.set()
                cancelled.wait(timeout=4)
            return super().generate(request, words, emit, cancelled, deadline)

    application = create_app(ServiceSettings(demo=True, api_key=key), PreviewBackend)
    started = threading.Event()

    class TestServer(uvicorn.Server):
        async def startup(self, sockets=None):
            await super().startup(sockets)
            started.set()

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        server = TestServer(uvicorn.Config(application, access_log=False, log_level="warning"))
        thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
        thread.start()
        try:
            assert started.wait(timeout=10), "Local test server failed to start"
            yield f"http://127.0.0.1:{port}", key, entered
        finally:
            server.should_exit = True
            thread.join(timeout=10)
            assert not thread.is_alive(), "Test server did not stop"


def test_real_browser_workflow(browser_app):
    from playwright.sync_api import sync_playwright, expect

    url, key, entered = browser_app
    artifacts = Path(__file__).resolve().parents[1] / "results" / "product-browser"
    artifacts.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1080})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.add_init_script("window.cspViolations = []; document.addEventListener('securitypolicyviolation', event => window.cspViolations.push(event.violatedDirective));")
        page.goto(url)
        expect(page.get_by_role("dialog", name="Your workspace starts here.")).to_be_visible()
        page.get_by_label("API key", exact=False).fill(key)
        page.get_by_role("button", name="Connect to workspace", exact=True).click()
        expect(page.locator("#connection-label")).to_have_text("Connected")
        expect(page.locator("#demo-banner")).to_be_visible()
        page.screenshot(path=str(artifacts / "desktop.png"), full_page=True)
        page.get_by_role("button", name="Find the next steps", exact=False).click()
        assert "Northstar" in page.locator("#prompt").input_value()
        assert page.locator("#messages").locator("article").count() == 0
        page.get_by_role("button", name="Send message", exact=False).click()
        expect(page.locator("#request-state")).to_have_text("Response complete")
        expect(page.locator(".message.assistant .message-content")).to_contain_text("scripted preview")
        expect(page.get_by_role("button", name="Copy response", exact=True)).to_be_visible()
        assert page.locator("#metric-rate").is_hidden()
        assert page.evaluate("Object.keys(localStorage).length + Object.keys(sessionStorage).length") == 0
        page.get_by_role("link", name="Connection & API", exact=False).click()
        assert key not in page.locator("#api-example").inner_text()
        page.get_by_role("button", name="View authenticated API schema", exact=False).click()
        expect(page.get_by_role("dialog", name="Service API schema")).to_be_visible()
        expect(page.locator("#schema-output")).to_contain_text("/v1/chat/completions")
        page.get_by_role("button", name="Close API schema", exact=True).click()
        page.get_by_role("link", name="Workbench", exact=False).click()
        page.locator("#prompt").fill("Wait for cancellation")
        page.get_by_role("button", name="Send message", exact=False).click()
        assert entered.wait(timeout=5)
        page.get_by_role("button", name="Stop", exact=False).click()
        expect(page.locator("#request-state")).to_have_text("Stopped · partial excluded")
        expect(page.get_by_role("button", name="Edit & retry", exact=True)).to_be_visible()
        page.wait_for_function("async key => { const response = await fetch('/api/status', {headers: {Authorization: 'Bearer ' + key}}); const status = await response.json(); return status.requests.cancelled === 1 && status.queue.active === 0; }", arg=key)
        page.set_viewport_size({"width": 390, "height": 844})
        page.evaluate("window.scrollTo(0, 0)")
        page.screenshot(path=str(artifacts / "mobile.png"), full_page=True)
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"), "Mobile horizontal overflow"
        page.get_by_role("link", name="Connection & API", exact=False).click()
        page.get_by_role("button", name="Disconnect", exact=True).click()
        expect(page.locator("#connection-label")).to_have_text("Disconnected")
        assert page.locator("#messages").locator("article").count() == 0
        assert not page.evaluate("window.cspViolations"), "Unexpected content-security-policy violations"
        assert not errors, errors
        browser.close()