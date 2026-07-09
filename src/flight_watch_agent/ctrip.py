from __future__ import annotations

import gzip
import json
import os
import threading
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import FlightEvidence, FlightSearchIntent, SearchResult


CTRIP_SCHEME = "ctrip-selenium"
DEFAULT_CTRIP_COOKIES_FILE = "data/ctrip_cookies.json"
REQUIRED_CTRIP_COOKIES = ["AHeadUserInfo", "DUID", "IsNonUser", "_udl", "cticket", "login_type", "login_uid"]


class CtripRouteSearchTool:
    def search(self, query: str) -> list[SearchResult]:
        intent = _intent_from_query(query)
        if intent is None:
            return []
        url = build_ctrip_selenium_url(intent)
        return [
            SearchResult(
                title=f"Ctrip Selenium route {intent.origin} to {intent.destination}",
                url=url,
                snippet="Constructed Ctrip SeleniumWire crawl task.",
                source_name="flights.ctrip.com",
            )
        ]


class CtripSeleniumWirePageExtractor:
    def __init__(
        self,
        *,
        browser: str = "edge",
        headless: bool = True,
        timeout_seconds: int = 30,
        direct_only: bool = False,
        max_results: int = 5,
        login_allowed: bool = False,
        accounts: list[str] | None = None,
        passwords: list[str] | None = None,
        cookies_file: str | os.PathLike[str] = DEFAULT_CTRIP_COOKIES_FILE,
        login_wait_seconds: int = 300,
        manual_verification_wait_seconds: int = 0,
    ) -> None:
        self.browser = browser
        self.headless = headless
        self.timeout_seconds = timeout_seconds
        self.direct_only = direct_only
        self.max_results = max_results
        self.login_allowed = login_allowed
        self.accounts = accounts or []
        self.passwords = passwords or []
        self.cookies_file = Path(cookies_file)
        self.login_wait_seconds = login_wait_seconds
        self.manual_verification_wait_seconds = manual_verification_wait_seconds

    def supports(self, url: str) -> bool:
        return urllib.parse.urlparse(url).scheme == CTRIP_SCHEME

    def extract(self, url: str) -> list[FlightEvidence]:
        intent = parse_ctrip_selenium_url(url)
        driver = _init_seleniumwire_driver(browser=self.browser, headless=self.headless)
        errors: list[str] = []
        fallback_evidence: list[FlightEvidence] = []
        try:
            if self.login_allowed:
                _CtripLoginSession(
                    accounts=self.accounts,
                    passwords=self.passwords,
                    cookies_file=self.cookies_file,
                    timeout_seconds=self.timeout_seconds,
                    login_wait_seconds=self.login_wait_seconds,
                ).ensure_login(driver)
            for search_url in _build_ctrip_search_urls(intent):
                try:
                    del driver.requests
                except AttributeError:
                    pass
                try:
                    driver.get(search_url)
                    payload = _wait_for_ctrip_search_payload(
                        driver,
                        timeout_seconds=self.timeout_seconds,
                        manual_verification_wait_seconds=self.manual_verification_wait_seconds,
                    )
                    evidence = parse_ctrip_batch_search_payload(
                        payload,
                        intent,
                        source_url=search_url,
                        direct_only=self.direct_only,
                        max_results=self.max_results,
                    )
                    if not evidence:
                        evidence = _retry_after_manual_verification_if_present(
                            driver,
                            intent,
                            source_url=search_url,
                            direct_only=self.direct_only,
                            max_results=self.max_results,
                            timeout_seconds=self.timeout_seconds,
                            manual_verification_wait_seconds=self.manual_verification_wait_seconds,
                        )
                    if evidence:
                        if _evidence_satisfies_requested_time(evidence, intent):
                            return evidence
                        if not fallback_evidence:
                            fallback_evidence = evidence
                        errors.append(f"{search_url}:NoTimePreferenceMatch:{intent.time_preference}")
                        continue
                    errors.append(f"{search_url}:NoFlightEvidence:batchSearch returned no parsable itineraries")
                except Exception as exc:
                    errors.append(f"{search_url}:{type(exc).__name__}:{str(exc).split('Stacktrace:')[0]}")
                    continue
            try:
                del driver.requests
            except AttributeError:
                pass
            try:
                search_url = "https://flights.ctrip.com/online/channel/domestic"
                _drive_ctrip_homepage_search(driver, intent, timeout_seconds=self.timeout_seconds)
                payload = _wait_for_ctrip_search_payload(
                    driver,
                    timeout_seconds=self.timeout_seconds,
                    manual_verification_wait_seconds=self.manual_verification_wait_seconds,
                )
                evidence = parse_ctrip_batch_search_payload(
                    payload,
                    intent,
                    source_url=search_url,
                    direct_only=self.direct_only,
                    max_results=self.max_results,
                )
                if not evidence:
                    evidence = _retry_after_manual_verification_if_present(
                        driver,
                        intent,
                        source_url=search_url,
                        direct_only=self.direct_only,
                        max_results=self.max_results,
                        timeout_seconds=self.timeout_seconds,
                        manual_verification_wait_seconds=self.manual_verification_wait_seconds,
                    )
                if evidence:
                    if _evidence_satisfies_requested_time(evidence, intent):
                        return evidence
                    if not fallback_evidence:
                        fallback_evidence = evidence
                    errors.append(f"homepage_ui:NoTimePreferenceMatch:{intent.time_preference}")
                else:
                    errors.append("homepage_ui:NoFlightEvidence:batchSearch returned no parsable itineraries")
            except Exception as exc:
                errors.append(f"homepage_ui:{type(exc).__name__}:{str(exc).split('Stacktrace:')[0]}")
            if fallback_evidence:
                return fallback_evidence
            if errors:
                raise RuntimeError("Ctrip SeleniumWire extraction failed; attempts=" + " | ".join(errors))
            return []
        finally:
            driver.quit()


def build_ctrip_selenium_url(intent: FlightSearchIntent) -> str:
    params = {
        "origin": intent.origin,
        "destination": intent.destination,
        "travel_date": intent.travel_date.isoformat(),
        "currency": intent.currency,
    }
    if intent.time_preference:
        params["time_preference"] = intent.time_preference
    query = urllib.parse.urlencode(params)
    return f"{CTRIP_SCHEME}://flight?{query}"


def parse_ctrip_selenium_url(url: str) -> FlightSearchIntent:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != CTRIP_SCHEME or parsed.netloc != "flight":
        raise ValueError(f"Unsupported Ctrip Selenium URL: {url}")
    query = urllib.parse.parse_qs(parsed.query)
    return FlightSearchIntent(
        origin=_required_query(query, "origin"),
        destination=_required_query(query, "destination"),
        travel_date=datetime.fromisoformat(_required_query(query, "travel_date")).date(),
        time_preference=query.get("time_preference", [None])[0] or None,
        currency=query.get("currency", ["CNY"])[0],
    )


def decode_ctrip_response_body(body: bytes) -> dict[str, Any]:
    try:
        text = gzip.decompress(body).decode("utf-8")
    except OSError:
        text = body.decode("utf-8")
    return json.loads(text)


def _wait_for_ctrip_search_payload(
    driver,
    *,
    timeout_seconds: int,
    manual_verification_wait_seconds: int,
) -> dict[str, Any]:
    if _is_manual_verification_present(driver) and manual_verification_wait_seconds > 0:
        _wait_for_manual_verification(driver, manual_verification_wait_seconds)

    try:
        request = driver.wait_for_request(
            "/international/search/api/search/batchSearch?.*",
            timeout=timeout_seconds,
        )
        if request.response is not None:
            return decode_ctrip_response_body(request.response.body)
    except Exception as exc:
        initial_error = exc
    else:
        initial_error = RuntimeError("batchSearch request had no response body.")

    payload = _extract_latest_ctrip_search_payload(driver)
    if payload is not None:
        return payload

    if manual_verification_wait_seconds <= 0:
        raise initial_error

    _wait_for_manual_verification(driver, manual_verification_wait_seconds)
    payload = _extract_latest_ctrip_search_payload(driver)
    if payload is not None:
        return payload

    request = driver.wait_for_request(
        "/international/search/api/search/batchSearch?.*",
        timeout=timeout_seconds,
    )
    if request.response is None:
        raise RuntimeError("batchSearch request had no response body after manual verification.")
    return decode_ctrip_response_body(request.response.body)


def _retry_after_manual_verification_if_present(
    driver,
    intent: FlightSearchIntent,
    *,
    source_url: str,
    direct_only: bool,
    max_results: int,
    timeout_seconds: int,
    manual_verification_wait_seconds: int,
) -> list[FlightEvidence]:
    if manual_verification_wait_seconds <= 0 or not _is_manual_verification_present(driver):
        return []
    _wait_for_manual_verification(driver, manual_verification_wait_seconds)
    try:
        payload = _wait_for_ctrip_search_payload(
            driver,
            timeout_seconds=timeout_seconds,
            manual_verification_wait_seconds=0,
        )
    except Exception:
        payload = _extract_latest_ctrip_search_payload(driver)
    if payload is None:
        return []
    return parse_ctrip_batch_search_payload(
        payload,
        intent,
        source_url=source_url,
        direct_only=direct_only,
        max_results=max_results,
    )


def _extract_latest_ctrip_search_payload(driver) -> dict[str, Any] | None:
    requests = list(getattr(driver, "requests", []) or [])
    for request in reversed(requests):
        url = getattr(request, "url", "")
        response = getattr(request, "response", None)
        if response is None or "batchSearch" not in url:
            continue
        try:
            return decode_ctrip_response_body(response.body)
        except Exception:
            continue
    return None


def _wait_for_manual_verification(driver, wait_seconds: int) -> None:
    if wait_seconds <= 0:
        return
    print(
        "Ctrip may require manual verification. "
        f"Please complete it in the browser within {wait_seconds} seconds..."
    )
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if not _is_manual_verification_present(driver):
            time.sleep(2)
            return
        time.sleep(1)


def _is_manual_verification_present(driver) -> bool:
    try:
        page_source = driver.page_source or ""
    except Exception:
        return False
    verification_markers = (
        "为保障您的安全访问",
        "请完成以下操作",
        "依次点击图标验证",
        "安全访问",
    )
    if any(marker in page_source for marker in verification_markers):
        return True
    try:
        return bool(
            driver.execute_script(
                """
                const text = document.body ? document.body.innerText : "";
                return text.includes("为保障您的安全访问")
                  || text.includes("请完成以下操作")
                  || text.includes("依次点击图标验证");
                """
            )
        )
    except Exception:
        return False


def parse_ctrip_batch_search_payload(
    payload: dict[str, Any],
    intent: FlightSearchIntent,
    *,
    source_url: str,
    direct_only: bool,
    max_results: int,
) -> list[FlightEvidence]:
    itineraries = payload.get("data", {}).get("flightItineraryList", [])
    evidence: list[FlightEvidence] = []
    captured_at = datetime.now(timezone.utc)
    for itinerary in itineraries:
        segments = itinerary.get("flightSegments", [])
        if not segments:
            continue
        itinerary_segments = _itinerary_segments(segments)
        if not itinerary_segments:
            continue
        transfer_count = _transfer_count(segments, itinerary_segments)
        if direct_only and transfer_count != 0:
            continue
        price = _lowest_price(itinerary.get("priceList", []))
        if price is None:
            continue
        first_segment = itinerary_segments[0]
        last_segment = itinerary_segments[-1]
        evidence.append(
            FlightEvidence(
                source_name="flights.ctrip.com",
                url=source_url,
                price=price,
                currency=intent.currency or "CNY",
                departure_time=_parse_ctrip_datetime(first_segment.get("departure_time")),
                arrival_time=_parse_ctrip_datetime(last_segment.get("arrival_time")),
                captured_at=captured_at,
                origin=intent.origin,
                destination=intent.destination,
                travel_date=intent.travel_date,
                metadata=_flight_metadata(
                    itinerary_segments=itinerary_segments,
                    transfer_count=transfer_count,
                    itinerary=itinerary,
                ),
            )
        )
    return _rank_ctrip_evidence(evidence, intent)[:max_results]


def _intent_from_query(query: str) -> FlightSearchIntent | None:
    pieces = [piece for piece in query.split() if piece]
    if len(pieces) < 3:
        return None
    travel_date = None
    for piece in pieces:
        try:
            travel_date = datetime.fromisoformat(piece).date()
            break
        except ValueError:
            continue
    if travel_date is None:
        return None
    date_index = pieces.index(travel_date.isoformat())
    if date_index < 2:
        return None
    return FlightSearchIntent(
        origin=pieces[0],
        destination=pieces[1],
        travel_date=travel_date,
        time_preference=_time_preference_from_query_pieces(pieces),
    )


def _build_ctrip_search_urls(intent: FlightSearchIntent) -> list[str]:
    origin = intent.origin.lower()
    destination = intent.destination.lower()
    international_query = urllib.parse.urlencode(
        {
            "depdate": intent.travel_date.isoformat(),
            "cabin": "y_s",
            "adult": "1",
            "child": "0",
            "infant": "0",
        }
    )
    legacy_query = urllib.parse.urlencode({"depdate": intent.travel_date.isoformat()})
    return [
        f"https://flights.ctrip.com/international/search/oneway-{origin}-{destination}?{international_query}",
        f"https://flights.ctrip.com/online/list/oneway-{origin}-{destination}?{legacy_query}",
    ]


def _time_preference_from_query_pieces(pieces: list[str]) -> str | None:
    for piece in pieces:
        preference = piece.strip().lower()
        if preference in {"morning", "afternoon", "evening"}:
            return preference
    return None


def _init_seleniumwire_driver(
    *,
    browser: str,
    headless: bool,
):
    try:
        from seleniumwire import webdriver
    except ImportError as exc:
        raise RuntimeError(
            "Ctrip crawler requires optional dependency selenium-wire. "
            "Install with: python -m pip install -e \".[ctrip]\""
        ) from exc

    normalised_browser = browser.strip().lower()
    if normalised_browser == "chrome":
        options = webdriver.ChromeOptions()
        if headless:
            options.add_argument("--headless=new")
        _configure_browser_options(options)
        return webdriver.Chrome(options=options)
    if normalised_browser == "edge":
        options = webdriver.EdgeOptions()
        if headless:
            options.add_argument("--headless=new")
        _configure_browser_options(options)
        return webdriver.Edge(options=options)
    raise ValueError(f"Unsupported browser for Ctrip crawler: {browser}")


def _configure_browser_options(options) -> None:
    options.add_argument("--incognito")
    options.add_argument("--remote-debugging-port=0")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-extensions")
    options.add_argument("--pageLoadStrategy=eager")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-software-rasterizer")
    options.add_argument("--ignore-certificate-errors")
    options.add_argument("--ignore-ssl-errors")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])


class _CtripLoginSession:
    def __init__(
        self,
        *,
        accounts: list[str],
        passwords: list[str],
        cookies_file: Path,
        timeout_seconds: int,
        login_wait_seconds: int,
    ) -> None:
        self.accounts = [account for account in accounts if account]
        self.passwords = [password for password in passwords if password]
        self.cookies_file = cookies_file
        self.timeout_seconds = timeout_seconds
        self.login_wait_seconds = login_wait_seconds

    def ensure_login(self, driver) -> None:
        if not self.accounts:
            raise RuntimeError("Ctrip login is enabled but no account is configured.")
        if not self.passwords:
            raise RuntimeError("Ctrip login is enabled but no password is configured.")

        for index, account in enumerate(self.accounts):
            password = self.passwords[index % len(self.passwords)]
            if self._try_cookie_login(driver, account):
                return
            if self._try_password_login(driver, account, password):
                self._save_required_cookies(account, driver.get_cookies())
                return
        raise RuntimeError("Ctrip login failed for all configured accounts.")

    def _try_cookie_login(self, driver, account: str) -> bool:
        cookies = self._load_cookies(account)
        if not cookies:
            return False
        driver.get("https://www.ctrip.com/")
        for cookie in cookies:
            try:
                driver.add_cookie(cookie)
            except Exception:
                continue
        if self._is_logged_in(driver):
            return True
        self._delete_cookies(account)
        return False

    def _try_password_login(self, driver, account: str, password: str) -> bool:
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.support.ui import WebDriverWait

        wait = WebDriverWait(driver, self.timeout_seconds)
        driver.get("https://flights.ctrip.com/online/channel/domestic")
        self._open_login_panel(driver, wait)
        account_input = wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, ".r_input.bbz-js-iconable-input"))
        )
        _replace_input_value(account_input, account)
        password_input = wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "div[data-testid='accountPanel'] input[data-testid='passwordInput']"))
        )
        _replace_input_value(password_input, password)

        for selector in ('[for="checkboxAgreementInput"]',):
            elements = driver.find_elements(By.CSS_SELECTOR, selector)
            if elements:
                elements[0].click()
                break

        submit = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".form_btn.form_btn--block")))
        submit.click()
        self._handle_double_auth_if_present(driver)
        return self._is_logged_in(driver)

    def _open_login_panel(self, driver, wait) -> None:
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support import expected_conditions as EC

        if driver.find_elements(By.CLASS_NAME, "lg_loginbox_modal"):
            return
        login_buttons = driver.find_elements(By.CLASS_NAME, "tl_nfes_home_header_login_wrapper_siwkn")
        if login_buttons:
            login_buttons[0].click()
            wait.until(EC.presence_of_element_located((By.CLASS_NAME, "lg_loginwrap")))
            return
        driver.get("https://passport.ctrip.com/user/login")

    def _handle_double_auth_if_present(self, driver) -> None:
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.support.ui import WebDriverWait

        wait = WebDriverWait(driver, self.timeout_seconds)
        selector = "[data-testid='doubleAuthSwitcherBox']"
        try:
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
        except Exception:
            return

        send_button = wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, f"{selector} dl[data-testid='dynamicCodeInput'] a.btn-primary-s"))
        )
        send_button.click()
        code = self._wait_for_console_input("Please enter the Ctrip verification code: ")
        if not code:
            raise RuntimeError("Timed out waiting for Ctrip verification code.")
        code_input = wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, f"{selector} input[data-testid='verifyCodeInput']"))
        )
        code_input.send_keys(code)
        verify_button = wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, f"{selector} dl[data-testid='dynamicVerifyButton'] input[type='submit']"))
        )
        verify_button.click()

    def _wait_for_console_input(self, prompt: str) -> str | None:
        result: list[str | None] = [None]
        completed = threading.Event()

        def read_input() -> None:
            try:
                result[0] = input(prompt).strip()
            except EOFError:
                result[0] = None
            finally:
                completed.set()

        thread = threading.Thread(target=read_input, daemon=True)
        thread.start()
        thread.join(timeout=self.login_wait_seconds)
        if not completed.is_set():
            return None
        return result[0]

    def _is_logged_in(self, driver) -> bool:
        try:
            driver.get("https://my.ctrip.com/myinfo/home")
            deadline = time.time() + self.timeout_seconds
            while time.time() < deadline:
                if driver.current_url.startswith("https://my.ctrip.com/myinfo/home"):
                    return True
                time.sleep(0.25)
        except Exception:
            return False
        return False

    def _load_cookies(self, account: str) -> list[dict[str, Any]] | None:
        if not self.cookies_file.exists():
            return None
        try:
            cookies_by_account = json.loads(self.cookies_file.read_text(encoding="utf-8"))
            cookies = cookies_by_account.get(account)
            return cookies if isinstance(cookies, list) else None
        except Exception:
            return None

    def _save_required_cookies(self, account: str, cookies: list[dict[str, Any]]) -> None:
        self.cookies_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            cookies_by_account = json.loads(self.cookies_file.read_text(encoding="utf-8"))
            if not isinstance(cookies_by_account, dict):
                cookies_by_account = {}
        except Exception:
            cookies_by_account = {}
        cookies_by_account[account] = [
            cookie for cookie in cookies if cookie.get("name") in REQUIRED_CTRIP_COOKIES
        ]
        self.cookies_file.write_text(
            json.dumps(cookies_by_account, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _delete_cookies(self, account: str) -> None:
        if not self.cookies_file.exists():
            return
        try:
            cookies_by_account = json.loads(self.cookies_file.read_text(encoding="utf-8"))
            if isinstance(cookies_by_account, dict) and account in cookies_by_account:
                del cookies_by_account[account]
                self.cookies_file.write_text(
                    json.dumps(cookies_by_account, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        except Exception:
            return


def _drive_ctrip_homepage_search(driver, intent: FlightSearchIntent, *, timeout_seconds: int) -> None:
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait

    wait = WebDriverWait(driver, timeout_seconds)
    driver.get("https://flights.ctrip.com/online/channel/domestic")
    wait.until(EC.presence_of_element_located((By.CLASS_NAME, "pc_home-jipiao")))
    driver.find_element(By.CLASS_NAME, "pc_home-jipiao").click()
    wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, "radio-label")))
    driver.find_elements(By.CLASS_NAME, "radio-label")[0].click()

    wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, "form-input-v3")))
    inputs = driver.find_elements(By.CLASS_NAME, "form-input-v3")
    _replace_input_value(inputs[0], _ctrip_display_name(intent.origin))
    _replace_input_value(inputs[1], _ctrip_display_name(intent.destination))

    date_inputs = driver.find_elements(By.CSS_SELECTOR, "[aria-label=请选择日期]")
    if date_inputs:
        driver.execute_script(
            """
            const input = arguments[0];
            input.value = arguments[1];
            input.dispatchEvent(new Event('input', { bubbles: true }));
            input.dispatchEvent(new Event('change', { bubbles: true }));
            """,
            date_inputs[0],
            intent.travel_date.isoformat(),
        )

    buttons = driver.find_elements(By.CLASS_NAME, "search-btn")
    if buttons:
        buttons[0].click()
    else:
        inputs[1].send_keys(Keys.ENTER)


def _replace_input_value(element, value: str) -> None:
    from selenium.webdriver.common.keys import Keys

    element.click()
    element.send_keys(Keys.CONTROL + "a")
    element.send_keys(value)


_CTRIP_DISPLAY_NAMES = {
    "SIN": "新加坡",
    "TFU": "成都",
    "CTU": "成都",
    "BJS": "北京",
    "PEK": "北京",
    "PKX": "北京",
    "SHA": "上海",
    "PVG": "上海",
}


def _ctrip_display_name(value: str) -> str:
    return _CTRIP_DISPLAY_NAMES.get(value.strip().upper(), value)


def _lowest_price(price_list: list[dict[str, Any]]) -> float | None:
    prices: list[float] = []
    for item in price_list:
        adult_price = item.get("adultPrice")
        if adult_price is None:
            continue
        adult_tax = item.get("adultTax")
        if adult_tax is None:
            sort_price = item.get("sortPrice", adult_price)
            adult_tax = 0 if item.get("freeOilFeeAndTax") else sort_price - adult_price
        prices.append(float(adult_price) + float(adult_tax))
    if not prices:
        return None
    return min(prices)


def _rank_ctrip_evidence(evidence: list[FlightEvidence], intent: FlightSearchIntent) -> list[FlightEvidence]:
    preference = (intent.time_preference or "").strip().lower()
    if preference not in {"morning", "afternoon", "evening"}:
        return sorted(evidence, key=lambda item: item.price)
    return sorted(
        evidence,
        key=lambda item: (
            0 if _matches_time_preference(item.departure_time, preference) else 1,
            item.price,
        ),
    )


def _evidence_satisfies_requested_time(evidence: list[FlightEvidence], intent: FlightSearchIntent) -> bool:
    preference = (intent.time_preference or "").strip().lower()
    if preference not in {"morning", "afternoon", "evening"}:
        return True
    return any(_matches_time_preference(item.departure_time, preference) for item in evidence)


def _matches_time_preference(departure_time: datetime | None, preference: str) -> bool:
    if departure_time is None:
        return False
    hour = departure_time.hour
    if preference == "morning":
        return hour < 12
    if preference == "afternoon":
        return 12 <= hour < 18
    return hour >= 18


def _flight_metadata(
    *,
    itinerary_segments: list[dict[str, object]],
    transfer_count: int,
    itinerary: dict[str, Any],
) -> dict[str, object]:
    first = itinerary_segments[0]
    last = itinerary_segments[-1]
    flight_no = "+".join(str(segment["flight_no"]) for segment in itinerary_segments if segment.get("flight_no"))
    airline_names = [str(segment["airline"]) for segment in itinerary_segments if segment.get("airline")]
    aircraft_names = [str(segment["aircraft"]) for segment in itinerary_segments if segment.get("aircraft")]
    return {
        "flight_no": flight_no or itinerary.get("itineraryId"),
        "airline": "+".join(airline_names) if airline_names else None,
        "aircraft": "+".join(aircraft_names) if aircraft_names else None,
        "departure_airport": first.get("departure_airport"),
        "departure_airport_code": first.get("departure_airport_code"),
        "departure_terminal": first.get("departure_terminal"),
        "arrival_airport": last.get("arrival_airport"),
        "arrival_airport_code": last.get("arrival_airport_code"),
        "arrival_terminal": last.get("arrival_terminal"),
        "transfer_count": transfer_count,
        "is_direct": transfer_count == 0,
        "itinerary_id": itinerary.get("itineraryId"),
        "segments": itinerary_segments,
    }


def _itinerary_segments(segments: list[dict[str, Any]]) -> list[dict[str, object]]:
    itinerary_segments: list[dict[str, object]] = []
    for segment in segments:
        for flight in segment.get("flightList", []):
            itinerary_segments.append(_segment_metadata(flight))
    return itinerary_segments


def _segment_metadata(flight: dict[str, Any]) -> dict[str, object]:
    return {
        "flight_no": flight.get("flightNo"),
        "operate_flight_no": flight.get("operateFlightNo"),
        "airline": flight.get("marketAirlineName"),
        "operate_airline": flight.get("operateAirlineName"),
        "aircraft": flight.get("aircraftName") or flight.get("planeType"),
        "departure_time": flight.get("departureDateTime"),
        "arrival_time": flight.get("arrivalDateTime"),
        "departure_airport": flight.get("departureAirportName"),
        "departure_airport_code": flight.get("departureAirportCode"),
        "departure_terminal": flight.get("departureTerminal"),
        "arrival_airport": flight.get("arrivalAirportName"),
        "arrival_airport_code": flight.get("arrivalAirportCode"),
        "arrival_terminal": flight.get("arrivalTerminal"),
        "duration": flight.get("duration"),
    }


def _transfer_count(segments: list[dict[str, Any]], itinerary_segments: list[dict[str, object]]) -> int:
    explicit_counts = [segment.get("transferCount") for segment in segments if segment.get("transferCount") is not None]
    if explicit_counts:
        return int(max(explicit_counts))
    return max(len(itinerary_segments) - 1, 0)


def _parse_ctrip_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _first(value: list[Any]) -> Any | None:
    if not value:
        return None
    return value[0]


def _required_query(query: dict[str, list[str]], name: str) -> str:
    value = query.get(name, [""])[0]
    if not value:
        raise ValueError(f"Missing required Ctrip Selenium URL query parameter: {name}")
    return value
