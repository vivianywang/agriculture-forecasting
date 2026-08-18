import sys
import time
from pathlib import Path

import requests

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import API_BASE_URL

TEST_CASES = [
    {"city": "Regina", "expect": "ok", "note": "SK, major city"},
    {"city": "Calgary", "expect": "ok", "note": "AB, major city"},
    {"city": "Winnipeg", "expect": "ok", "note": "MB, major city"},
    {"city": "Toronto", "expect": "ok", "note": "ON, major city"},
    {"city": "Montreal", "expect": "either", "note": "QC, major city, accent handling varies"},
    {"city": "Montréal", "expect": "either", "note": "QC, accented spelling"},
    {"city": "Moose Jaw", "expect": "ok", "note": "SK, smaller city"},
    {"city": "Brandon", "expect": "ok", "note": "MB, smaller city"},
    {"city": "regina", "expect": "ok", "note": "lowercase, tests case-insensitivity"},
    {"city": "  Regina  ", "expect": "ok", "note": "extra whitespace"},
    {"city": "Vancouver", "expect": "404", "note": "BC, uncovered province"},
    {"city": "Halifax", "expect": "404", "note": "NS, uncovered province"},
    {"city": "Asdfghjkl", "expect": "404", "note": "nonexistent city"},
    {"city": "", "expect": "error", "note": "empty city"},
]

REQUEST_DELAY_SECONDS = 0.3


def run_predict(city):
    try:
        return requests.post(f"{API_BASE_URL}/predict", json={"city": city}, timeout=15), None
    except requests.exceptions.RequestException as e:
        return None, str(e)


def run_history(city):
    try:
        return requests.get(f"{API_BASE_URL}/history", params={"city": city}, timeout=15), None
    except requests.exceptions.RequestException as e:
        return None, str(e)


def check_result(case, resp):
    expect = case["expect"]
    if resp is None:
        return "FAIL", "no response"

    if expect == "ok":
        if resp.status_code == 200:
            return "PASS", f"200, yield={resp.json().get('predicted_yield_t_ha')} t/ha"
        return "FAIL", f"expected 200, got {resp.status_code}: {resp.text[:100]}"

    if expect == "404":
        if resp.status_code == 404:
            return "PASS", f"404 as expected: {resp.json().get('detail', '')[:80]}"
        return "FAIL", f"expected 404, got {resp.status_code}"

    if expect == "either":
        return "INFO", f"{resp.status_code}: {resp.text[:100]}"

    if expect == "error":
        if resp.status_code >= 400:
            return "PASS", f"{resp.status_code} as expected"
        return "FAIL", f"expected an error status, got {resp.status_code}"

    return "INFO", f"{resp.status_code}"


def main():
    print(f"Testing API at {API_BASE_URL}")
    print("(if you're running uvicorn with --reload, its file watcher can drop")
    print(" in-flight requests if it restarts mid-test; --reload is for development,")
    print(" consider running without it for a clean test pass)\n")
    results = []

    for case in TEST_CASES:
        city = case["city"]
        resp, error = run_predict(city)
        if error:
            print(f"[FAIL] predict('{city}') - {case['note']}")
            print(f"       connection error: {error}")
            results.append("FAIL")
            print()
            time.sleep(REQUEST_DELAY_SECONDS)
            continue

        status, detail = check_result(case, resp)
        results.append(status)
        print(f"[{status}] predict('{city}') - {case['note']}")
        print(f"       {detail}")

        if status in ("PASS", "INFO") and resp is not None and resp.status_code == 200:
            time.sleep(REQUEST_DELAY_SECONDS)
            hist_resp, hist_error = run_history(city)
            if hist_error:
                print(f"       history: connection error: {hist_error}")
            elif hist_resp.status_code == 200:
                n = len(hist_resp.json().get("history", []))
                print(f"       history: {n} years returned")
            else:
                print(f"       history: FAILED (status {hist_resp.status_code})")
        print()
        time.sleep(REQUEST_DELAY_SECONDS)

    passed = results.count("PASS")
    failed = results.count("FAIL")
    info = results.count("INFO")
    print(f"Summary: {passed} passed, {failed} failed, {info} informational")


if __name__ == "__main__":
    main()