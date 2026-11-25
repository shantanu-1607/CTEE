"""Download the exact shared feature CSV, checking its size and SHA-256."""

import argparse
import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


class DownloadForm(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_form = False
        self.values = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "form" and attrs.get("id") == "download-form":
            self.in_form = attrs.get("action") == "https://drive.usercontent.google.com/download"
        if tag == "input" and self.in_form and attrs.get("type") == "hidden":
            self.values[attrs["name"]] = attrs.get("value", "")

    def handle_endtag(self, tag):
        if tag == "form":
            self.in_form = False


def download(destination):
    source = json.loads((ROOT / "data/source.json").read_text())
    expected = source["raw_csv_sha256"]
    destination = Path(destination)
    if destination.exists():
        with destination.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual == expected:
            print("The verified CSV is already present.")
            return
        raise FileExistsError("Destination exists with different contents; choose a new path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.exists():
        raise FileExistsError(f"An earlier partial download exists: {temporary}")
    url = "https://drive.google.com/uc?" + urlencode({"export": "download", "id": source["drive_file_id"]})
    response = urlopen(url, timeout=90)
    if "text/html" in response.headers.get("Content-Type", ""):
        form = DownloadForm()
        form.feed(response.read().decode("utf-8"))
        response.close()
        if form.values.get("id") != source["drive_file_id"] or form.values.get("confirm") != "t":
            raise ValueError("Drive returned an access or download page instead of the CSV")
        response = urlopen("https://drive.usercontent.google.com/download?" + urlencode(form.values), timeout=90)
    digest, total = hashlib.sha256(), 0
    try:
        with response, temporary.open("xb") as output:
            while block := response.read(1024 * 1024):
                output.write(block)
                digest.update(block)
                total += len(block)
        if total != source["raw_csv_bytes"] or digest.hexdigest() != expected:
            raise ValueError("Downloaded file differs from the audited CSV")
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(f"Downloaded and verified {total:,} bytes: {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/raw/all_features.csv")
    download(parser.parse_args().output)
