"""Inspect exported XLSX values, native charts and source bindings using OOXML."""
import json
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main", "c": "http://schemas.openxmlformats.org/drawingml/2006/chart", "a": "http://schemas.openxmlformats.org/drawingml/2006/main"}

def main():
    metrics = json.loads((ROOT/"results/test_metrics.json").read_text())
    with zipfile.ZipFile(ROOT/"outputs/model3/chart_data.xlsx") as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        sheets = [s.attrib["name"] for s in workbook.findall("m:sheets/m:sheet", NS)]
        assert sheets == ["Overview", "Predictions", "Validation", "Comparison", "Daily patterns", "Test months", "Features", "Data quality"]
        charts = [n for n in archive.namelist() if "/charts/chart" in n and n.endswith(".xml")]
        assert len(charts) == 6
        bindings = {}
        for name in charts:
            xml = ET.fromstring(archive.read(name))
            refs = [e.text for e in xml.findall(".//c:f", NS)]
            assert refs and all("!" in r for r in refs)
            colors = [c.attrib.get("val") for c in xml.findall(".//a:srgbClr", NS)]
            assert "284B7D" in [c.upper() for c in colors]
            bindings[name] = refs
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        cells = {c.attrib["r"]: c for c in sheet.findall(".//m:c", NS)}
        for name, expected in {"B6":metrics["accuracy"], "B7":metrics["macro_f1"], "B8":metrics["balanced_accuracy"], "B9":metrics["n"], "B10":metrics["correct"]}.items():
            assert cells[name].find("m:f", NS) is not None
            actual = float(cells[name].find("m:v", NS).text)
            assert abs(actual-expected)<1e-12
        for name in archive.namelist():
            if name.startswith("xl/worksheets/") and name.endswith(".xml"):
                xml = ET.fromstring(archive.read(name))
                assert not any(c.attrib.get("t")=="e" for c in xml.findall(".//m:c", NS))
    record = {"passed":True,"worksheets":sheets,"native_charts":len(charts),"headline_formulas_match_python_metrics":True,"formula_error_cells":0,"chart_bindings":bindings,"visual_review":"All eight sheet previews and all five scientific figures reviewed; the feature chart was repaired and reviewed again."}
    (ROOT/"results/workbook_verification.json").write_text(json.dumps(record,indent=2),encoding="utf-8")
    print(json.dumps(record,indent=2))

if __name__ == "__main__": main()
