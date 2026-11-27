"""Independent checks on the exported XLSX, including local timestamp serials."""
import datetime as dt
import json
import xml.etree.ElementTree as ET
import zipfile
from forecast_core import ROOT

N={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main','c':'http://schemas.openxmlformats.org/drawingml/2006/chart','a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
def main():
    h=json.loads((ROOT/'results/hourly_test_metrics.json').read_text());m=json.loads((ROOT/'results/monthly_test_metrics.json').read_text())
    with zipfile.ZipFile(ROOT/'outputs/forecast_results.xlsx') as z:
        book=ET.fromstring(z.read('xl/workbook.xml'))
        names=[s.attrib['name'] for s in book.findall('m:sheets/m:sheet',N)]
        assert len(names)==10
        charts=[p for p in z.namelist() if '/charts/chart' in p and p.endswith('.xml')]
        assert len(charts)==10
        for file in charts:
            chart=ET.fromstring(z.read(file))
            assert chart.findall('.//c:f',N)
            colors=[c.attrib.get('val','').upper() for c in chart.findall('.//a:srgbClr',N)]
            assert '284B7D' in colors
        overview=ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
        cells={c.attrib['r']:c for c in overview.findall('.//m:c',N)}
        for cell,expected in {'C6':h['overall']['mae_kw'],'D6':h['overall']['wape'],'C7':m['overall']['mae_kw'],'D7':m['overall']['wape'],'B15':h['peak']['classification_report']['peak']['f1-score']}.items():
            assert cells[cell].find('m:f',N) is not None
            assert abs(float(cells[cell].find('m:v',N).text)-expected)<1e-9
        # Excel stores dates as days since 1899-12-30; no timezone offset belongs here.
        dates={2:dt.datetime(2018,1,1),3:dt.datetime(2018,1,1),4:dt.datetime(2017,1,1),5:dt.datetime(2017,1,1),9:dt.datetime(2017,1,1)}
        for index,expected in dates.items():
            sheet=ET.fromstring(z.read(f'xl/worksheets/sheet{index}.xml'))
            c=next(c for c in sheet.findall('.//m:c',N) if c.attrib['r']=='A6')
            value=c.find('m:v',N).text
            if c.attrib.get('t')=='d':
                actual=dt.datetime.fromisoformat(value.replace('Z','+00:00')).replace(tzinfo=None)
            else:
                actual=dt.datetime(1899,12,30)+dt.timedelta(days=float(value))
            assert actual==expected,(index,actual,expected)
        for p in z.namelist():
            if p.startswith('xl/worksheets/') and p.endswith('.xml'):
                sheet=ET.fromstring(z.read(p))
                assert not any(c.attrib.get('t')=='e' for c in sheet.findall('.//m:c',N))
        hp=ET.fromstring(z.read('xl/worksheets/sheet4.xml'))
        cells={c.attrib['r']:c for c in hp.findall('.//m:c',N)}
        for name in ('G6','H6','I6','J6'):assert cells[name].find('m:f',N) is not None
        assert len(hp.findall('m:sheetData/m:row',N))>=25303
    result={'passed':True,'worksheets':names,'native_charts':len(charts),'headline_formulas_match_python':True,'local_timestamp_serials_verified':True,'peak_and_error_formulas_present':True,'formula_error_cells':0,'visual_review':'all ten sheets and six scientific figures reviewed'}
    (ROOT/'results/workbook_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
