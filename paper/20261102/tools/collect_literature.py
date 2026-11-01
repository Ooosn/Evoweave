"""Cache primary-source PDFs with hashes and extract their complete text."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

from pypdf import PdfReader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for spec in json.loads(args.catalog.read_text(encoding='utf-8'))['papers']:
        pdf = args.output / (spec['key'] + '.pdf')
        record = dict(spec)
        try:
            if not pdf.exists():
                request = urllib.request.Request(spec['pdf'], headers={'User-Agent': 'EvoWeave-literature-review/1.0'})
                with urllib.request.urlopen(request, timeout=120) as response:
                    body = response.read(70 * 1024 * 1024 + 1)
                if not body.startswith(b'%PDF') or len(body) > 70 * 1024 * 1024:
                    raise ValueError('not a bounded PDF response')
                pdf.write_bytes(body)
            reader = PdfReader(pdf)
            text = '\n\n'.join('PAGE ' + str(i + 1) + '\n' + (page.extract_text() or '')
                               for i, page in enumerate(reader.pages))
            (args.output / (spec['key'] + '.txt')).write_text(text, encoding='utf-8', errors='replace')
            record.update(pages=len(reader.pages), bytes=pdf.stat().st_size,
                          sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),
                          extracted_characters=len(text), status='downloaded_and_extracted')
        except Exception as error:
            record.update(status='unavailable', error=str(error))
        records.append(record)
        print(json.dumps({k: record[k] for k in ('key', 'status', 'pages') if k in record}), flush=True)
        (args.output / 'receipt.json').write_text(json.dumps({
            'observed_utc': datetime.now(timezone.utc).isoformat(), 'papers': records}, indent=2) + '\n', encoding='utf-8')
    if any(record['status'] != 'downloaded_and_extracted' for record in records):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
