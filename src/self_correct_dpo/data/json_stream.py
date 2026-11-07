"""Read an official JSON array of objects without retaining its whole text/list."""
import json
from pathlib import Path
import time


def object_records(path, *, chunk_size=64*1024, max_record_chars=16*1024**2):
    """Bound parser memory by one record; reject truncation and trailing content.

    HPDv3 annotations are objects (not arbitrary scalar JSON values), so an
    object can be decoded as soon as its closing brace is available.
    """
    path = Path(path)
    decoder = json.JSONDecoder()
    with path.open(encoding='utf-8') as handle:
        buffer = ''
        pos = count = 0
        eof = False
        last_log = time.monotonic()

        def refill():
            nonlocal buffer, pos, eof
            buffer = buffer[pos:]
            pos = 0
            block = handle.read(chunk_size)
            eof = not block
            buffer += block

        def token():
            nonlocal pos
            while True:
                while pos < len(buffer) and buffer[pos] in ' \t\r\n':
                    pos += 1
                if pos < len(buffer):
                    return buffer[pos]
                if eof:
                    return ''
                refill()

        if token() != '[':
            raise ValueError(f'Expected JSON array: {path}')
        pos += 1
        if token() != ']':
            while True:
                if token() != '{':
                    raise ValueError(f'Expected object at row {count}: {path}')
                while True:
                    try:
                        record, end = decoder.raw_decode(buffer, pos)
                    except json.JSONDecodeError as error:
                        if eof:
                            raise ValueError(f'Invalid/truncated object at row {count}: {path}') from error
                        if len(buffer)-pos >= max_record_chars:
                            raise ValueError(f'Object at row {count} exceeds parser limit: {path}') from error
                        refill()
                    else:
                        if end-pos > max_record_chars:
                            raise ValueError(f'Object at row {count} exceeds parser limit: {path}')
                        pos = end
                        break
                count += 1
                yield record
                now = time.monotonic()
                if now-last_log >= 15:
                    print(json.dumps({'phase':'scan_annotations','file':path.name,
                                      'scanned_rows':count}), flush=True)
                    last_log = now
                next_token = token()
                if next_token == ']':
                    break
                if next_token != ',':
                    raise ValueError(f'Expected comma/closing bracket after row {count}: {path}')
                pos += 1
        pos += 1
        if token():
            raise ValueError(f'Trailing content after JSON array: {path}')
    print(json.dumps({'phase':'annotations_complete','file':path.name,'scanned_rows':count}), flush=True)
