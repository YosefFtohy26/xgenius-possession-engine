"""Download licensed external tracking only, from a pinned SkillCorner revision."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import shutil

REVISION = '4340d274572876239c154c90bc507a9b3250a656'
MATCHES = ('1874553','1886347','1899585','1925299','1927964','1953632','1959846',
           '1986691','1996435','1996436','2006229','2006363','2007448','2007721',
           '2010085','2011166','2013725','2015213','2016236','2017461')


def download(root: Path):
    evaluation = set(MATCHES[:6])
    ids = [x for x in MATCHES if x not in evaluation]+[x for x in MATCHES if x in evaluation]
    transfer = shutil.which('curl') or shutil.which('curl.exe')
    if not transfer:
        raise RuntimeError('Install curl to download the external dataset')

    def one(match_id):
        folder = root / match_id
        folder.mkdir(parents=True,exist_ok=True)
        records = {}
        for suffix in ('match.json', 'phases_of_play.csv', 'tracking_extrapolated.jsonl'):
            filename = f'{match_id}_{suffix}'
            path = folder / filename
            valid = path.exists() and path.stat().st_size >= 500
            if valid and suffix.endswith('jsonl'):
                expected = json.loads(Path(__file__).with_name('external_sources.json').read_text())['tracking'][match_id]
                valid = path.stat().st_size == expected['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest() == expected['sha256']
            if not valid:
                host = 'media.githubusercontent.com/media' if suffix.endswith('jsonl') else 'raw.githubusercontent.com'
                url = f'https://{host}/SkillCorner/opendata/{REVISION}/data/matches/{match_id}/{filename}'
                temporary = path.with_suffix(path.suffix + '.partial')
                subprocess.run([transfer,'-sS','-L','--fail','--retry','4','--retry-all-errors',
                                '--speed-limit','1024','--speed-time','45',
                                '--connect-timeout','30','--max-time','600',
                                '--output',str(temporary),url],check=True,
                               creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                temporary.replace(path)
            if suffix.endswith('jsonl'):
                expected = json.loads(Path(__file__).with_name('external_sources.json').read_text())['tracking'][match_id]
                if hashlib.sha256(path.read_bytes()).hexdigest() != expected['sha256']:
                    raise ValueError(f'External source checksum mismatch: {match_id}')
            records[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f'External match {match_id} ready', flush=True)
        return match_id, records

    with ThreadPoolExecutor(max_workers=4) as pool:
        manifest = dict(pool.map(one, ids))
    (root.parent / 'download_manifest.json').write_text(json.dumps({'revision': REVISION, 'files': manifest}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('data/external/skillcorner-opendata/data/matches'))
    download(parser.parse_args().root)
