import os
from urllib.request import urlopen, Request, build_opener, HTTPRedirectHandler
from tqdm import tqdm


class FollowAllRedirects(HTTPRedirectHandler):
    """Handle 301/302/303/307/308 redirects."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return Request(newurl, headers=req.headers, method=req.get_method())


files = [
    'model.safetensors',
    'preprocessor_config.json',
    'config.json',
]

# Try mirrors in order — first one that responds wins
mirrors = [
    'https://hf-mirror.com',
    'https://huggingface.co',
]

dest = os.path.expanduser('~/.cache/huggingface/hub/models--briaai--RMBG-2.0/snapshots/main')
os.makedirs(dest, exist_ok=True)

opener = build_opener(FollowAllRedirects())

for f in files:
    downloaded = False
    for mirror in mirrors:
        url = f'{mirror}/briaai/RMBG-2.0/resolve/main/{f}'
        try:
            with opener.open(url, timeout=10) as r:
                total = int(r.headers.get('Content-Length', 0))
                path = os.path.join(dest, f)
                with open(path, 'wb') as out, tqdm(desc=f, total=total, unit='B', unit_scale=True) as pbar:
                    while True:
                        chunk = r.read(8192)
                        if not chunk:
                            break
                        out.write(chunk)
                        pbar.update(len(chunk))
            print(f'  ✓ {f}  (from {mirror})')
            downloaded = True
            break
        except Exception as e:
            print(f'  ✗ {mirror}: {e}')
    if not downloaded:
        print(f'\nERROR: Could not download {f} from any mirror.')
        raise SystemExit(1)

print('\nDone — all files downloaded.')
