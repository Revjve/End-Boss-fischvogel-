#!/usr/bin/env python3
"""
Builds "Soul of the End.zip" from the folder of the same name.

    python3 tools/build.py [--sim N]

  1. converts the original .bbmodel files (Source Files) for ModelEngine and
     verifies the result (convert_models.py / verify_models.py):
       plugins/ModelEngine/blueprints/fv_endsoul*.bbmodel  (legacy 4.10)
       Source Files/fv_endsoul*.bbmodel                    (Blockbench 5, editable)
  2. runs the static checker against the vanilla data in tools/vanilla_data
  3. --sim N: plays N simulated fights per scenario (simulate_fight.py)
  4. writes the zip: one top folder, directory entries, CRLF guide, fixed
     timestamps - so the same sources always give the same zip.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NAME = "Soul of the End"
PACK = os.path.join(ROOT, NAME)
PLUGINS = os.path.join(PACK, 'plugins')
STAMP = (2026, 10, 1, 12, 0, 0)


def run(*args):
    print('>', ' '.join(os.path.basename(a) if i == 1 else a for i, a in enumerate(args)))
    subprocess.run([sys.executable] + list(args), check=True)


def main():
    argv = sys.argv[1:]
    # 1. models
    with tempfile.TemporaryDirectory() as tmp:
        run(os.path.join(HERE, 'convert_models.py'), tmp)
        run(os.path.join(HERE, 'verify_models.py'), tmp)
        bp = os.path.join(PLUGINS, 'ModelEngine', 'blueprints')
        os.makedirs(bp, exist_ok=True)
        for f in ('fv_endsoul.bbmodel', 'fv_endsoul_projectile.bbmodel'):
            shutil.copyfile(os.path.join(tmp, f), os.path.join(bp, f))
        src = os.path.join(PACK, 'Source Files')
        for f in os.listdir(src):
            os.remove(os.path.join(src, f))
        for m in ('fv_endsoul', 'fv_endsoul_projectile'):
            shutil.copyfile(os.path.join(tmp, m + ' (Blockbench 5).bbmodel'), os.path.join(src, m + '.bbmodel'))
    # 2. checks
    run(os.path.join(HERE, 'check_pack.py'), PLUGINS, os.path.join(HERE, 'vanilla_data'))
    # 3. simulation
    if '--sim' in argv:
        n = argv[argv.index('--sim') + 1]
        run(os.path.join(HERE, 'simulate_fight.py'), PLUGINS, n)
    # 4. guide with CRLF line endings (Windows Notepad friendly)
    guide = os.path.join(PACK, 'Installation Guide.txt')
    with open(guide, 'rb') as f:
        txt = f.read().replace(b'\r\n', b'\n')
    with open(guide, 'wb') as f:
        f.write(txt.replace(b'\n', b'\r\n'))
    # 5. zip
    out = os.path.join(ROOT, NAME + '.zip')
    entries = []
    for dp, dns, fns in os.walk(PACK):
        dns.sort()
        rel = os.path.relpath(dp, ROOT)
        entries.append((rel.replace(os.sep, '/') + '/', None))
        for fn in sorted(fns):
            entries.append((os.path.join(rel, fn).replace(os.sep, '/'), os.path.join(dp, fn)))
    entries.sort(key=lambda e: e[0])
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for arc, src in entries:
            zi = zipfile.ZipInfo(arc, STAMP)
            if src is None:
                zi.external_attr = (0o40755 << 16) | 0x10
                z.writestr(zi, b'')
            else:
                zi.external_attr = 0o644 << 16
                zi.compress_type = zipfile.ZIP_DEFLATED
                with open(src, 'rb') as f:
                    z.writestr(zi, f.read())
    print(f'wrote {out} ({os.path.getsize(out) / 1024:.0f} KB, {len(entries)} entries)')


if __name__ == '__main__':
    main()
