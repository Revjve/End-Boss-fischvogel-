#!/usr/bin/env python3
"""
Checks the converted blueprints against the originals.

  A. Every keyframe channel that convert_models.py did not bake is identical
     (same times, values, interpolation) to the original.
  B. Every baked channel stays inside the original's random envelope: for each
     tick, the original channel is evaluated many times with fresh rolls and
     the baked value must fall inside [min, max] of those rolls (+ tolerance).
  C. Loading the legacy (4.10) file the way Blockbench 5 does (sign flip on
     load) gives exactly the Blockbench 5 file's animation data back.
  D. Geometry, textures and UVs are unchanged.

Usage: python3 verify_models.py <converted_dir>
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import convert_models as cm  # noqa: E402

SRC = cm.SRC


def channels(model):
    out = {}
    for a in model['animations']:
        for uid, an in a.get('animators', {}).items():
            for k in an.get('keyframes', []):
                out.setdefault((a['name'], an['name'], k['channel']), []).append(k)
    for v in out.values():
        v.sort(key=lambda k: k['time'])
    return out


def geometry(model):
    return json.dumps({'elements': model['elements'], 'textures': model['textures'],
                       'resolution': model.get('resolution')}, sort_keys=True)


def main(conv_dir):
    ok = True
    orig = json.load(open(os.path.join(SRC, 'endboss (original).bbmodel')))
    new5 = json.load(open(os.path.join(conv_dir, 'fv_endboss (Blockbench 5).bbmodel')))
    legacy = json.load(open(os.path.join(conv_dir, 'fv_endboss.bbmodel')))
    porig = json.load(open(os.path.join(SRC, 'projectile (original).bbmodel')))
    pnew5 = json.load(open(os.path.join(conv_dir, 'fv_endboss_projectile (Blockbench 5).bbmodel')))
    plegacy = json.load(open(os.path.join(conv_dir, 'fv_endboss_projectile.bbmodel')))

    # D - geometry untouched
    for a, b, label in ((orig, new5, 'endboss'), (porig, pnew5, 'projectile')):
        if geometry(a) != geometry(b):
            print('FAIL D: geometry/texture differs in', label)
            ok = False
    print('D geometry/textures/uv identical:', ok)

    # A / B
    co, cn = channels(orig), channels(new5)
    anim_len = {a['name']: (a['length'], a['loop']) for a in orig['animations']}
    added = {k for k in cn if k not in co}
    expected_added = {('idle', e, 'scale') for e in ('eye1', 'eye2', 'eye3')} | {k for k in cn if k[0] == 'dormant'}
    if added != expected_added:
        print('FAIL A: unexpected new channels', sorted(added - expected_added))
        ok = False
    baked, same = 0, 0
    worst = 0.0
    for key, kfs in co.items():
        nk = cn.get(key)
        if nk is None:
            print('FAIL A: channel vanished', key)
            ok = False
            continue
        is_random = any(cm.has_expression(k) for k in kfs)
        if not is_random:
            a = [(k['time'], k['interpolation'], [str(k['data_points'][0].get(x)).strip() for x in 'xyz']) for k in kfs]
            b = [(k['time'], k['interpolation'], [str(k['data_points'][0].get(x)).strip() for x in 'xyz']) for k in nk]
            if a != b:
                print('FAIL A: deterministic channel changed', key)
                ok = False
            same += 1
            continue
        baked += 1
        length, loop = anim_len[key[0]]
        rng = random.Random(7)
        calc = cm.Molang(rng)
        for k in nk:
            t = k['time']
            rolls = [cm.interpolate(kfs, t, loop, calc) for _ in range(400)]
            v = [float(k['data_points'][0][x]) for x in 'xyz']
            for i in range(3):
                lo = min(r[i] for r in rolls)
                hi = max(r[i] for r in rolls)
                span = hi - lo
                tol = max(1e-3, 0.08 * span)   # 400 rolls never quite reach the true extremes
                if not (lo - tol <= v[i] <= hi + tol):
                    worst = max(worst, min(abs(v[i] - lo), abs(v[i] - hi)))
                    print('FAIL B: %s t=%.2f axis %d baked %.4f outside [%.4f, %.4f]' % (key, t, i, v[i], lo, hi))
                    ok = False
    print('A deterministic channels unchanged: %d   B baked channels within envelope: %d' % (same, baked))

    # C - legacy round trip
    for l, n, label in ((legacy, new5, 'endboss'), (plegacy, pnew5, 'projectile')):
        back = cm.from_legacy(l)
        cb, cnn = channels(back), channels(n)
        if set(cb) != set(cnn):
            print('FAIL C: channel set differs', label)
            ok = False
        for key in cnn:
            a = [[float(k['data_points'][0][x]) for x in 'xyz'] + [k['time']] for k in cnn[key]]
            b = [[float(k['data_points'][0][x]) for x in 'xyz'] + [k['time']] for k in cb[key]]
            if a != b:
                print('FAIL C: legacy round trip differs', label, key)
                ok = False
        if 'groups' in l or l['meta']['format_version'] != '4.10':
            print('FAIL C: not a legacy layout', label)
            ok = False

        # outliner hierarchy identical
        def tree(m):
            if 'groups' in m:
                g = {x['uuid']: x for x in m['groups']}

                def w(nd):
                    if isinstance(nd, str):
                        return nd
                    gg = g[nd['uuid']]
                    return [gg['name'], gg['origin'], gg.get('rotation'), [w(c) for c in nd.get('children', [])]]
            else:
                def w(nd):
                    if isinstance(nd, str):
                        return nd
                    return [nd['name'], nd['origin'], nd.get('rotation'), [w(c) for c in nd.get('children', [])]]
            return [w(x) for x in m['outliner']]
        if tree(l) != tree(n):
            print('FAIL C: outliner differs', label)
            ok = False
    print('C legacy 4.10 <-> Blockbench 5 round trip exact')

    # E - every animation overrides lower layers (ModelEngine layering)
    for m, label in ((new5, 'endboss'), (pnew5, 'projectile'), (legacy, 'endboss legacy'), (plegacy, 'projectile legacy')):
        bad = [a['name'] for a in m['animations'] if a.get('override') is not True]
        if bad:
            print('FAIL E: override missing', label, bad)
            ok = False
    print('E override flag set on all animations')
    # F - loop modes untouched
    lo = {a['name']: a['loop'] for a in orig['animations']}
    ln = {a['name']: a['loop'] for a in new5['animations'] if a['name'] in lo}
    if lo != ln:
        print('FAIL F: loop modes changed', lo, ln)
        ok = False
    print('F loop modes as authored:', lo)

    print('RESULT:', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
