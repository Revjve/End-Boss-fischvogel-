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
  G. The laser beam (attack1 / attack2) ends up exactly where Blockbench draws
     it. Reference: the original project evaluated the way Blockbench does it
     (bones in group-list order, the laser's "rotate in global space" flag
     cancelling whatever was already rotated). Converted: plain hierarchy,
     evaluated both ways an engine can stack bones - full matrices, and
     rotation * per-axis scale (no skew, how display-entity engines work).
     Every corner of every beam face, every tick the beam is visible, must be
     within one model pixel. (Blockbench's own pose is slightly skewed where
     root is not exactly on a quarter turn; the per-axis-scale numbers include
     that, and the two ticks where root is mid-turn are reported, not judged.)

Usage: python3 verify_models.py <converted_dir>
"""
import copy
import json
import math
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import convert_models as cm  # noqa: E402

SRC = cm.SRC


# ---- bone transforms (Blockbench 5 / three.js conventions) -----------------
def rot(deg):
    x, y, z = (math.radians(float(v)) for v in deg)
    rx = np.array([[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]])
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rz = np.array([[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]])
    return rz @ ry @ rx          # Euler order ZYX


def mat(pos, r, s):
    m = np.eye(4)
    m[:3, :3] = r @ np.diag(s)
    m[:3, 3] = pos
    return m


class Rig:
    """Evaluates one animation of a Blockbench 5 project at a given time."""

    def __init__(self, model, anim_name):
        self.groups = {g['uuid']: g for g in model['groups']}
        self.by_name = {g['name']: g['uuid'] for g in model['groups']}
        self.parent = {}

        def walk(n, p):
            if isinstance(n, dict):
                self.parent[n['uuid']] = p
                for c in n.get('children', []):
                    walk(c, n['uuid'])
        for n in model['outliner']:
            walk(n, None)
        self.anim = next(a for a in model['animations'] if a['name'] == anim_name)
        self.calc = cm.Molang(random.Random(0))

    def chain(self, name):
        u, out = self.by_name[name], []
        while u is not None:
            out.append(u)
            u = self.parent[u]
        return out[::-1]

    def channel(self, uid, ch, t):
        an = self.anim['animators'].get(uid)
        kfs = [k for k in (an or {}).get('keyframes', []) if k['channel'] == ch]
        return cm.interpolate(kfs, t, self.anim['loop'], self.calc) if kfs else None

    def local(self, uid, t, animated=True):
        g = self.groups[uid]
        p = self.parent[uid]
        pos = np.array(g['origin'], float) - (np.array(self.groups[p]['origin'], float) if p else 0)
        r = [float(v) for v in g.get('rotation', [0, 0, 0])]
        s = [1.0, 1.0, 1.0]
        if animated:
            dp = self.channel(uid, 'position', t)
            dr = self.channel(uid, 'rotation', t)
            ds = self.channel(uid, 'scale', t)
            if dp:
                pos = pos + np.array(dp)
            if dr:
                r = [a + b for a, b in zip(r, dr)]
            if ds:
                s = [v or 0.00001 for v in ds]
        return pos, rot(r), np.array(s)

    def global_flag(self, uid):
        an = self.anim['animators'].get(uid)
        return bool(an and an.get('rotation_global'))

    def world_matrix(self, name, t, order=None):
        """Full matrices. With order (Blockbench's group list) the global-space
        flag is applied the way Blockbench does it."""
        w = np.eye(4)
        locs = {}
        for u in self.chain(name):
            pos, r, s = self.local(u, t)
            if order is not None and self.global_flag(u):
                # parent world rotation as it stands when this bone is reached:
                # bones later in the group list are still in their rest pose
                pw = np.eye(3)
                for a in self.chain(name)[:-1]:
                    done = order.index(self.groups[a]['name']) < order.index(self.groups[u]['name'])
                    pa, ra, sa = self.local(a, t, animated=done)
                    pw = pw @ ra
                r = pw.T @ r
            locs[u] = (pos, r, s)
            w = w @ mat(pos, r, s)
        return w

    def world_trs(self, name, t):
        """rotation and per-axis scale stacked separately (no skew)."""
        wp, wr, ws = np.zeros(3), np.eye(3), np.ones(3)
        for u in self.chain(name):
            pos, r, s = self.local(u, t)
            wp = wp + wr @ (ws * pos)
            wr = wr @ r
            ws = ws * s
        return wp, wr, ws


def element_points(model, group_uuid):
    """Corners of every element directly in the group, relative to its pivot."""
    g = next(x for x in model['groups'] if x['uuid'] == group_uuid)
    els = {e['uuid']: e for e in model['elements']}
    pts = []

    def kids(n):
        if isinstance(n, dict):
            if n['uuid'] == group_uuid:
                return n.get('children', [])
            for c in n.get('children', []):
                r = kids(c)
                if r is not None:
                    return r
        return None
    for n in model['outliner']:
        ch = kids(n)
        if ch is not None:
            break
    for c in ch:
        if not isinstance(c, str):
            continue
        e = els[c]
        o = np.array(e.get('origin', [0, 0, 0]), float)
        r = rot(e.get('rotation', [0, 0, 0]))
        for x in (e['from'][0], e['to'][0]):
            for y in (e['from'][1], e['to'][1]):
                for z in (e['from'][2], e['to'][2]):
                    pts.append(r @ (np.array([x, y, z], float) - o) + o - np.array(g['origin'], float))
    return np.array(pts)


def check_beam(orig, new5):
    """G - see the module docstring. Returns ok."""
    ref_model = copy.deepcopy(orig)
    cm.clean_numbers(ref_model)
    cm.bake_random(ref_model, [])        # same seeds -> the same rolls as the build
    order = [g['name'] for g in orig['groups']]
    ok = True
    for anim in ('attack1', 'attack2'):
        ref, new = Rig(ref_model, anim), Rig(new5, anim)
        laser = new.by_name['laser']
        pts = element_points(new5, laser)
        root = new.by_name['root']
        length = float(ref.anim['length'])
        worst_m = worst_t = 0.0
        edge, n = [], 0
        for i in range(int(round(length / cm.TICK)) + 1):
            t = round(i * cm.TICK, 4)
            sc = ref.channel(laser, 'scale', t)
            if not sc or min(abs(v) for v in sc) < 1e-6:
                continue
            n += 1
            wr = ref.world_matrix('laser', t, order)
            a = (wr[:3, :3] @ pts.T).T + wr[:3, 3]
            wm = new.world_matrix('laser', t)
            b = (wm[:3, :3] @ pts.T).T + wm[:3, 3]
            p, r, s = new.world_trs('laser', t)
            c = (r @ (s * pts).T).T + p
            dm = float(np.abs(a - b).max())
            dt = float(np.abs(a - c).max())
            worst_m = max(worst_m, dm)
            ang = ref.channel(root, 'rotation', t)[0]
            if abs(ang - 90 * round(ang / 90)) > 0.5:
                # root is between quarter turns: Blockbench's own pose is skewed
                # here, which no per-axis-scale engine can draw - report only
                edge.append((t, round(ang, 2), round(dt, 2)))
            else:
                worst_t = max(worst_t, dt)
            if t in (2.5, 4.0, 7.0) and anim == 'attack2':
                d = (a[:, 1].min(), a[:, 1].max())
                print('G  attack2 t=%.2f beam spans y %.1f .. %.1f px (Blockbench)' % (t, d[0], d[1]))
        # within one model pixel (1/16 block). A wrong axis, sign or scale
        # in the bake moves the beam by tens of pixels.
        if worst_m > 1.0 or worst_t > 1.0:
            print('FAIL G: %s beam differs from Blockbench (matrix %.3f px, per-axis %.3f px)' % (anim, worst_m, worst_t))
            ok = False
        print('G  %s: %d visible ticks, max deviation %.3f px (matrices) / %.3f px (per-axis scale)%s'
              % (anim, n, worst_m, worst_t, ('; root mid-turn at ' + ', '.join('t=%.2f (%.2f deg, %.2f px)' % e for e in edge)) if edge else ''))
    for a in new5['animations']:
        for an in a.get('animators', {}).values():
            if an.get('rotation_global'):
                print('FAIL G: rotation_global still set on', a['name'], an['name'])
                ok = False
    return ok


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
    new5 = json.load(open(os.path.join(conv_dir, 'fv_endsoul (Blockbench 5).bbmodel')))
    legacy = json.load(open(os.path.join(conv_dir, 'fv_endsoul.bbmodel')))
    porig = json.load(open(os.path.join(SRC, 'projectile (original).bbmodel')))
    pnew5 = json.load(open(os.path.join(conv_dir, 'fv_endsoul_projectile (Blockbench 5).bbmodel')))
    plegacy = json.load(open(os.path.join(conv_dir, 'fv_endsoul_projectile.bbmodel')))

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
    expected_added = {('idle', e, 'scale') for e in ('eye1', 'eye2', 'eye3')} | {k for k in cn if k[0] == 'dormant'} \
        | {('attack1', 'laser', 'rotation')}
    # global-space rotation bake (check G proves the beam is unchanged)
    rebuilt = {('attack2', 'rootlaser', 'rotation')}
    swapped = {('attack2', 'rootlaser', 'scale')}
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
            if a != b and key in rebuilt:
                print('A  %s %s %s rebuilt from the global-space rotation (see G)' % key)
            elif a != b:
                print('FAIL A: deterministic channel changed', key)
                ok = False
            same += 1
            continue
        baked += 1
        length, loop = anim_len[key[0]]
        rng = random.Random(7)
        calc = cm.Molang(rng)
        if key in swapped:
            print('B  %s %s %s: Y/Z swapped by the global-space bake, checked swapped back' % key)
        for k in nk:
            t = k['time']
            rolls = [cm.interpolate(kfs, t, loop, calc) for _ in range(400)]
            v = [float(k['data_points'][0][x]) for x in 'xyz']
            if key in swapped:
                v = [v[0], v[2], v[1]]
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

    # G - the beam where Blockbench draws it, in any engine
    if not check_beam(orig, new5):
        ok = False

    print('RESULT:', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
