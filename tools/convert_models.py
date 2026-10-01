#!/usr/bin/env python3
"""
Prepare Fischvogel's End Boss blueprints for ModelEngine.

FUNCTIONAL changes only - geometry, textures, UVs and the designed motion are
left exactly as authored. What this script does, per model:

  endboss
    1. Bakes every keyframe channel that uses math.random() into plain numeric
       keyframes, one per server tick (0.05 s). Blockbench re-rolls
       math.random() on every rendered frame, which is what makes the "shaker"
       bones jitter. ModelEngine does not re-roll Molang randomness the same
       way, so the jitter is pre-rolled here using Blockbench's own
       interpolation rules (linear / step / uniform catmull-rom) with the very
       same min/max ranges. ModelEngine animates at 20 fps internally, so one
       key per tick is the full resolution the game can show.
    2. Strips stray whitespace from numeric values (e.g. "-3\n" -> "-3").
    3. idle: hides eye1/eye2/eye3 explicitly. All four eye cubes occupy the
       exact same space with different, fully opaque textures. Blockbench
       draws eye4 last so the editor shows eye4; in game each bone is its own
       display entity and identical faces z-fight (flicker). Hiding the three
       cubes that are covered anyway gives exactly the editor's look.
    4. Adds a static "dormant" pose: the closed block with every effect bone
       hidden and the closed eye (eye1) showing - i.e. on_spawn's first frame
       minus the light rays that burst out when it wakes up. No new geometry.
    5. Sets "override" on every animation. ModelEngine combines all states
       that are playing at once unless an animation overrides the ones below
       it, so without the flag the idle pose would be ADDED on top of
       attack1/attack2/shooting (corner pieces at 5 px + 8 px, double root
       motion, ...). In Blockbench the flag only matters when two animations
       are previewed together; a single animation looks exactly the same.
    6. Writes the file in Blockbench's legacy 4.10 project format, using the
       exact transformation of Blockbench's own "File > Export Legacy Project"
       (groups inlined into the outliner, position-X and rotation-X/Y keyframe
       values sign-flipped, because Blockbench 5 changed how those are
       stored). Every ModelEngine build reads 4.x projects; only newer ones
       understand the 5.0 layout.

  projectile
    2, 5 and 6.

  Loop modes are deliberately left as authored: Blockbench's catmull-rom
  wraps neighbour keyframes for looping animations, so changing a loop mode
  would reshape curves. Transitions are timed from MythicMobs instead.

Usage:  python3 convert_models.py <out_dir>
Writes: fv_endboss.bbmodel, fv_endboss_projectile.bbmodel            (legacy, for ModelEngine)
        fv_endboss (Blockbench 5).bbmodel, ... (Blockbench 5)      (same edits, editable)
"""
import copy
import hashlib
import json
import math
import os
import random
import re
import sys
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', "Fischvogel's End Boss", 'Source Files')
TICK = 0.05
SEED = 20261001

NUM_RE = re.compile(r'^\s*-?\d+(\.\d+)?f?\s*$')
STRICT_NUM_RE = re.compile(r'^-?\d+(\.\d+f?)?$')   # Blockbench isStringNumber
RANDOM_RE = re.compile(r'math\.random\(\s*([^,()]+?)\s*,\s*([^,()]+?)\s*\)')


# ---------------------------------------------------------------------------
# number formatting identical to JavaScript's Number.prototype.toString for
# the value ranges used here (Blockbench writes keyframe values that way)
# ---------------------------------------------------------------------------
def js_num(x):
    if x == 0:
        return '0'
    if float(x).is_integer() and abs(x) < 1e21:
        return str(int(x))
    r = repr(float(x))
    if 'e' in r:  # not expected with 4-decimal rounding, keep it safe
        return format(float(x), '.10f').rstrip('0').rstrip('.')
    return r


def det_uuid(*parts):
    h = hashlib.md5('|'.join(str(p) for p in parts).encode()).hexdigest()
    return f'{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}'


# ---------------------------------------------------------------------------
# Blockbench 5 keyframe interpolation (js/animations/timeline_animators.js)
# ---------------------------------------------------------------------------
class Molang:
    def __init__(self, rng):
        self.rng = rng

    def __call__(self, v):
        if v is None:
            return 0.0
        s = str(v).strip()
        if s == '':
            return 0.0
        if NUM_RE.match(s):
            return float(s.rstrip('f'))

        def roll(m):
            lo, hi = float(m.group(1)), float(m.group(2))
            return '(%.12f)' % (lo + self.rng.random() * (hi - lo))
        expr = RANDOM_RE.sub(roll, s)
        if 'math.' in expr or re.search(r'[a-zA-Z_]', expr):
            raise ValueError('unsupported molang: %r' % s)
        return float(eval(expr, {'__builtins__': {}}, {}))


def catmull(t, p0, p1, p2, p3):
    v0 = (p2 - p0) * 0.5
    v1 = (p3 - p1) * 0.5
    t2 = t * t
    t3 = t * t2
    return (2 * p1 - 2 * p2 + v0 + v1) * t3 + (-3 * p1 + 3 * p2 - 2 * v0 - v1) * t2 + v0 * t + p1


def spline_y(ys, t):
    p = (len(ys) - 1) * t
    ip = math.floor(p)
    w = p - ip
    p0 = ys[ip if ip == 0 else ip - 1]
    p1 = ys[ip]
    p2 = ys[len(ys) - 1 if ip > len(ys) - 2 else ip + 1]
    p3 = ys[len(ys) - 1 if ip > len(ys) - 3 else ip + 2]
    return catmull(w, p0, p1, p2, p3)


def interpolate(kfs, time, loop_mode, calc):
    eps = 1 / 1200
    before = after = None
    bt = at = 0.0
    for k in kfs:
        if k['time'] < time:
            if before is None or k['time'] > bt:
                before, bt = k, k['time']
        else:
            if after is None or k['time'] < at:
                after, at = k, k['time']
    val = lambda k, ax: calc(k['data_points'][0].get(ax))
    result = None
    if before is not None and abs(bt - time) < eps:
        result = before
    elif after is not None and abs(at - time) < eps:
        result = after
    elif before is not None and before['interpolation'] == 'step':
        result = before
    elif before is not None and after is None:
        result = before
    elif after is not None and before is None:
        result = after
    elif before is None and after is None:
        return None
    else:
        alpha = (time - bt) / (at - bt)
        if before['interpolation'] == 'linear' and after['interpolation'] in ('linear', 'step'):
            out = []
            for ax in 'xyz':
                a = val(before, ax)
                out.append(a + (val(after, ax) - a) * alpha)
            return out
        srt = sorted(kfs, key=lambda k: k['time'])
        bi = srt.index(before)
        bp = srt[bi - 1] if bi - 1 >= 0 else None
        ap = srt[bi + 2] if bi + 2 < len(srt) else None
        if loop_mode == 'loop' and len(srt) >= 3:
            if bp is None:
                bp = srt[-2]
            if ap is None:
                ap = srt[1]
        out = []
        for ax in 'xyz':
            ys = []
            if bp is not None:
                ys.append(val(bp, ax))
            ys.append(val(before, ax))
            ys.append(val(after, ax))
            if ap is not None:
                ys.append(val(ap, ax))
            out.append(spline_y(ys, (alpha + (1 if bp is not None else 0)) / (len(ys) - 1)))
        return out
    return [val(result, ax) for ax in 'xyz']


# ---------------------------------------------------------------------------
# edits
# ---------------------------------------------------------------------------
def has_expression(kf):
    for dp in kf['data_points']:
        for ax in 'xyz':
            v = dp.get(ax)
            if v is not None and str(v).strip() != '' and not NUM_RE.match(str(v)):
                return True
    return False


def clean_numbers(model):
    n = 0
    for anim in model.get('animations', []):
        for an in anim.get('animators', {}).values():
            for kf in an.get('keyframes', []):
                for dp in kf['data_points']:
                    for ax in 'xyz':
                        v = dp.get(ax)
                        if isinstance(v, str) and NUM_RE.match(v) and v != v.strip():
                            dp[ax] = v.strip()
                            n += 1
    return n


def bake_random(model, log):
    for anim in model['animations']:
        length = float(anim['length'])
        ticks = int(round(length / TICK))
        for uid, an in anim.get('animators', {}).items():
            kfs = an.get('keyframes', [])
            for channel in ('position', 'rotation', 'scale'):
                ch = [k for k in kfs if k['channel'] == channel]
                if not ch or not any(has_expression(k) for k in ch):
                    continue
                rng = random.Random('%d|%s|%s|%s' % (SEED, anim['name'], an['name'], channel))
                calc = Molang(rng)
                baked = []
                for i in range(ticks + 1):
                    t = round(i * TICK, 4)
                    v = interpolate(ch, t, anim['loop'], calc)
                    baked.append({
                        'channel': channel,
                        'data_points': [{ax: js_num(round(v[j], 4)) for j, ax in enumerate('xyz')}],
                        'uuid': det_uuid(anim['name'], an['name'], channel, i),
                        'time': t,
                        'color': -1,
                        'interpolation': 'linear',
                    })
                    if channel == 'scale':
                        baked[-1]['uniform'] = False
                an['keyframes'] = [k for k in kfs if k['channel'] != channel] + baked
                kfs = an['keyframes']
                log.append('baked %-9s %-12s %-8s -> %d keys (was %d)' % (anim['name'], an['name'], channel, len(baked), len(ch)))


def group_uuid(model, name):
    groups = model['groups'] if 'groups' in model else None
    if groups:
        hits = [g['uuid'] for g in groups if g['name'] == name]
    else:
        hits = []

        def walk(n):
            if isinstance(n, dict):
                if n.get('name') == name:
                    hits.append(n['uuid'])
                for c in n.get('children', []):
                    walk(c)
        for n in model['outliner']:
            walk(n)
    if len(hits) != 1:
        raise SystemExit('expected exactly one group named %s, found %d' % (name, len(hits)))
    return hits[0]


def animator(anim, model, bone):
    uid = group_uuid(model, bone)
    an = anim['animators'].get(uid)
    if an is None:
        an = {'name': bone, 'type': 'bone', 'rotation_global': False, 'quaternion_interpolation': False, 'keyframes': []}
        anim['animators'][uid] = an
    an.setdefault('keyframes', [])
    return an


def scale_key(anim_name, bone, value, interp='step'):
    v = js_num(value)
    return {'channel': 'scale', 'data_points': [{'x': v, 'y': v, 'z': v}],
            'uuid': det_uuid('fv-edit', anim_name, bone), 'time': 0, 'color': -1,
            'uniform': True, 'interpolation': interp}


def fix_idle_eyes(model, log):
    idle = next(a for a in model['animations'] if a['name'] == 'idle')
    for eye in ('eye1', 'eye2', 'eye3'):
        an = animator(idle, model, eye)
        if any(k['channel'] == 'scale' for k in an['keyframes']):
            raise SystemExit('idle already animates %s scale - refusing to touch it' % eye)
        an['keyframes'].append(scale_key('idle', eye, 0))
        log.append('idle: %s scale 0 (eye4 stays the visible eye, as in Blockbench)' % eye)


def set_loops(model, modes, log):
    for a in model['animations']:
        if a['name'] in modes and a['loop'] != modes[a['name']]:
            log.append('loop mode %s: %s -> %s' % (a['name'], a['loop'], modes[a['name']]))
            a['loop'] = modes[a['name']]


def add_dormant(model, log):
    if any(a['name'] == 'dormant' for a in model['animations']):
        return
    anim = {
        'uuid': det_uuid('fv-edit', 'dormant'), 'name': 'dormant', 'loop': 'loop', 'override': False,
        'length': 1, 'snapping': 20, 'selected': False, 'group_name': '', 'scope': 0,
        'anim_time_update': '', 'blend_weight': '', 'start_delay': '', 'loop_delay': '', 'animators': {},
    }
    hidden = ['eye2', 'eye3', 'eye4', 'smalllight1', 'smalllight2', 'smalllight3', 'smalllight4',
              'laser', 'groundimpact', 'rootshine']
    animator(anim, model, 'eye1')['keyframes'].append(scale_key('dormant', 'eye1', 1))
    for b in hidden:
        animator(anim, model, b)['keyframes'].append(scale_key('dormant', b, 0))
    model['animations'].append(anim)
    log.append('added animation "dormant" (static closed-block pose, effect bones hidden)')


def set_override(model, log):
    changed = [a['name'] for a in model['animations'] if not a.get('override')]
    for a in model['animations']:
        a['override'] = True
    if changed:
        log.append('%s: override=true on %s' % (model['name'], ', '.join(changed)))


# ---------------------------------------------------------------------------
# Blockbench "Export Legacy Project" (js/formats/bbmodel.js) re-implemented
# ---------------------------------------------------------------------------
def invert_molang(v):
    if isinstance(v, (int, float)):
        return -v
    if v == '' or v == '0':
        return v
    if STRICT_NUM_RE.match(v):
        return js_num(-float(v.rstrip('f')))
    if 'return ' in v or ';' in v or '=' in v:
        raise SystemExit('expression too complex for this converter: %r' % v)
    # Blockbench's char-walker for a plain expression
    invert = True
    depth = 0
    last_op = None
    res = ''
    for ch in v:
        if not depth:
            op = None
            had_input = True
            if ch == '-' and last_op not in ('*', '/'):
                if not invert and not last_op:
                    res += '+'
                invert = False
                continue
            elif ch in (' ', '\n'):
                had_input = False
            elif ch == '+' and last_op not in ('*', '/'):
                res += '-'
                invert = False
                continue
            elif ch in '?:':
                invert = True
                op = ch
            elif invert:
                res += '-'
                invert = False
            elif ch in '+-*/&|':
                op = ch
            if had_input:
                last_op = op
        if ch in '{([':
            depth += 1
        elif ch in '})]':
            depth -= 1
        res += ch
    return res


def to_legacy(model):
    m = copy.deepcopy(model)
    m['meta']['format_version'] = '4.10'
    if 'groups' in m:
        groups = {g['uuid']: g for g in m['groups']}

        def build(node):
            if isinstance(node, str):
                return node
            g = copy.deepcopy(groups[node['uuid']])
            g.pop('_static', None)
            g['children'] = [build(c) for c in node.get('children', [])]
            if 'isOpen' in node:
                g['isOpen'] = node['isOpen']
            return g
        m['outliner'] = [build(n) for n in m['outliner']]
        del m['groups']
    for anim in m.get('animations', []):
        for an in anim.get('animators', {}).values():
            for kf in an.get('keyframes', []):
                for dp in kf.get('data_points', []):
                    if kf['channel'] in ('rotation', 'position') and dp.get('x'):
                        dp['x'] = invert_molang(dp['x'])
                    if kf['channel'] == 'rotation' and dp.get('y'):
                        dp['y'] = invert_molang(dp['y'])
    return m


def from_legacy(model):
    """Blockbench 5's loader path for <5.0 files (used by the verifier)."""
    m = copy.deepcopy(model)
    for anim in m.get('animations', []):
        for an in anim.get('animators', {}).values():
            for kf in an.get('keyframes', []):
                for dp in kf.get('data_points', []):
                    if kf['channel'] in ('rotation', 'position') and dp.get('x'):
                        dp['x'] = invert_molang(dp['x'])
                    if kf['channel'] == 'rotation' and dp.get('y'):
                        dp['y'] = invert_molang(dp['y'])
    return m


def dump(model, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(model, f, ensure_ascii=False, separators=(',', ':'))


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    log = []

    boss = json.load(open(os.path.join(SRC, 'endboss (original).bbmodel'), encoding='utf-8'))
    boss['name'] = 'fv_endboss'
    log.append('endboss: cleaned %d numeric strings' % clean_numbers(boss))
    bake_random(boss, log)
    fix_idle_eyes(boss, log)
    add_dormant(boss, log)
    set_override(boss, log)

    proj = json.load(open(os.path.join(SRC, 'projectile (original).bbmodel'), encoding='utf-8'))
    proj['name'] = 'fv_endboss_projectile'
    log.append('projectile: cleaned %d numeric strings' % clean_numbers(proj))
    set_override(proj, log)

    for m in (boss, proj):
        for anim in m['animations']:
            for an in anim.get('animators', {}).values():
                for kf in an.get('keyframes', []):
                    if has_expression(kf):
                        raise SystemExit('expression survived in %s/%s' % (anim['name'], an['name']))

    dump(boss, os.path.join(out_dir, 'fv_endboss (Blockbench 5).bbmodel'))
    dump(proj, os.path.join(out_dir, 'fv_endboss_projectile (Blockbench 5).bbmodel'))
    dump(to_legacy(boss), os.path.join(out_dir, 'fv_endboss.bbmodel'))
    dump(to_legacy(proj), os.path.join(out_dir, 'fv_endboss_projectile.bbmodel'))
    print('\n'.join(log))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', 'build', 'models'))
