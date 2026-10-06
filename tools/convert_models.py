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
       Plus "blank", an empty animation for ModelEngine's default states.
    4b. attack1 / attack2: the laser's "rotate in global space" flag is
       replaced by plain keyframes that give the same pose (see
       bake_global_rotation) - ModelEngine handles the flag differently from
       Blockbench, which turned attack2's downward beam sideways in game.
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
Writes: fv_soulcube.bbmodel, fv_soulcube_projectile.bbmodel            (legacy, for ModelEngine)
        fv_soulcube (Blockbench 5).bbmodel, ... (Blockbench 5)      (same edits, editable)
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
SRC = os.path.join(HERE, 'originals')
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


def x_only(kfs):
    return all(float(k['data_points'][0].get(ax) or 0) == 0 for k in kfs for ax in 'yz')


def bake_global_rotation(model, log):
    """attack1 and attack2 turn the laser bone "in global space" (Blockbench's
    rotation_global animator flag). Blockbench evaluates bones in the order of
    the project's group list, and the laser comes after root but before its
    own parent rootlaser - so in the editor the laser cancels root's rotation
    and keeps rootlaser's:  laser = root(t) * rootlaser(t) * root(t)^-1.
      attack1  root tilts, rootlaser 0         -> the beam stays level
      attack2  root flips -89.87, rootlaser -90 -> the beam points straight
               DOWN, scaled to exactly the drop from the eye to the floor
    ModelEngine does not reproduce that order trick: a global-space bone
    there simply ignores every parent rotation, so attack2's beam stayed
    level ("sideways"). The same motion is written here as plain keyframes
    and the flag is removed, so the hierarchy alone gives the editor's look
    in every engine:
      attack1  laser rotation = -root(t)                 (root cancelled)
      attack2  rootlaser rotation = -90 - root(t), and its random scale's
               Y/Z swapped (the quarter turn that used to sit between that
               scale and the beam now sits above it)
    Both are exact for engines that compose bones as matrices and for the
    ones that multiply scales per axis (verify_models.py check G)."""
    order = [g['name'] for g in model['groups']]
    parent = {}

    def walk(n, p):
        if isinstance(n, dict):
            parent[n['uuid']] = p
            for c in n.get('children', []):
                walk(c, n['uuid'])
    for n in model['outliner']:
        walk(n, None)
    gby = {g['name']: g for g in model['groups']}
    for anim in model['animations']:
        for uid, an in list(anim.get('animators', {}).items()):
            if not an.get('rotation_global'):
                continue
            if an['name'] != 'laser':
                raise SystemExit('rotation_global on unexpected bone %s' % an['name'])
            lz, rl, rt = gby['laser'], gby['rootlaser'], gby['root']
            if parent[lz['uuid']] != rl['uuid'] or parent[rl['uuid']] != rt['uuid']:
                raise SystemExit('laser hierarchy changed - global rotation bake needs root > rootlaser > laser')
            if not order.index('root') < order.index('laser') < order.index('rootlaser'):
                raise SystemExit('unexpected Blockbench bone order around the laser')
            above, p = [], parent[rt['uuid']]
            while p is not None:
                above.append(p)
                p = parent[p]
            if any(anim['animators'].get(u, {}).get('keyframes') for u in above):
                raise SystemExit('%s: a bone above root is animated' % anim['name'])
            chain = [lz, rl, rt] + [g for g in model['groups'] if g['uuid'] in above]
            if any(g.get('rotation', [0, 0, 0]) != [0, 0, 0] for g in chain) or lz['origin'] != rl['origin']:
                raise SystemExit('laser / rootlaser / root rest pose changed')
            if any(k['channel'] in ('rotation', 'position') for k in an['keyframes']):
                raise SystemExit('%s: laser has its own rotation/position keys' % anim['name'])
            root = anim['animators'][rt['uuid']]
            rlan = animator(anim, model, 'rootlaser')
            rk = sorted([k for k in root['keyframes'] if k['channel'] == 'rotation'], key=lambda k: k['time'])
            lk = [k for k in rlan['keyframes'] if k['channel'] == 'rotation']
            if not rk or not x_only(rk) or not x_only(lk):
                raise SystemExit('%s: only X rotations are handled' % anim['name'])
            consts = {float(k['data_points'][0]['x']) for k in lk} or {0.0}
            if len(consts) != 1:
                raise SystemExit('%s: rootlaser rotation is not constant' % anim['name'])
            c = consts.pop()
            # root's angle wherever the beam can be seen
            calc = Molang(random.Random(0))
            lsc = [k for k in an['keyframes'] if k['channel'] == 'scale']
            seen = []
            for i in range(int(round(float(anim['length']) / TICK)) + 1):
                t = round(i * TICK, 4)
                s = interpolate(lsc, t, anim['loop'], calc)
                if s and min(abs(v) for v in s) > 1e-6:
                    seen.append(interpolate(rk, t, anim['loop'], calc)[0])
            quarter = {round(r / 90) for r in seen}
            if len(quarter) != 1:
                raise SystemExit('%s: root turns through a quarter while the beam is visible' % anim['name'])
            q = quarter.pop()

            def key(kf, value, tag):
                return {'channel': 'rotation', 'data_points': [{'x': js_num(round(value, 4)), 'y': '0', 'z': '0'}],
                        'uuid': det_uuid('fv-edit', anim['name'], tag, kf['time']), 'time': kf['time'], 'color': -1,
                        'interpolation': kf['interpolation']}
            if c == 0 and q == 0:
                # the laser itself undoes root's rotation
                an['keyframes'] += [key(k, -float(k['data_points'][0]['x']), 'laser') for k in rk]
                how = 'laser rotation = -root (%d keys)' % len(rk)
            else:
                rlan['keyframes'] = [k for k in rlan['keyframes'] if k['channel'] != 'rotation'] + \
                    [key(k, c - float(k['data_points'][0]['x']), 'rootlaser') for k in rk]
                how = 'rootlaser rotation = %s - root (%d keys)' % (js_num(c), len(rk))
                if q % 2:
                    n = 0
                    for k in rlan['keyframes']:
                        if k['channel'] == 'scale':
                            dp = k['data_points'][0]
                            dp['y'], dp['z'] = dp['z'], dp['y']
                            k['uniform'] = False
                            n += 1
                    how += ', rootlaser scale Y/Z swapped (%d keys)' % n
            an['rotation_global'] = False
            log.append('%s: laser global-space rotation baked: %s' % (anim['name'], how))


def set_loops(model, modes, log):
    for a in model['animations']:
        if a['name'] in modes and a['loop'] != modes[a['name']]:
            log.append('loop mode %s: %s -> %s' % (a['name'], a['loop'], modes[a['name']]))
            a['loop'] = modes[a['name']]


def add_dormant(model, log):
    """A frozen copy of on_spawn's very first frame: the closed block exactly
    as it looks the moment the boss starts waking up (light rays not out
    yet, small side lights tucked in). EVERY bone/channel that any animation moves gets an explicit key,
    so nothing else (e.g. ModelEngine's automatic default idle) can leak
    into the pose. No new geometry."""
    if any(a['name'] == 'dormant' for a in model['animations']):
        return
    anim = {
        'uuid': det_uuid('fv-edit', 'dormant'), 'name': 'dormant', 'loop': 'loop', 'override': False,
        'length': 1, 'snapping': 20, 'selected': False, 'group_name': '', 'scope': 0,
        'anim_time_update': '', 'blend_weight': '', 'start_delay': '', 'loop_delay': '', 'animators': {},
    }
    spawn = next(a for a in model['animations'] if a['name'] == 'on_spawn')
    names = {}
    for a in model['animations']:
        for an in a.get('animators', {}).values():
            for k in an.get('keyframes', []):
                if k['channel'] in ('position', 'rotation', 'scale'):
                    names.setdefault(an['name'], set()).add(k['channel'])
    first = {}
    calc = Molang(random.Random(0))
    for an in spawn['animators'].values():
        by = {}
        for k in an.get('keyframes', []):
            by.setdefault(k['channel'], []).append(k)
        for ch, kfs in by.items():
            if ch in ('position', 'rotation', 'scale'):
                first[(an['name'], ch)] = interpolate(kfs, 0.0, spawn['loop'], calc)
    rest = {'position': [0, 0, 0], 'rotation': [0, 0, 0], 'scale': [1, 1, 1]}
    # the four small side lights are already half out in that frame - tucked
    # away here so the block is a clean block (they pop out as it wakes)
    tucked = {('smalllight%d' % i, 'scale'): [0, 0, 0] for i in range(1, 5)}
    n = 0
    for bone in sorted(names):
        for ch in sorted(names[bone]):
            v = tucked.get((bone, ch), first.get((bone, ch), rest[ch]))
            animator(anim, model, bone)['keyframes'].append({
                'channel': ch, 'data_points': [{'x': js_num(v[0]), 'y': js_num(v[1]), 'z': js_num(v[2])}],
                'uuid': det_uuid('fv-edit', 'dormant', bone, ch), 'time': 0, 'color': -1,
                'uniform': ch == 'scale' and v[0] == v[1] == v[2], 'interpolation': 'step'})
            n += 1
    model['animations'].append(anim)
    log.append('added animation "dormant" (on_spawn frame 0 frozen, %d channels pinned)' % n)


def add_blank(model, log):
    """An animation with no keyframes. ModelEngine plays its default states
    (idle / walk / death ...) on its own; pointing them at "blank" (skill
    defaultstate) means only the states the skills choose ever move a bone."""
    if any(a['name'] == 'blank' for a in model['animations']):
        return
    model['animations'].append({
        'uuid': det_uuid('fv-edit', model['name'], 'blank'), 'name': 'blank', 'loop': 'loop', 'override': False,
        'length': 1, 'snapping': 20, 'selected': False, 'group_name': '', 'scope': 0,
        'anim_time_update': '', 'blend_weight': '', 'start_delay': '', 'loop_delay': '', 'animators': {},
    })
    log.append('%s: added empty animation "blank"' % model['name'])


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
    boss['name'] = 'fv_soulcube'
    log.append('endboss: cleaned %d numeric strings' % clean_numbers(boss))
    bake_random(boss, log)
    fix_idle_eyes(boss, log)
    bake_global_rotation(boss, log)
    add_dormant(boss, log)
    add_blank(boss, log)
    set_override(boss, log)

    proj = json.load(open(os.path.join(SRC, 'projectile (original).bbmodel'), encoding='utf-8'))
    proj['name'] = 'fv_soulcube_projectile'
    log.append('projectile: cleaned %d numeric strings' % clean_numbers(proj))
    add_blank(proj, log)
    set_override(proj, log)

    for m in (boss, proj):
        for anim in m['animations']:
            for an in anim.get('animators', {}).values():
                for kf in an.get('keyframes', []):
                    if has_expression(kf):
                        raise SystemExit('expression survived in %s/%s' % (anim['name'], an['name']))

    dump(boss, os.path.join(out_dir, 'fv_soulcube (Blockbench 5).bbmodel'))
    dump(proj, os.path.join(out_dir, 'fv_soulcube_projectile (Blockbench 5).bbmodel'))
    dump(to_legacy(boss), os.path.join(out_dir, 'fv_soulcube.bbmodel'))
    dump(to_legacy(proj), os.path.join(out_dir, 'fv_soulcube_projectile.bbmodel'))
    print('\n'.join(log))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', 'build', 'models'))
