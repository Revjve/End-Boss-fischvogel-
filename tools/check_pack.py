#!/usr/bin/env python3
"""
Static checker for Soul of the End.

    python3 check_pack.py <plugins_dir> [vanilla_data_dir]

<plugins_dir> is the folder that holds MythicMobs/ and ModelEngine/.
[vanilla_data_dir] (optional) holds minecraft-data sounds_<ver>.json and
particles_<ver>.json files (PrismarineJS format) to validate vanilla names.

Checks
  * every YAML file parses; no duplicate top-level keys across files
  * every line: balanced {} [] and quotes, known mechanic / targeter / inline
    condition / trigger names (catches typos MythicMobs would only log)
  * references: skill{s=}, randomskill{skills=}, projectile onTick/onHit/onEnd,
    summon{type=}, projectile mob= -> must exist
  * ModelEngine: model{mid=} blueprint exists, state{s=} animation exists in it
  * sounds: vanilla only - every event exists in every given Minecraft
    version, no custom (namespaced) sounds
  * particles exist in every given Minecraft version
  * variables: every variable that is read is also written somewhere (skill
    scope separately; caster and target scope both live on an entity);
    INTEGER variables never get a literal decimal; variableinrange ranges are well formed (low <= high)
  * math in quoted values: balanced parentheses, known functions only
  * auras tested with hasaura are applied somewhere
  * skills that nothing references (warnings)
"""
import glob
import json
import os
import re
import sys

import yaml

MECHANICS = {
    'skill', 'randomskill', 'setvariable', 'variableadd', 'variableunset', 'delay', 'state',
    'model', 'bodyrotation', 'brightness', 'renderinit', 'teleport', 'rotatetowards',
    'setrotation', 'summon', 'remove', 'signal', 'cancelevent', 'modifydamage', 'damage',
    'throw', 'aura', 'sound', 'particles', 'particlering', 'particleline', 'projectile',
    'sendactionmessage', 'setvarloc', 'defaultstate', 'sudoskill', 'cancelskill', 'recoil',
}
TARGETERS = {
    'self', 'selflocation', 'nearestplayer', 'playersinradius', 'forward', 'variablelocation',
    'owner', 'trigger', 'origin', 'playersnearorigin', 'mobsinradius', 'flooroftargets',
}
CONDITIONS = {
    'variableequals', 'variableinrange', 'variableisset', 'playerwithin', 'entitytype',
    'gamemode', 'hasaura', 'mobsinradius', 'blocktype',
}
TRIGGERS = {'onspawn', 'ontimer', 'ondamaged', 'oninteract', 'onsignal', 'ondeath', 'onload'}
MATH_FUNCS = {'floor', 'ceil', 'sqrt', 'abs', 'sin', 'cos', 'atan2', 'toradian', 'todegree',
              'min', 'max', 'signum', 'random'}
PARTICLE_KEYS = ('particle', 'p')

errors = []
warnings = []


def err(where, msg):
    errors.append(f'{where}: {msg}')


def warn(where, msg):
    warnings.append(f'{where}: {msg}')


# --------------------------------------------------------------------------
#  line parsing
# --------------------------------------------------------------------------
def split_top(s, sep=';'):
    out, depth, q, cur = [], 0, False, ''
    for ch in s:
        if ch == '"':
            q = not q
        elif not q and ch in '{[':
            depth += 1
        elif not q and ch in '}]':
            depth -= 1
        if ch == sep and depth == 0 and not q:
            out.append(cur)
            cur = ''
        else:
            cur += ch
    out.append(cur)
    return out


def parse_args(body):
    args = {}
    for part in split_top(body):
        if not part.strip():
            continue
        if '=' not in part:
            args[part.strip()] = None
            continue
        k, v = part.split('=', 1)
        v = v.strip()
        if len(v) >= 2 and v[0] == '"' and v[-1] == '"':
            v = v[1:-1]
        args[k.strip().lower()] = v
    return args


TOKEN = re.compile(r'(?P<pre>[@?~!]*)(?P<name>[A-Za-z_:][A-Za-z0-9_:.]*)(?P<body>\{)?')


def tokenize(line):
    """Split a skill line into tokens: (prefix, name, args-dict or None)."""
    toks, i, n = [], 0, len(line)
    while i < n:
        if line[i].isspace():
            i += 1
            continue
        m = TOKEN.match(line, i)
        if not m:
            # bare number (delay 20 / chance) or something odd
            m2 = re.match(r'-?[0-9.]+', line[i:])
            if m2:
                toks.append(('#', m2.group(0), None))
                i += len(m2.group(0))
                continue
            raise ValueError(f'cannot parse at {line[i:]!r}')
        pre, name = m.group('pre'), m.group('name')
        j = m.end()
        args = None
        if m.group('body'):
            depth, q, k = 1, False, j
            while k < n and depth:
                if line[k] == '"':
                    q = not q
                elif not q and line[k] == '{':
                    depth += 1
                elif not q and line[k] == '}':
                    depth -= 1
                k += 1
            if depth:
                raise ValueError('unbalanced {}')
            args = parse_args(line[j:k - 1])
            j = k
        toks.append((pre, name, args))
        i = j
    return toks


def parse_line(line):
    toks = tokenize(line)
    if not toks:
        raise ValueError('empty line')
    pre, mech, margs = toks[0]
    if pre:
        raise ValueError('line does not start with a mechanic')
    out = {'mech': mech.lower(), 'args': margs or {}, 'targeter': None, 'conds': [],
           'trigger': None, 'numbers': []}
    for pre, name, args in toks[1:]:
        if pre == '#':
            out['numbers'].append(name)
        elif pre == '@':
            if out['targeter']:
                raise ValueError('two targeters')
            out['targeter'] = (name.lower(), args or {})
        elif pre.startswith('?'):
            out['conds'].append((pre, name.lower(), args or {}))
        elif pre == '~':
            out['trigger'] = name
        else:
            raise ValueError(f'unexpected token {pre}{name}')
    return out


def parse_condition(text):
    """'variableequals{var=..;value=..} true' -> (name, args, bool)"""
    m = re.match(r'\s*([A-Za-z_]+)(\{.*\})?\s*(true|false)?\s*$', text)
    if not m:
        raise ValueError(f'bad condition {text!r}')
    args = parse_args(m.group(2)[1:-1]) if m.group(2) else {}
    return m.group(1).lower(), args, (m.group(3) or 'true') == 'true'


# --------------------------------------------------------------------------
def balanced(s):
    st, q = [], False
    pairs = {'}': '{', ']': '[', ')': '('}
    for ch in s:
        if ch == '"':
            q = not q
            continue
        if ch in '{[(':
            st.append(ch)
        elif ch in '}])':
            if not st or st[-1] != pairs[ch]:
                return False
            st.pop()
    return not st and not q


def check_math(where, expr):
    if not balanced(expr):
        err(where, f'unbalanced parentheses in {expr!r}')
    for f in re.findall(r'([A-Za-z_][A-Za-z0-9_]*)\s*\(', expr):
        if f.lower() not in MATH_FUNCS:
            err(where, f'unknown math function {f}() in {expr!r}')


VAR_READ = re.compile(r'<(caster|target|skill)\.var\.([A-Za-z0-9_]+)>')
PLACEHOLDER = re.compile(r'<([^<>]+)>')
# Only the caster's own position / variables: MythicMobs documents exact
# coordinates for the caster only, so other entities are measured with the
# probe (sudoskill) instead of <target.l.*> / <target.var.*> placeholders.
KNOWN_PH = re.compile(r'^caster\.(l\.(x|y|z)\.double|l\.yaw|hp|mhp)$|^(caster|skill)\.var\.[A-Za-z0-9_]+$|^#[0-9a-fA-F]{6}$|^/?(gray|dark_gray|red|white|gradient:[^>]*|/gradient)$')


# Interruptible chains: every metaskill that waits (delay) remembers the
# attack epoch before its first delay and stops after each delay when the
# boss was interrupted (hit reaction, reset, death) in the meantime.
EPOCH_PH = '<skill.var.ep>'
EPOCH_SET = 'setvariable{var=skill.ep;type=INTEGER;value=<caster.var.fv_endsoul_epoch>}'
EPOCH_GUARD = 'cancelskill ?!variableequals{var=caster.fv_endsoul_epoch;value=<skill.var.ep>}'
UNGUARDED = {'fv_endsoul_block_wake', 'fv_endsoul_block_peek', 'fv_endsoul_wake_timeline',
             'fv_endsoul_wake_handover', 'fv_endsoul_death_timeline', 'fv_endsoul_snap_floor'}


def check_guards(name, where, lines):
    if name in UNGUARDED:
        return
    delays = [i for i, l in enumerate(lines) if re.match(r'delay\b', l.strip())]
    if not delays:
        return
    if EPOCH_SET not in lines[:delays[0]]:
        err(where, 'waits (delay) without remembering the attack epoch first (skill.ep)')
    for i in delays:
        if i + 1 >= len(lines) or lines[i + 1].strip() != EPOCH_GUARD:
            err(where, f'no epoch guard right after "{lines[i]}"')


def main():
    plugins = sys.argv[1]
    vdir = sys.argv[2] if len(sys.argv) > 2 else None
    mm = os.path.join(plugins, 'MythicMobs', 'packs', 'fv_endsoul')
    me = os.path.join(plugins, 'ModelEngine', 'blueprints')

    # ---- vanilla data ----
    vsounds, vparts = [], []
    if vdir:
        for f in sorted(glob.glob(os.path.join(vdir, 'sounds_*.json'))):
            vsounds.append((os.path.basename(f), {x['name'] for x in json.load(open(f))}))
        for f in sorted(glob.glob(os.path.join(vdir, 'particles_*.json'))):
            try:
                vparts.append((os.path.basename(f), {x['name'] for x in json.load(open(f))}))
            except Exception:
                pass

    # ---- blueprints ----
    anims = {}
    for f in glob.glob(os.path.join(me, '*.bbmodel')):
        d = json.load(open(f, encoding='utf-8'))
        mid = os.path.splitext(os.path.basename(f))[0]
        anims[mid] = {a['name'] for a in d.get('animations', [])}
        if d['meta'].get('format_version') != '4.10' or 'groups' in d:
            err(f, 'blueprint is not in the legacy 4.10 layout')

    # ---- yaml ----
    defs, mobs, origin = {}, {}, {}
    files = sorted(glob.glob(os.path.join(mm, '**', '*.yml'), recursive=True))
    docs = {}
    for f in files:
        try:
            d = yaml.safe_load(open(f, encoding='utf-8'))
        except Exception as e:
            err(f, f'YAML: {e}')
            continue
        docs[f] = d
        if os.path.basename(f) == 'packinfo.yml':
            continue
        kind = 'mob' if os.sep + 'mobs' + os.sep in f else 'skill'
        for k in d:
            if k in origin:
                err(f, f'duplicate key {k} (also in {origin[k]})')
            origin[k] = f
            (mobs if kind == 'mob' else defs)[k] = d[k]

    refs = {}            # skill name -> set(referrers)
    mob_refs = {}
    writes = {}          # (scope, name) -> [where]
    reads = {}
    int_vars = set()
    aura_set, aura_read = set(), {}

    def add_ref(n, w):
        refs.setdefault(n, set()).add(w)

    def note_read(scope, name, where):
        reads.setdefault((scope, name), []).append(where)

    def scan_values(where, args):
        for v in (args or {}).values():
            if v is None:
                continue
            for sc, nm in VAR_READ.findall(v):
                note_read(sc, nm, where)
            for ph in PLACEHOLDER.findall(v):
                if not KNOWN_PH.match(ph):
                    if re.match(r'^(target|trigger)\.', ph):
                        err(where, f'<{ph}> - only caster placeholders are reliable, use the probe (sudoskill)')
                    else:
                        err(where, f'unknown placeholder <{ph}>')

    def check_var_attr(where, args, writing=False, typ=None):
        v = args.get('var') or args.get('variable')
        if not v:
            err(where, 'missing var=')
            return
        if '.' not in v:
            err(where, f'variable {v} has no scope (caster./target./skill.)')
            return
        sc, nm = v.split('.', 1)
        if sc not in ('caster', 'target', 'skill'):
            err(where, f'bad variable scope in {v}')
        if writing:
            writes.setdefault((sc, nm), []).append(where)
        else:
            note_read(sc, nm, where)

    def check_condition(where, name, args):
        if name not in CONDITIONS:
            err(where, f'unknown condition {name}')
        if name.startswith('variable'):
            check_var_attr(where, args)
        if name == 'variableinrange':
            m = re.match(r'^(-?[0-9.]+)to(-?[0-9.]+)$', args.get('value', ''))
            if not m:
                err(where, f'bad range {args.get("value")!r}')
            elif float(m.group(1)) > float(m.group(2)):
                err(where, f'empty range {args.get("value")!r}')
        if name == 'variableequals' and '<' in (args.get('value') or '') and \
                (args.get('value'), args.get('var')) != (EPOCH_PH, 'caster.fv_endsoul_epoch'):
            err(where, 'placeholder in a condition value - use a math variable instead')
        if name == 'hasaura':
            a = args.get('auraname') or args.get('aura')
            aura_read.setdefault(a, []).append(where)
        scan_values(where, args)

    def check_line(where, line, in_mob):
        if not balanced(line):
            err(where, f'unbalanced brackets/quotes: {line}')
            return
        try:
            p = parse_line(line)
        except ValueError as e:
            err(where, f'{e}: {line}')
            return
        mech, a = p['mech'], p['args']
        if mech not in MECHANICS:
            err(where, f'unknown mechanic {mech}')
        if p['trigger'] and not in_mob:
            err(where, f'trigger ~{p["trigger"]} inside a metaskill')
        if in_mob and not p['trigger']:
            err(where, 'mob skill line without a trigger')
        if p['trigger']:
            t = p['trigger'].split(':')[0].lower()
            if t not in TRIGGERS:
                err(where, f'unknown trigger {p["trigger"]}')
        if p['targeter']:
            tn, ta = p['targeter']
            if tn not in TARGETERS:
                err(where, f'unknown targeter @{tn}')
            if tn == 'variablelocation':
                check_var_attr(where, ta)
            if tn == 'mobsinradius':
                for t in (ta.get('types') or ta.get('type') or '').split(','):
                    mob_refs.setdefault(t, []).append(where)
            scan_values(where, ta)
        for pre, cn, ca in p['conds']:
            if pre not in ('?', '?!', '?~', '?~!'):
                err(where, f'bad inline condition prefix {pre}')
            check_condition(where, cn, ca)
            v = ca.get('var', '')
            if v.startswith('target.') and pre.startswith('?') and not pre.startswith('?~'):
                err(where, 'inline ?condition reads a target variable - it is tested on the caster')
        scan_values(where, a)
        # ---- mechanic specifics
        if mech in ('skill', 'sudoskill'):
            s = a.get('s') or a.get('skill')
            add_ref(s, where)
        elif mech == 'randomskill':
            for s in (a.get('skills') or a.get('s') or '').split(','):
                add_ref(s, where)
        elif mech == 'projectile':
            for k in ('ontick', 'onhit', 'onend', 'onstart'):
                if k in a:
                    add_ref(a[k], where)
            if a.get('bullettype', '').upper() == 'MOB':
                mob_refs.setdefault(a.get('mob'), []).append(where)
        elif mech == 'summon':
            mob_refs.setdefault(a.get('type'), []).append(where)
        elif mech in ('setvariable', 'variableadd', 'variableunset', 'setvarloc'):
            check_var_attr(where, a, writing=(mech != 'variableunset'))
            if mech == 'setvariable':
                typ = (a.get('type') or 'INTEGER').upper()
                val = a.get('value', a.get('val', ''))
                if typ not in ('INTEGER', 'FLOAT', 'STRING'):
                    err(where, f'unknown variable type {typ}')
                if typ == 'INTEGER' and re.fullmatch(r'-?[0-9]+\.[0-9]+', val or ''):
                    err(where, f'INTEGER variable set to a decimal {val}')
                if typ in ('INTEGER', 'FLOAT') and val and re.search(r'[()+*/^%<>=-]', val.lstrip('-')):
                    check_math(where, val)
                if typ == 'INTEGER':
                    int_vars.add(a.get('var'))
        elif mech == 'delay':
            if p['numbers']:
                if not re.fullmatch(r'[0-9]+', p['numbers'][0]):
                    err(where, f'delay must be a whole, non-negative number of ticks: {p["numbers"][0]}')
            elif not a.get('ticks'):
                err(where, 'delay without ticks')
        elif mech == 'defaultstate':
            mid = a.get('mid') or a.get('m')
            st = a.get('state') or a.get('s')
            if (a.get('type') or '').upper() not in ('IDLE', 'WALK', 'JUMP_START', 'JUMP', 'JUMP_END', 'SPAWN', 'DEATH'):
                err(where, f'unknown default state type {a.get("type")}')
            if mid not in anims or st not in anims[mid]:
                err(where, f'default state animation {st} not in {mid}')
        elif mech in ('state', 'model', 'bodyrotation', 'brightness', 'renderinit'):
            mid = a.get('mid') or a.get('m') or a.get('model')
            if mid not in anims:
                err(where, f'unknown blueprint {mid}')
            elif mech == 'state':
                s = a.get('s') or a.get('state')
                if s not in anims[mid]:
                    err(where, f'animation {s} not in {mid}')
        elif mech == 'sound':
            s = a.get('s') or a.get('sound')
            if ':' in s and not s.startswith('minecraft:'):
                err(where, f'custom sound {s} - the pack uses vanilla sounds only')
            else:
                s2 = s.replace('minecraft:', '')
                for vn, vs in vsounds:
                    if s2 not in vs:
                        err(where, f'vanilla sound {s2} does not exist in {vn}')
            for k in ('p', 'pitch'):
                if k in a and not (0.5 <= float(a[k]) <= 2.0):
                    warn(where, f'pitch {a[k]} outside 0.5-2 (Minecraft clamps it)')
        elif mech in ('particles', 'particlering', 'particleline'):
            pn = next((a[k] for k in PARTICLE_KEYS if k in a), None)
            if not pn:
                err(where, 'particle missing')
            else:
                for vn, vp in vparts:
                    if pn.lower() not in vp:
                        err(where, f'particle {pn} does not exist in {vn}')
        elif mech == 'aura':
            aura_set.add(a.get('auraname') or a.get('aura'))

    # ---- walk everything ----
    for name, body in defs.items():
        where = f'{os.path.basename(origin[name])}:{name}'
        if not isinstance(body, dict) or 'Skills' not in body:
            err(where, 'metaskill without Skills')
            continue
        for k in body:
            if k not in ('Skills', 'Conditions', 'TargetConditions', 'Cooldown'):
                err(where, f'unknown metaskill key {k}')
        for c in body.get('Conditions', []) + body.get('TargetConditions', []):
            try:
                cn, ca, _ = parse_condition(c)
                check_condition(where, cn, ca)
            except ValueError as e:
                err(where, str(e))
        for line in body['Skills']:
            check_line(where, line, in_mob=False)
        check_guards(name, where, body['Skills'])
    for name, body in mobs.items():
        where = f'{os.path.basename(origin[name])}:{name}'
        if 'Template' in body:
            if body['Template'] not in mobs:
                err(where, f'unknown template {body["Template"]}')
        for line in body.get('Skills', []) or []:
            check_line(where, line, in_mob=True)

    # ---- cross checks ----
    for s, w in refs.items():
        if s not in defs:
            err(sorted(w)[0], f'skill {s} is not defined')
    for m, w in mob_refs.items():
        if m not in mobs:
            err(w[0], f'mob {m} is not defined')
    used = set(refs)
    for name in defs:
        if name not in used:
            warn(name, 'skill is never used')
    # caster and target variables both live on an entity: the boss writes
    # target.x on a player and the player (casting via sudoskill) reads it as
    # caster.x, and the other way round
    ent_writes = {nm for (sc, nm) in writes if sc in ('caster', 'target')}
    for (sc, nm), w in reads.items():
        if (sc, nm) not in writes and not (sc in ('caster', 'target') and nm in ent_writes):
            err(w[0], f'variable {sc}.{nm} is read but never set')
    for a, w in aura_read.items():
        if a not in aura_set:
            err(w[0], f'aura {a} is tested but never applied')

    for w in warnings:
        print('WARN ', w)
    for e in errors:
        print('ERROR', e)
    print(f'{len(defs)} skills, {len(mobs)} mobs, {len(anims)} blueprints, '
          f'vanilla sounds checked against {len(vsounds)} sound lists / '
          f'{len(vparts)} particle lists: {len(errors)} errors, {len(warnings)} warnings')
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
