#!/usr/bin/env python3
"""
Fight simulator for FischVogel's End Boss.

Runs the real MythicMobs files of the pack through a small interpreter of
the mechanics they use (tick by tick, with delays, timers, signals,
cooldowns, conditions, projectiles and ModelEngine animation states) and a
few scripted players who walk around, dodge a little and hit the boss when
it is open. Many fights with different random seeds are played and a set of
invariants is checked:

  * exactly one attack chain at a time (pick -> attack -> after_attack)
  * the fight-loop watchdog never has to step in
  * the same attack TYPE is never picked twice in a row
  * every attack ends with its tick loops stopped (beam, down beam, shots)
  * the model always has a body animation playing (no rest-pose frame)
  * the boss' height in pixels (fv_soulcube_hpx) always matches its real height
  * no variable is ever read before it was set
  * phases 2 / 3 start at 66% / 33% health, each roar plays once
  * death hands the model to the effect carrier, which lives through the
    whole death animation; the block cools down and re-forms; "everyone
    left" resets the fight; a chunk unload / restart makes the block re-form
  * the block stays hidden while its boss is waking up or fighting
  * only the caster's own position / variables are read through
    placeholders (everything else goes through the sudoskill probe)

~onTimer skills run on a global clock (random phase per fight), as in
MythicMobs - not counted from the mob's spawn.

Semantics that MythicMobs does not document precisely are simulated BOTH
ways (skill-scope variables shared with or copied into sub-skills); the
pack must pass in every mode.

    python3 simulate_fight.py <plugins_dir> [fights] [--verbose]
"""
import heapq
import json
import math
import os
import random
import re
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_pack import parse_line, parse_condition  # noqa: E402

FLOOR = 64.0
LOWEST = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_lowest.json')))['lowest']
ARENA_R = 22.0


class SimError(Exception):
    pass


# --------------------------------------------------------------------------
class Loc:
    __slots__ = ('x', 'y', 'z', 'yaw')

    def __init__(self, x, y, z, yaw=0.0):
        self.x, self.y, self.z, self.yaw = x, y, z, yaw

    def copy(self):
        return Loc(self.x, self.y, self.z, self.yaw)


class Ent:
    _next = 1

    def __init__(self, sim, mtype, loc, owner=None, player=False, name=None):
        self.sim = sim
        self.id = Ent._next
        Ent._next += 1
        self.mtype = mtype
        self.name = name or mtype
        self.x, self.y, self.z, self.yaw = loc.x, loc.y, loc.z, loc.yaw
        self.owner = owner
        self.player = player
        self.vars, self.exp = {}, {}
        self.alive, self.removed = True, False
        self.model, self.anims = None, {}
        self.auras = {}
        self.cooldowns = {}
        self.spawn_tick = sim.tick
        self.hp = self.mhp = 1.0
        self.gamemode = 'SURVIVAL'
        self.damage_taken = {}

    @property
    def loc(self):
        return Loc(self.x, self.y, self.z, self.yaw)

    def var(self, name):
        if name in self.exp and self.exp[name] is not None and self.sim.tick >= self.exp[name]:
            self.vars.pop(name, None)
            self.exp.pop(name, None)
        return self.vars.get(name)

    def isset(self, name):
        return self.var(name) is not None

    def __repr__(self):
        return f'<{self.name}#{self.id}>'


class Meta:
    def __init__(self, caster, targets, trigger=None, origin=None, svars=None, event=None,
                 chain=None):
        self.caster, self.targets = caster, targets
        self.trigger, self.origin, self.event = trigger, origin, event
        self.svars = {} if svars is None else svars
        self.chain = chain or []

    def child(self, targets, name, share):
        return Meta(self.caster, targets, self.trigger, self.origin,
                    self.svars if share else dict(self.svars), self.event,
                    (self.chain + [name])[-12:])


# --------------------------------------------------------------------------
class Sim:
    def __init__(self, plugins, seed, share_skill_vars=True, players=2, verbose=False,
                 scenario='kill'):
        self.rng = random.Random(seed)
        self.seed = seed
        self.share = share_skill_vars
        self.verbose = verbose
        self.scenario = scenario
        self.tick = 0
        self.clock0 = self.rng.randint(0, 9999)
        self.queue, self.seq = [], 0
        self.ents, self.projs = [], []
        self.errors, self.log = [], []
        self.stats = {'attacks': [], 'damage': {}, 'phase_breaks': 0, 'watchdog': 0,
                      'shots': 0, 'hits_on_boss': 0, 'cancelled_hits': 0}
        self.active_attacks = 0
        self._load(plugins)
        self.n_players = players

    # ---- loading ----
    def _load(self, plugins):
        mm = os.path.join(plugins, 'MythicMobs', 'packs', 'fv_soulcube')
        self.skills, self.mobs, self.parsed = {}, {}, {}
        for root, _, files in os.walk(mm):
            for f in files:
                if not f.endswith('.yml') or f == 'packinfo.yml':
                    continue
                d = yaml.safe_load(open(os.path.join(root, f)))
                (self.mobs if 'mobs' in root else self.skills).update(d)
        for name, sk in self.skills.items():
            self.parsed[name] = [parse_line(l) for l in sk['Skills']]
            sk['_cond'] = [parse_condition(c) for c in sk.get('Conditions', [])]
            sk['_tcond'] = [parse_condition(c) for c in sk.get('TargetConditions', [])]
        # resolve templates
        for name, m in self.mobs.items():
            lines = list(m.get('Skills') or [])
            if 'Template' in m:
                lines = list(self.mobs[m['Template']].get('Skills') or []) + lines
            m['_lines'] = [parse_line(l) for l in lines]
        self.blueprints = {}
        for f in os.listdir(os.path.join(plugins, 'ModelEngine', 'blueprints')):
            d = json.load(open(os.path.join(plugins, 'ModelEngine', 'blueprints', f)))
            self.blueprints[f[:-8]] = {a['name']: (a['loop'], a['length']) for a in d['animations']}

    # ---- scheduling ----
    def at(self, tick, fn):
        self.seq += 1
        heapq.heappush(self.queue, (tick, self.seq, fn))

    def error(self, msg, meta=None):
        where = ' > '.join(meta.chain[-6:]) if meta else ''
        e = f'[seed {self.seed} t={self.tick}] {msg}' + (f'   ({where})' if where else '')
        if e not in self.errors:
            self.errors.append(e)

    # ---- entities ----
    def spawn(self, mtype, loc, owner=None, trigger=None):
        e = Ent(self, mtype, loc, owner)
        m = self.mobs[mtype]
        e.hp = e.mhp = float(m.get('Health', 1))
        e.despawn = str((m.get('Options') or {}).get('Despawn', ''))
        if 'Template' in m:
            e.despawn = str((self.mobs[m['Template']].get('Options') or {}).get('Despawn', e.despawn))
        self.ents.append(e)
        self.fire(e, 'onspawn', trigger)
        return e

    def fire(self, e, trig, trigger=None, event=None, signal=None):
        if e.removed:
            return
        for p in self.mobs[e.mtype]['_lines']:
            t = p['trigger'].lower()
            if signal is not None:
                if t != f'onsignal:{signal.lower()}':
                    continue
            elif t != trig:
                continue
            meta = Meta(e, [e], trigger=trigger, event=event, chain=[f'{e.mtype}~{t}'])
            self.exec_line(p, meta, f'{e.mtype}')

    def remove(self, e):
        if e.removed:
            return
        if e.mtype.endswith('_deathfx') and self.tick - e.spawn_tick < 150:
            self.error(f'death effect removed after {self.tick - e.spawn_tick} ticks - the death animation is cut short')
        e.removed = True
        e.alive = False
        e.model = None
        e.anims = {}

    # ---- placeholders and math ----
    def ph(self, text, meta, target):
        if text is None:
            return text

        def rep(m):
            key = m.group(1)
            parts = key.split('.')
            if parts[0] == 'skill' and parts[1] == 'var':
                if parts[2] not in meta.svars:
                    self.error(f'skill variable {parts[2]} read before it was set', meta)
                    return '0'
                return str(meta.svars[parts[2]])
            ent = meta.caster if parts[0] == 'caster' else target
            if parts[0] != 'caster' and parts[1] in ('l', 'var'):
                # MythicMobs only documents exact coordinates (and reliable
                # variable placeholders) for the caster of a skill
                self.error(f'<{key}> - only caster placeholders are reliable, use the probe (sudoskill)', meta)
            if not isinstance(ent, Ent):
                self.error(f'<{key}> has no entity', meta)
                return '0'
            if parts[1] == 'var':
                v = ent.var(parts[2])
                if v is None:
                    self.error(f'{parts[0]} variable {parts[2]} read while unset', meta)
                    return '0'
                return str(v)
            if parts[1] == 'l':
                return repr({'x': ent.x, 'y': ent.y, 'z': ent.z, 'yaw': ent.yaw}[parts[2]])
            if parts[1] == 'hp':
                return repr(ent.hp)
            if parts[1] == 'mhp':
                return repr(ent.mhp)
            self.error(f'unknown placeholder <{key}>', meta)
            return '0'
        return re.sub(r'<((?:caster|target|skill)\.[A-Za-z0-9_.]+)>', rep, text)

    def math(self, expr, meta):
        e = expr.replace('^', '**')
        env = {'floor': math.floor, 'ceil': math.ceil, 'sqrt': math.sqrt, 'abs': abs,
               'sin': math.sin, 'cos': math.cos, 'atan2': math.atan2,
               'toradian': math.radians, 'todegree': math.degrees, 'min': min, 'max': max,
               'signum': lambda v: (v > 0) - (v < 0),
               'random': lambda a, b: self.rng.uniform(a, b)}
        try:
            v = eval(e, {'__builtins__': {}}, env)
        except Exception as ex:
            self.error(f'math failed: {expr!r} ({ex})', meta)
            return 0.0
        return float(v)

    def num(self, text, meta, target):
        t = self.ph(text, meta, target)
        try:
            return float(t)
        except ValueError:
            return self.math(t, meta)

    # ---- conditions ----
    def cond(self, name, args, meta, ent):
        if name in ('variableequals', 'variableinrange', 'variableisset'):
            sc, vn = args['var'].split('.', 1)
            holder = meta.svars if sc == 'skill' else (meta.caster if sc == 'caster' else ent)
            if sc == 'skill':
                v = holder.get(vn)
            else:
                if not isinstance(holder, Ent):
                    return False
                v = holder.var(vn)
            if name == 'variableisset':
                return v is not None
            if v is None:
                return False
            if name == 'variableequals':
                want = self.ph(args['value'], meta, ent)
                try:
                    return abs(float(v) - float(want)) < 1e-9
                except (TypeError, ValueError):
                    return str(v) == str(want)
            lo, hi = re.match(r'^(-?[0-9.]+)to(-?[0-9.]+)$', args['value']).groups()
            try:
                return float(lo) <= float(v) <= float(hi)
            except (TypeError, ValueError):
                self.error(f'variableinrange on non-number {args["var"]}={v!r}', meta)
                return False
        if name == 'playerwithin':
            d = float(args['d'])
            return any(self.dist(ent, p) <= d for p in self.players())
        if name == 'entitytype':
            return isinstance(ent, Ent) and ent.player and args['t'].upper() == 'PLAYER'
        if name == 'gamemode':
            return isinstance(ent, Ent) and ent.player and ent.gamemode in args['m'].upper().split(',')
        if name == 'hasaura':
            a = args.get('auraname') or args.get('aura')
            return isinstance(ent, Ent) and ent.auras.get(a, -1) > self.tick
        self.error(f'condition {name} not simulated', meta)
        return True

    # ---- geometry ----
    @staticmethod
    def dist(a, b):
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)

    def players(self):
        return [e for e in self.ents if e.player and not e.removed]

    def targeter(self, tg, meta):
        name, a = tg
        c = meta.caster
        if name == 'self':
            return [c]
        if name == 'selflocation':
            l = c.loc
            l.y += float(a.get('y', 0))
            return [l]
        if name == 'nearestplayer':
            r = float(a.get('r', 16))
            ps = [p for p in self.players() if self.dist(c, p) <= r]
            return [min(ps, key=lambda p: self.dist(c, p))] if ps else []
        if name == 'playersinradius':
            r = float(a.get('r', 16))
            return [p for p in self.players() if self.dist(c, p) <= r]
        if name == 'forward':
            f = float(a.get('f', 1))
            yr = math.radians(c.yaw)
            return [Loc(c.x - math.sin(yr) * f, c.y + float(a.get('yo', a.get('y', 0))),
                        c.z + math.cos(yr) * f, c.yaw)]
        if name == 'variablelocation':
            sc, vn = a['var'].split('.', 1)
            v = c.var(vn)
            if v is None:
                self.error(f'location variable {vn} unset', meta)
                return []
            l = v.copy()
            l.y += float(a.get('y', 0))
            return [l]
        if name == 'owner':
            return [c.owner] if c.owner is not None and not c.owner.removed else []
        if name == 'trigger':
            return [meta.trigger] if meta.trigger is not None else []
        if name == 'origin':
            return [meta.origin] if meta.origin is not None else [c.loc]
        if name == 'mobsinradius':
            r = float(a.get('r', 5))
            types = (a.get('types') or a.get('type')).split(',')
            return [e for e in self.ents if not e.removed and not e.player and e.mtype in types
                    and self.dist(c, e) <= r]
        if name == 'playersnearorigin':
            o = meta.origin or c.loc
            r = float(a.get('r', 5))
            return [p for p in self.players() if self.dist(o, p) <= r]
        self.error(f'targeter {name} not simulated', meta)
        return []

    # ---- skills ----
    def run_skill(self, name, meta):
        if name not in self.skills:
            self.error(f'unknown skill {name}', meta)
            return
        sk = self.skills[name]
        c = meta.caster
        for cn, ca, want in sk['_cond']:
            if self.cond(cn, ca, meta, c) != want:
                return
        if 'Cooldown' in sk:
            if self.tick < c.cooldowns.get(name, -1):
                return
            c.cooldowns[name] = self.tick + float(sk['Cooldown']) * 20
        if sk['_tcond']:
            meta.targets = [t for t in meta.targets
                            if all(self.cond(cn, ca, meta, t) == want for cn, ca, want in sk['_tcond'])]
        self.hook(name, meta)
        self.run_lines(name, 0, meta)

    def run_lines(self, name, i, meta):
        lines = self.parsed[name]
        while i < len(lines):
            p = lines[i]
            if p['mech'] == 'delay':
                if p['numbers']:
                    d = int(p['numbers'][0])
                else:
                    d = self.num(p['args']['ticks'], meta, meta.caster)
                    if d != int(d) or d < 0:
                        self.error(f'bad delay {d}', meta)
                    d = int(d)
                if d <= 0:
                    self.error('delay of 0 ticks', meta)
                self.at(self.tick + max(0, d), lambda i=i: self.run_lines(name, i + 1, meta))
                return
            self.exec_line(p, meta, name)
            i += 1

    def exec_line(self, p, meta, name):
        d = p['args'].get('delay')
        if d:
            self.at(self.tick + int(d), lambda: self.exec_now(p, meta, name))
        else:
            self.exec_now(p, meta, name)

    def exec_now(self, p, meta, name):
        c = meta.caster
        for pre, cn, ca in p['conds']:
            ent = meta.trigger if pre.startswith('?~') else c
            ok = self.cond(cn, ca, meta, ent)
            if '!' in pre:
                ok = not ok
            if not ok:
                return
        targets = self.targeter(p['targeter'], meta) if p['targeter'] else meta.targets
        m, a = p['mech'], p['args']
        if m in ('skill', 'randomskill'):
            if m == 'skill':
                s = a.get('s')
            else:
                s = self.rng.choice((a.get('skills') or a.get('s')).split(','))
            self.run_skill(s, meta.child(targets, s, self.share))
        elif m == 'sudoskill':
            for t in targets:
                if not isinstance(t, Ent) or t.removed:
                    continue
                trig = c if a.get('cat', a.get('setcasterastrigger')) == 'true' else meta.trigger
                self.run_skill(a['s'], Meta(t, [], trigger=trig, origin=t.loc, event=None,
                                            svars={} , chain=(meta.chain + ['sudo:' + a['s']])[-12:]))
        elif m in ('setvariable', 'variableadd', 'variableunset', 'setvarloc'):
            sc, vn = a['var'].split('.', 1)
            tlist = targets if sc == 'target' else (targets[:1] or [c])
            for t in tlist:
                holder = meta.svars if sc == 'skill' else (c if sc == 'caster' else t)
                if m == 'variableunset':
                    if sc == 'skill':
                        holder.pop(vn, None)
                    else:
                        holder.vars.pop(vn, None)
                        holder.exp.pop(vn, None)
                    continue
                if m == 'setvarloc':
                    val = self.targeter(('selflocation', {}), meta)[0] \
                        if a['val'].lower().startswith('@selflocation') else None
                    if val is None:
                        self.error('setvarloc val not simulated', meta)
                        continue
                elif m == 'variableadd':
                    cur = holder.get(vn) if sc == 'skill' else holder.var(vn)
                    if cur is None:
                        self.error(f'variableadd on unset {a["var"]}', meta)
                        cur = 0
                    val = float(cur) + self.num(a.get('amount', '1'), meta, t)
                    if isinstance(cur, int):
                        val = int(val)
                else:
                    typ = (a.get('type') or 'INTEGER').upper()
                    raw = a.get('value', '')
                    if typ == 'STRING':
                        val = self.ph(raw, meta, t)
                    else:
                        val = self.num(raw, meta, t)
                        if typ == 'INTEGER':
                            if abs(val - round(val)) > 1e-6:
                                self.error(f'INTEGER {a["var"]} got non-integer {val} from {raw!r}', meta)
                            val = int(math.floor(val + 1e-9))
                if sc == 'skill':
                    holder[vn] = val
                else:
                    holder.vars[vn] = val
                    dur = a.get('duration')
                    holder.exp[vn] = self.tick + int(dur) if dur else None
        elif m == 'state':
            for t in targets:
                self.me_state(t, a, meta)
        elif m == 'model':
            for t in targets:
                if a.get('remove') == 'true':
                    t.model, t.anims = None, {}
                else:
                    t.model = a['mid']
        elif m == 'defaultstate':
            for t in targets:
                if isinstance(t, Ent) and t.model is None and t.alive:
                    self.error(f'defaultstate on {t} without a model', meta)
                elif isinstance(t, Ent):
                    st = a.get('state') or a.get('s')
                    if st not in self.blueprints[t.model]:
                        self.error(f'defaultstate animation {st} not in {t.model}', meta)
                    t.defaults = getattr(t, 'defaults', {})
                    t.defaults[a['type'].upper()] = st
        elif m in ('bodyrotation', 'brightness', 'renderinit'):
            for t in targets:
                if t.model is None and t.alive:
                    self.error(f'{m} on {t} without a model', meta)
        elif m == 'teleport':
            if targets:
                t = targets[0]
                c.x, c.y, c.z = t.x, t.y, t.z
                self.hook('teleport', meta)
        elif m == 'rotatetowards':
            if targets:
                t = targets[0]
                want = math.degrees(math.atan2(-(t.x - c.x), t.z - c.z))
                d = ((want - c.yaw) % 360 + 540) % 360 - 180
                mx = float(a.get('maxyaw', 360))
                c.yaw += max(-mx, min(mx, d))
        elif m == 'setrotation':
            for t in targets:
                y = float(a.get('yaw', 0))
                t.yaw = t.yaw + y if a.get('relative') == 'true' else y
        elif m == 'summon':
            for t in targets:
                owner = c
                e = self.spawn(a['type'], t.copy() if isinstance(t, Loc) else t.loc, owner, trigger=c)
                self.hook('summon:' + a['type'], meta, e)
        elif m == 'remove':
            for t in targets:
                if isinstance(t, Ent):
                    self.remove(t)
        elif m == 'signal':
            for t in targets:
                self.fire(t, None, trigger=c, signal=a['s'])
        elif m == 'cancelevent':
            if meta.event is None:
                self.error('cancelevent outside an event', meta)
            else:
                meta.event['cancelled'] = True
        elif m == 'modifydamage':
            if meta.event is None:
                self.error('modifydamage outside an event', meta)
            else:
                meta.event['damage'] *= float(a['a'])
        elif m == 'damage':
            for t in targets:
                if isinstance(t, Ent) and t.player:
                    src = meta.chain[-1] if meta.chain else '?'
                    self.stats['damage'][src] = self.stats['damage'].get(src, 0) + float(a['amount'])
                    t.damage_taken[src] = t.damage_taken.get(src, 0) + float(a['amount'])
                elif isinstance(t, Ent):
                    self.error(f'damage on non-player {t}', meta)
        elif m == 'aura':
            for t in targets:
                if isinstance(t, Ent):
                    t.auras[a.get('auraname')] = self.tick + int(a.get('duration', 1))
        elif m == 'projectile':
            for t in targets:
                self.launch(a, t, meta)
        elif m in ('sound', 'particles', 'particlering', 'particleline', 'throw', 'sendactionmessage'):
            pass
        else:
            self.error(f'mechanic {m} not simulated', meta)

    # ---- ModelEngine ----
    def me_state(self, e, a, meta):
        if not isinstance(e, Ent) or e.removed:
            return
        if e.model is None:
            if e.alive:
                self.error(f'state {a.get("s")} on {e} without a model', meta)
            return
        s = a['s']
        if s not in self.blueprints[e.model]:
            self.error(f'unknown animation {s}', meta)
            return
        if a.get('r') == 'true':
            e.anims.pop(s, None)
            return
        if s in e.anims and a.get('force') != 'true':
            return
        loop, length = self.blueprints[e.model][s]
        speed = float(a.get('speed', 1))
        e.anims[s] = {'start': self.tick, 'speed': speed, 'loop': loop, 'length': length,
                      'li': int(float(a.get('li', 0))),
                      'end': self.tick + length * 20 / speed if loop == 'once' else None}

    def coverage(self):
        for e in self.ents:
            if e.removed or e.model is None:
                continue
            live = [s for s, st in e.anims.items() if st['end'] is None or self.tick < st['end']]
            if not live:
                self.error(f'{e} model {e.model} has NO animation playing (rest pose shows)')
                continue
            if e.model != 'fv_soulcube':
                continue
            # ModelEngine plays its own default states unless they are re-pointed
            if not getattr(e, 'defaults', {}).get('IDLE') or not e.defaults.get('WALK'):
                self.error(f'{e.mtype}: ModelEngine default idle/walk not re-pointed - they would play under the skills')
            # the frame must never sink into the floor
            name = max(live, key=lambda n: e.anims[n]['start'])
            st = e.anims[name]
            if self.tick - st['start'] < st['li']:
                continue          # still blending in from the previous pose
            sec = (self.tick - st['start']) * st['speed'] / 20
            if st['loop'] == 'loop':
                sec = sec % st['length']
            idx = min(int(round(sec * 20)), len(LOWEST[name]) - 1)
            low = LOWEST[name][idx]
            height = (e.y - FLOOR) * 16
            if low is not None and height + low < -0.5:
                self.error(f'{e.mtype}: frame sinks {-(height + low):.1f} px into the floor during "{name}" '
                           f'(standing {height:.1f} px up)')

    # ---- projectiles ----
    def launch(self, a, target, meta):
        c = meta.caster
        yr = math.radians(c.yaw)
        sfo = float(a.get('sfo', 0))
        start = Loc(c.x - math.sin(yr) * sfo, c.y + float(a.get('syo', 0)), c.z + math.cos(yr) * sfo)
        tl = target.loc if isinstance(target, Ent) else target
        tp = Loc(tl.x, tl.y + float(a.get('tyo', 0)), tl.z)
        dx, dy, dz = tp.x - start.x, tp.y - start.y, tp.z - start.z
        L = math.sqrt(dx * dx + dy * dy + dz * dz) or 1
        v = float(a['v']) / 20
        pr = {'pos': start, 'vel': (dx / L * v, dy / L * v, dz / L * v), 'meta': meta, 'a': a,
              'age': 0, 'travel': 0.0, 'alive': True}
        if a.get('bullettype', '').upper() == 'MOB':
            pr['bullet'] = self.spawn(a['mob'], Loc(start.x, start.y + float(a.get('byo', 0)), start.z), c)
        self.projs.append(pr)
        self.stats['shots'] += 1

    def step_projectiles(self):
        home = self.home
        for pr in self.projs:
            if not pr['alive']:
                continue
            a, meta = pr['a'], pr['meta']
            p = pr['pos']
            vx, vy, vz = pr['vel']
            hit = None
            for sub in range(4):
                p.x += vx / 4
                p.y += vy / 4
                p.z += vz / 4
                for pl in self.players():
                    if math.hypot(pl.x - p.x, pl.z - p.z) <= float(a['hr']) + 0.3 and \
                            pl.y - float(a['vr']) <= p.y <= pl.y + 1.8 + float(a['vr']):
                        hit = pl
                        break
                if hit:
                    break
            pr['age'] += 1
            pr['travel'] += math.sqrt(vx * vx + vy * vy + vz * vz)
            if 'bullet' in pr:
                b = pr['bullet']
                b.x, b.y, b.z = p.x, p.y + float(a.get('byo', 0)), p.z
            if a.get('ontick'):
                self.run_skill(a['ontick'], Meta(meta.caster, [p.copy()], origin=p.copy(),
                                                 svars=dict(meta.svars), chain=['proj']))
            end = False
            if hit is not None:
                self.run_skill(a['onhit'], Meta(meta.caster, [hit], origin=p.copy(),
                                                svars=dict(meta.svars), chain=['proj_hit']))
                end = True
            if math.hypot(p.x - home.x, p.z - home.z) > ARENA_R or p.y < FLOOR:
                end = True
            if pr['travel'] >= float(a.get('mr', 40)) or pr['age'] >= int(a.get('duration', 400)):
                end = True
            if end:
                pr['alive'] = False
                if 'bullet' in pr:
                    self.remove(pr['bullet'])
                self.run_skill(a['onend'], Meta(meta.caster, [p.copy()], origin=p.copy(),
                                                svars=dict(meta.svars), chain=['proj_end']))
        self.projs = [pr for pr in self.projs if pr['alive']]

    # ---- hooks: bookkeeping for the invariants ----
    def hook(self, name, meta, extra=None):
        c = meta.caster
        if name == 'fv_soulcube_pick':
            self.active_attacks += 1
            if self.active_attacks > 1:
                self.error('a second attack chain started while one is running', meta)
            for v in ('fv_soulcube_beam_n', 'fv_soulcube_db_n', 'fv_soulcube_shots'):
                if (c.var(v) or 0) > 0:
                    self.error(f'attack picked while {v}={c.var(v)}', meta)
        elif name == 'fv_soulcube_after_attack':
            self.active_attacks -= 1
            if self.active_attacks < 0:
                self.error('after_attack without a running attack', meta)
                self.active_attacks = 0
            for v in ('fv_soulcube_beam_n', 'fv_soulcube_db_n', 'fv_soulcube_shots'):
                if (c.var(v) or 0) > 0:
                    self.error(f'attack ended with {v}={c.var(v)} still running', meta)
            if c.var('fv_soulcube_hpx_target') not in (20,):
                self.error(f'attack ended without heading back to hover height '
                           f'(target {c.var("fv_soulcube_hpx_target")})', meta)
        elif name == 'fv_soulcube_watchdog':
            self.stats['watchdog'] += 1
            self.error('watchdog had to restart the fight loop', meta)
            self.active_attacks = 0
        elif name == 'fv_soulcube_phase_break':
            self.stats['phase_breaks'] += 1
        elif name in ('fv_soulcube_laser_begin', 'fv_soulcube_downbeam', 'fv_soulcube_shoot_begin'):
            typ = {'fv_soulcube_laser_begin': 'LASER', 'fv_soulcube_downbeam': 'DOWNBEAM',
                   'fv_soulcube_shoot_begin': 'SHOOT'}[name]
            pat = meta.chain[-2] if len(meta.chain) > 1 else '?'
            if self.stats['attacks'] and self.stats['attacks'][-1][0] == typ:
                self.error(f'same attack type twice in a row: {typ}', meta)
            self.stats['attacks'].append((typ, pat, c.var('fv_soulcube_phase')))

    # ---- players ----
    def make_players(self):
        for i in range(self.n_players):
            ang = self.rng.uniform(0, 2 * math.pi)
            r = self.rng.uniform(6, 10)
            p = Ent(self, 'PLAYER', Loc(self.home.x + math.cos(ang) * r, FLOOR,
                                        self.home.z + math.sin(ang) * r), player=True,
                    name=f'player{i + 1}')
            p.dirsign = self.rng.choice([-1, 1])
            p.next_hit = 0
            p.jump_t = 0
            p.leaving = False
            self.ents.append(p)

    def move_players(self, boss):
        for p in self.players():
            if p.leaving:
                ang = math.atan2(p.z - self.home.z, p.x - self.home.x)
                p.x += math.cos(ang) * 0.28
                p.z += math.sin(ang) * 0.28
                continue
            # jumping (feet height)
            if p.jump_t > 0:
                p.jump_t -= 1
                k = 12 - p.jump_t
                p.y = FLOOR + max(0.0, 0.42 * k - 0.04 * k * k)
            else:
                p.y = FLOOR
                if self.rng.random() < 0.02:
                    p.jump_t = 12
            if boss is None or boss.removed:
                continue
            mode, state = boss.var('fv_soulcube_mode'), boss.var('fv_soulcube_state')
            dx, dz = p.x - boss.x, p.z - boss.z
            d = math.hypot(dx, dz) or 0.01
            want = 2.2 if (state == 'FIGHT' and mode in ('OPEN', 'STAGGER')) else 7.0
            # player1 is careless during the down beam and walks right under
            # it, so its damage path gets exercised too
            if p.name == 'player1' and boss.var('fv_soulcube_lasttype') == 'DOWNBEAM' and mode == 'IMMUNE':
                want = 0.0
            if self.rng.random() < 0.01:
                p.dirsign *= -1
            radial = max(-0.2, min(0.2, (want - d) * 0.25))
            tang = 0.18 * p.dirsign
            ux, uz = dx / d, dz / d
            p.x += ux * radial - uz * tang
            p.z += uz * radial + ux * tang
            hx, hz = p.x - self.home.x, p.z - self.home.z
            hd = math.hypot(hx, hz)
            if hd > ARENA_R - 1.5:
                p.x = self.home.x + hx / hd * (ARENA_R - 1.5)
                p.z = self.home.z + hz / hd * (ARENA_R - 1.5)
            # hit the boss
            if self.tick >= p.next_hit and d <= 3.0 and abs(boss.y + 0.5 - (p.y + 1.6)) <= 3.0:
                p.next_hit = self.tick + 12
                self.damage_boss(boss, p, 7.0)

    def damage_boss(self, boss, attacker, amount):
        ev = {'damage': amount, 'cancelled': False}
        self.fire(boss, 'ondamaged', trigger=attacker, event=ev)
        if ev['cancelled']:
            self.stats['cancelled_hits'] += 1
            return
        self.stats['hits_on_boss'] += 1
        boss.hp -= ev['damage']
        if boss.hp <= 0 and boss.alive:
            boss.alive = False
            self.fire(boss, 'ondeath', trigger=attacker)
            self.at(self.tick + 20, lambda: self.remove(boss))

    # ---- main loop ----
    def run(self, max_ticks=24000):
        self.home = Loc(0.5, FLOOR, 0.5, self.rng.uniform(-180, 180))
        block = self.spawn('fv_soulcube_block', self.home.copy())
        # shorter respawn for the test
        self.skills['fv_soulcube_block_settings']['Skills'][0] = \
            'setvariable{var=caster.fv_soulcube_respawn;type=INTEGER;value=8;save=true}'
        self.parsed['fv_soulcube_block_settings'] = [parse_line(l) for l in self.skills['fv_soulcube_block_settings']['Skills']]
        self.make_players()
        click_at = 40 + self.rng.randint(0, 60)
        leave_at = None
        unload_at = None
        if self.scenario == 'leave':
            leave_at = click_at + self.rng.randint(400, 2400)
        if self.scenario == 'unload':
            unload_at = click_at + self.rng.randint(300, 2000)
        boss = None
        died_at = None
        restored = False
        phase_at_hp = {}
        while self.tick < max_ticks:
            # scheduled work
            while self.queue and self.queue[0][0] <= self.tick:
                _, _, fn = heapq.heappop(self.queue)
                fn()
            # timers
            for e in list(self.ents):
                if e.removed or e.player or not e.alive:
                    continue
                for p in self.mobs[e.mtype]['_lines']:
                    t = p['trigger'].lower()
                    if t.startswith('ontimer:'):
                        n = int(t.split(':')[1])
                        # global clock, not counted from the mob's spawn
                        if (self.tick + self.clock0) % n == 0 and self.tick > e.spawn_tick:
                            self.exec_line(p, Meta(e, [e], chain=[f'{e.mtype}~timer']), e.mtype)
            # projectiles
            self.step_projectiles()
            # the click
            if self.tick == click_at:
                pl = self.players()[0]
                self.fire(block, 'oninteract', trigger=pl, event={'damage': 0, 'cancelled': False})
            bosses = [e for e in self.ents if e.mtype == 'fv_soulcube' and not e.removed]
            boss = bosses[0] if bosses else None
            if len(bosses) > 1:
                self.error('more than one boss alive')
            if boss is not None and boss.alive:
                # phase bookkeeping
                ph = boss.var('fv_soulcube_phase')
                if ph and ph not in phase_at_hp:
                    phase_at_hp[ph] = boss.hp / boss.mhp
                # height invariant
                hpx = boss.var('fv_soulcube_hpx')
                if hpx is not None and boss.var('fv_soulcube_state') == 'FIGHT':
                    if not (0 <= hpx <= 30):
                        self.error(f'hpx out of range: {hpx}')
                    real = (boss.y - FLOOR) * 16
                    if abs(real - hpx) > 1e-6:
                        self.error(f'height drift: hpx={hpx} but the boss is {real:.3f} px up')
                # leash
                away = math.hypot(boss.x - self.home.x, boss.z - self.home.z)
                self.stats['max_away'] = max(self.stats.get('max_away', 0), away)
                if away > 24 and not self.stats.get('strayed'):
                    self.stats['strayed'] = True
                    last = self.stats['attacks'][-1] if self.stats['attacks'] else None
                    near = min((math.hypot(p.x - self.home.x, p.z - self.home.z) for p in self.players()), default=-1)
                    self.error(f'boss strayed more than 24 blocks from its block during {last} '
                               f'(db_n {boss.var("fv_soulcube_db_n")}, nearest player {near:.1f} from home)')
            self.move_players(boss)
            if leave_at is not None and self.tick == leave_at:
                for p in self.players():
                    p.leaving = True
            if unload_at is not None and self.tick == unload_at and boss is not None:
                # chunk unloads: the non-persistent boss is gone, the block reloads
                self.remove(boss)
                self.projs = []
                self.fire(block, 'onload')
                unload_at = None
            self.coverage()
            # the block must stay hidden while its boss is up
            if boss is not None and boss.alive and not boss.removed and \
                    boss.var('fv_soulcube_state') in ('WAKING', 'FIGHT') and \
                    block.var('fv_soulcube_block') != 'AWAKE' and not self.stats.get('reformed_early'):
                self.stats['reformed_early'] = True
                self.error(f'block is {block.var("fv_soulcube_block")} while its boss is still fighting')
            # end conditions
            if boss is not None and not boss.alive and died_at is None:
                died_at = self.tick
            if block.var('fv_soulcube_block') == 'READY' and self.tick > click_at + 10:
                if died_at is not None or leave_at is not None or self.scenario == 'unload':
                    restored = True
                    if not any(e.mtype == 'fv_soulcube' and not e.removed for e in self.ents):
                        break
            self.tick += 1
        # ---- end-of-fight checks ----
        if self.scenario == 'kill':
            if died_at is None:
                self.error(f'boss did not die within {max_ticks} ticks')
            if 2 not in phase_at_hp or 3 not in phase_at_hp:
                self.error(f'phases reached: {sorted(phase_at_hp)}')
            else:
                if phase_at_hp[2] > 0.66 + 1e-9 or phase_at_hp[3] > 0.33 + 1e-9:
                    self.error(f'phase thresholds off: {phase_at_hp}')
            # two phase changes during one attack share one roar
            if self.stats['phase_breaks'] not in (1, 2):
                self.error(f'{self.stats["phase_breaks"]} phase roars')
        if not restored:
            self.error(f'block did not re-form (state {block.var("fv_soulcube_block")})')
        if block.model is None or not block.anims:
            self.error('block has no model / animation at the end')
        if any(e.mtype in ('fv_soulcube_bullet', 'fv_soulcube_deathfx') and not e.removed for e in self.ents) \
                and self.tick < max_ticks - 1:
            pass
        self.stats['ticks'] = self.tick
        self.stats['died_at'] = died_at
        return self


def main():
    plugins = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 30
    verbose = '--verbose' in sys.argv
    all_errors = []
    summary = {}
    pat_count = {}
    dmg = {}
    durations = []
    for share in (True, False):
        for scenario in ('kill', 'leave', 'unload'):
            runs = n if scenario == 'kill' else max(3, n // 5)
            for seed in range(runs):
                players = 1 + seed % 3
                sim = Sim(plugins, seed * 7 + (0 if share else 1000), share, players, verbose,
                          scenario).run()
                all_errors += [f'[{"shared" if share else "copied"} skill vars, {scenario}] {e}'
                               for e in sim.errors]
                key = (share, scenario)
                summary.setdefault(key, [0, 0])
                summary[key][0] += 1
                summary[key][1] += 1 if not sim.errors else 0
                if scenario == 'kill':
                    for typ, pat, ph in sim.stats['attacks']:
                        pat_count[(ph, pat)] = pat_count.get((ph, pat), 0) + 1
                    for k, v in sim.stats['damage'].items():
                        dmg[k] = dmg.get(k, 0) + v
                    if sim.stats['died_at']:
                        durations.append((sim.stats['died_at'], players))
    seen = set()
    for e in all_errors:
        k = re.sub(r'\[seed \d+ t=\d+\] ', '', e)
        if k in seen and not verbose:
            continue
        seen.add(k)
        print('ERROR', e)
    for (share, scen), (runs, ok) in summary.items():
        print(f'{"shared" if share else "copied"} skill vars / {scen:6s}: {ok}/{runs} fights clean')
    print('patterns used (phase, pattern): ' + ', '.join(f'{k[0]}:{k[1]}={v}' for k, v in sorted(pat_count.items(), key=lambda x: (x[0][0] or 0, x[0][1]))))
    if durations:
        for np_ in (1, 2, 3):
            ds = [d for d, p in durations if p == np_]
            if ds:
                print(f'{np_} player(s): kill time {min(ds) / 20:.0f}-{max(ds) / 20:.0f} s (avg {sum(ds) / len(ds) / 20:.0f} s)')
    print('damage dealt to players by source: ' + ', '.join(f'{k}={v:.0f}' for k, v in sorted(dmg.items(), key=lambda x: -x[1])))
    return 1 if all_errors else 0


if __name__ == '__main__':
    sys.exit(main())
