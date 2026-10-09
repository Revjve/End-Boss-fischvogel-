# Soul of the End - Soul of the End

A MythicMobs + ModelEngine + Nexo boss: a soul trapped in a block.
Spawn `fv_endsoul_block` (the block); a left or right click wakes it into `fv_endsoul` (the boss).

**Download:** [`Soul of the End.zip`](Soul%20of%20the%20End.zip) - drag the contents of its
`plugins` folder into the server's `plugins` folder. Everything else is in the
[Installation Guide](Soul%20of%20the%20End/Installation%20Guide.txt).

## Layout

| Path | What it is |
|---|---|
| `Soul of the End/` | Exactly what goes into the zip (the hand-over pack) - the source of truth. Nothing from `tools/` or `docs/` ships. |
| `Soul of the End/plugins/MythicMobs/packs/fv_endsoul/` | Mobs and skills: `core` (block, wake-up, loop, phases, damage rules, height, death), `laser`, `downbeam`, `shooting`, `fx` (every sound / particle cue). |
| `Soul of the End/plugins/ModelEngine/blueprints/` | The two blueprints, generated from the originals by `tools/convert_models.py`. |
| `Soul of the End/Source Files/` | The converted models in Blockbench 5 format, for editing (same file names as the blueprints). |
| `tools/originals/` | The `.bbmodel` files as originally delivered - the converter's input. |
| `tools/` | Build, conversion, static checker and fight simulator. |

## Where it sits in the game

![Side view of every state at its in-game height](docs/in-game-heights.png)

Heights are in model pixels above the floor (16 px = 1 block). The boss floats at eye
level (24 px) and drops to 12 px for the jump-rope sweep and to 3 px for the down beam (its beam then
ends exactly on the floor). Every
height was measured against the blueprint (`tools/model_lowest.json`, the lowest point of
the cube and frame per animation tick) so the frame never sinks into the floor; the fight
simulator checks this every tick.

## Tools

```sh
python3 tools/build.py              # convert + verify models, check, write the zip
python3 tools/build.py --sim 20     # ... and play 20 simulated fights per scenario
```

* `convert_models.py` / `verify_models.py` - functional-only blueprint conversion (bakes
  `math.random()` per tick, hides the z-fighting eye cubes in `idle`, adds the `dormant`
  and `blank` poses, writes the laser's "rotate in global space" setting out as plain
  keyframes, sets `override`, writes Blockbench's legacy 4.10 format) and its proof that
  geometry, textures and every authored keyframe are unchanged - and that the laser beam
  sits exactly where Blockbench draws it, every tick, in both ways an engine can stack
  bone transforms.
* `check_pack.py` - every skill reference, animation, sound and particle name (against the
  1.21.1, 1.21.4 and 26.1 game data in `tools/vanilla_data`, from
  [PrismarineJS/minecraft-data](https://github.com/PrismarineJS/minecraft-data)), variable
  reads vs. writes, math and bracket syntax, and that no skill reads another entity's
  position through a placeholder (MythicMobs only documents exact coordinates for the
  caster - other entities are measured with a `sudoskill` probe).
* `simulate_fight.py` - interprets the real skill files tick by tick with scripted players
  and checks the fight's invariants (one attack chain at a time, no animation gaps, exact
  heights, phases, hit reactions that cancel the running attack, the block landing on the
  floor, death / reset / chunk-unload flows) under both possible skill-variable semantics.
  `check_pack.py` also requires every waiting skill to carry the epoch guard that lets a
  hit reaction stop it.

Python 3.10+ with `numpy` and `pyyaml`.
