# Scrapland UI/Menu System (SScorer) Notes

This document summarizes how Scrapland's UI/menu stack works based on scripts found in `data/data*.packed`.

## What was inspected

- Packed Python sources (`.py`) and bytecode (`.pyc`) in the game packs.
- Main UI/menu script modules:
  - `scripts/scorer/menu.py`
  - `scripts/scorer/pcmenu.py`
  - `scripts/scorer/xbmenu.py`
  - `scripts/scorer/xboxmenu.py`
  - `scripts/scorer/scorer.py`
  - `scripts/scorer/charscorer.py`
  - `scripts/scorer/racerscorer.py`
  - `levels/menu/scripts/map.py`
  - `scripts/init.py`

## Ghidra artifacts in packs

- No explicit Ghidra script/artifact paths were found in the packed file index (no `ghidra`-named entries).
- The pack content is dominated by assets plus Python gameplay/UI scripts (`.py`/`.pyc`).
- Native engine behavior (the `SScorer` module itself) appears to be provided by engine-side code, with Python scripts acting as high-level UI logic.

## High-level architecture

The UI system is a script-driven layer over native module `SScorer`.

- `SScorer` is the render/control backend (create controls, set properties, focus/default, callbacks, transitions).
- `Menu` (`scripts/scorer/menu.py`) is the shared UI framework and style library.
- `PCMenu` and `XBMenu`/`XboxMenu` are platform policy layers (PC vs Xbox flows).
- `Scorer`, `CharScorer`, and `RacerScorer` build in-game HUDs and overlays.
- `levels/menu/scripts/map.py` boots the 3D menu scene and hands control to `Menu`.

## Boot flow and ownership

### Script preload stage (`scripts/init.py`)

`init.py` preloads scorer/menu libraries from the scorer pack and configures platform-specific modules:

- Loads shared modules: `Menu`, `Scorer`, `CharScorer`.
- Loads `PCMenu` on PC and `XBMenu` on Xbox.
- For menu map startup, opens level scripts pack and loads `Map.pyc`/`MapSnd.pyc`.
- Sets `EscapeEvent` to `Menu.Init`, then calls `Menu.Initialize()`.

### Menu-level stage (`levels/menu/scripts/map.py`)

`levels/menu/scripts/map.py` builds the animated background scene and activates menu mode:

- `StartMenu()` sets `SInput` action set to `"Menu"`, calls `Menu.Init(0)`, and disables blur/noise side effects.
- `Init()` creates camera entities (`DemoCam`, linked camera), sets background debris/asteroid effects, and starts menu music.
- If credits mode is active, switches input to `"Inactive"`, uses `Menu.MovieScorer(0)`, and runs credit sequencing.

## SScorer programming model

From the scripts, SScorer behaves as an immediate/config-driven UI graph:

- Create controls:
  - `SScorer.Add(id, name, type)`
  - `SScorer.New(id, name, prefab)`
- Configure controls:
  - `SScorer.Set(id, name, key, value)`
  - `SScorer.Get(id, name, key)`
- Manage context/focus:
  - `SScorer.Clear(id)`
  - `SScorer.SetDefault(id, control)`
  - `SScorer.SetCursor(id, control)`
  - `SScorer.SetOnCancel/SetOnNext/SetOnPrev(...)`
- Visual/system services:
  - `SScorer.PreloadTexture(...)`
  - `SScorer.SetCinema(...)`
  - `SScorer.SetMsgText(...)`
  - `SScorer.AddModel(...)` (portrait/model previews)

Callback dispatch is string-based (`"Menu.OptionsMenu"`, `"PCMenu.MultiPlayerMenu"`, etc.), so menu items are data rows with function-name strings.

## Control types and composition

Observed control families used by scripts:

- Menu controls: `Text`, `Button`, `Sprite`, `Tab`, `Hint`, `Circuit`.
- HUD/gameplay controls: `Status`, `Radar`, `Monitor`, `CopMeter`, `Mission`, `Talk`.
- Specialized typing/credits controls: `TextTyping`.

Controls are composed by assigning many properties (position, color, alpha, sprite index, behavior flags, callbacks), often with reusable helper functions.

## Navigation model

Navigation is explicit graph wiring, not automatic layout focus.

- Direction links are set with string neighbors: `Up`, `Down`, `Left`, `Right`.
- Helpers in `menu.py` (`LinkUD`, `LinkLR`, etc.) wire menu items and tabs.
- `SetDefault` determines focused control on entry.
- Cancel/back behavior is bound via `SetOnCancel`.

`VerticalMenu(...)` in `menu.py` is the primary constructor for list-style menus:

- Creates item controls from `(label, on_accept, optional_name, ...)` tuples.
- Applies shared typography/colors/effects.
- Wires directional links and default control.
- Handles PC/Xbox differences for back button presentation.

## Core menu framework (`scripts/scorer/menu.py`)

`menu.py` is the central framework with these responsibilities:

- Session setup (`InitMenuSys`): captures prior action set, switches to `"Menu"`, controls timescale.
- Frame reset (`StartNewMenu`): clears previous controls and recreates cursor while preserving prior cursor position when possible.
- Shared visual chrome:
  - `DrawBackMainMenu`, `DrawBackSubMenu`, `DrawBackOptionMenu`
  - `DrawCircuitMenu`, `DrawMenuTitleBar`
- Common screen builders:
  - `MainMenu` / `CreateMainMenu`
  - `VerticalMenu`, `YesNoMenu`, mission info panels
- Modal movie/credits overlay path via `MovieScorer`.
- Preloading strategy in `Initialize()`, with conditional texture sets by game mode/platform.

## Platform split

### PC path (`scripts/scorer/pcmenu.py`)

PC menu layer extends shared menu features with:

- Multiplayer browser/join/create flows.
- Video/audio/options/control submenus.
- Key binding editor with live define-list refresh:
  - `WaitForKeyPress(...)` enters `SInput.ListenToDefine(...)` and temporarily switches action set to `"Inactive"`.
  - `RefreshControlMenu()` rebuilds visible bindings and enforces max binding count behavior.

### Xbox path (`scripts/scorer/xbmenu.py`, `scripts/scorer/xboxmenu.py`)

Xbox logic separates launcher/profile/start semantics from shared menu rendering:

- `XboxMenu.Init(...)` overrides main menu creation callback (`Menu.onCreateMainMenu = CreateMainMenu`).
- `XboxMenu.MainMenu()` starts in a "Press Start" style screen and uses different action-set/escape behavior.
- Profile/save-device flow gates transition to actual main menu (`StartMainMenu`).
- `XBMenu` provides Xbox-specific options/cheat/debug branches and control/audio menus.

## HUD/scorer side (in-game)

The same `SScorer` backend also renders gameplay HUD layers:

- `scripts/scorer/scorer.py`: shared widgets (status, mission, radar, monitor, talk, cop meter).
- `scripts/scorer/charscorer.py`: on-foot/character overlays and special actions.
- `scripts/scorer/racerscorer.py`: vehicle/race overlays (weapon, target, waypoint, racer HUD).

This indicates one unified UI system for both menus and runtime HUD, switched by mode and input action set.

## Key takeaways

- Scrapland UI is heavily data/config driven through `SScorer.Set(...)` property maps.
- Menu behavior is script-controlled with string callback routing and explicit focus graph wiring.
- `menu.py` is the reusable framework; PC/Xbox modules provide platform policy and flow variants.
- The menu map script (`levels/menu/scripts/map.py`) is responsible for 3D backdrop + audio + mode handoff, while `Menu` owns actual UI construction.
- No standalone Ghidra scripts were found inside the `.packed` archives; the actionable UI logic is in the shipped Python scripts.
