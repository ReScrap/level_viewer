# Scrap.exe Keyboard Shortcuts

This file documents keyboard shortcuts recovered from the `Scrap.exe` menu and accelerator resources in Ghidra (`/remaster_update/Scrap.exe [main]`).

## Main shortcuts (menu-visible)

| Key | Command ID | Action |
|---|---:|---|
| `F1` | `0x9c41` | Open **About** dialog (`Acerca de...`). |
| `F2` | `0x9c42` | Open **Video/Device setup** dialog (`Config. video...`). |
| `F11` | `0x9c45` | Toggle **step-by-step** mode (`Paso a paso`). |
| `F12` | `0x9c44` | Toggle **pause/continue** (`Pausa/continuar`). |
| `Shift+Esc` | `0x9c46` | Exit application (`Salir`). |
| `Ctrl+F3` | `0x9c57` | Toggle Gouraud-related render mode (`gouraud`). |
| `Ctrl+F4` | `0x9c56` | Toggle mesh/wireframe draw mode (`dibujar malla`). |
| `Ctrl+F5` | `0x9c59` | Toggle vertex-buffer related path (`buffer de vertices`). |
| `Ctrl+F6` | `0x9c5a` | Toggle `buffer w`. |
| `Ctrl+F7` | `0x9c54` | Toggle vertex specular lighting (`ilum. especular`). |
| `F4` | `0x9c66` | Reload textures / refresh render state (`recargar texturas`). |
| `F5` | `0x9c5f` | Toggle double-sided rendering (`solo doble cara`). |
| `F6` | `0x9c5e` | Open **node/model tree info** dialog (`arbol de nodos...`). |
| `F7` | `0x9c64` | Toggle ground-area visualization (`areas suelo`). |
| `F8` | `0x9c62` | Toggle fog (`niebla`). |
| `Alt+F5` | `0x9c65` | Toggle sharpen. |

## Hidden/alternate accelerator shortcuts

These are bound in accelerator tables and routed through `AppWndProc`/`FUN_006345a0`.

| Key | Command ID | Action |
|---|---:|---|
| `F5` | `0x9ca4` | Cycle `R_ShowInfo` basic modes (`(value + 1) % 3`). |
| `F7` | `0x9ca6` | Toggle focus between game window and console window. |
| `F9` | `0x9ca4` | Same as above: cycle `R_ShowInfo` basic modes. |
| `F11` | `0x9ca6` | Same as above: toggle focus game <-> console. |
| `F12` | `0x9ca5` | Trigger screenshot (`DoScreenshot_ = 1`). |
| `Ctrl+F8` | `0x9ca7` | Cycle texture filter mode: `Point -> Linear -> Anisotropic`. |
| `Alt+F7` | `0x9ca8` | Toggle stencil shadows (`M3D::Sombras_Stencil::Usar`). |
| `Alt+F9` | `0x9ca9` | Cycle extended `R_ShowInfo` modes (`(value + 1) % 6`). |
| `Ctrl+Alt+Shift+F12` | `0x9caa` | Alternate `R_ShowInfo` toggle path (binary-like state clamp). |

## Notes

- The `Camaras` menu is populated dynamically at runtime from world camera nodes and does not have fixed static shortcuts.
- Some keys appear in multiple accelerator tables/profiles; behavior can vary by runtime context (main window vs console vs alternate mode).
- Command IDs above are retained for quick cross-reference in reverse engineering notes.
