# Changelog
All notable changes to this project will be documented in this file. See [conventional commits](https://www.conventionalcommits.org/) for commit guidelines.

- - -
## 0.3.0 - 2026-03-04
#### Features
- (**flycam**) Adjust flycam controls - (367b67e) - Daniel Seiller
- (**linux_support**) Add linux target - (0437632) - Daniel Seiller
- (**parser**) properly handle NaN node scale - (6c9a29b) - Daniel Seiller
- (**parser**) support duplicate INI keys via markers and add serde_path_to_error diagnostics - (d4fa3bd) - Daniel Seiller
- (**parser**) properly handle empty lines in INI blocks - (b28c2d6) - Daniel Seiller
- (**parser**) Add deserialize impl - (fa92477) - Daniel Seiller
- (**parser**) Fix string handling - (b094ebb) - Daniel Seiller
- (**parser**) Improve string encoding and decoding - (a038d58) - Daniel Seiller
- (**parser**) Add missing brw annotations - (167d0a3) - Daniel Seiller
- (**parser**) add binwrite support across parser structs - (6a74ba5) - Daniel Seiller
- (**parser**) expose more parser internals - (3cc9f2a) - Daniel Seiller
- (**render**) nicer autofocus - (38de811) - Daniel Seiller
- (**viewer**) add hierarchical browser panel and flight toggles - (f0fea12) - Daniel Seiller
#### Bug Fixes
- (**flycam**) gate auto-level behind idle roll input and switch keyboard controls to explicit virtual axes - (550c436) - Daniel Seiller
- (**parser**) comment out failing test - (d67566f) - Daniel Seiller
- (**parser**) enable serde_json float roundtripping - (5567bfb) - Daniel Seiller
- (**parser**) serialize MD3D skin weight NaN as null for JSON roundtrip - (bde0400) - Daniel Seiller
- (**parser**) handle INI empty lines as empty strings for JSON roundtrip compatibility - (e6df725) - Daniel Seiller
- (**parser**) align MAT NAM and ANI layouts with binary - (8948ea6) - Daniel Seiller
- (**parser**) align MAT and NAM serialization with engine - (5662f85) - Daniel Seiller
#### Documentation
- (**parser**) document all parser structures with Ghidra cross-checks - (413ad8b) - Daniel Seiller
#### Build system
- Add potentially missing cargo features - (85a778f) - Daniel Seiller
- Add linux dependencies - (f516914) - Daniel Seiller
- Fix post-bump hook - (9c1eb3b) - Daniel Seiller
#### Refactoring
- (**parser**) model MAP texture fx types - (dd7f45f) - Daniel Seiller
- (**parser**) clarify collision and zone field semantics - (2477cc8) - Daniel Seiller
- (**parser**) rename reverse-engineered chunk fields - (ff88e2e) - Daniel Seiller
#### Miscellaneous Chores
- (**debug**) capture parser roundtrip mismatch artifacts - (c49714c) - Daniel Seiller
- (**release**) 0.3.0 - (cfb95c0) - Daniel Seiller
- (**version**) 0.2.0 - (2019ecf) - Daniel Seiller
- (**version**) 0.1.0 - (29d0402) - Daniel Seiller
- (**version**) 0.0.3 - (5273e7e) - Daniel Seiller
- (**version**) 0.0.2 - (9efa123) - Daniel Seiller
- (**version**) 0.0.1 - (31b7e9b) - Daniel Seiller
- (**version**) 0.0.1 - (23660c0) - Daniel Seiller
- code cleanup, remove unnecessary dependencies - (c70ff98) - Daniel Seiller
- Fix post-bump hook - (a72354b) - Daniel Seiller
- fix post bump hook - (9d241ae) - Daniel Seiller
- cocogitto setup - (15d1d14) - Daniel Seiller
- cargo-dist setup - (4250077) - Daniel Seiller
- Initial Commit - (60f656e) - Daniel Seiller
#### Style
- (**parser**) apply rustfmt cleanup - (0eb538a) - Daniel Seiller

- - -

## 0.2.0 - 2025-01-03
#### Features
- **(parser)** expose more parser internals - (dc8baed) - Daniel Seiller
- **(render)** nicer autofocus - (57ece24) - Daniel Seiller
#### Miscellaneous Chores
- **(version)** 0.1.0 - (62987c8) - Daniel Seiller
- code cleanup, remove unnecessary dependencies - (88bff51) - Daniel Seiller

- - -

## 0.1.0 - 2024-07-11
#### Build system
- Add potentially missing cargo features - (a915549) - Daniel Seiller
- Add linux dependencies - (1b893ee) - Daniel Seiller
- Fix post-bump hook - (9c64bc3) - Daniel Seiller
#### Features
- **(linux_support)** Add linux target - (94e5b73) - Daniel Seiller
#### Miscellaneous Chores
- **(version)** 0.0.3 - (5ce6df7) - Daniel Seiller
- **(version)** 0.0.2 - (f27c585) - Daniel Seiller
- **(version)** 0.0.1 - (6c40fc2) - Daniel Seiller
- **(version)** 0.0.1 - (231b9ca) - Daniel Seiller
- Fix post-bump hook - (69ce905) - Daniel Seiller
- fix post bump hook - (182831b) - Daniel Seiller
- cocogitto setup - (178b2dc) - Daniel Seiller
- cargo-dist setup - (5aba83e) - Daniel Seiller
- Initial Commit - (60f656e) - Daniel Seiller

- - -

## 0.0.3 - 2024-07-11
#### Build system
- Add linux dependencies - (6c664e9) - Daniel Seiller

- - -

## 0.0.2 - 2024-07-11
#### Build system
- Fix post-bump hook - (cce5831) - Daniel Seiller
#### Miscellaneous Chores
- Fix post-bump hook - (83bde9a) - Daniel Seiller
- fix post bump hook - (c7f6bd5) - Daniel Seiller

- - -

## 0.0.1 - 2024-07-11
#### Features
- **(linux_support)** Add linux target - (09b2385) - Daniel S
#### Miscellaneous Chores
- **(version)** 0.0.1 - (ebea57c) - Daniel Seiller
- cocogitto setup - (50a463f) - Daniel Seiller
- cargo-dist setup - (268d1eb) - Daniel Seiller
- Initial Commit - (60f656e) - Daniel Seiller

- - -

## 0.0.1 - 2024-07-10
#### Miscellaneous Chores
- cocogitto setup - (bc00749) - Daniel Seiller
- cargo-dist setup - (f64de73) - Daniel Seiller
- Initial Commit - (24edaab) - Daniel Seiller

- - -

Changelog generated by [cocogitto](https://github.com/cocogitto/cocogitto).