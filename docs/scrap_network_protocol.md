# Scrap.exe network protocol (reverse-engineered)

Source binary: `/remaster_update/Scrap.exe [main]` (x86, win32)

This document captures the protocol behavior recovered from static analysis of `Scrap.exe`.

## 1) Transport overview

Function refs: `init_server_net_manager`, `init_client_net_manager`, `recv_packet`, `send_data_1`, `send_data_2`.

- Protocol: UDP (`socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)`)
- Socket mode: non-blocking (`ioctlsocket(FIONBIO, 1)`)
- Game/default server port: `0x6db6` (`28086`)
- Master-server default port: `0x6cfc` (`27900`)
- Modes:
  - `NET_MASTER` (host/server)
  - `NET_CLIENT` (player client)
  - `NET_BROWSE` (server browser)

Two different UDP protocols are present:

1. Gameplay/session UDP protocol (encrypted/authenticated payload wrapper + game messages).
2. Master/query UDP protocol:
   - in-game master/listing protocol to `MasterServerAddress:MasterServerPort`
   - ASE-compatible public query protocol (`master.udpsoft.com`/`master2.udpsoft.com` heartbeat and query replies)

## 2) Gameplay/session UDP framing

Function refs: `FUN_005909b0`, `net_recv`, `handle_packet`, `FUN_0058c850`.

### 2.1 Outer datagram crypto format

Function refs: `FUN_005909b0`, `net_recv`, `decrypt_chacha20`, `FUN_005887e0`, `generate_chacha20_block`, `calc_poly1305_mac`.

Outgoing datagrams are created in `FUN_005909b0` and incoming are parsed in `net_recv`.

The packet format is:

- `nonce12` (12 bytes)
- `ciphertext` (encrypted gameplay payload)
- `tag16` (16-byte Poly1305 MAC)

Cipher/MAC implementation details from code:

- ChaCha20 core constants are standard (`"expand 32-byte k"` words in `FUN_005887e0`).
- Poly1305 is used for authentication (`calc_poly1305_mac`).
- The Poly1305 one-time key is derived from ChaCha20 block 0 with the packet nonce.
- MAC verification failure logs `"Bad Poly1305"` and drops payload.

Nonce generation:

- `nonce12` is generated from an internal PRNG state (`DAT_00865870` via `thunk_FUN_0058ce20`).
- No explicit key-exchange was found in these routines; keying material is initialized by internal RNG setup.

### 2.2 Decrypted gameplay payload envelope

Function refs: `net_recv`, `handle_packet`, `handle_packet_from_client`, `handle_packet_from_server`.

After decryption, payload is interpreted in `handle_packet`.

Layout:

- Byte `0`: channel/sequence byte
  - low 7 bits = sequence counter
  - high bit = direction bit check (`NET_MASTER`/`NET_CLIENT` path checks opposite polarity)
- Bytes `1..2` (`u16` LE): `usr_len_plus_1`
- Byte `3`: user-subchannel sequence byte (`& 0x7f` is checked for monotonic increment)
- Bytes `4 .. 4 + (usr_len_plus_1 - 1) - 1`: user/control payload (`usr_buffer`)
- Remaining bytes: entity/world state payload (`pkt_buffer`)

`handle_packet` computes:

- `usr_size = usr_len_plus_1 - 1`
- `pkt_size = total_len - usr_len_plus_1 - 3`

and dispatches:

- user/control payload to user-message handler
- entity/world payload to entity VMT handlers

### 2.3 Outgoing queue wrapper before encryption

Function refs: `FUN_0058c850`, `send_data_1`, `send_data_2`, `FUN_005909b0`.

`FUN_0058c850` assembles the plaintext before `FUN_005909b0` encryption:

- first byte = per-peer sequence (`next & 0x7f`) with optional high-bit flag
- optional queued control frame (from `NetData` linked queue) is inserted first
- then current frame payload bytes

Send path:

- server mode: `send_data_1` (to specific client slot address)
- client mode: `send_data_2` (to server address)

## 3) User/control message opcodes (decrypted `usr_buffer`)

Function refs: `handle_packet_from_client`, `handle_packet_from_server`, `make_join_packet`, `handle_resources`, `ClientSetConfig`.

These are the first byte of `usr_buffer` parsed by:

- server side: `handle_packet_from_client`
- client side: `handle_packet_from_server`

### 3.0 Primitive encodings used by opcode payloads

Function refs: `write_str`, `read_str`, `handle_packet_from_client`, `handle_packet_from_server`.

- Endianness: little-endian for integer and floating-point primitives.
- `u8/u16/u32`: fixed-width unsigned integers.
- `i16`: signed 16-bit integer (used by quantized deltas).
- `f32`: IEEE754 single-precision float.
- `str`: `write_str`/`read_str` helper format (NUL-terminated C-string on the wire).
  - Practical parser rule: read bytes until `0x00`, decode as ANSI/Windows codepage text.
  - Empty string is encoded as a single `0x00` byte.

These primitive assumptions are confirmed by decompiled serializer/deserializer symmetry and `fread`/`fwrite` call widths in network handlers.

### 3.1 Client -> Server opcodes

Function refs: `handle_packet_from_client`, `make_join_packet`.

- `0x00` / `0x06` : player init/modify packet (`make_join_packet` uses `0x06`)
  - payload fields (ordered):
    - `u16 NET_GAME_ID` (`0xceba`)
    - `u16 NET_GAME_VERSION` (`0x0101`)
    - `str client_password`
    - `str player_name`
    - `str player_model`
    - `u16 player_max_life`
    - `str pilot_model`
    - `str motor0_model`
    - `str motor1_model`
    - `str motor2_model`
    - `str motor3_model`
    - `str weapon_bay_list`
    - `u32 player_team_id`
- `0x02` : goodbye/disconnect
- `0x03` : chat string (`write_str`)
- `0x04` : user string (`write_str`)
- `0x05` : full/config request (server resets send phase and emits init/config packets)
- `0x08` : remote console command
  - payload:
    - `str remote_password`
    - `str command`

Compact schema form:

- `0x00/0x06`: `u16 game_id, u16 version, str password, str player_name, str player_model, u16 max_life, str pilot, str motor0, str motor1, str motor2, str motor3, str weapon_bay, u32 team`
- `0x02`: no body
- `0x03`: `str chat`
- `0x04`: `str user_text`
- `0x05`: no body (request full/config sync)
- `0x08`: `str rcon_password, str command`

### 3.2 Server -> Client opcodes

Function refs: `handle_packet_from_server`, `handle_resources`, `ClientSetConfig`.

- `0x00` : map/type init
  - payload:
    - `str map_name`
    - `str server_type`
- `0x01` : resource dictionary chunk
  - payload:
    - `u16 start_index`
    - `u8 count`
    - `count * str resource_name`
- `0x02` : generic control/ack path (handled, no extra fields parsed there)
- `0x03` : chat string
- `0x04` : user string
- `0x05` : per-player config table (`ClientSetConfig`)
  - begins with `u16 num_entries`, then repeated entries by `net_id`
- `0x06` : remove player/entity (`u8 player_idx`, resolves `net%d` and removes)
- `0x07` : force reset/disconnect path (client tears down and re-inits net state)

Compact schema form:

- `0x00`: `str map_name, str server_type`
- `0x01`: `u16 start_index, u8 count, str names[count]`
- `0x02`: control/ack (handler consumes opcode; no stable public body recovered)
- `0x03`: `str chat`
- `0x04`: `str user_text`
- `0x05`: `u16 count, player_cfg[count]` where each entry starts with `u16 net_id`
- `0x06`: `u8 player_idx`
- `0x07`: reset/disconnect control (no extra stable body recovered)

## 4) Entity/world state payload (`pkt_buffer`)

Function refs: `handle_packet_from_server`, `build_packet`, `cVehicleEntity::handle_packet`, `cItemEntity::handle_packet`.

In `handle_packet_from_server`, world-state payload starts with:

- `u16 ply_id`
- `u16 num_vals`
- `f32 pos_x`
- `f32 pos_y`
- `f32 pos_z`
- `u8 player_idx`
- `u8 rtt`

Then repeated `num_vals` entries:

- `3 bytes`: `net_name(16 bits)` + `type_idx(8 bits)`
- followed by entity-specific serialized data consumed by entity virtual `handle_packet`

If an entity does not exist, client auto-creates:

- `net_id < 0x201` -> vehicle (`Car`)
- otherwise type resolved via resource table (`GetResourceName(type_idx)`)

On server send path (`build_packet`), each entity contributes via virtual serializer (`VMT field41`/`field42`).

## 4.1 Entity types and type IDs

Function refs: `CreateResource`, `GetResourceName`, `handle_resources`, `handle_packet_from_server`.

Entity identity in world-state records is encoded as:

- `net_name` (16-bit): logical network entity id
- `type_idx` (8-bit): resource/type id

Observed behavior:

- `net_name < 0x201` (`0..512`) is treated as vehicle/player namespace (`"net%d"`, type `Car`).
- `net_name >= 0x201` is object/resource namespace and uses `type_idx`.

`type_idx` meaning is runtime-mapped through `NetResources[0..255]`:

- Server creates entries via `CreateResource(name)`.
- Clients receive resource table updates via user opcode `0x01` (resource chunk).
- `GetResourceName(type_idx)` resolves this table on client.

So entity "type ids" are dynamic per session and not a fixed enum in binary.

## 4.2 Known entity payload families

Function refs: `cVehicleEntity::handle_packet`, `cNetVController::handle_packet`, `cVehicleEntity::method_164`, `cNetVController::method_008`, `cItemEntity::handle_packet`, `cNetController::handle_packet`, `cItemEntity::method_164`, `cNetController::method_008`.

At the top level, each entity update starts with:

- `3 bytes`: `[net_name_lo, net_name_hi, type_idx]`

The payload that follows is entity-class specific and handled by virtual methods.

Recovered handlers:

- Vehicles:
  - deserialize chain: `cVehicleEntity::handle_packet` -> `cNetVController::handle_packet`
  - serialize chain: `cVehicleEntity::method_164` -> `cNetVController::method_008`
- Generic items/objects/missiles:
  - deserialize chain: `cItemEntity::handle_packet` -> `cNetController::handle_packet`
  - serialize chain: `cItemEntity::method_164` -> `cNetController::method_008`

## 4.3 Base cNetController delta payload

Function refs: `RTTI::cNetController::handle_packet` (`0x004fbb30`), `RTTI::cNetController::method_008` (`0x004fbf00`).

`cNetController::method_008` / `cNetController::handle_packet` define a shared compact/full delta format.

First byte in payload is a `flags` bitfield.

- `bit0` (`0x01`): precision mode
  - clear = compact quantized encoding
  - set = fuller precision encoding
- `bit1` (`0x02`): position block present
- `bit2` (`0x04`): velocity component block A present
- `bit3` (`0x08`): velocity component block B present
- `bit4` (`0x10`): angular block present
- `bit5` (`0x20`): extra motion block(s) present

Block semantics (from read/write symmetry):

- Position block (`0x02`):
  - compact: 3x `i16` delta from global origin scale constants
  - full: 3x `f32` absolute
- Velocity/rotation/motion blocks:
  - compact mode uses byte- or short-quantized values scaled by constants
  - full mode uses short/float precision depending on block

This base format is reused by multiple networked entity classes.

Canonical parser order for base controller payload:

1. Read `u8 flags`.
2. If `flags & 0x02`, read position block (`3xi16` compact or `3xf32` full).
3. If `flags & 0x04`, read velocity block A (compact quantized or fuller precision).
4. If `flags & 0x08`, read velocity block B (compact quantized or fuller precision).
5. If `flags & 0x10`, read angular block.
6. If `flags & 0x20`, read extra motion block(s).

Exact per-subblock scalar widths vary by compact/full mode branch, but gate bits and ordering above are stable.

## 4.4 Vehicle extension payload (cNetVController)

Function refs: `RTTI::cNetVController::handle_packet` (`0x00518720`), `RTTI::cNetVController::method_008` (`0x00518f20`).

Vehicle controller adds a second flags layer (`some_net_flags`, 16-bit), then optional fields.

Recovered high-level bits:

- `0x100` / `0x200` / `0x400` / `0x800`: team/AI state influence masks
- `0x1000`: zero-speed/idle indicator path
- `0x2000`: lock-on target by `net` name
- `0x4000`: direct target position delta (3x `i16`)
- `0x8000`: extra packed control nibble byte appended
- `0x0080`: fire-delay byte present
- `0x0020`: thrust/boost related values present

Additional known fields inside vehicle payload:

- target/current focus entity id (`u16`, usually parsed from `"net%d"`)
- team/group byte
- camera/aim linkage ids (`u16` ids or compact xyz deltas)
- current weapon/camera slot byte + owning `net id` (`u16`)
- optional packed control state byte (high/low nibble change tracking)

After vehicle-specific fields, base `cNetController` payload is appended.

Canonical parser order for vehicle extension:

1. Read `u16 some_net_flags`.
2. Conditionally read vehicle-specific fields according to `some_net_flags` bits (`0x0020`, `0x0080`, `0x2000`, `0x4000`, `0x8000`, and state mask bits).
3. Parse trailing base `cNetController` payload (section 4.3).

## 4.5 Init/config payloads for entities

Function refs: `make_join_packet`, `handle_packet_from_client`, `ClientSetConfig`, `FUN_0058e0c0`, `FUN_0058dfd0`, `write_end_id`, `handle_resources`.

Outside per-tick deltas, there are explicit config payloads.

### Player join/config packet (`usr opcode 0x00/0x06`)

Function refs: `make_join_packet`, `handle_packet_from_client`.

Sent by `make_join_packet`; parsed in `handle_packet_from_client`:

- `u16 NET_GAME_ID`
- `u16 NET_GAME_VERSION`
- `str client_password`
- `str player_name`
- `str player_model`
- `u16 player_max_life`
- `str pilot_model`
- `str motor0_model`
- `str motor1_model`
- `str motor2_model`
- `str motor3_model`
- `str weapon_bay_list`
- `u32 team`

### Server -> client player config table (`usr opcode 0x05`)

Function refs: `ClientSetConfig`, `FUN_0058d9d0`, `FUN_0058e0c0`, `FUN_0058dfd0`, `write_end_id`.

Parsed by `ClientSetConfig`:

- `u16 count`
- repeated `count` times:
  - `u16 net_id`
  - vehicle config blob from `FUN_0058d9d0` (invoked immediately)

The same config shape is used when writing per-entity init in:

- `FUN_0058e0c0`
- `FUN_0058dfd0`
- `write_end_id`

Recovered `FUN_0058e0c0` serialized fields:

- `str player_name`
- `str ship_model_name`
- `u16 ship_health`
- `str pilot_model`
- `str engine_1_model`
- `str engine_2_model`
- `str engine_3_model`
- `str engine_4_model`
- `str weapon/loadout`
- `u32 team`

### Resource payload (`usr opcode 0x01`)

Function refs: `handle_resources`, `CreateResource`, `GetResourceName`.

`handle_resources` payload:

- `u16 start_index`
- `u8 count`
- `count * str resource_name`

This updates `NetResources` and therefore controls interpretation of `type_idx` in world-state entries.

## 4.6 Protocol-level field types (controller payloads)

Function refs: `RTTI::cNetController::handle_packet`, `RTTI::cNetController::method_008`, `RTTI::cNetVController::handle_packet`, `RTTI::cNetVController::method_008`.

This section intentionally describes only wire-level types and ordering.

Base controller payload assumptions:

- Leading `u8` bitmask controls presence/encoding of subsequent blocks.
- Position block: either `3xi16` quantized deltas (compact) or `3xf32` absolute values (full).
- Velocity/rotation/motion blocks: compact branches use quantized byte/short values; full branches use short/float forms depending on block.

Vehicle extension payload assumptions:

- Leading vehicle extension bitmask is `u16`.
- Optional vehicle fields are gated by `u16` bits (`0x0020`, `0x0080`, `0x2000`, `0x4000`, `0x8000`, plus state-mask bits).
- Known optional scalars include `u16` entity ids, `u8` state bytes, and compact `3xi16` target-position deltas.
- Base controller payload follows after vehicle-specific optional fields.

Stable parser/export typing rules:

- Treat kinematic values as logical `f32` state, with optional quantized wire encoding.
- Treat controller flags as bitmasks (`u8` base flags, `u16` vehicle-extension flags).
- Treat network entity ids as `u16` (`net%d` namespace), with `0x200` used as sentinel/invalid in several paths.
- Treat `type_idx` as `u8` indexing the session `NetResources` map.

## 5) Master/listing protocol (in-game)

Function refs: `ConnectToMasterServer`, `FUN_00590640`, `FUN_00590170`, `FUN_0058f600`, `FUN_005909b0`, `C2S_handle_packet`.

Master connection setup:

- `ConnectToMasterServer` resolves and stores master endpoint in `DAT_00866a80`.
- Calls are made through normal encrypted send wrapper (`FUN_005909b0`) using `LastMasterCommandSent` text.

Outbound command examples:

- Heartbeat from server loop (`C2S_handle_packet`):
  - format: `HB=%d,%d:%d|%s`
  - values observed: game id `0xceba`, version `0x101`, server port, browser code

Inbound master message parser (`FUN_00590170`) expects datagram prefix:

- first four bytes must be `00 00 00 00`
- command discriminator at byte `[4]`

Supported master command types:

- `'|'` : redirect/update master endpoint
  - payload includes `host:port` text
  - updates `MasterServerAddress`/`MasterServerPort`
- `'}'` : server list response
  - sequence of endpoint tuples (4-byte IPv4 + 2-byte port)
  - each endpoint is fed into browser ping pipeline (`FUN_0058f600`)
- `'~'` : text command/notification
  - passed to callback `MasterCommandPtr`

## 6) ASE/public query protocol (udpsoft/master style)

Function refs: `FUN_004028f0`, `FUN_00589300`, `FUN_00589270`, `FUN_005890c0`.

Implemented in `FUN_004028f0` (enabled when `EngineVars::ASEPublic` is set).

Server heartbeat sent periodically to:

- `master.udpsoft.com:27900`
- `master2.udpsoft.com:27900` (when resolved)

Heartbeat wire text is assembled as:

- `"\\hb\\<port>\\<servername>"`

Incoming one-byte command handlers:

- `'p'` : ping -> reply `'P'`
- `'g'` : short game name query -> reply `'G' + ASEName` text
- `'v'` (len 9, from master addr) : challenge verification -> reply `'V'` + 32-bit computed value
- `'s'` : status query -> reply `'EYE1'` structured status payload

`'s'` reply includes (built via `FUN_00589300`/`FUN_00589270`/`FUN_005890c0` and helpers):

- server name, type, map, version string (`1.1`), password flag, current/max players
- key/value extras like `fraglimit`, `dedicated`, `ForceServerVehicle`
- per-player blocks (`name`, `team`, vehicle/model fields, scorer, ping)

Large `'s'` replies are chunked into segments of up to `0x550` bytes with continuation markers.

## 7) Notable constants

Function refs: `make_join_packet`, `C2S_handle_packet`, `ConnectToMasterServer`, `init_server_net_manager`.

- Game ID: `0xCEBA`
- Net/game version: `0x0101`
- Default gameplay UDP port: `28086`
- Default master UDP port: `27900`

## 8) Directional packet reference (full wire structure)

Function refs: `FUN_005909b0`, `net_recv`, `handle_packet`, `handle_packet_from_client`, `handle_packet_from_server`, `FUN_00590170`, `FUN_004028f0`.

This section is a parser-oriented, direction-specific wire map.

### 8.1 Gameplay UDP datagram (both directions)

Function refs: `FUN_005909b0`, `net_recv`, `decrypt_chacha20`, `calc_poly1305_mac`.

Outer encrypted/authenticated datagram:

1. `nonce12` : `u8[12]`
2. `ciphertext` : `u8[n]` (encrypted plaintext envelope)
3. `poly1305_tag` : `u8[16]`

Decrypted plaintext envelope:

1. `seq_flags` : `u8`
   - `seq = seq_flags & 0x7f`
   - `dir_bit = seq_flags & 0x80`
2. `usr_len_plus_1` : `u16` (LE)
3. `usr_seq_flags` : `u8`
   - `usr_seq = usr_seq_flags & 0x7f`
4. `usr_payload` : `u8[usr_len_plus_1 - 1]`
5. `pkt_payload` : `u8[remaining_bytes]`

Derived sizes used by game code:

- `usr_size = usr_len_plus_1 - 1`
- `pkt_size = total_plaintext_size - usr_len_plus_1 - 3`

### 8.2 Client -> Server gameplay payload

Function refs: `handle_packet_from_client`, `make_join_packet`, `build_packet`.

`usr_payload` format:

1. `usr_opcode` : `u8`
2. `usr_body` : opcode-specific

Client->Server opcode bodies:

- `0x00` or `0x06` (join/config)
  1. `game_id` : `u16` (`0xCEBA`)
  2. `game_version` : `u16` (`0x0101`)
  3. `client_password` : `str`
  4. `player_name` : `str`
  5. `player_model` : `str`
  6. `player_max_life` : `u16`
  7. `pilot_model` : `str`
  8. `motor0_model` : `str`
  9. `motor1_model` : `str`
  10. `motor2_model` : `str`
  11. `motor3_model` : `str`
  12. `weapon_bay_list` : `str`
  13. `team_id` : `u32`
- `0x02` (goodbye/disconnect)
  - no body fields
- `0x03` (chat)
  1. `chat_text` : `str`
- `0x04` (user text)
  1. `user_text` : `str`
- `0x05` (full/config request)
  - no body fields
- `0x08` (remote console)
  1. `remote_password` : `str`
  2. `command` : `str`

`pkt_payload` in Client->Server direction:

- Entity/world delta stream emitted by client-controlled entities.
- Per-entity record wire header is still:
  1. `net_name` : `u16`
  2. `type_idx` : `u8`
  3. `entity_delta` : variable (class-specific)

### 8.3 Server -> Client gameplay payload

Function refs: `handle_packet_from_server`, `handle_resources`, `ClientSetConfig`, `build_packet`.

`usr_payload` format:

1. `usr_opcode` : `u8`
2. `usr_body` : opcode-specific

Server->Client opcode bodies:

- `0x00` (map/type init)
  1. `map_name` : `str`
  2. `server_type` : `str`
- `0x01` (resource table chunk)
  1. `start_index` : `u16`
  2. `count` : `u8`
  3. `resource_name[count]` : `str[]`
- `0x02` (control/ack)
  - body present in code path but no stable typed layout recovered
- `0x03` (chat)
  1. `chat_text` : `str`
- `0x04` (user text)
  1. `user_text` : `str`
- `0x05` (player config table)
  1. `count` : `u16`
  2. repeated `count` entries:
     - `net_id` : `u16`
     - `player_cfg_blob` : variable (starts with string/model fields described in section 4.5)
- `0x06` (remove player/entity)
  1. `player_idx` : `u8`
- `0x07` (reset/disconnect)
  - control opcode, no stable typed body recovered

`pkt_payload` in Server->Client direction (authoritative world update):

1. `ply_id` : `u16`
2. `num_vals` : `u16`
3. `player_pos_x` : `f32`
4. `player_pos_y` : `f32`
5. `player_pos_z` : `f32`
6. `player_idx` : `u8`
7. `rtt` : `u8`
8. repeated `num_vals` entity records:
   - `net_name` : `u16`
   - `type_idx` : `u8`
   - `entity_delta` : variable

### 8.4 Entity delta payload format (both directions)

Function refs: `cNetController::handle_packet`, `cNetController::method_008`, `cNetVController::handle_packet`, `cNetVController::method_008`.

Each entity record payload begins immediately after `(net_name, type_idx)`.

Base controller delta (shared families):

1. `base_flags` : `u8`
   - `0x01` precision mode (`0=compact`, `1=fuller precision`)
   - `0x02` position block present
   - `0x04` velocity block A present
   - `0x08` velocity block B present
   - `0x10` angular block present
   - `0x20` extra motion block(s) present
2. conditional blocks in this fixed gate order: `0x02`, `0x04`, `0x08`, `0x10`, `0x20`
   - compact branches: quantized `i16`/`u8`/`i8` style forms
   - full branches: `f32` and some short forms depending on block

Vehicle extension delta (for vehicle entities):

1. `veh_flags` : `u16`
2. optional fields gated by `veh_flags` bits
   - includes combinations of: `u16` target ids, `u8` state bytes, compact `3xi16` target deltas, packed nibble-control byte
3. trailing base controller delta (`base_flags` + gated blocks)

### 8.5 Master/listing protocol packets (in-game master server)

Function refs: `FUN_00590170`, `ConnectToMasterServer`, `FUN_00590640`, `FUN_0058f600`.

Master inbound datagram format (as parsed by `FUN_00590170`):

1. `prefix0` : `u8` must be `0x00`
2. `prefix1` : `u8` must be `0x00`
3. `prefix2` : `u8` must be `0x00`
4. `prefix3` : `u8` must be `0x00`
5. `master_cmd` : `u8` (`'|'`, `'}'`, `'~'`)
6. `master_body` : command-specific

Master command bodies:

- `'|'` redirect/update
  - `host_port_text` : `str`-like text (`"host:port"`)
- `'}'` server list
  - repeated tuples:
    1. `ipv4` : `u8[4]`
    2. `port` : `u16` (network tuple)
- `'~'` text command
  - `text` : command string passed to callback

Server heartbeat payload sent to master (text packet content):

- `"HB=%d,%d:%d|%s"`
  - fields are game-id, version, server port, browser code string

### 8.6 ASE/public query protocol packets (udpsoft style)

Function refs: `FUN_004028f0`, `FUN_00589300`, `FUN_00589270`, `FUN_005890c0`.

Inbound one-byte command at offset 0:

- `'p'` ping request -> `'P'` reply
- `'g'` game-name request -> `'G' + ASEName` reply
- `'v'` challenge verify (len 9, master source) -> `'V' + u32` computed reply
- `'s'` status request -> `'EYE1'` status block reply

`'s'` response is key/value text blocks plus per-player blocks, segmented when length exceeds `0x550` bytes.

## 9) Confidence and gaps

Function refs: `FUN_005909b0`, `net_recv`, `handle_packet_from_client`, `handle_packet_from_server`, `FUN_00590170`, `FUN_004028f0`.

High confidence:

- socket setup, mode transitions, ports
- encrypted outer datagram structure (nonce/ciphertext/tag)
- decrypted envelope field boundaries
- user/control opcodes and most payload schemas
- master/ASE command strings and dispatch

Remaining low-level gaps:

- exact semantics of some timeout/state fields in per-peer `NetData`
- full per-entity binary schema for every object type (serialization is virtual per-entity class)
- some runtime-initialized constants stored in BSS-like regions that are zero in static image

## 10) Confidence tags by sub-area

Function refs: consolidated from sections 1-9; primary anchors are `FUN_005909b0`, `net_recv`, `handle_packet*`, `FUN_00590170`, `FUN_004028f0`, `cNetController::*`, `cNetVController::*`.

- `[HIGH]` Transport/socket behavior, default ports, mode switching.
- `[HIGH]` Outer crypto framing (`nonce12 + ciphertext + tag16`) and Poly1305 failure behavior.
- `[HIGH]` Decrypted envelope split (`seq`, `usr_len_plus_1`, user payload, world payload).
- `[HIGH]` User opcode ids and field order for `0x00/0x01/0x03/0x04/0x05/0x06/0x08`.
- `[MEDIUM]` Exact body semantics for control opcodes `0x02` and `0x07`.
- `[HIGH]` Entity record header shape (`u16 net_name + u8 type_idx`) and dynamic `NetResources` mapping.
- `[MEDIUM]` Base controller flag ordering and block gating; `[LOW-MEDIUM]` exact scalar widths in every compact/full branch.
- `[MEDIUM]` `cNetController`/`cNetVController` struct field names/semantics; widths/offsets are stronger than labels.
- `[HIGH]` ASE-style query command characters and heartbeat/query endpoints.
