# Changelog

## 0.1.0 (2026-11-28)


### ⚠ BREAKING CHANGES

* adjacent records with one number join even when no `next` links them and their nominal lengths differ, which changes the ordinals of the floating measures that follow.
* read Listen Here! recordings as seconds timelines
* every conversion map mints the cmap prefix
* score timelines carry no ticks C-Map by default; pass ppq to ScoreStore.create_timeline() or get_cmaps() to attach one.
* **timelines:** `FlowControlRegistry.elements`, `jumps`, `jump_targets`, `breaks`, `section_breaks` and `barlines` return tuples; `FlowControlRegistry.clear()` is removed; adding an element or a place outside the axis, setting such a length, and rejecting an element that a standing element names raise ValueError; a Break with `ends.passes` as a list no longer ends the flow at arrivals beyond the highest pass listed.
* **timelines:** ComputedFlows has a third field `findings`; create_flow_maps_from_flow_control() no longer raises for sections named by a FlowMap of another origin or by a jump, raises without `actor` where a section's stretch is gone, and keeps instead of deleting the Regions of former sections; Region.is_atomic_section reads the rejection of the type statement.
* **core:** trim FlowManifestation to the manifestations the element classes admit
* **timelines:** return findings for the default flow only
* **timelines:** `FlowMapSection`, `FlowMap.from_qb_sections`, the placement form of `FlowMap.from_dict`, `FlowMap.inverse`, `FlowMap.unfold_coordinate`, `create_unfolded_timeline`, `unfold_via_flowmap`, `Timeline.apply_flow`, `Timeline.unfold_coordinate`, `Timeline.fold`, `Timeline.has_flow_map`, `Timeline.n_flow_maps` and `TimelineGroup.apply_flow` are removed without replacement: an applied FlowMap is a virtual timeline, not a copy of the timeline. `Timeline.create_flow_map` takes chunks instead of intervals, `Timeline.get_flow_map` raises `KeyError` for an unknown ID, and `Gap` holds one positive `length` and no label.
* **timelines:** `provenance` of a flow-control element is a tuple of entries and is written as a list; `active_elements()` and `play()` without a believer leave out the elements any actor rejects; `Jump.targets` are in canonical order.
* **loader:** write flow control from measures and source signs, drop the flow controller
* **timelines:** `Marker`, `ActivationCondition`, `control_type`, `condition`, `is_active`, `remaining_activations`, `Jump.from_coordinate`, `Jump.to_coordinate`, the registry's `add_break` / `add_jump` / `add_marker` and its `markers` list, and `ScoreFlowController.get_breaks` / `get_jumps` / `get_markers` / `get_flow_control_registry` are removed without aliases. The enum of manifestations is renamed `FlowManifestation` (it gains `volta` and `ossia`); the name `FlowControlElement` is now the elements' base class. An element written with `type`, `target` or its own `repeat_count` is refused with a message naming the form to write.
* **core:** ColumnNaming.name is now ColumnNaming.label, ColumnRole.name is now ColumnRole.label and ColumnRole.partition is now ColumnRole.partition_by. The string values are unchanged ("name", "partition"): passing, storing, serializing or comparing the strings behaves exactly as before, and ColumnRole("name") is ColumnRole.label.
* **graphical:** the axis parameter and canvases="first"|"all" are removed, create_timeline() without paths is a TypeError, uid and name are keyword-only, and IIIFCanvasInfo.extent and IIIFManifestLoader.name are gone.

### Features

* a placed chunk knows its FlowMap entry; look up the chunk at a played position ([d5d2c3e](https://github.com/TimeToAlign/TimeToAlign/commit/d5d2c3ea0af370ef34d5460d55362a3cd313fac3))
* accept a name on a CHUNK side of a MatchClaim ([edebc9b](https://github.com/TimeToAlign/TimeToAlign/commit/edebc9bf94034bb3a5b79b2e146e2a740297e866))
* **alignment:** write a bundle's Notes in a Note table beside the manifest ([c6b757f](https://github.com/TimeToAlign/TimeToAlign/commit/c6b757fe21a39b70b6640311f8883a027d5a6f1e))
* carry extractions with the events loaders read, and locate Boxes and table rows ([fb5e10e](https://github.com/TimeToAlign/TimeToAlign/commit/fb5e10e1113a75979f62d31238ee3eb677cf8998))
* check Box events against their image and write them as a package table ([19581a7](https://github.com/TimeToAlign/TimeToAlign/commit/19581a74c3b25517ac12ccdaabed204e42b16cd9))
* **core:** graphical shapes and the Box event on a timeline's axis ([ad7feaa](https://github.com/TimeToAlign/TimeToAlign/commit/ad7feaa37b583ec6cd2abb0bbbc293b28216e5bb))
* **core:** let a measure's volta state one pass or a list of passes ([eaf4107](https://github.com/TimeToAlign/TimeToAlign/commit/eaf4107f01440ae3d06a05215cc29552e6a44f77))
* **core:** move staff and voice from Note to a ScoreNote subtype ([9b84504](https://github.com/TimeToAlign/TimeToAlign/commit/9b84504558dd10de9dae92fe4f0142bbc43a949c))
* **core:** warn when from_field() ignores keyword arguments ([456dc48](https://github.com/TimeToAlign/TimeToAlign/commit/456dc48ac3eae6bd33a7358a9bc076a96cbd880a))
* every conversion map mints the cmap prefix ([69488e6](https://github.com/TimeToAlign/TimeToAlign/commit/69488e6e531d1ac2f775bd7dea69d607875fc62d))
* **flow:** report the findings of every convention, with the cause of a stretch single playback leaves out ([4f255ac](https://github.com/TimeToAlign/TimeToAlign/commit/4f255ac70532d7f01613d9ea076c567622521735))
* give children created from regions the slices of the measure map ([a46a5be](https://github.com/TimeToAlign/TimeToAlign/commit/a46a5be25b8574158c83d00e7ef2d32b5ba40158))
* **graphical:** IIIF canvases become timelines only along a given time path ([dfeeac2](https://github.com/TimeToAlign/TimeToAlign/commit/dfeeac2c324a236cc7572ab3102d950ee35ecfbd))
* **graphical:** one builder of volume, page and accolade timelines from time paths ([3d76f79](https://github.com/TimeToAlign/TimeToAlign/commit/3d76f79d9531de5dc6b1ab99a46f293602e33cd9))
* **graphical:** piecewise time paths, paths from boxes and one accolade rule ([689b43b](https://github.com/TimeToAlign/TimeToAlign/commit/689b43ba011ee079f5fb0649c75d9bf477c960d4))
* **graphical:** read DMOS pages along the time path of their systems ([36777e3](https://github.com/TimeToAlign/TimeToAlign/commit/36777e345a58f4650484f01c4c35c78ec0c0ace9))
* **graphical:** read MEI facsimile surfaces as pages of accolades along a time path ([1f264b7](https://github.com/TimeToAlign/TimeToAlign/commit/1f264b715669da0d2a9f3525eaac3836f4994e66))
* join a split bar numbered as two measures and warn about measures that do not add up ([1c9c0ad](https://github.com/TimeToAlign/TimeToAlign/commit/1c9c0ad07c4564463653bade923ae0935f288714))
* keep names out of the registry and write packages through the public registry doors ([1c6a1fe](https://github.com/TimeToAlign/TimeToAlign/commit/1c6a1fe286b8ff68928e2182adf500e679543974))
* load Musical Theme Dataset themes as an AlignmentBundle with MtdLoader ([62850da](https://github.com/TimeToAlign/TimeToAlign/commit/62850da78938ace852e1cafdbb34b7a928dd47f6))
* **loader:** lock score timelines whose measures state their length ([219b65a](https://github.com/TimeToAlign/TimeToAlign/commit/219b65aa8bed5aade6c3ef00e858c7b24ae7203b))
* **loader:** lock the timelines AudioLoader creates at the file's length ([e2f02c9](https://github.com/TimeToAlign/TimeToAlign/commit/e2f02c9984655f0e814c6b31a35757ba11839b05))
* **loader:** read a whole IIIF volume as a pixel segment line and accept time-based canvases ([f394536](https://github.com/TimeToAlign/TimeToAlign/commit/f3945363a5def01585343a4bf826e7027071a15e))
* **loader:** read DMOS page layouts and MEI facsimile zones as Box events ([7e132a7](https://github.com/TimeToAlign/TimeToAlign/commit/7e132a7e8268cd1321801415941680fad2c94f73))
* **loader:** read sources directly from ZIP archive members ([43ea024](https://github.com/TimeToAlign/TimeToAlign/commit/43ea024064aa38263afdb29936a836544cba3706))
* loaders mint every id from one minter seeded by their source ([c8a06be](https://github.com/TimeToAlign/TimeToAlign/commit/c8a06be38edbf38efc1b835ce5a5d5ef152b1827))
* **loader:** write a Fine no leap names and a D.C. beside a repeat end without start ([9012929](https://github.com/TimeToAlign/TimeToAlign/commit/90129298ec8f3ddae4ef5349d4890f77cae538d2))
* **loader:** write flow control from measures and source signs, drop the flow controller ([c3bbb4f](https://github.com/TimeToAlign/TimeToAlign/commit/c3bbb4f9c17f459105d889a41a532f3e22a149bc))
* **loader:** write the order of measures a source states as a FlowMap ([e1d67d4](https://github.com/TimeToAlign/TimeToAlign/commit/e1d67d4da8371070b87792c2bcce452e943c4121))
* locate every loaded event by its place and the key its source states ([bb7a81b](https://github.com/TimeToAlign/TimeToAlign/commit/bb7a81b3b20e6e53622189e7eb5e7586973062ee))
* locate the Projections a MAPS file makes in the file and its extraction ([c3e29e2](https://github.com/TimeToAlign/TimeToAlign/commit/c3e29e2050d4de5804b1897ce412906bff8e5e57))
* MIDI timelines carry the file's tempo map and bars ([7546aeb](https://github.com/TimeToAlign/TimeToAlign/commit/7546aeb65d56ccb360d7229caeefea59f03a53ba))
* mint event, timeline and MatchClaim ids from seeded minters ([7d5fe37](https://github.com/TimeToAlign/TimeToAlign/commit/7d5fe37db1a8dad202e43cc0dbb5e5d4dfe79971))
* mint every id at construction and refuse collisions without renaming ([5187587](https://github.com/TimeToAlign/TimeToAlign/commit/5187587818015f1f602a278ce7292faf5e429e14))
* MtdLoader makes one timeline per MTD file, grouped by encoding ([f52ae9a](https://github.com/TimeToAlign/TimeToAlign/commit/f52ae9ae1b49c0a2aa19f98cdc6419a5edd75348))
* name MatchClaim sides without an event by the sentinels NOMATCH and CHUNK ([ceb2bd7](https://github.com/TimeToAlign/TimeToAlign/commit/ceb2bd7c72a4a3854d077709ab5942619a4258eb))
* one event table per timeline tree; joins keep each row's number type ([3209969](https://github.com/TimeToAlign/TimeToAlign/commit/3209969054454e25157308d693d4931d0b3b0df8))
* per-row number-type column kernels and one cell validator ([fa4faa3](https://github.com/TimeToAlign/TimeToAlign/commit/fa4faa3aca106dfdff9484bc2e294e6bfdf1e852))
* play a timeline's flow control together with its descendants' ([11ee8d0](https://github.com/TimeToAlign/TimeToAlign/commit/11ee8d0f120093856b83684d10cf16e3dadb0ab8))
* Projections as registry rows, parses for extractions ([4d417fc](https://github.com/TimeToAlign/TimeToAlign/commit/4d417fc20301c3f3f981427a6afad197502fdc00))
* public registry doors for writers, package readers and updates ([4e2de37](https://github.com/TimeToAlign/TimeToAlign/commit/4e2de371247c8817564ce2028574b86e9134c24a))
* read every table resource of a type and folder resources of a package ([d3ce7c5](https://github.com/TimeToAlign/TimeToAlign/commit/d3ce7c5d5f6be453ef1fe0c9ebaad2a83a72e0d3))
* read Listen Here! recordings as seconds timelines ([99bd90c](https://github.com/TimeToAlign/TimeToAlign/commit/99bd90c4bac03844b7788c24f6fae2df7c35a32d))
* read the records of one measure number as one floating measure ([49f5dc4](https://github.com/TimeToAlign/TimeToAlign/commit/49f5dc40960cdd00b952af71d147d824b6119739))
* record an extraction per source file and a locator per event in the loaders ([f53f2ea](https://github.com/TimeToAlign/TimeToAlign/commit/f53f2eae5b7e287a9eaba386a987a2a94c7e9aee))
* seeded random-id minter, whole-id pattern resolution ([cb038ba](https://github.com/TimeToAlign/TimeToAlign/commit/cb038ba1647e0d66eba3b01c811c5a161f2d0c16))
* slice a measure map and give a child the slice of its parent's ([74206f1](https://github.com/TimeToAlign/TimeToAlign/commit/74206f18d942e432f1cf800db894f0b7751914cd))
* **storage:** read Notes and ScoreNotes back with get_field ([4a78afd](https://github.com/TimeToAlign/TimeToAlign/commit/4a78afd485a9ce480994d7dd5a7dcd0bfecd8247))
* **storage:** read the events of an event type back with get_field ([d2c6856](https://github.com/TimeToAlign/TimeToAlign/commit/d2c68561de19976a59131844220c3a00e511fd94))
* **timelines:** hold flow-control provenance as entries and leave rejected elements out ([1540189](https://github.com/TimeToAlign/TimeToAlign/commit/154018922c953dc6d77d49e8ae9f6162f811d915))
* **timelines:** keep provenance entries of flow control and FlowMaps as the format admits them ([707998d](https://github.com/TimeToAlign/TimeToAlign/commit/707998d87b5fd7447a1ec733e4cf9a6991aa750e))
* **timelines:** return the findings of a flow and keep section Regions when flows are computed again ([e6f9f06](https://github.com/TimeToAlign/TimeToAlign/commit/e6f9f068a55202f0d5341d100bb7c8568b8829c9))
* **timelines:** state flow control as five element classes and play it ([a9f94f1](https://github.com/TimeToAlign/TimeToAlign/commit/a9f94f11f818fbd31499c1a4c655d389d15805a2))
* **timelines:** write a child timeline's computed flows on its root ([70d3b3f](https://github.com/TimeToAlign/TimeToAlign/commit/70d3b3feed78f1b181c6e1761618099431736147))
* **timelines:** write FlowMaps as chunk references and state virtual timelines ([bf44d1a](https://github.com/TimeToAlign/TimeToAlign/commit/bf44d1aa4b1bac88937bce0b5281ed5643570ac6))
* virtual timelines as group members, FlowMaps applied through a group ([f1e82ec](https://github.com/TimeToAlign/TimeToAlign/commit/f1e82ecf23f24de9b9bea6a002b8832b74a43bcd))
* write a package's registry as one Parquet file ([4319362](https://github.com/TimeToAlign/TimeToAlign/commit/4319362241d77572ac68e343c91d2fd32ce147c3))
* write and read packages through the event registry, and resolve IDs ([2f75516](https://github.com/TimeToAlign/TimeToAlign/commit/2f755166cd6c5cfaaa6d9eb045f4e89c728abedd))
* write every event type as event files and add the Package reader and writer ([f1ddb90](https://github.com/TimeToAlign/TimeToAlign/commit/f1ddb90d749f1d68870d504fe5641984721d00cc))
* write Listen Here! grid claims as CHUNK on both sides ([6827ab7](https://github.com/TimeToAlign/TimeToAlign/commit/6827ab76b41f82db18233f6504704f3d370bd4f9))


### Bug Fixes

* a joined row keeps its start and duration on an exact table ([67ce427](https://github.com/TimeToAlign/TimeToAlign/commit/67ce4271fa9301f29f4c408e0c6cb6a1a3c830c0))
* a MIDI timeline is as long as its end of track, and locked ([827ee9b](https://github.com/TimeToAlign/TimeToAlign/commit/827ee9b27d4a858cd4ceb0e887259f386a21a359))
* a record named by next joins its bar across a metre change ([af2e6f1](https://github.com/TimeToAlign/TimeToAlign/commit/af2e6f19b8bb8088717e2dbeca46c2afb377b769))
* a spanning cast keys changed rows by their axis; one column rule for joins ([f7ae2b9](https://github.com/TimeToAlign/TimeToAlign/commit/f7ae2b9106531c93887f8fd3e998ee2cec986101))
* accept a ZIP archive's root as a loader source ([31a70ff](https://github.com/TimeToAlign/TimeToAlign/commit/31a70ffecce8f0ef07432d4d12fdc56da22b3f16))
* **alignment:** a MatchStamp converts through the one map its spec names ([fd1e2b3](https://github.com/TimeToAlign/TimeToAlign/commit/fd1e2b375b42ac4fae00820d2b1f460b45bdb268))
* **alignment:** compare MatchClaim ids through the equal kernel as given ([c54c25d](https://github.com/TimeToAlign/TimeToAlign/commit/c54c25dba6d267f99f58365c970b226e656e37e1))
* **alignment:** let write_match_file write the MatchLine it is given ([5e131de](https://github.com/TimeToAlign/TimeToAlign/commit/5e131dee3472762fee862eebaf4745243547ba74))
* **alignment:** map interval claims exactly and read them from group siblings ([828fb4d](https://github.com/TimeToAlign/TimeToAlign/commit/828fb4db85148eeb8699ecf087073bc9f9a96087))
* **alignment:** orient a MatchStamp's pairs by the stamp's timeline order ([9731343](https://github.com/TimeToAlign/TimeToAlign/commit/97313437ba70ccf26cb4da2aeb5355facf96977e))
* **alignment:** refuse a Box read from a package in the words add_events uses ([33c69fe](https://github.com/TimeToAlign/TimeToAlign/commit/33c69fe87c376eec0ac02795ddcddc431baa0363))
* **alignment:** refuse a null row when a MatchClaimField is constructed ([e5c3705](https://github.com/TimeToAlign/TimeToAlign/commit/e5c3705c9711a9ed46cf5046a9cf6bf993dbdc00))
* **alignment:** refuse a position a nested timeline cannot reach before any Projection is added ([8eaedb1](https://github.com/TimeToAlign/TimeToAlign/commit/8eaedb10e06be09394aafc2a7a472ddd1d572389))
* **alignment:** refuse an unstorable float before create_match_claims adds anything ([dfd8319](https://github.com/TimeToAlign/TimeToAlign/commit/dfd8319f3ebd45b8d7c24393e2e98f34c79e28f6))
* **alignment:** refuse floats on samples, backwards spans, positions past a locked end and held IDs with another span in create_match_claims ([ba7c96d](https://github.com/TimeToAlign/TimeToAlign/commit/ba7c96de91341cc508ff49d361c0c68c000097ba))
* **alignment:** show a MatchStamp without raising when a claim it lists can no longer be read ([5a0a467](https://github.com/TimeToAlign/TimeToAlign/commit/5a0a4677303fb41251184207e3aa93b6babe6522))
* **alignment:** skip a recorded name whose row names no timeline ([7653b66](https://github.com/TimeToAlign/TimeToAlign/commit/7653b669041a79dd67b0770d8da2dc3305bc8b1f))
* **alignment:** write and read no Note table beside a package's manifest ([4d7d3da](https://github.com/TimeToAlign/TimeToAlign/commit/4d7d3da152769f8250ace548b914ae9ad64c93fb))
* **aton:** give holes sharing an @ID distinct event IDs ([bc2d4d1](https://github.com/TimeToAlign/TimeToAlign/commit/bc2d4d19ae4daa847a9b3590439a10af1e40bf74))
* **audio:** count an MP4's samples as its edit list states them ([fa97ad5](https://github.com/TimeToAlign/TimeToAlign/commit/fa97ad58f141390b63cd87fc8b2f211fd9b69d1a))
* call a column function once, passing name only when its signature takes it ([6b39e67](https://github.com/TimeToAlign/TimeToAlign/commit/6b39e67e041d8469594d4bd3cedebadaca84ca92))
* carry tables of other rows into packages and clean up cut-short writes ([a27a917](https://github.com/TimeToAlign/TimeToAlign/commit/a27a9173b2105889e12a1877c775490a7a67ba2e))
* check every Box row alike and write its shape as JSON numbers ([63c584a](https://github.com/TimeToAlign/TimeToAlign/commit/63c584a8365390356b25e9af8b30dd1a75344da9))
* compare cells by value, name event holders at every depth, refuse broken narrowings ([fd675cb](https://github.com/TimeToAlign/TimeToAlign/commit/fd675cb9eee150960e38c1dc0a6556841a60b539))
* compare Projections read from a table and keep a user's Segment rows in copies ([cdf0227](https://github.com/TimeToAlign/TimeToAlign/commit/cdf0227a60582e51e56fd0fb989b94cdd116454f))
* complete interval end and duration in the table's number type ([ad03b4f](https://github.com/TimeToAlign/TimeToAlign/commit/ad03b4fcb296e603de34642c9cb167ca09b010ee))
* compute exact column arithmetic without intermediate overflow ([36a3b2b](https://github.com/TimeToAlign/TimeToAlign/commit/36a3b2b7a181be6d8743b50e68b153c8b901238b))
* **core:** a pitch field without data has no pitch columns ([a36efea](https://github.com/TimeToAlign/TimeToAlign/commit/a36efea6874dff3eed8e35c73021386721263085))
* **core:** accept nulls among the iterable ids taken by id minting ([37a93c2](https://github.com/TimeToAlign/TimeToAlign/commit/37a93c2c4ff64f33cb01e0b61ab755e1b312182d))
* **core:** answer interval endpoints and protocol checks on empty fields ([56bdb8f](https://github.com/TimeToAlign/TimeToAlign/commit/56bdb8f932ed066ddf3540e4a16df2ccae0f2dcf))
* **core:** build converted pitch structs from any column the field holds ([05e5e4b](https://github.com/TimeToAlign/TimeToAlign/commit/05e5e4b37c49639f417742e52089f9c30fa09694))
* **core:** let an unpaired shared-axis field lack semantic_type again ([d5bfca1](https://github.com/TimeToAlign/TimeToAlign/commit/d5bfca13140b45dcbbab732185083d1264c1c00a))
* **core:** let axis fields declared without data answer and print as unbound ([7333c92](https://github.com/TimeToAlign/TimeToAlign/commit/7333c92bf7f01482abb78a513b5d32d8c262fedb))
* **core:** let Box.shape answer None on an unvalidated Box again ([d573903](https://github.com/TimeToAlign/TimeToAlign/commit/d57390343c0a985ab030036becaa8f00df1e28f5))
* **core:** let raw blueprints cast a plain sequence in from_array again ([bba4b91](https://github.com/TimeToAlign/TimeToAlign/commit/bba4b91f49b7ef65916dc1081d52dbb19bf333ea))
* **core:** never round shape geometry; read SVG corner lists in any order ([686ed02](https://github.com/TimeToAlign/TimeToAlign/commit/686ed024ee8629a41790de5f97c2694d837270f8))
* **core:** pitch fields answer null on a null row ([34c62b4](https://github.com/TimeToAlign/TimeToAlign/commit/34c62b47124243cdfae8a99aa449ca0e2c0fb899))
* **core:** refuse a DCML label naming no chord type with the parse error ([bc3a9c5](https://github.com/TimeToAlign/TimeToAlign/commit/bc3a9c534561a7a1396dab4486d4b2d93709151a))
* **core:** refuse a timestamp cell that carries no number ([9b30c97](https://github.com/TimeToAlign/TimeToAlign/commit/9b30c97e5508889201ccac3065deac33c8db7b64))
* **core:** refuse non-finite shape numbers and ratios on float axes, read closed SVG rings ([2e2db11](https://github.com/TimeToAlign/TimeToAlign/commit/2e2db11f0670e66e4ee82d260d20a073e722ff5f))
* **core:** type cell_coordinate on an optional cell without a scalar as Coordinate | None ([48fac15](https://github.com/TimeToAlign/TimeToAlign/commit/48fac15e0dcb43dd68daf6ec840ef55b74c027c6))
* create timelines from storage after a bare import timetoalign ([8d70f71](https://github.com/TimeToAlign/TimeToAlign/commit/8d70f7180a6d2cfff302949f44a1e754eee4c40d))
* decide a float position against exact child boundaries on the queried axis ([b2fd492](https://github.com/TimeToAlign/TimeToAlign/commit/b2fd4920c90ac282b8818469ae56158478a37e94))
* decide interval consistency exactly on exact axes ([9515707](https://github.com/TimeToAlign/TimeToAlign/commit/95157071fcf7dad8be4d4ff45332318073d799a4))
* document MTD seconds outside an excerpt and name two-movement tracks ([f5baff5](https://github.com/TimeToAlign/TimeToAlign/commit/f5baff5c6767dc596b226fdd6644d4dc557254d5))
* give a child the measure views of its parent in the parent's number type ([4b7082a](https://github.com/TimeToAlign/TimeToAlign/commit/4b7082a7025618f55cd4a14f5172bee72afae17f))
* give every conversion-map column of a timestamp table its own name ([28c9770](https://github.com/TimeToAlign/TimeToAlign/commit/28c97709c5e1a99580e93a770f0e3673c8b82836))
* **graphical:** leave DMOS systems without width and measures out of the time path ([fa88db4](https://github.com/TimeToAlign/TimeToAlign/commit/fa88db4fc8386114bc781795e6d5e5bfc9ce4ec5))
* **graphical:** settle the ids of a page-image call before building ([d3bf011](https://github.com/TimeToAlign/TimeToAlign/commit/d3bf011da11938790817b876efaca31d8256ba12))
* grouped timelines keep the extractions of their loaded rows ([674c8a5](https://github.com/TimeToAlign/TimeToAlign/commit/674c8a58a04cdc146301ccb339b63170391d5037))
* **iiif:** open manifests that state no [@context](https://github.com/context) by their structure ([33696d0](https://github.com/TimeToAlign/TimeToAlign/commit/33696d0b2b0e8943a6c97d04502626a9a88bc3be))
* join a bar split at a repeat sign into one floating measure ([edaaece](https://github.com/TimeToAlign/TimeToAlign/commit/edaaece2573a2a7ae7e3ebb762f31721a1f24eb7))
* join event tables of any number type; float queries compare as keys ([7c71a66](https://github.com/TimeToAlign/TimeToAlign/commit/7c71a6670d2e58c5c14770a8b18f3d1daa8e8cb6))
* keep each row's number type in joined tables and harden the event registry ([1251264](https://github.com/TimeToAlign/TimeToAlign/commit/12512645d69738cd2b95c65d5f209587d110136e))
* keep one number type and one spelling per EventData table ([27304f0](https://github.com/TimeToAlign/TimeToAlign/commit/27304f0b8836132a947fa144cfe96d87b3f42b65))
* keep packages whole and write one table file per type ([9517d61](https://github.com/TimeToAlign/TimeToAlign/commit/9517d612eb1394f3cb72c57030e155681c0de689))
* keep the source bundles' resources when merging bundles ([759c312](https://github.com/TimeToAlign/TimeToAlign/commit/759c312e6f8ca20b41de551322086cb943888ad3))
* leave a score MIDI's velocity null; name the graphical extra in install hints ([4e6e43d](https://github.com/TimeToAlign/TimeToAlign/commit/4e6e43d523b2283ff19c62a0f2f1db669c64aeb1))
* let a match timestamp be a groupstamp on a grouped timeline ([093cf97](https://github.com/TimeToAlign/TimeToAlign/commit/093cf97b1e31fe51cb5c5e821945f8e0b018f467))
* let timelines join whatever Projection IDs they hold ([e2229d8](https://github.com/TimeToAlign/TimeToAlign/commit/e2229d8606a41363e6d66284b9010826bbffbf70))
* let unexpected errors out of the bundle's fallbacks; read timeline ids once ([8adb997](https://github.com/TimeToAlign/TimeToAlign/commit/8adb997362bcb7ae98e586c5bbdc0a17c42abbe0))
* **loader:** declare the event payload as the dict the loader accepts ([c8b6914](https://github.com/TimeToAlign/TimeToAlign/commit/c8b69142ed3affefae5c4adc16861cf8c656251e))
* **loader:** give a measure of only &lt;mSpace&gt; its meter's length in VerovioLoader ([85395e3](https://github.com/TimeToAlign/TimeToAlign/commit/85395e3969cde09c1bc089b15d01ceeb68f5d41e))
* **loader:** import every loader subpackage without its optional backends ([2895f6b](https://github.com/TimeToAlign/TimeToAlign/commit/2895f6bafaf50ff7428307b7ce5718df27e4f4a7))
* **loader:** keep archive members out of metadata strings and read audio and RDF members in place ([99476da](https://github.com/TimeToAlign/TimeToAlign/commit/99476da63acf77af3f7e5730fd6ad6c99c611b25))
* **loader:** keep Partitura's float times when joining MIDI velocities ([cb3fc94](https://github.com/TimeToAlign/TimeToAlign/commit/cb3fc94beae11621defd40e25064d60cf4ee22c2))
* **loader:** let Verovio time a measure of only &lt;mSpace&gt; as a measure rest ([8a520ee](https://github.com/TimeToAlign/TimeToAlign/commit/8a520eeecabe9f358df1f54abaa454128b23b60e))
* **loader:** pass over an empty ms3 companion instead of refusing the load ([5e0d88a](https://github.com/TimeToAlign/TimeToAlign/commit/5e0d88acecad9b8f8901072c76aa7318fbf208c9))
* **loader:** refuse an ms3 TSV file that holds no table, naming it ([f2aa78c](https://github.com/TimeToAlign/TimeToAlign/commit/f2aa78c1099ec6d082f03b2cd1f789963532a23b))
* **loader:** refuse DMOS measures outside their system and export the page-image readers ([b4d1b71](https://github.com/TimeToAlign/TimeToAlign/commit/b4d1b71e24acc43ee6f3f8974dffa6bdd8c8c73f))
* **loader:** refuse MpmLoader.create_timeline after an incomplete load ([5f3eebb](https://github.com/TimeToAlign/TimeToAlign/commit/5f3eebb34b1124cbb46dfe5551fdf75202746ba2))
* **loader:** refuse one time path for a time-based IIIF canvas ([71f6f29](https://github.com/TimeToAlign/TimeToAlign/commit/71f6f2976e2aba22dee54814265edac4ab69ce28))
* **loader:** refuse sources whose measures music21 gives no length ([c1a1d4e](https://github.com/TimeToAlign/TimeToAlign/commit/c1a1d4ea5e3b72a1d276fda39c661514ae837911))
* **loader:** state only what score sources state; one optional-dependency idiom ([17eca9e](https://github.com/TimeToAlign/TimeToAlign/commit/17eca9e5ebca8d4d79edbabc72b1050989b9d1c7))
* **loader:** state the resolved number type in score store metadata ([08815c6](https://github.com/TimeToAlign/TimeToAlign/commit/08815c609ba0e8d6b3a8643a14e376c9f4c53ce1))
* **loader:** stop copying audio file metadata onto timelines ([dcca70c](https://github.com/TimeToAlign/TimeToAlign/commit/dcca70cd3758be7f8fb9d0805e21d8abd5491aaa))
* **maps:** answer TableMap.is_invertible as a Python bool ([0e674c6](https://github.com/TimeToAlign/TimeToAlign/commit/0e674c69be9b6942fb1b772e9177a12325f6fb62))
* **maps:** keep exact measure starts exact in from_measure_data ([9655f8d](https://github.com/TimeToAlign/TimeToAlign/commit/9655f8dd57ab8dd751f0f793538a760757e3d974))
* measure, grow and query timeline events in the axis's number type ([1a5e347](https://github.com/TimeToAlign/TimeToAlign/commit/1a5e3477a3b9a064f4b3599bbc3e693c824fcced))
* **midi:** leave an unjoinable score velocity empty with a warning ([ed0f6bf](https://github.com/TimeToAlign/TimeToAlign/commit/ed0f6bf3384d5493b809301d29d3c303470999de))
* **midi:** read note velocities on the default score parser ([a5b4996](https://github.com/TimeToAlign/TimeToAlign/commit/a5b4996eb12f955f00e20e221ae8cccf2e463e80))
* mint everything a loader builds from its own minter ([58dbd18](https://github.com/TimeToAlign/TimeToAlign/commit/58dbd187afcef48fac53f81e09c44c075f8be29d))
* mint Projection IDs unique across the manifest ([5c1efeb](https://github.com/TimeToAlign/TimeToAlign/commit/5c1efebb4467d0d2098c21e7dc1f8a11488cff38))
* MTD participation children take the excerpt's exact length; document what the groups refuse ([68d7bb2](https://github.com/TimeToAlign/TimeToAlign/commit/68d7bb2ed26a9d1f5567f22d0321845a653c5d71))
* MTD takes each MIDI timeline as the MIDI loader states it ([8748c35](https://github.com/TimeToAlign/TimeToAlign/commit/8748c35d2f370bf4285ed52d06341ae1070b39a5))
* name IEEE 1599 groups by their timeline's role, not its minted ID ([b76ec41](https://github.com/TimeToAlign/TimeToAlign/commit/b76ec413b990fccc310679689b55e54223b4115c))
* number casts read each row from the side it holds ([6c1b1c8](https://github.com/TimeToAlign/TimeToAlign/commit/6c1b1c8ffc0fd93182a48ff3e32713c69f74361e))
* **package:** keep one local table per type ([9319a5b](https://github.com/TimeToAlign/TimeToAlign/commit/9319a5be764c06aea2f1d72b43fa7d2e3ba4ece8))
* **package:** refuse a Projection of a timeline no manifest holds ([a306869](https://github.com/TimeToAlign/TimeToAlign/commit/a3068695815662581641acba2ffc064a2af5f94a))
* pass exact boundaries to the event paths and reduce cells only on entry ([c7de76b](https://github.com/TimeToAlign/TimeToAlign/commit/c7de76b1ac40e1062e97de75e1db5dcb810245a6))
* pass the keyword-only arguments of load through Loader.from_file ([17b00ba](https://github.com/TimeToAlign/TimeToAlign/commit/17b00ba0254052bef24b0eb5b2025b810d5bd0ac))
* read a dictionary-encoded start or end column in coordinate_range ([049f14b](https://github.com/TimeToAlign/TimeToAlign/commit/049f14b14597f9f7931fdcbb841727a2d5d044ac))
* refuse a child whose Projection IDs its new manifest holds ([5718c76](https://github.com/TimeToAlign/TimeToAlign/commit/5718c76059ba7a356b0f94bbdd907f40481a39b7))
* refuse a groupstamp lying in no group, on reading and writing a manifest ([02752e0](https://github.com/TimeToAlign/TimeToAlign/commit/02752e08fa965839e02818b376d85fa8a2120c96))
* refuse a sentinel named as an event by a claim side or a Projection ([a3488bf](https://github.com/TimeToAlign/TimeToAlign/commit/a3488bf58bc20561fa87a1b5bc560ba2843115f9))
* refuse a shared Projection ID when timelines join, project inside the bundle ([4ecfa46](https://github.com/TimeToAlign/TimeToAlign/commit/4ecfa46d993e5a9a667304d9e1a8fe84e300e824))
* Rekordbox beat grids build floating measures without notation warnings ([974e498](https://github.com/TimeToAlign/TimeToAlign/commit/974e49843044cfee5e28968236ef9293f25d82d0))
* report offset mismatches and every pass of a repeat, and warn where a score has no floating measures ([72d805d](https://github.com/TimeToAlign/TimeToAlign/commit/72d805d767f14364b40edd02762be79ea8b389fd))
* respell pitch-name columns of every type without Series.replace ([c999345](https://github.com/TimeToAlign/TimeToAlign/commit/c99934551b6a3d20e21e6a9fa3c006d4cf535824))
* score loaders honour ppq and attach_cmaps ([9f13aab](https://github.com/TimeToAlign/TimeToAlign/commit/9f13aab66d7d1da506279873ce11f8e8f0597dc4))
* score loaders no longer invent a 480-PPQ ticks map ([a31e778](https://github.com/TimeToAlign/TimeToAlign/commit/a31e7786a9232795dd4b24a5b2ee80d85d5cd909))
* seed loader IDs per load call and from configuration; mint IDs sources do not state ([373c520](https://github.com/TimeToAlign/TimeToAlign/commit/373c5208a525e6786abb438f9e539e1059997045))
* seed minted ids from complete content and one minter per seeded call ([82d078d](https://github.com/TimeToAlign/TimeToAlign/commit/82d078dee0509ea63d3c5bd9ea4e9e2082be26ff))
* serialize the manifest before writing any package table ([b56f6b9](https://github.com/TimeToAlign/TimeToAlign/commit/b56f6b9e8a052c6256b1997e4cb72a3536f46ff8))
* spell pitch names with sharp and flat signs at every loader boundary ([8feed8d](https://github.com/TimeToAlign/TimeToAlign/commit/8feed8d8f1009eb1cff3585d060741ace3924383))
* stamp displays never raise and C-Map specs are read map by map ([222086b](https://github.com/TimeToAlign/TimeToAlign/commit/222086b5ccfbb0ed7449ce4b2effced17be29482))
* **storage:** compose an event type's field only where the table holds its data ([75d9ebb](https://github.com/TimeToAlign/TimeToAlign/commit/75d9ebb659ffa00737b64347d962dcaa8ac7ac77))
* **storage:** derive the scalar-to-field map at each lookup ([954776e](https://github.com/TimeToAlign/TimeToAlign/commit/954776e407e66831464fc410fd9357a308d0512f))
* **storage:** give no Note field where no column holds a pitch the Note reads ([5f34174](https://github.com/TimeToAlign/TimeToAlign/commit/5f34174fbe50b95b3af543b3e522e9be62829ef9))
* **storage:** keep a null event id null in EventData.prefix_ids ([590e6ec](https://github.com/TimeToAlign/TimeToAlign/commit/590e6ecd194031824c64c57b9811458f9d429d4f))
* **storage:** name non-string column labels in EventData.from_dataframe ([2fb175e](https://github.com/TimeToAlign/TimeToAlign/commit/2fb175e2c4e5d8a6a206441d0c29695dd3132eee))
* **storage:** read Box shape cells through the shape constructor and refuse what the Box table cannot hold ([913f149](https://github.com/TimeToAlign/TimeToAlign/commit/913f14994cd4173ebebbde7a2b068d7d44d99ce1))
* **storage:** read numeric literals in computed-field formulas as numbers ([d6b685e](https://github.com/TimeToAlign/TimeToAlign/commit/d6b685ede76ef8604aff9c2646f250d0d3ef9fc4))
* **storage:** refuse a computed-field formula whose terms do not alternate ([ec2bbb6](https://github.com/TimeToAlign/TimeToAlign/commit/ec2bbb6a0c1dce932d0ad0b3e7e8deac83173a26))
* the tree store keeps row order and every valid tree, joins never refuse ([f0e64db](https://github.com/TimeToAlign/TimeToAlign/commit/f0e64dbe2e39862e676f4c6941ca0ba40b2d6640))
* **timelines:** accept use_conversion_map in SegmentLine.add_child ([f6e5d25](https://github.com/TimeToAlign/TimeToAlign/commit/f6e5d257e3afc369c99637643285c751277d0d67))
* **timelines:** carry a position to a descendant down the queried tree ([3ed3161](https://github.com/TimeToAlign/TimeToAlign/commit/3ed3161979290721099b14e832d24d11ff9004dd))
* **timelines:** cut children from overlapping regions without doubling events ([64a84d2](https://github.com/TimeToAlign/TimeToAlign/commit/64a84d2318308c58503bcdeead4c43c4e07e2523))
* **timelines:** derive C-Map table columns by one rule in both builders ([dabf8bc](https://github.com/TimeToAlign/TimeToAlign/commit/dabf8bcf1e4885b734f25101b9eb9b63ba325b76))
* **timelines:** keep Boxes valid when slicing, splitting or cutting children ([31e8cc6](https://github.com/TimeToAlign/TimeToAlign/commit/31e8cc625a480717177dae78646a8cc52ded7301))
* **timelines:** keep flow control on its axis and its registry closed to anything but adding and believing ([57c8566](https://github.com/TimeToAlign/TimeToAlign/commit/57c856634de5ba7103af41290e13b4acbafe79ad))
* **timelines:** let a locked timeline give a child from one of its regions ([1bc727f](https://github.com/TimeToAlign/TimeToAlign/commit/1bc727f71b107ddedbc8d6127c8a73b5bcb9df24))
* **timelines:** read a NumPy float tempo weight as the exact value of its double ([c95af84](https://github.com/TimeToAlign/TimeToAlign/commit/c95af84f63010ea3fadb40fde047e412d7ef52ad))
* **timelines:** read a Projections table from a file object again ([bf0ffe7](https://github.com/TimeToAlign/TimeToAlign/commit/bf0ffe7b84389bee35c6f289b85318ac4071a04a))
* **timelines:** read meta["box"] on cuts only of timelines holding Boxes ([4be7456](https://github.com/TimeToAlign/TimeToAlign/commit/4be7456a5476feb18a48d8edcd32fb986793c080))
* **timelines:** refuse a played flow holding a gap when writing a measure order ([c4597f3](https://github.com/TimeToAlign/TimeToAlign/commit/c4597f3f515aa4f40212fe644e8134a3f1c95a2f))
* **timelines:** refuse a second jump at a position of the root's axis when it is added ([d5eb7d8](https://github.com/TimeToAlign/TimeToAlign/commit/d5eb7d83dc8e21cceef1cea3485869df302e6caf))
* **timelines:** refuse cuts that move a descendant's Box onto a timeline stating no area ([d687a86](https://github.com/TimeToAlign/TimeToAlign/commit/d687a86ae44c312885d02436e59be21162e674b7))
* **timelines:** refuse from_segmentation on a line class without segment type ([2eda2d1](https://github.com/TimeToAlign/TimeToAlign/commit/2eda2d1a09b0f9599e91ffb55ded7f0376fa2bab))
* **timelines:** return findings for the default flow only ([ee3ef11](https://github.com/TimeToAlign/TimeToAlign/commit/ee3ef11f6902420544081344383779c39979045c))
* **timelines:** write the prefixes a FlowMap's provenance uses into the manifest's context ([dcff283](https://github.com/TimeToAlign/TimeToAlign/commit/dcff28370d188e75b0b4dc50874794d1be571b73))
* validate a manifest file as it is written ([fa5b8f7](https://github.com/TimeToAlign/TimeToAlign/commit/fa5b8f7cc47d4a5183e14043705f9605deef6544))
* validate every manifest read and written with the format's validator ([f3f4d60](https://github.com/TimeToAlign/TimeToAlign/commit/f3f4d60dd30d6bb17e841aec7bf90f722faf7d35))
* **verovio:** name the missing element when an MEI file holds a header and no music ([23e8d1d](https://github.com/TimeToAlign/TimeToAlign/commit/23e8d1d7556730c4cd4f619eca76bac4a8f221eb))


### Performance Improvements

* find EventData.coordinate_range with array kernels ([016225c](https://github.com/TimeToAlign/TimeToAlign/commit/016225cb823b857eb25ed13ec983c089b0d33c93))
* **graphical:** draw any number of shapes on one copy of a page image ([852a8e0](https://github.com/TimeToAlign/TimeToAlign/commit/852a8e02cda5ad896b4bde82b7a2a9bec212074e))
* read and write bundle manifests in linear time in the number of roots ([53d15b7](https://github.com/TimeToAlign/TimeToAlign/commit/53d15b78f370c14aec1b071d147b9a8cf9e35b4e))


### Documentation

* add a how-to on graphical boxes, their readers and the Box table ([65a1604](https://github.com/TimeToAlign/TimeToAlign/commit/65a16045929bb58c31da1e20b877419ca953a40a))
* add CHUNK to the glossary and name both sentinels' table strings ([b4b84e8](https://github.com/TimeToAlign/TimeToAlign/commit/b4b84e8e6933b63b672f5932395f8c7d0c695225))
* add reference pages for shapes, Box, Note, ScoreNote, the graphical loaders and time-axis paths ([a753080](https://github.com/TimeToAlign/TimeToAlign/commit/a7530805ebeab7803caabd9cf1f1cb159d14896b))
* **alignment:** state how MatchFileContext.from_dataframes names notes by index label ([a7dc49d](https://github.com/TimeToAlign/TimeToAlign/commit/a7dc49d94a3c054d18c74fe42aa316c77cd0a52e))
* complete exact note ends from the whole-note column in the tabular loaders how-to ([7be114c](https://github.com/TimeToAlign/TimeToAlign/commit/7be114c5361c2d5022e13eb06e56de311792bac3))
* complete the Raises sections of the time-field constructors ([e7adf4a](https://github.com/TimeToAlign/TimeToAlign/commit/e7adf4a41a14a7610b71a8884b09d84dd853f593))
* **core:** order the axis field's members and name its parent in docstrings ([14e1350](https://github.com/TimeToAlign/TimeToAlign/commit/14e13508ce9d9aa54b7e3b271b21a19187e58927))
* **core:** say how Projection.from_row refuses a missing start ([53e6b0b](https://github.com/TimeToAlign/TimeToAlign/commit/53e6b0b2736777d3c4bbf866961f130731284f01))
* correct the retry, walk and rounding-bound helper docstrings ([2ffcda0](https://github.com/TimeToAlign/TimeToAlign/commit/2ffcda050720481c539c83583899be7518447b7c))
* describe flow control elements, findings, FlowMaps over chunks and virtual timelines in the glossary and reference ([b9151f5](https://github.com/TimeToAlign/TimeToAlign/commit/b9151f53d862de1c98047e43f28ce193f2857f1d))
* describe flow maps, beats and IDs as the code behaves ([c6e507e](https://github.com/TimeToAlign/TimeToAlign/commit/c6e507edd2fd27da85066b74591f5f174515f082))
* describe MatchClaims addressed by ID, within keeping the queried timeline and measure-map slices for children ([36363f6](https://github.com/TimeToAlign/TimeToAlign/commit/36363f6283cbc5b3ca0c56f002b38e1b6072a69c))
* describe minted ids as seeded random suffixes that are never renamed ([283f385](https://github.com/TimeToAlign/TimeToAlign/commit/283f38512544e3d0e16822568f34cd82501c5cbf))
* describe several resources of a type and folder resources in the glossary ([d433909](https://github.com/TimeToAlign/TimeToAlign/commit/d43390933eed74ff57f1c1f76b2df72436600a59))
* describe the local build and the per-version publishing of the site ([582b725](https://github.com/TimeToAlign/TimeToAlign/commit/582b72560afcb4848cf80249d13ba2301f23e7e8))
* describe the MatchClaim producers in the core test README plainly ([a4c3a7e](https://github.com/TimeToAlign/TimeToAlign/commit/a4c3a7e888fa439be2d9731b161a395b8896cd46))
* document beliefs, the package form of MatchClaim tables and the within filter ([d1bfc7f](https://github.com/TimeToAlign/TimeToAlign/commit/d1bfc7f87ece9e8a0da38f78b19b6cdee043e66b))
* document playing a timeline with its descendants' flow control ([b7979ff](https://github.com/TimeToAlign/TimeToAlign/commit/b7979fffab17bb806417686293815a83afa12c68))
* document the CHUNK sentinel in the concepts page and the API reference ([7973f77](https://github.com/TimeToAlign/TimeToAlign/commit/7973f7740fa6a2c2edc97787cb25b14f9e4119fd))
* document the pitch, harmony and note scalars of the events module ([e425605](https://github.com/TimeToAlign/TimeToAlign/commit/e42560526fd046e7aa546c6c14aa87f0e4d41d2a))
* document the remaining undocumented helpers outside the core scalars ([3943ce7](https://github.com/TimeToAlign/TimeToAlign/commit/3943ce77b94a942488019176b6563712e69f2d78))
* document the time scalars, their fields and the raw field factories ([eaf7ac5](https://github.com/TimeToAlign/TimeToAlign/commit/eaf7ac5e9341d28801e59d0c2adb4fb9f563b8f2))
* event files and packages, with a how-to writing and sharing tables ([52af2f2](https://github.com/TimeToAlign/TimeToAlign/commit/52af2f207dd6a4e7e59cd34f8d49c2972c70d2ad))
* give every flow convention's findings and their cause in the glossary and the API reference ([a4a3d19](https://github.com/TimeToAlign/TimeToAlign/commit/a4a3d1977450b459a6f4c2d8603cbb2b151711f0))
* guide the registry by its own name and re-execute the notebooks ([6fda12a](https://github.com/TimeToAlign/TimeToAlign/commit/6fda12aafb992659eaeb8f6ee7a2a66779a12f2e))
* how to load Musical Theme Dataset themes aligned to recordings ([24dc6f3](https://github.com/TimeToAlign/TimeToAlign/commit/24dc6f3a5044c3ad9e6a559fd1294b10c9548078))
* how-to guides write and read the registry, its pre-parsed tables and packages ([d1fa3b7](https://github.com/TimeToAlign/TimeToAlign/commit/d1fa3b7a33413c1be4f02357a9074692e3e27461))
* how-to reads MTD themes as one timeline per file, grouped by encoding ([dc20a39](https://github.com/TimeToAlign/TimeToAlign/commit/dc20a39891969ceba58c3be37f1812134612a7d7))
* leave AlignmentBundle.transfer out of the API reference ([dbe3bdd](https://github.com/TimeToAlign/TimeToAlign/commit/dbe3bdd49132e932a95a98058900a739673527b9))
* list the mpm extra in the package docstring ([01e3f81](https://github.com/TimeToAlign/TimeToAlign/commit/01e3f81a72f96c7331c38813b37a680c4efb177e))
* **loader:** give SourceLike a docstring the API reference reads ([305ed97](https://github.com/TimeToAlign/TimeToAlign/commit/305ed976a1d593ef9e94094d09b1997178306cc8))
* **loader:** say how to list a Rekordbox beat grid's uneven bars ([fb10a70](https://github.com/TimeToAlign/TimeToAlign/commit/fb10a7061cfe1d2a7b3a6b8c5de775c26148c53b))
* MTD how-to names the tempo map and bars of its MIDI timelines ([3cf3b67](https://github.com/TimeToAlign/TimeToAlign/commit/3cf3b671624786d4db607c09b0e60c4eff662fe0))
* name allow_expansion and the notated extent in the Locked entry ([8a480cb](https://github.com/TimeToAlign/TimeToAlign/commit/8a480cbba37cb7b36314ade7e6f568a70393f222))
* name the float subtraction in the children README without workflow words ([0f88a17](https://github.com/TimeToAlign/TimeToAlign/commit/0f88a1719c0ea65ffcff8f9a2e828ac9c0d05cd7))
* name the Note table in the package how-tos ([b5046eb](https://github.com/TimeToAlign/TimeToAlign/commit/b5046eb22a053ea7791e1e0c3fc0744cbb990f08))
* Projections in the registry, parses, no table for an ID-only kind ([36e14d6](https://github.com/TimeToAlign/TimeToAlign/commit/36e14d6014ee66a1551f6d560b5187c5d773404e))
* publish the flow-control how-to and re-execute the notebooks that use flow control ([c448e8e](https://github.com/TimeToAlign/TimeToAlign/commit/c448e8e827f12aa5e414743d5acdca21abbea98b))
* re-execute how-tos whose outputs follow manifest-wide Projection IDs ([3884c97](https://github.com/TimeToAlign/TimeToAlign/commit/3884c977593ca511f1d6527617ab31d412f08d27))
* re-execute stale pages; prose follows the narrowed MatchClaim ([d646cf9](https://github.com/TimeToAlign/TimeToAlign/commit/d646cf92381be43089e8d30d4660d9db17e395ce))
* re-execute the alignment bundle tutorial and the MatchClaim package how-to ([e3183bc](https://github.com/TimeToAlign/TimeToAlign/commit/e3183bc960a3d12fcd2bec2dbf590ef14feba76b))
* re-execute the Beethoven multimodal and note-alignment how-tos ([10d2fb4](https://github.com/TimeToAlign/TimeToAlign/commit/10d2fb4a998c5a5d4d66c2a6c205929997db4a8e))
* re-execute the flow-control how-to ([343a234](https://github.com/TimeToAlign/TimeToAlign/commit/343a23470efda37f52845144eee41eebe02fda2c))
* re-execute the graphical boxes how-to and the nesting tutorial ([3337db2](https://github.com/TimeToAlign/TimeToAlign/commit/3337db21974e7583506b133e213ca77119cef952))
* re-execute the loading how-tos for respelled pitch names and null score fields ([a0ba65a](https://github.com/TimeToAlign/TimeToAlign/commit/a0ba65a75d49ca1051bb0388d9b9db4bec96f2d2))
* re-execute the nesting and alignment-bundle tutorials and the MatchClaim package how-to ([e0929c6](https://github.com/TimeToAlign/TimeToAlign/commit/e0929c6494c11966acae04c10ad8759a53ac14dd))
* re-execute the notebook pages that show the NOMATCH and CHUNK sentinels ([8c59165](https://github.com/TimeToAlign/TimeToAlign/commit/8c5916581c446386788ed5419cc41f675e34b32a))
* re-execute the tabular loaders how-to ([1faba73](https://github.com/TimeToAlign/TimeToAlign/commit/1faba7324dd2525859b2761c99042e21039157a3))
* re-run the piano-roll how-to with distinct hole event IDs ([3a3ff58](https://github.com/TimeToAlign/TimeToAlign/commit/3a3ff58ace617fe4d60a9a4a56fd1c0b69cc9112))
* re-sync the package, box and MTD how-tos after the manifest file rename ([f988031](https://github.com/TimeToAlign/TimeToAlign/commit/f98803163aafb34306f8e73767bea90b8eaede22))
* read MTD timelines and claims from the bundle in the MTD how-to ([23593c8](https://github.com/TimeToAlign/TimeToAlign/commit/23593c833f580fa46eea44559fe9ffd518b23ab5))
* read the Boxes of an accolade from its Box rows in the graphical boxes how-to ([6e5698a](https://github.com/TimeToAlign/TimeToAlign/commit/6e5698a563a5487712b72a76a3736b09de7accdc))
* rebuild reference pages for MatchClaim IDs, measure slices and Projection IDs; add MeasureMap ([25c7959](https://github.com/TimeToAlign/TimeToAlign/commit/25c7959940894e17b6e586bb800b0a25f4e9e1a8))
* rebuild reference pages for one-sided ends, narrowing refusals and event holders ([e896a29](https://github.com/TimeToAlign/TimeToAlign/commit/e896a29bb5622e7b60ca44b6c4de5e5e9a97c391))
* rebuild the MatchClaim and MatchClaimField reference pages for names on CHUNK sides ([3357693](https://github.com/TimeToAlign/TimeToAlign/commit/33576930b0d8560881f82f2acd936dc550f76d65))
* rebuild the reference pages whose docstrings changed ([72e909c](https://github.com/TimeToAlign/TimeToAlign/commit/72e909cdb631270e103184e1171744c9885963ab))
* refer to minted IDs through objects, roles and whole-ID patterns in the notebooks ([5a06620](https://github.com/TimeToAlign/TimeToAlign/commit/5a06620fecf8e669a7846716dceec32036b629e9))
* reference pages for Parse and the registry's parses ([7380d6c](https://github.com/TimeToAlign/TimeToAlign/commit/7380d6ce1f8490f99c9aef5316ca7a9aaa3e9b2c))
* reference pages for resolve, the event registry and extractions ([e66ff64](https://github.com/TimeToAlign/TimeToAlign/commit/e66ff64f4b27baab42bf58ea3094b4abb8679319))
* regenerate the API reference for random IDs and drop ScopedId ([1b61529](https://github.com/TimeToAlign/TimeToAlign/commit/1b61529bb4fbfff0bf195721e59f02d27adddbec))
* regenerate the floating-measure and Ms3Loader reference pages ([cc0a759](https://github.com/TimeToAlign/TimeToAlign/commit/cc0a759a2e4dd6d02e4b634e9618ac70e99686cd))
* relate spelled-pitch rows to the line of fifths and correct the bass_note fallback ([1255814](https://github.com/TimeToAlign/TimeToAlign/commit/1255814c87cc2e5f1b15a26cfdf58f2880f27220))
* republish four how-to pages whose output had drifted from the library ([704d63a](https://github.com/TimeToAlign/TimeToAlign/commit/704d63af7ef475b042d107b2dcf4be2b9a6dc9cd))
* say a CHUNK side of create_match_claims keeps its name ([eb5b822](https://github.com/TimeToAlign/TimeToAlign/commit/eb5b822f5a3865edc4b9d1ba1c392d8afba6f5f4))
* say in CONTRIBUTING that the validator checks every manifest ([52a69eb](https://github.com/TimeToAlign/TimeToAlign/commit/52a69eb96810de8a8405e9d439befa1f04b54cda))
* show Listen Here! in seconds and the named Beethoven section boundaries ([0a379ed](https://github.com/TimeToAlign/TimeToAlign/commit/0a379ed185538cc1a4a2a36a1ca530a4cbee9267))
* show loading from a ZIP archive member in the loading how-to ([6a8746b](https://github.com/TimeToAlign/TimeToAlign/commit/6a8746b4a5a9efe9159379c1d535d2117d81305f))
* show the conversion maps a score states, not an assumed ticks map ([fbe08b3](https://github.com/TimeToAlign/TimeToAlign/commit/fbe08b3789f4c58df0d722e5fb6608fab91a62b9))
* show the findings of each convention in the flow-control how-to ([9c50020](https://github.com/TimeToAlign/TimeToAlign/commit/9c50020df5a7ec75d18798993e167faed0c1089f))
* state that the format's validator checks every manifest read and written ([024cbc7](https://github.com/TimeToAlign/TimeToAlign/commit/024cbc728f06414c5804f76e0f4de984a264913a))
* state what a locked timeline refuses, the ScoreNote, and is_segment_line as a property ([ea8839f](https://github.com/TimeToAlign/TimeToAlign/commit/ea8839fa51f2bcb6b690e70005e4864790efd18a))
* teach flow control elements, played flows and FlowMaps over chunks in the notebooks ([a42a691](https://github.com/TimeToAlign/TimeToAlign/commit/a42a6917195662d23bcb527b3b8049454ea96a46))
* teach page images along their time path ([4445e58](https://github.com/TimeToAlign/TimeToAlign/commit/4445e581ff6f6f1df50c2438e21eab30509a8bc9))
* teach the NOMATCH and CHUNK sentinels in the notebooks ([a490be8](https://github.com/TimeToAlign/TimeToAlign/commit/a490be836ce7bd2280b13901c567c618a4d95154))
* **tests:** state the nine remaining approx sites of the map property tests ([6cd9828](https://github.com/TimeToAlign/TimeToAlign/commit/6cd98283157e7a51062e6fecb536ca0791b7a3b7))
* **timelines:** annotate the Projections table reader with what it reads ([75e7adf](https://github.com/TimeToAlign/TimeToAlign/commit/75e7adf4700a3c4f60f65859dbd4e92fb6834e7d))
* word the Rekordbox identity test without the word lanes ([0117c0f](https://github.com/TimeToAlign/TimeToAlign/commit/0117c0f16f025a379e4ed60f7323a2cff2a168e6))


### Code Refactoring

* adjacent records with one measure number form one floating measure ([efabc85](https://github.com/TimeToAlign/TimeToAlign/commit/efabc85b82a5d193ed2273376418e9e523cf4351))
* **core:** rename the column enum members that shadow str and Enum attributes ([56e9a12](https://github.com/TimeToAlign/TimeToAlign/commit/56e9a126bc9e30b1ed27f05c52989ab18f468ae4))
* **core:** trim FlowManifestation to the manifestations the element classes admit ([bb793d6](https://github.com/TimeToAlign/TimeToAlign/commit/bb793d6baa70fecb91713dcfd814b3abc7c7f1d7))

## [1.2.0](https://github.com/TimeToAlign/timetoalign/compare/v1.1.0...v1.2.0) (2026-08-08)


### Features

* **flow:** add FlowMap.from_dict and fill placed gaps with empty segments ([e8c2fce](https://github.com/TimeToAlign/timetoalign/commit/e8c2fce1292fbf0b8a2390a5f5df82d622b94ba3))
* **flow:** place FlowMap spans at given coordinates instead of appending ([1fcd274](https://github.com/TimeToAlign/timetoalign/commit/1fcd274a6fd62714debf77f50b3865aa899d034f))
* **timelines:** guard the default-flow traversal and report traversal diagnostics ([c5ff3a2](https://github.com/TimeToAlign/timetoalign/commit/c5ff3a28365f1ad0844d78a37be780e5b6e5dde9))


### Bug Fixes

* **core:** accept integer-valued float ratio members at the rational wire boundary ([bfe8d77](https://github.com/TimeToAlign/timetoalign/commit/bfe8d77ca59a6011310b5af48e8d9b12bdf8e3ac))
* **testdata:** extract corpora atomically and verify completeness before trusting the cache ([50cadab](https://github.com/TimeToAlign/timetoalign/commit/50cadabfd0594a57b7f6ee8906bd9a1c61555fa7))
* **tests:** point MIDI integration specimens at the midi corpus ([0c23fb3](https://github.com/TimeToAlign/timetoalign/commit/0c23fb3b046c529b34e2130fa74f1c5ec35b444b))


### Performance Improvements

* **midi:** add a mido-backed parser option to ScoreMidiLoader ([bffed90](https://github.com/TimeToAlign/timetoalign/commit/bffed90e029d192a6cd512f550775e66cf50a6ea))


### Documentation

* **core:** document exact binary-expansion semantics for float-to-Fraction conversion ([991032c](https://github.com/TimeToAlign/timetoalign/commit/991032c3eee8f373d4bdf4b0f22089383ff4ad03))
* **howto:** demonstrate traversal diagnostics on the Choros flows notebook ([3465a00](https://github.com/TimeToAlign/timetoalign/commit/3465a006774479ebb09e35488a20413f7e28f58f))
* show placing flows and inverted cuts in the FlowMap how-to ([4c274bf](https://github.com/TimeToAlign/timetoalign/commit/4c274bf13f724e99647869e478f90732dbd5cab0))

## [1.1.0](https://github.com/TimeToAlign/timetoalign/compare/v1.0.1...v1.1.0) (2026-07-30)


### Features

* **display:** recurse timeline diagrams with a depth control ([e1eca24](https://github.com/TimeToAlign/timetoalign/commit/e1eca2483757ee13efc6eca0e38c5b60d82f7361))
* **timelines:** harmonize child creation across concrete classes ([7db621c](https://github.com/TimeToAlign/timetoalign/commit/7db621c610cac63c73baf0a18b19d784fe130fe3))
* **timelines:** parameterize SegmentLine over its segment class ([d9d9d17](https://github.com/TimeToAlign/timetoalign/commit/d9d9d17f1e5de5f375604135f86ce1b7f84b9f03))
* **timelines:** resolve descendant coordinates through get_coordinate ([407c738](https://github.com/TimeToAlign/timetoalign/commit/407c7382c62bca9bbaee81c098d42deecc3b6fba))


### Bug Fixes

* **timelines:** reject appends to length-locked timelines ([b4ba551](https://github.com/TimeToAlign/timetoalign/commit/b4ba55141bac928ac82223faefb9b602124d9ad1))


### Documentation

* construct concrete timeline types in notebooks ([30eac1b](https://github.com/TimeToAlign/timetoalign/commit/30eac1b54193ad4e5f6b524fecdd0bc89bcb7699))
* demonstrate the hierarchical child API across notebooks and reference ([a070a83](https://github.com/TimeToAlign/timetoalign/commit/a070a8364ef03ec18e4229a9275fb62a6699ef80))
* register SegmentLine and inherited timeline methods in the reference ([0533ab3](https://github.com/TimeToAlign/timetoalign/commit/0533ab3fd7cd7a9853a550948ea2665dfacd6518))

## [1.0.1](https://github.com/TimeToAlign/timetoalign/compare/v1.0.0...v1.0.1) (2026-07-24)


### Documentation

* full site render with re-executed notebooks ([41d0c2e](https://github.com/TimeToAlign/timetoalign/commit/41d0c2e04076ed4c160668d637a8a018881c5a95))
* improves project links ([949a8c7](https://github.com/TimeToAlign/timetoalign/commit/949a8c7df26b0b570d126eec320dfd8fa76a5eec))
* open the landing page with the article's lead figure ([699046e](https://github.com/TimeToAlign/timetoalign/commit/699046ee90b65811c43b7a9210e7094e06fd7a89))
* regenerate reference and site after the docstring restatement ([56c4c6c](https://github.com/TimeToAlign/timetoalign/commit/56c4c6c3ede366099e305b475dbcf4fa4cc61746))
* regenerate the API reference with Google-style docstring parsing ([8bc6d6d](https://github.com/TimeToAlign/timetoalign/commit/8bc6d6d5eaa5a957a4121d72859192d5defd4000))
* render Google-style docstrings with a Yields-capable renderer ([62500b0](https://github.com/TimeToAlign/timetoalign/commit/62500b07ab2a2638373b5b136a890e95ac631dcb))
* state model definitions in the library's own voice ([75a12cf](https://github.com/TimeToAlign/timetoalign/commit/75a12cf3d107bf92550781859a6e71b24a598db2))


### Miscellaneous Chores

* release 1.0.1 ([50af5f8](https://github.com/TimeToAlign/timetoalign/commit/50af5f8e892cee5bf3988701794fc9f12cdafe3d))

## [0.2.0](https://github.com/TimeToAlign/timetoalign/compare/v0.1.0...v0.2.0) (2026-05-13)


### Features

* fetch test data on demand via pooch + timetoalign.testdata ([a334d43](https://github.com/TimeToAlign/timetoalign/commit/a334d4330e3bcfa0463fd40bf8599f15980217f1))


### Bug Fixes

* **bundle:** resolve internal IDs and interval components in matchstamp lookup ([f1b0546](https://github.com/TimeToAlign/timetoalign/commit/f1b05467233bdb67ba5e686f85cd54767d175fd7))


### Documentation

* adds how01_flow_control.py ([95290d7](https://github.com/TimeToAlign/timetoalign/commit/95290d74420655e0ea46aa6e1de7780b01c960cd))

## 0.1.0 (2026-05-13)


### Features

* uniform units display across all timestamp APIs ([220c6a5](https://github.com/TimeToAlign/timetoalign/commit/220c6a5cc0aa49f0ee7df757da755189a99c9e7e))


### Bug Fixes

* populate start/end/duration temporal fields in NoteEventData (fixes [#9](https://github.com/TimeToAlign/timetoalign/issues/9), fixes [#10](https://github.com/TimeToAlign/timetoalign/issues/10)) ([00c3734](https://github.com/TimeToAlign/timetoalign/commit/00c3734f492b10dbc12bef76489743b2edd46194))
* populate temporal coordinates in MeasureData, ControlEventData, and AnnotationEventData (fixes [#7](https://github.com/TimeToAlign/timetoalign/issues/7)) ([c6b2eb3](https://github.com/TimeToAlign/timetoalign/commit/c6b2eb36cf2499d32e35ddd835d8da6b13f71540))
* report consistent unit/number_type in ScoreLoader metadata (fixes [#8](https://github.com/TimeToAlign/timetoalign/issues/8)) ([786b65c](https://github.com/TimeToAlign/timetoalign/commit/786b65c790a4ae14530c86cdd4d2c0f3b3e31e9c))
* use quarter_map instead of beat_map in PartituraLoader (fixes [#12](https://github.com/TimeToAlign/timetoalign/issues/12)) ([dcf23c0](https://github.com/TimeToAlign/timetoalign/commit/dcf23c055f5a51774c82fb1dd730f0bba2c5ecd4))


### Documentation

* add redirect index.html and disable Jekyll ([bc71e64](https://github.com/TimeToAlign/timetoalign/commit/bc71e64d7969bc5b8f420c364b1d68d4cd59d2f0))
* add redirect index.html and disable Jekyll ([cb80dd5](https://github.com/TimeToAlign/timetoalign/commit/cb80dd5f1798f7f763eacffd4d2f951b1653234c))
* broaden 'Voice and Staff Information' beyond piano music (fixes [#16](https://github.com/TimeToAlign/timetoalign/issues/16)) ([b918686](https://github.com/TimeToAlign/timetoalign/commit/b918686d6782cd1aadbf69ae93ccc2cf8bd1d88f))
* clarify interpolation methods with distinct y-values and 'next' strategy (fixes [#19](https://github.com/TimeToAlign/timetoalign/issues/19)) ([145a7af](https://github.com/TimeToAlign/timetoalign/commit/145a7afca4638533adde1d9ba04b35a2eff28e93))
* correct path ([7eeecb9](https://github.com/TimeToAlign/timetoalign/commit/7eeecb92cc9c8d4944b187774d98d6e29265544d))
* document large file timeout handling in test README ([fd545a2](https://github.com/TimeToAlign/timetoalign/commit/fd545a2290808950749b96536a4c4efd7891c50b))
* replace deprecated 'bundle' wording with 'store' (fixes [#17](https://github.com/TimeToAlign/timetoalign/issues/17)) ([7ed2ffc](https://github.com/TimeToAlign/timetoalign/commit/7ed2ffc36643254da941f5f4dca53bc42fc029ac))
* replace unreliable septuplet float example with 0.1*10 (fixes [#3](https://github.com/TimeToAlign/timetoalign/issues/3)) ([13f8567](https://github.com/TimeToAlign/timetoalign/commit/13f8567c3d47e4e1b4ae7c33a79abe89e50030fa))
* split notebooks into tuto-notebooks and howto-notebooks (Diátaxis) ([b1dc9e6](https://github.com/TimeToAlign/timetoalign/commit/b1dc9e6d3e58cd3cc4367a09b013ec362c4d300e))
* update test READMEs to reflect cleanup changes ([c481aca](https://github.com/TimeToAlign/timetoalign/commit/c481aca47e55617095bb2741c6db7c1d94b7996f))
* updates autodocs ([592fbdc](https://github.com/TimeToAlign/timetoalign/commit/592fbdc8d094b10f5982f4ae44303bc02da4f406))
* updates homepage ([4402640](https://github.com/TimeToAlign/timetoalign/commit/44026408749517d527008b363d6ca7c881e3d937))
* updates tutorial page ([3d5aa5f](https://github.com/TimeToAlign/timetoalign/commit/3d5aa5faab5cdf04afea0fc48169e01f5a660601))
* writes reduced tutorial notebooks, factoring out older content to how-to notebooks ([faa3fbf](https://github.com/TimeToAlign/timetoalign/commit/faa3fbf7f797ae73bcc573785273f82acea9b1f7))
