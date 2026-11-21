# Changelog

## 1.0.1 (2026-11-21)


### ⚠ BREAKING CHANGES

* **loader:** AudioLoader.create_timeline() returns a DiscretePhysicalTimeline in samples with a SamplesToSeconds map; call AudioLoader.create_continuous_timeline() for the timeline in seconds.
* manifests declaring TimeToShare 0.1.0 are refused.
* **maps:** a TableMap with no target unit and exact anchors now returns a Fraction where it returned a float.
* accept any id without control characters, unique across a manifest
* manifests carry the marker "timetoshare": "0.1.0"; documents with the marker "timeskeleton" are refused as not a TimeToShare manifest. TIMESKELETON_VERSION is renamed TIMETOSHARE_VERSION.
* manifests use the revised conversion-map, measure-map and flow-control forms; former map type names, the timeline keys measure_map and metric_hierarchy, and the breaks/jumps/markers object are no longer read. Break and Jump lose condition, Break loses repeat_count, Marker requires control_type, MeasureMap.to_dict/from_dict use the measure_maps entry.
* TimeUnit.whole_note and the unit string "whole_note" are replaced by TimeUnit.whole_notes and "whole_notes".
* TimelineGroup.add_timeline no longer takes start, end or allow_extension (a partial member is a child); the group's lock, n_timestamps, timestamps, get_timestamp_at_index and GroupTimestamp are removed, and TimeStamp.row_index is removed; AlignmentBundle.add_timeline and add_group no longer take uid or uid_map, and as_group names the group while its id is minted; default group and bundle ids are group_1 and bundle_1; loader-made child timelines no longer use their label as id; Timeline.to_dict writes is_segment_line and plain class names.
* timeline ids follow `{prefix}_{N}` (`clt_1`, `score:clt_1`); regions are looked up by id; `Jump.target_name` is now `Jump.target`; flow-control markers are `Marker` objects.
* **timelines:** measure ids are row IDs ("1", not "m1"); rows must use the JSON number forms; add_measure_map refuses non-quarters timelines; Rekordbox track timelines carry no measure map.
* **timelines:** Timeline.to_dict()["measure_map"] is a row list, never an object with "measures".
* **core:** rational_to_dict / rational_from_dict / is_rational_dict no longer exist and to_dict() output changed shape for every number and coordinate.
* **core:** rational_to_wire, wire_to_rational, is_rational_wire, coordinate_wire_entry and coordinate_from_wire_entry no longer exist, and to_dict() output for integer coordinates now carries the ratio.
* **timelines:** MetricHierarchy.from_sections is removed; MetricHierarchy is constructed with measure_map= and its wire keys are authored_readings, authored_forest and tempo_maps, restored with from_dict(data, measure_map=); MetricNode(level, span) replaces the optional level and proportion; TempoEntry.at is required; the tempo map kind "indication" is renamed "tempo_indication".
* **timelines:** BeatPolicy has no bpm field; a rate is a Tempo inside a TempoMap. MetricHierarchy.from_sections takes Tempo values and is the only constructor — from_beat_policies, create_sections and the named policy registry are gone, as are MetricHierarchyComponent and Address.skeleton_id. policy_for_metro is gone; read a grid meter with BeatPolicy.from_time_signature(metro).as_divisions(). BeatPolicy now refuses a division and a beat_size that disagree, records the signature it was read from, and serializes its beat size through the shared duration wire. Beat.level admits negative levels.
* **timelines:** timetoalign.alignment no longer exports MeasureMap, SectionHierarchy, MetricHierarchy or MetricHierarchyComponent; import them from timetoalign.timelines (the root facade is unchanged). A measures store whose measures cannot form a map now raises at create_timeline instead of yielding a timeline without structure.
* **timelines:** the metrical-timeline BeatGrid (quarters axis, meter-map family, from_tempo/materialize_*/wire format) is replaced; ContinuousPhysicalTimeline.create_metrical_grid, create_metrical_region and MetricalResult are removed. Build grids with BeatGrid.from_tempo or take them from a loader-built TimeSkeleton via create_beatgrid().
* **alignment:** Measure's field set now follows the Measure Map standard (id/count/qstamp/number/name/time_signature/nominal_length/actual_length/ start_repeat/end_repeat/next, plus volta); TimeUnit.beats is removed; loading an ms3 measures table no longer produces measure events.
* **retrieval:** `to_dataframe()` is removed from Timeline, TimelineGroup and AlignmentBundle in favour of `format="table"|"dataframe"` on the table getters; `get_timestamp_of`/`get_timestamps_of` are renamed to `_for`; the `coordinates=` keyword on every table getter is renamed to `at`, which now also accepts event keys; `as_fractions` is deleted because it let a caller override an axis's declared number type; `get_matchstamps` keeps only its claims path; and `Timeline.get_timestamp_table` no longer emits an `axis` column, which was byte-identical to the receiver's own column.
* **retrieval:** raw-float stamp accessors, coordinate pair tuples, and the untyped WarpMap call surface are removed; retrieval defaults return IdCoordinate. Full suite: 4589 passed, 24 skipped.
* **core:** TimelineGroup.get_timestamp_at and AlignmentBundle.get_matchstamp_at now express the axis in the addressed timeline's declared number type (a float query on a rational axis yields an exact fraction; an exact query on a float axis yields a float), and fractional queries on discrete axes round before interpolation.
* **core:** query and derived coordinates on rational axes now surface as exact fractions rather than floats; get_timestamp(9.5) on a quarters timeline yields Fraction(19, 2).
* **core:** RationalField, CoordinateParser and the public DISCRETE_UNITS constant are removed; storage/parsing.py and timelines/engines/coordinate_ops.py are gone. NumberType defaults are now derived from the unit rather than per-timeline/per-loader settings. to_int() defaults to "round" (was "truncate"). Stamp.axis carries the canonical numeric value instead of always-float. Scalar constructors coerce to the unit's representation, so Coordinate(1.5, quarters) now holds Fraction(3, 2); a non-integral exact fraction in a discrete unit raises instead of silently rounding.
* **alignment:** `AlignmentBundle.add_timeline(aligned_to=...)` is now `AlignmentBundle.add_timeline(grouped_with=...)`. No alias is kept; existing call sites must be renamed.

### Features

* accept any id without control characters, unique across a manifest ([1d61e02](https://github.com/TimeToAlign/TimeToAlign/commit/1d61e026f6dcda78e062db1cacff04656c666a2d))
* accept coordinate collections in bundle batch matchstamp queries ([2856f9d](https://github.com/TimeToAlign/TimeToAlign/commit/2856f9d4893804f1db7bc80bf5488cec9e03f55b))
* add EventData.head() for pandas-style row previews ([c34bc7f](https://github.com/TimeToAlign/TimeToAlign/commit/c34bc7fee8132379331181ef52a3c2afbe32319b))
* **alignment:** add MatchClaimField columnar claim store ([f9dbf60](https://github.com/TimeToAlign/TimeToAlign/commit/f9dbf60c634ac2f46e28e91444ebb37da6051a94))
* **alignment:** affordance "Try" footer on timestamp and claim displays ([2e9cf57](https://github.com/TimeToAlign/TimeToAlign/commit/2e9cf57d346a162c025d9bfc07bf8c1eea575196))
* **alignment:** affordance "Try" footer on timestamp and claim displays ([426c03b](https://github.com/TimeToAlign/TimeToAlign/commit/426c03b28a432dca9183fe945854925f0afee882))
* **alignment:** allow NOMATCH claims in create_match_claims ([76924cf](https://github.com/TimeToAlign/TimeToAlign/commit/76924cf5f1def40928b8cb2b1362ffb2661d3888))
* **alignment:** conform GroupTimestamp to the Stamp contract ([9d4d76a](https://github.com/TimeToAlign/TimeToAlign/commit/9d4d76adca76a42b030cac5051dd2a163b9f7135))
* **alignment:** derive a ClaimType for every MatchClaim ([532dc50](https://github.com/TimeToAlign/TimeToAlign/commit/532dc5029e6da053bd0b7fa2617acfaaeb4a40b6))
* **alignment:** give TimeSkeleton structural equality and a materialization round trip ([2fc63a8](https://github.com/TimeToAlign/TimeToAlign/commit/2fc63a8b23d491de16a7cea00c12204adc04014b))
* **alignment:** introduce the TimeSkeleton shared temporal structure ([000d2ed](https://github.com/TimeToAlign/TimeToAlign/commit/000d2ed91ec822035e7ba4e57355556600b37aad))
* **alignment:** MatchClaims name events on their timelines' axes, with provenance and beliefs ([c1d4034](https://github.com/TimeToAlign/TimeToAlign/commit/c1d4034fefc81ca1fba2f29d837a2f158a27924f))
* **alignment:** resolve event ids in create_match_claims pairs ([9fd962b](https://github.com/TimeToAlign/TimeToAlign/commit/9fd962b8c6015935c34cd9e54ab2892fc6e1a2e7))
* **alignment:** resolve interval claims in matchstamp retrieval ([318a250](https://github.com/TimeToAlign/TimeToAlign/commit/318a2502abc8ec95a30f794213f65c0183183f96))
* **alignment:** unify MatchStamp with the Stamp contract and single cross-group resolution API ([90dfad4](https://github.com/TimeToAlign/TimeToAlign/commit/90dfad4e8d44fc5e4cc944bd51dd82d07ce07568))
* **alignment:** unit-aware coordinate inputs and canonical claim filters ([4d51481](https://github.com/TimeToAlign/TimeToAlign/commit/4d51481eeb85b495b3f3370752cee29b56de45ca))
* **alignment:** unit-bearing Coordinate anchors through claims and columnar storage ([93e734a](https://github.com/TimeToAlign/TimeToAlign/commit/93e734afec70ef1503b06a4271d908e3f096c43b))
* **alignment:** write MatchClaimField tables with one column per claim property ([a0a0c5b](https://github.com/TimeToAlign/TimeToAlign/commit/a0a0c5b7a11ed002205628f277c45bc358bb6e7b))
* carry TimeToShare links and schemes on bundles, timelines and regions ([ff43780](https://github.com/TimeToAlign/TimeToAlign/commit/ff43780af2506630f7ef526ad24cffc220e2ba8b))
* check a manifest's statements when reading and writing it ([40161e4](https://github.com/TimeToAlign/TimeToAlign/commit/40161e42ba59b2f558c2c69240aae586679d2ff7))
* **core,loader,timelines,alignment:** type-based IDs, curated facade, timeline registry, WarpMap family API ([827a34f](https://github.com/TimeToAlign/TimeToAlign/commit/827a34f0f638bcfd51719409397fd34fa5ccdb06))
* **core:** adopt the TimeSkeleton JSON number forms (BOPP Fraction pairs, typed coordinates) ([cc7c4e5](https://github.com/TimeToAlign/TimeToAlign/commit/cc7c4e55d443627991369ed5e7c59b875375ddf4))
* **core:** canonical BoundingBox scalar and field ([db5be12](https://github.com/TimeToAlign/TimeToAlign/commit/db5be1214c02802d2f918767f49cfdbed0902de4))
* **core:** derive number representation from the unit, end to end ([ddf9832](https://github.com/TimeToAlign/TimeToAlign/commit/ddf9832002a2b8d92eb468c7d393d48cc5fbe463))
* **core:** introduce Stamp base class unifying the timestamp family ([3c1ee8b](https://github.com/TimeToAlign/TimeToAlign/commit/3c1ee8b982630f341d344e16fa75cba4961510c8))
* **core:** one Arrow cell for coordinates with unit and number type ([ec51690](https://github.com/TimeToAlign/TimeToAlign/commit/ec51690f655858854e00e5ceea5406d6e16c2c22))
* **core:** one Arrow cell for coordinates with unit and number type ([4164dad](https://github.com/TimeToAlign/TimeToAlign/commit/4164dad7e6273ae2ec0588e7a6cd50c317e70fbd))
* **core:** one coordinate-resolution canon with a strict unit policy ([ffae3d9](https://github.com/TimeToAlign/TimeToAlign/commit/ffae3d9b7f21851cb9f023bb9bcf53ab0e8c7532))
* **core:** write integers as exact ratios in the rational dict and retire the "wire" vocabulary ([1bacf50](https://github.com/TimeToAlign/TimeToAlign/commit/1bacf504f5ca73fa1c05771f55475bcfd605a456))
* create_calibrated_child builds a child timeline and its conversion map from calibration pairs ([6997129](https://github.com/TimeToAlign/TimeToAlign/commit/6997129b53abc0ab31f021c23fc09a3c222d52db))
* css_pixels and user_units graphical units ([6a94999](https://github.com/TimeToAlign/TimeToAlign/commit/6a949993bc8fb69ab8f60c348883baab509e9001))
* customizable atomic-section naming with volta-suffix labels ([db2ea7e](https://github.com/TimeToAlign/TimeToAlign/commit/db2ea7e43d810a36e567413891a6acec79c50760))
* **display:** recurse timeline diagrams with a depth control ([e1eca24](https://github.com/TimeToAlign/TimeToAlign/commit/e1eca2483757ee13efc6eca0e38c5b60d82f7361))
* **display:** shared affordance-card _repr_html_ for loaders, EventData, fields ([115f6d7](https://github.com/TimeToAlign/TimeToAlign/commit/115f6d72a843b59a409b47149bea2444cb17a864))
* **display:** shared affordance-card _repr_html_ for loaders, EventData, fields ([7f73a94](https://github.com/TimeToAlign/TimeToAlign/commit/7f73a94ec3c059679274b0239c9b193272b315ed))
* **events:** add the Projection event type and unique event IDs in add_events ([0a918a1](https://github.com/TimeToAlign/TimeToAlign/commit/0a918a1e3ebc2eb16f996796d1f65ac443e32422))
* **events:** Projections stored in coordinate cells; one minter that never reissues an ID ([f35ebd1](https://github.com/TimeToAlign/TimeToAlign/commit/f35ebd127cdfee55a7e4b0c0abf6755195052124))
* every manifest written carries its inline JSON-LD context ([7b28f2c](https://github.com/TimeToAlign/TimeToAlign/commit/7b28f2cab678df3b5f0636672563948dedb98481))
* **flow:** add FlowMap.from_dict and fill placed gaps with empty segments ([e8c2fce](https://github.com/TimeToAlign/TimeToAlign/commit/e8c2fce1292fbf0b8a2390a5f5df82d622b94ba3))
* **flow:** add structural-invariant diagnostics to FlowController ([c8b609d](https://github.com/TimeToAlign/TimeToAlign/commit/c8b609d197c1fbed71b69f587221877a638bae73))
* **flow:** build FlowMap from any interval-like input ([7fc3c4a](https://github.com/TimeToAlign/TimeToAlign/commit/7fc3c4a0a7cfce1afe63f0fc1a3ccf781acd7805))
* **flow:** place FlowMap spans at given coordinates instead of appending ([1fcd274](https://github.com/TimeToAlign/TimeToAlign/commit/1fcd274a6fd62714debf77f50b3865aa899d034f))
* give every manifest element an id from one generator ([f0a58ce](https://github.com/TimeToAlign/TimeToAlign/commit/f0a58ce1e4f546aefe042cb6319b485ee1a3b006))
* **ieee1599:** nest engraved editions as pages of accolades ([2078b9c](https://github.com/TimeToAlign/TimeToAlign/commit/2078b9ce83dc2c00395cb8c914ad776fdf737375))
* import an MEI header as statements ([4d2dfbf](https://github.com/TimeToAlign/TimeToAlign/commit/4d2dfbf4380d594c348490b6177d23013673bd8f))
* import MeLODy's Turtle as statements ([41cabf4](https://github.com/TimeToAlign/TimeToAlign/commit/41cabf4041f49d786955ac002e991e9a117529a2))
* keep timeline band orders in graphical.order and reserve band ids ([4aacfe8](https://github.com/TimeToAlign/TimeToAlign/commit/4aacfe8e732e98266cab13fa9155999d33da8942))
* keep what links name valid in every manifest, and let links change ([ee49ffe](https://github.com/TimeToAlign/TimeToAlign/commit/ee49ffec3a8b744a4d3009be803918dab5fffd79))
* let a timeline belong to several groups; one id prefix per flow-control type ([636fd9c](https://github.com/TimeToAlign/TimeToAlign/commit/636fd9cef510fa42685db28125bb2749c40860af))
* **loader:** add ListenHereLoader for audio-to-audio alignment ([decc57c](https://github.com/TimeToAlign/TimeToAlign/commit/decc57cdf44478f6d3d575e572e875a2579bcf80))
* **loader:** add RekordboxLoader with beat-grid measure derivation ([3d2a944](https://github.com/TimeToAlign/TimeToAlign/commit/3d2a9446b81aa2d57c3c4d71b8acc2a27455d002))
* **loader:** afford EnharmonicPitch pitch view across every loader ([7123ea0](https://github.com/TimeToAlign/TimeToAlign/commit/7123ea0b38f704975a369d0900c9b3f4bf641942))
* **loader:** alignment loaders state MatchClaims between events ([cc77eb8](https://github.com/TimeToAlign/TimeToAlign/commit/cc77eb8b0d5ccad22bd48e2124e8d1c7c5174073))
* **loader:** derive Rekordbox timeline identity from file names and expose sample axes ([9ffd4f6](https://github.com/TimeToAlign/TimeToAlign/commit/9ffd4f6ca9a1b422b3a6fa345c1944db1d3a3b24))
* **loader:** IEEE 1599 alignment loader ([b16389f](https://github.com/TimeToAlign/TimeToAlign/commit/b16389f0cfe587b20abf20596ebfbb173f0c9ded))
* **loader:** reach ListenHere claims via get_field and query them columnar ([f637004](https://github.com/TimeToAlign/TimeToAlign/commit/f637004163a3759afa1f322555c65fbd801de6ae))
* **loader:** reach ListenHere claims via get_field and query them columnar ([1ff11e3](https://github.com/TimeToAlign/TimeToAlign/commit/1ff11e35b098e6ba443ba6adb6bd9a0cee30ab23))
* **loader:** resolve IEEE 1599 structural analyses into spine external references ([46ddb4a](https://github.com/TimeToAlign/TimeToAlign/commit/46ddb4ac48e02e4ad3bd6a757ca01b89bed43cea))
* **loader:** uid as the one timeline-identifier name; live id_pattern ([c494328](https://github.com/TimeToAlign/TimeToAlign/commit/c4943280c21da9645910a503856acf1f3040c795))
* lock a root timeline after construction and keep group and bundle styling ([996de8c](https://github.com/TimeToAlign/TimeToAlign/commit/996de8c23f7a7684bfa474fcbdaa8f8e6c2e1644))
* **manifest:** carry named timestamps in a manifest's graphical object ([33ee9ef](https://github.com/TimeToAlign/TimeToAlign/commit/33ee9efd07d8f2e3915d1bc8fff7e21cb335f094))
* **manifest:** keep event rows and external references on a file round trip ([536dd82](https://github.com/TimeToAlign/TimeToAlign/commit/536dd82f85083c24262d56f413f0e8c5326efe1b))
* manifests write every ID as #id; graphical rules ([dd2d829](https://github.com/TimeToAlign/TimeToAlign/commit/dd2d82915a5df8505d306c2f7d5b7b582f3a5da7))
* **manifest:** write a named timestamp's display in canonical order ([a3e0d25](https://github.com/TimeToAlign/TimeToAlign/commit/a3e0d256f0c13ba847a4baa712050104cea02ab9))
* **maps:** add MetricMap.quarters_at for direct MC -&gt; downbeat lookup ([5ac3792](https://github.com/TimeToAlign/TimeToAlign/commit/5ac37928e236456031f124c970af845a7b34c958))
* MapsLoader keeps every row trompa-align writes, and builds the bundle ([7b52118](https://github.com/TimeToAlign/TimeToAlign/commit/7b52118605c26524b9ced65b9f8c9007932f5c0f))
* MapsLoader reads MAPS score-to-performance alignments as MatchClaims ([7c86c05](https://github.com/TimeToAlign/TimeToAlign/commit/7c86c05e8ce00d3e35a383da602ad598eba337c1))
* never lock a segment, lock segment lines in place, refuse negative child offsets ([1d90e3f](https://github.com/TimeToAlign/TimeToAlign/commit/1d90e3fff80e4f51e20350e4b170f2ba7348dcd6))
* read a MatchClaimField back from its Arrow table ([cd1ef3d](https://github.com/TimeToAlign/TimeToAlign/commit/cd1ef3ddb33c82d0c08f2db6e28de95be6fcb2f1))
* read and write manifest files as they came, minimal by default ([f946f0f](https://github.com/TimeToAlign/TimeToAlign/commit/f946f0f45c5b056f8b2610a4ff01b0349147b4f9))
* read and write TimeSkeleton 0.7.0 manifests, with unknown length and unlocked children ([6b7e4e9](https://github.com/TimeToAlign/TimeToAlign/commit/6b7e4e98d6134f71e9b0942032f8299b78441dd6))
* read and write TimeToShare 0.1.0 manifests ([bea05b0](https://github.com/TimeToAlign/TimeToAlign/commit/bea05b0d9e9d9e9d94c2ff6bc435b9d21b4bdc3a))
* read and write TimeToShare manifests of the next schema version ([790f4d7](https://github.com/TimeToAlign/TimeToAlign/commit/790f4d71308f21b5857163066e267801401a2f23))
* read every JSON number by its declared number type ([0e77588](https://github.com/TimeToAlign/TimeToAlign/commit/0e77588ca94c04c5cfd9929ad154dd90d6f7b450))
* record beliefs in held MatchClaims and filter coordinate queries by belief ([ace400a](https://github.com/TimeToAlign/TimeToAlign/commit/ace400a3ac683710c06d5f1b5eb8d250efcd3cd8))
* **rekordbox:** loader creates each track's beat grid and measure map ([873297e](https://github.com/TimeToAlign/TimeToAlign/commit/873297ea30727bed8519459e2373d3881acbebf9))
* rename the unit whole_note to whole_notes ([d7c2d86](https://github.com/TimeToAlign/TimeToAlign/commit/d7c2d86efd0218e8eea20a109bda53f792160772))
* **retrieval:** complete the retrieval grid across stamps and tables ([897e3d2](https://github.com/TimeToAlign/TimeToAlign/commit/897e3d26ade6609d957f4add217ee9c3fd68de55))
* **retrieval:** typed coordinate retrieval with format vocabulary, Interval scalar, and typed stamp storage ([b9aebd2](https://github.com/TimeToAlign/TimeToAlign/commit/b9aebd21145cb6b07fed06f55e88b3a8e845d3a1))
* ScoreWarpLoader reads a ScoreWarp SVG's drawn time axis as a calibrated child timeline ([09d39fd](https://github.com/TimeToAlign/TimeToAlign/commit/09d39fdb5a4f1fa4faf71b982920f7506cfdef9f))
* serialize timeline groups and alignment bundles ([6116d31](https://github.com/TimeToAlign/TimeToAlign/commit/6116d31e6aad8af66168798e5594d76489e2bca3))
* serialize timelines in a canonical order ([5fca756](https://github.com/TimeToAlign/TimeToAlign/commit/5fca756dcf2fffdce4c6f36b712cba85a61c9b60))
* statements and provenance on bundles, timelines and regions ([b7f5022](https://github.com/TimeToAlign/TimeToAlign/commit/b7f5022425dd8979f14f18ae22ef56e7e6453713))
* **statements:** candidates and beliefs replace statuses on statement provenance ([6047b28](https://github.com/TimeToAlign/TimeToAlign/commit/6047b28e228ef22fa6403870747393b885615d59))
* surface converted coordinates in MatchStamp display and table ([0189bb4](https://github.com/TimeToAlign/TimeToAlign/commit/0189bb4ae6b3dd1726a7e23e9725968e1898c94a))
* surface every attached C-Map in timestamps across the whole subtree ([4adccc0](https://github.com/TimeToAlign/TimeToAlign/commit/4adccc0db94c3d0125a9d091e8d8ff7a434b2e65))
* Timeline.apply_flow() yields the unfolded timeline ([ce51906](https://github.com/TimeToAlign/TimeToAlign/commit/ce5190690abe8f301b0f3e1a79e7728a934507eb))
* **timelines:** anchor the metric hierarchy on the measure map ([919fedd](https://github.com/TimeToAlign/TimeToAlign/commit/919feddaab53ad5d407508de103b9855166eea67))
* **timelines:** external reference tables and slim to_dict serialization ([a50bf7e](https://github.com/TimeToAlign/TimeToAlign/commit/a50bf7e94b9c237f40d78ab5b100c10a3a03a63b))
* **timelines:** guard the default-flow traversal and report traversal diagnostics ([c5ff3a2](https://github.com/TimeToAlign/TimeToAlign/commit/c5ff3a28365f1ad0844d78a37be780e5b6e5dde9))
* **timelines:** harmonize child creation across concrete classes ([7db621c](https://github.com/TimeToAlign/TimeToAlign/commit/7db621c610cac63c73baf0a18b19d784fe130fe3))
* **timelines:** lossless Measure Map rows with row IDs and one measure classifier ([d4326f0](https://github.com/TimeToAlign/TimeToAlign/commit/d4326f0f91e61097bb4381c953d0676d5543c5f2))
* **timelines:** parameterize SegmentLine over its segment class ([d9d9d17](https://github.com/TimeToAlign/TimeToAlign/commit/d9d9d17f1e5de5f375604135f86ce1b7f84b9f03))
* **timelines:** promote BeatGrid with faithful serialization and beat-unit-aware queries ([1d27d02](https://github.com/TimeToAlign/TimeToAlign/commit/1d27d02b007b8de0edab66ac67b4d1b0a79243b7))
* **timelines:** rebuild BeatGrid as tempo-anchored beat generators ([481ea03](https://github.com/TimeToAlign/TimeToAlign/commit/481ea030e626ad4f6becc9dc626496875458e0c6))
* **timelines:** resolve descendant coordinates through get_coordinate ([407c738](https://github.com/TimeToAlign/TimeToAlign/commit/407c7382c62bca9bbaee81c098d42deecc3b6fba))
* **timelines:** separate the metrical lattice from tempo realization ([c48b81e](https://github.com/TimeToAlign/TimeToAlign/commit/c48b81e9f9e83204e218e1b8e0cfec1401b18f90))
* **timelines:** serialize measure maps as compressed Measure Map rows ([95b983b](https://github.com/TimeToAlign/TimeToAlign/commit/95b983b60c5d67e6db0832ef3c8ad3481dacb0cf))
* **timelines:** store measure structure and flow control on every timeline ([c777df2](https://github.com/TimeToAlign/TimeToAlign/commit/c777df2dea072e4c598b46abd62491555a8325d4))
* **timelines:** unfold by appending named children and regions ([a4e5d72](https://github.com/TimeToAlign/TimeToAlign/commit/a4e5d72b6519825c173605499d21ba11018cd2a5))
* uu abbreviates the user_units graphical unit ([a85763a](https://github.com/TimeToAlign/TimeToAlign/commit/a85763a2a8a7e3de346ad3d7dba68feebe6682cf))
* VerovioLoader reads an engraved score through Verovio's timemap ([97ff33f](https://github.com/TimeToAlign/TimeToAlign/commit/97ff33f7a7ed949c26f5390025d587bca94d5781))
* VerovioLoader reads repeat marks, voices and chords; mints movement ids ([c5d51e6](https://github.com/TimeToAlign/TimeToAlign/commit/c5d51e67c5aaaf277f4657426d7df6b1e6bcabb1))
* write and read a bundle's MatchClaim and Projections tables beside its manifest ([f64f688](https://github.com/TimeToAlign/TimeToAlign/commit/f64f688ecbb095757fe2f90bb2cf33fad807c7e2))
* write and read a merged bundle's MatchClaims with their scopes ([27fbb13](https://github.com/TimeToAlign/TimeToAlign/commit/27fbb13da412177ae9e685213432a7c92a0fdf59))
* write conversion maps, measure maps and flow control in their TimeSkeleton 0.7.0 forms ([ce97970](https://github.com/TimeToAlign/TimeToAlign/commit/ce97970380c040e77d612c0c18a6d27c0bf94484))
* write the minimal form by default from every element writer ([45986d6](https://github.com/TimeToAlign/TimeToAlign/commit/45986d68b2ff38934fc8e2d7a57ef66b02e257ca))


### Bug Fixes

* a bundle answers for every timeline in its trees, nested ones included ([6961deb](https://github.com/TimeToAlign/TimeToAlign/commit/6961debeffbc32b2d049e90d2ba66b33bcf6ba1d))
* a given VerovioLoader root id keeps its number from the movements ([3619384](https://github.com/TimeToAlign/TimeToAlign/commit/3619384bcd341a01fe918ad2774cfc19c705b8c2))
* a map declared to write fractions refuses a float input ([dac45d2](https://github.com/TimeToAlign/TimeToAlign/commit/dac45d290354b4263593cac63abc1fa196fb429b))
* accept bracketed Verovio build suffixes ([1c0ce6e](https://github.com/TimeToAlign/TimeToAlign/commit/1c0ce6e1d68c8413b9fb228c85ee68ceebd102ed))
* **alignment:** count MatchClaims from the table; compare claim actors under the bundle's context ([735c81e](https://github.com/TimeToAlign/TimeToAlign/commit/735c81ea84733272b02a50a8520f0af48789904a))
* **alignment:** deepen MatchStamp guarantees and widen claim-factory coordinate inputs ([5b16e20](https://github.com/TimeToAlign/TimeToAlign/commit/5b16e20275990ef9267d3413a7a9c5bfbb5784f2))
* **alignment:** derive range-attach flow ids from the range, not the participant ([3ad9f59](https://github.com/TimeToAlign/TimeToAlign/commit/3ad9f591866c91785f8e424678ca77b2a17ed6b9))
* **alignment:** dispatch matchstamp retrieval on the query form ([0f3df89](https://github.com/TimeToAlign/TimeToAlign/commit/0f3df89ec5719b27a607a2737da31e57f73c7da8))
* **alignment:** read both claim stores in every AlignmentBundle query ([2a10c00](https://github.com/TimeToAlign/TimeToAlign/commit/2a10c002ad5e3ef261d6715fb5c02b7ba11c5a4b))
* **alignment:** record claim coordinates in the caller's unit ([38c7779](https://github.com/TimeToAlign/TimeToAlign/commit/38c77797e371403136b9e1bfaa02f86d50622952))
* **alignment:** refuse foreign units and rendition qualifiers in reference resolution ([2cbcf34](https://github.com/TimeToAlign/TimeToAlign/commit/2cbcf3482f6620053e7fc40e4210e388bc4ae814))
* **alignment:** reject ambiguous warp construction from non-monotone claims ([d3863fc](https://github.com/TimeToAlign/TimeToAlign/commit/d3863fc629f135606ec48b03d00c9fb526379a19))
* **alignment:** render claim coordinates exactly, with their units ([0b85da4](https://github.com/TimeToAlign/TimeToAlign/commit/0b85da410bbfecbbcf6fc2089047929345339718))
* **alignment:** return the transitive cross-group union from get_matchstamp_at ([67fc44e](https://github.com/TimeToAlign/TimeToAlign/commit/67fc44e40168850b17a6292155f2531727b3567e))
* **alignment:** the MatchClaim table is TimeToShare's MatchClaim table type ([d3d4bf8](https://github.com/TimeToAlign/TimeToAlign/commit/d3d4bf821f4f0e7a6d098657d2693089a6d6fd40))
* **alignment:** write MatchClaimField dictionary coordinates as coordinate dicts ([c6064b8](https://github.com/TimeToAlign/TimeToAlign/commit/c6064b8572c7d8b7897435b22bd9e04594079618))
* an ungrouped timeline warps as its own group in matchstamp queries ([1363c78](https://github.com/TimeToAlign/TimeToAlign/commit/1363c78acdb9c32e8a17211f6646d44d8c817d4d))
* bound a calibration pair by the error of a computed position relative to the span ([dc5daea](https://github.com/TimeToAlign/TimeToAlign/commit/dc5daea4c01ec00cd234997a6048cae8810c180e))
* bound a calibration pair by what computing its positions in doubles explains ([7a49d12](https://github.com/TimeToAlign/TimeToAlign/commit/7a49d12cb946bc76fdbe64d11f3e8de701dd6b7f))
* calibration pairs are checked against the exact line through the end pairs ([f1d7733](https://github.com/TimeToAlign/TimeToAlign/commit/f1d7733ecc7c86e606631c4444b7c64d4b48c3db))
* calibration positions are checked where they were rounded; maps declare their output number type ([b406fc0](https://github.com/TimeToAlign/TimeToAlign/commit/b406fc0b5caf8bb2d8deaba12be44aa0ecc5d69b))
* carry a bundle position up a tree only from a position on the queried timeline ([83d2c20](https://github.com/TimeToAlign/TimeToAlign/commit/83d2c20d18a1aa3938643f48f8900d154de047e0))
* carry a position through a timeline tree one offset step per edge ([69810cd](https://github.com/TimeToAlign/TimeToAlign/commit/69810cd462afb4566badfecdecb4857fd21542e2))
* carry a position to a descendant only through every timeline that holds it ([1e1f174](https://github.com/TimeToAlign/TimeToAlign/commit/1e1f1749849d1043341b944e17e3f73e4b48ab92))
* claims on an unregistered nested timeline are made and warped through ([038235e](https://github.com/TimeToAlign/TimeToAlign/commit/038235e3a0cf2305ad6c256c19e75e3166fa03a2))
* coerce Fraction timeline length to float at PyArrow boundary ([2354242](https://github.com/TimeToAlign/TimeToAlign/commit/2354242efcb5849a74b34d2e4dd3099f4e0c34ba))
* coerce Fraction timeline length to float at PyArrow boundary ([6eb05db](https://github.com/TimeToAlign/TimeToAlign/commit/6eb05dbfb23e948c78f82d3ea623f59976a4ed42))
* **core,alignment,loader:** close review findings on the input canon ([6fa672f](https://github.com/TimeToAlign/TimeToAlign/commit/6fa672fad3d9c0edb23892d5cb56617a8e0db707))
* **core:** accept integer-valued float ratio members at the rational wire boundary ([bfe8d77](https://github.com/TimeToAlign/TimeToAlign/commit/bfe8d77ca59a6011310b5af48e8d9b12bdf8e3ac))
* **core:** build the dyadic mirror without a platform-bound exponent ([39d8e59](https://github.com/TimeToAlign/TimeToAlign/commit/39d8e598b177958fdcce5b87ee2dc486769044f1))
* **core:** deep-copyable fields and one IRI comparison in the belief filter ([6a0c054](https://github.com/TimeToAlign/TimeToAlign/commit/6a0c054ac7436d5581e76ccafe3bbb8dcbf0be1d))
* **core:** exact duration ingestion and uniform stamp-axis number types ([ecc4e31](https://github.com/TimeToAlign/TimeToAlign/commit/ecc4e31c01eb471c4162b2f5afda1fe2ca1c8a39))
* **core:** express every derived value in its axis's declared number type ([8173232](https://github.com/TimeToAlign/TimeToAlign/commit/81732328be561fdd4b4baa239e4300d3549d5c26))
* **core:** express retrieved readings on their declared axes and make stamp order deterministic ([a4f64e4](https://github.com/TimeToAlign/TimeToAlign/commit/a4f64e4da3dedcc96ec230f6ba590ce7d2182799))
* **core:** keep exact rational coordinates exact through conversion and stamps ([ecfa12e](https://github.com/TimeToAlign/TimeToAlign/commit/ecfa12e4f5377904173347a3be3003cce64b5ef0))
* **core:** Provenance takes the format's method and match words and reads null beliefs ([f82e226](https://github.com/TimeToAlign/TimeToAlign/commit/f82e226d0b322e95bc8fe34251006614915f0a0a))
* **core:** register the BoundingBox field pairing and require finite corners ([d76182f](https://github.com/TimeToAlign/TimeToAlign/commit/d76182fa52eb9f9636151589d8402771ae90dcc3))
* **core:** stop reporting positions and pitches with more certainty than they carry ([c8272f7](https://github.com/TimeToAlign/TimeToAlign/commit/c8272f72652f0ac018e552d346c329db3735c9b8))
* **core:** surface metadata version errors in field shape matching ([d287562](https://github.com/TimeToAlign/TimeToAlign/commit/d2875622c9bbb26fa3f4fa08411869da051fa68a))
* count a map with float parameters as computing in doubles ([91b4297](https://github.com/TimeToAlign/TimeToAlign/commit/91b4297e9d9ced32bb3f7ebbddd817d6e44f5a1b))
* derive successors over whole endings, keep repeat edges at jumps ([8694923](https://github.com/TimeToAlign/TimeToAlign/commit/8694923bafc5548f6a6f6407f6bee29cc3db17d0))
* **display:** render nested children in the root's columns ([e6a5adc](https://github.com/TimeToAlign/TimeToAlign/commit/e6a5adc529287c6b276a0cc9d2ec424a5e5e57ac))
* evaluate CombinationMap outputs through each member's own evaluation ([10ceac9](https://github.com/TimeToAlign/TimeToAlign/commit/10ceac97aad7a4e6834fb853639ca3401b26f078))
* **flow:** repair default traversal for implicit, nested, and final-measure flow control ([4bf5dd3](https://github.com/TimeToAlign/TimeToAlign/commit/4bf5dd33055e93dc1d2c49dad24e682b7e22d370))
* **flow:** show every flow-control element in the score-map diagram ([a8fb41a](https://github.com/TimeToAlign/TimeToAlign/commit/a8fb41a4bdc3b84e0e924d21e90ade950ae42721))
* graph edges keep the claim id beside the claim they select by ([fef44ac](https://github.com/TimeToAlign/TimeToAlign/commit/fef44ac90f786e10a301863e7606575b53facdbe))
* group with a nested timeline of a bundle's trees ([a6c96e5](https://github.com/TimeToAlign/TimeToAlign/commit/a6c96e5112e0dfe37352fc3d570c62bb6bf31d6c))
* importers check every pair before changing a bundle and keep decisions on re-import ([e05a775](https://github.com/TimeToAlign/TimeToAlign/commit/e05a775d36b0a2a02281144173187e1d803fbfe1))
* keep exact numbers in IntervalToConstantMap.to_table_map ([511d9c1](https://github.com/TimeToAlign/TimeToAlign/commit/511d9c1e89af67d5aad58e33431d85696b2567f6))
* keep instant events interval-free and derive partitura coordinates from exact divisions ([b2e0c54](https://github.com/TimeToAlign/TimeToAlign/commit/b2e0c54cb22438ac21bd0824bb2f64bc0de69c0c))
* keep the styling read in TimeToShare's minimal form ([c826bd4](https://github.com/TimeToAlign/TimeToAlign/commit/c826bd4f96680c99052e1af7c57dad194523e07c))
* let conversion maps that compute in doubles answer doubles ([3e6c000](https://github.com/TimeToAlign/TimeToAlign/commit/3e6c000c3bacb03ba56ef42be3e6eadbc41aa9e4))
* **loader,storage,timelines:** harden the consolidated boundaries after review ([c484eda](https://github.com/TimeToAlign/TimeToAlign/commit/c484eda5710bc753e1f30fa5cee020962a86da1d))
* **loader:** align Rekordbox beat grids with the floating-measure lattice ([62a793f](https://github.com/TimeToAlign/TimeToAlign/commit/62a793fd5571b5d3e1ac422013c383fd4c303d60))
* **loader:** build conversion maps from exact parameters where the source is exact ([3e40047](https://github.com/TimeToAlign/TimeToAlign/commit/3e40047a2d7c0634c10c86105ed06e22725226c5))
* **loader:** carry exact rationals through matchfile and ms3 measure/duration parsing ([82f0e7e](https://github.com/TimeToAlign/TimeToAlign/commit/82f0e7ea38472e87d674dfcc93858e9246e0fd0f))
* **loader:** check soundfile's rate and channels and read its reason from soundfile ([a31f721](https://github.com/TimeToAlign/TimeToAlign/commit/a31f7210c08fd848ade976472b28a4f81e0be4a1))
* **loader:** count audio frames exactly instead of from the duration ([8bbc9d8](https://github.com/TimeToAlign/TimeToAlign/commit/8bbc9d8d80052e04dfc0a9aa8d6b96642a172981))
* **loader:** create audio timelines in samples with a seconds map ([8323b60](https://github.com/TimeToAlign/TimeToAlign/commit/8323b609360d32c46b1b4291809d12561e1a9c6a))
* **loader:** document the split-measure fm limitation instead of warning ([abcfd64](https://github.com/TimeToAlign/TimeToAlign/commit/abcfd642af511495331a35b9e142382e486d57f7))
* **loader:** keep audio metadata probing quiet on misnamed files ([1454024](https://github.com/TimeToAlign/TimeToAlign/commit/1454024511c96278c1f4ea513720f02eefaeba1b))
* **loader:** keep carried volta columns int64 across group unfolding ([632a76a](https://github.com/TimeToAlign/TimeToAlign/commit/632a76a532a05ae6c731705d1387aca5291b822b))
* **loader:** keep verbatim Listen Here seconds beside sample coordinates ([62c4b4a](https://github.com/TimeToAlign/TimeToAlign/commit/62c4b4a0aadf047046723426a29183bab79cd559))
* **loader:** key Rekordbox skeletons on the file-stem identity ([a7eea32](https://github.com/TimeToAlign/TimeToAlign/commit/a7eea3205aba7b09f0c4506705868fbcae47800a))
* **loader:** recover MEI repeat metadata from unnumbered measures ([0f739d7](https://github.com/TimeToAlign/TimeToAlign/commit/0f739d742b50368833817b67eafec6f4f45a00d5))
* **loader:** refuse a call with no source on single-source loaders ([1dfb41e](https://github.com/TimeToAlign/TimeToAlign/commit/1dfb41e4ad7a6d5347b487da27b209fe9638aee2))
* **loader:** refuse audio headers without a rate and name every backend's reason ([a4169cb](https://github.com/TimeToAlign/TimeToAlign/commit/a4169cbdc438093398b222257a4d6f72bc27b75e))
* **loader:** represent Listen Here audio on discrete sample timelines ([7d5dbe7](https://github.com/TimeToAlign/TimeToAlign/commit/7d5dbe77c8db9f9a3496d29d581661992ef65cff))
* **loader:** restore the notated extent of full-score ms3 timelines ([802cac6](https://github.com/TimeToAlign/TimeToAlign/commit/802cac621b9139461eb95f323626f4a434fea6fb))
* **loader:** retain full verbatim page coordinates and stable empty layers ([6ddd844](https://github.com/TimeToAlign/TimeToAlign/commit/6ddd844c505292dbb60e3a8873527e35e624df5d))
* **loader:** translate partitura navigation markers instead of dropping repeats ([3caa7bf](https://github.com/TimeToAlign/TimeToAlign/commit/3caa7bff2f6b2fb303dbd3bed30df03166ddeef3))
* **maps,timelines:** harden the unified map family after review ([9b222a2](https://github.com/TimeToAlign/TimeToAlign/commit/9b222a2bce761b4a5d77b6010fd98c8500b646eb))
* **maps:** answer a float constant as the float it is ([fbea279](https://github.com/TimeToAlign/TimeToAlign/commit/fbea2795b0de66766dcf19ac69203349e2a4e52f))
* **maps:** interpolate a TableMap exactly ([7f8895a](https://github.com/TimeToAlign/TimeToAlign/commit/7f8895ac49f69fc2d7ad41b78e5408e8f6e0f2a3))
* **maps:** keep nested metrical map identifiers on reload ([1217b1f](https://github.com/TimeToAlign/TimeToAlign/commit/1217b1f53058c2a193ef5e76ede7618c947e8e79))
* **metadata:** MEI-header and MeLODy imports offer candidates and hold no belief ([f3121be](https://github.com/TimeToAlign/TimeToAlign/commit/f3121beba06f4f8b012782b68827468ebbaefa01))
* mint claim ids in the scope that holds the claims ([c522618](https://github.com/TimeToAlign/TimeToAlign/commit/c522618dc90ee9d8a33e7daac65589f9e12b46ce))
* mint conversion-map IDs unique across nested maps ([34bc50f](https://github.com/TimeToAlign/TimeToAlign/commit/34bc50f9671f4103a55439cef638dd951233b079))
* mint implicit claim ids in the graph and carry source_claim_id through re-minting ([4654dd9](https://github.com/TimeToAlign/TimeToAlign/commit/4654dd98e2703d81edf350fa261b6ef61beeaf3d))
* Ms3Loader discovers companion facets in DCML facet directories ([3aa94ad](https://github.com/TimeToAlign/TimeToAlign/commit/3aa94add0a572a9ec1d703d185f8fa530b0184ed))
* Ms3Loader reads an empty tie cell as no tie ([eeb2088](https://github.com/TimeToAlign/TimeToAlign/commit/eeb20880927a79c5f8e304a6e273f627f9572a27))
* Ms3Loader refuses a measure without mc or length ([fb135d7](https://github.com/TimeToAlign/TimeToAlign/commit/fb135d7a65a479b9b040fb566aab82978ffd1ff8))
* name the MatchClaim belief add_match_claim_belief and read every declared prefix ([ccc985a](https://github.com/TimeToAlign/TimeToAlign/commit/ccc985a0b462e6ffbb510b434c0aafe346ddbb33))
* never copy a claim's bundle when the claim is copied ([8ebad4a](https://github.com/TimeToAlign/TimeToAlign/commit/8ebad4a36012bf69bf828d64f7d1a86a865f39d4))
* one containment rule for a position carried from a parent onto a child ([a7a1247](https://github.com/TimeToAlign/TimeToAlign/commit/a7a12472e304bda3ac6805dfeaf19f40ca9c2bad))
* prepare_abc_notes_for_matching keeps tie starts, not tie ends ([68908c1](https://github.com/TimeToAlign/TimeToAlign/commit/68908c188a53f8ad865911a35dab04e609619dea))
* preserve exact rational coordinates end-to-end instead of degrading to float ([5ad7dc4](https://github.com/TimeToAlign/TimeToAlign/commit/5ad7dc496dd9d7e2e39a82976e6d3a0f0a3c312d))
* read a child's end on a float axis as the double sum a reader computes ([e54ce60](https://github.com/TimeToAlign/TimeToAlign/commit/e54ce608c4e39872da66a24bf0b50ab6521ab254))
* read blank text tie cells as untied when selecting note onsets for matching ([9a4b3ac](https://github.com/TimeToAlign/TimeToAlign/commit/9a4b3ac42c3c5429c36c3a87c2a1fc2ff307e4a0))
* refuse a float input to a fraction-declared map as a ValueError ([f659f79](https://github.com/TimeToAlign/TimeToAlign/commit/f659f796e27beeaefb4ba199aa615a7db4c237f0))
* refuse a non-finite own-axis value with a ValueError before any rewrite suggestion ([554e2e1](https://github.com/TimeToAlign/TimeToAlign/commit/554e2e1ed7c8942025704c8c2937ed2669b11acf))
* refuse what the format refuses about named MatchStamps ([ea1fae8](https://github.com/TimeToAlign/TimeToAlign/commit/ea1fae83e4ec857a81bb2f180a80c0cc9859eb69))
* relabel only a map's float refusal as the timeline's query error ([11af652](https://github.com/TimeToAlign/TimeToAlign/commit/11af65259bde7b50ff3b4151eaf903eb41a6101c))
* renaming a group's minted ID renames the references to it ([86de1de](https://github.com/TimeToAlign/TimeToAlign/commit/86de1de2547e292358d0131c90418dd57d92a822))
* restore static type resolution for create_group/create_bundle ([883b9ae](https://github.com/TimeToAlign/TimeToAlign/commit/883b9ae7593be4fea834f58d4b061b017ed61986))
* ScoreWarpLoader mints its timeline id through the loader ([ef383fd](https://github.com/TimeToAlign/TimeToAlign/commit/ef383fd616b2b3539baf3eb5caee2bfbbb67cf10))
* **scorewarp:** mint the calibrated axis child's id through the loader ([e7a1ff9](https://github.com/TimeToAlign/TimeToAlign/commit/e7a1ff9d93e02df06d644ed9421071f8343875e0))
* **storage:** blueprint field resolution shares the class-path raw-column promotion ([04ad56f](https://github.com/TimeToAlign/TimeToAlign/commit/04ad56ff21528f5f8b8908ec8e852983de1a76b6))
* **testdata:** extract corpora atomically and verify completeness before trusting the cache ([50cadab](https://github.com/TimeToAlign/TimeToAlign/commit/50cadabfd0594a57b7f6ee8906bd9a1c61555fa7))
* **testdata:** make corpus extraction safe across processes ([ce2a6f5](https://github.com/TimeToAlign/TimeToAlign/commit/ce2a6f54ba905db4a6d65809285b81db53fcc6ae))
* **tests:** point MIDI integration specimens at the midi corpus ([0c23fb3](https://github.com/TimeToAlign/TimeToAlign/commit/0c23fb3b046c529b34e2130fa74f1c5ec35b444b))
* the minimal manifest form is TimeToShare's ([f889741](https://github.com/TimeToAlign/TimeToAlign/commit/f889741f2d5ae9751d14b6c3f4abdbab873fee2e))
* **timelines:** allow Region creation on locked timelines ([28fb4cc](https://github.com/TimeToAlign/TimeToAlign/commit/28fb4ccbc4d69b15e137b4163c83b07d598a1b96))
* **timelines:** enforce unit legality at timeline construction ([1540f7c](https://github.com/TimeToAlign/TimeToAlign/commit/1540f7c8da5be29330b5933add35e4218e059c9d))
* **timelines:** harden beat-grid queries and unify the published seconds lane ([b2e219e](https://github.com/TimeToAlign/TimeToAlign/commit/b2e219ed340f5f194cdf94a1bf39aad015e0612e))
* **timelines:** honour conversion_maps on every stamp producer and make row stamps round-trip ([1000577](https://github.com/TimeToAlign/TimeToAlign/commit/10005772247e0ab7cb3106be2002c0ab7be38291))
* **timelines:** reject appends to length-locked timelines ([b4ba551](https://github.com/TimeToAlign/TimeToAlign/commit/b4ba55141bac928ac82223faefb9b602124d9ad1))
* **timelines:** render discrete coordinates as integers in error messages ([c4ad085](https://github.com/TimeToAlign/TimeToAlign/commit/c4ad0855066bf9be39a16f5986730d97c934e8bd))
* **timelines:** resolve SegmentLine unit/number_type from segment_type ([7d8d5cb](https://github.com/TimeToAlign/TimeToAlign/commit/7d8d5cb3e41d5ca147ec092178c6e5c82d619f2b))
* **timelines:** resolve unit-qualified event coordinates on add ([60b9e67](https://github.com/TimeToAlign/TimeToAlign/commit/60b9e679550323b1ed501eac6d8ed3dd7ee20aa8))
* **timelines:** round-trip parameterized SegmentLine serializations ([541f8e5](https://github.com/TimeToAlign/TimeToAlign/commit/541f8e544b2156277724a03272d3a310bcf296a7))
* updates type hint ([6c6aabf](https://github.com/TimeToAlign/TimeToAlign/commit/6c6aabf5dbf0b0a34a24a757882383297f90c6b7))
* Verovio C-Map answers exact Fractions; movements minted with the loader's role ([33bd6fd](https://github.com/TimeToAlign/TimeToAlign/commit/33bd6fdfaf3041ed674a80a59ead32d89aa2c5d9))
* **verovio:** make a child timeline only for a named mdiv ([ac80009](https://github.com/TimeToAlign/TimeToAlign/commit/ac800092716cbbcc09b435c0cef88fd84621bfea))
* write a prefix into [@context](https://github.com/context) only where statements or the MatchClaim table use it ([f74e41e](https://github.com/TimeToAlign/TimeToAlign/commit/f74e41e02d42c1564153aad7d8de10388141a2b9))
* write every prefix the manifest declares into [@context](https://github.com/context) ([e46b7a6](https://github.com/TimeToAlign/TimeToAlign/commit/e46b7a60e7538b44b6a78cc1657b65741c115a3f))


### Performance Improvements

* a matchstamp reads only the claims naming each timeline it visits ([d758d87](https://github.com/TimeToAlign/TimeToAlign/commit/d758d87b4d3f23ca9a367987b5fb6e506e94df80))
* give an unregistered descendant its own warp unit only when a claim names it ([1771ec9](https://github.com/TimeToAlign/TimeToAlign/commit/1771ec93798f4292d8eb238632cfe5abddde253b))
* **midi:** add a mido-backed parser option to ScoreMidiLoader ([bffed90](https://github.com/TimeToAlign/TimeToAlign/commit/bffed90e029d192a6cd512f550775e66cf50a6ea))
* read a map's float arithmetic and stated number type once ([68b4a00](https://github.com/TimeToAlign/TimeToAlign/commit/68b4a006bd99b7be09fd8df51f8cd4dbd37ca2a8))
* take number-struct columns whole when building coordinate cells ([76ab7ba](https://github.com/TimeToAlign/TimeToAlign/commit/76ab7ba05ef9d906bd15d2cb6adf92ab619a07de))


### Reverts

* **testdata:** drop cross-process extraction file lock ([1217b5a](https://github.com/TimeToAlign/TimeToAlign/commit/1217b5a2b19ea50d0b28cea59239d49a09ec3323))


### Documentation

* add a how-to on describing the work a bundle is about ([d8e0695](https://github.com/TimeToAlign/TimeToAlign/commit/d8e0695e0c31470e1722c52f06342a62416636dc))
* adds how01_brazilian_flows.py ([2fd39d8](https://github.com/TimeToAlign/TimeToAlign/commit/2fd39d86d523f6f7be491b646344b44c7377d750))
* adopt release install instructions and public project links ([3c94352](https://github.com/TimeToAlign/TimeToAlign/commit/3c94352b8d90dc00b020979df0f7e685319c8f1e))
* build the part slot's reference with to_file_id ([0cb77a5](https://github.com/TimeToAlign/TimeToAlign/commit/0cb77a5c18b29ef622a88238e0f915b81aa3e203))
* **bundle:** make every AlignmentBundle doctest self-contained ([89b181a](https://github.com/TimeToAlign/TimeToAlign/commit/89b181a445c55d945c96ef78d51868db742ee3c6))
* cite the published TISMIR article and add CITATION.cff ([db78fdd](https://github.com/TimeToAlign/TimeToAlign/commit/db78fdd23bc1fa0a2f1df989bf8e557f2aed9511))
* claim ids, MatchClaimField.from_table and one containment rule for children ([36f59d5](https://github.com/TimeToAlign/TimeToAlign/commit/36f59d5d0c92cd0925b7aa0462d4cc36f7bf532b))
* classify MatchClaim kinds; orient how04 NOMATCH claims ([b48bbc9](https://github.com/TimeToAlign/TimeToAlign/commit/b48bbc9c167670dbdd8276fd519501f1a1dbadb4))
* consolidate glossary, fix stale paths, and complete packaging metadata ([f2316d8](https://github.com/TimeToAlign/TimeToAlign/commit/f2316d85318327728f8a22c566c7b1f8bda1abe0))
* construct concrete timeline types in notebooks ([30eac1b](https://github.com/TimeToAlign/TimeToAlign/commit/30eac1b54193ad4e5f6b524fecdd0bc89bcb7699))
* **core:** distinguish duration-less gap semantics between flow contexts ([e0911f9](https://github.com/TimeToAlign/TimeToAlign/commit/e0911f913c09e1e97acddb3c67d566519df2e0a0))
* **core:** document exact binary-expansion semantics for float-to-Fraction conversion ([991032c](https://github.com/TimeToAlign/TimeToAlign/commit/991032c3eee8f373d4bdf4b0f22089383ff4ad03))
* correct stored-uid queries, import homes, and the architecture page ([f33f018](https://github.com/TimeToAlign/TimeToAlign/commit/f33f0188b19e73ea3f0745b0bc6d64517fb87599))
* define the minimal form as the default of every writer ([3376740](https://github.com/TimeToAlign/TimeToAlign/commit/33767400f1fad422162129d54995fc8442074e6c))
* demonstrate the hierarchical child API across notebooks and reference ([a070a83](https://github.com/TimeToAlign/TimeToAlign/commit/a070a8364ef03ec18e4229a9275fb62a6699ef80))
* describe a bundle and its timelines by statements ([1b18f45](https://github.com/TimeToAlign/TimeToAlign/commit/1b18f45a05bcbf80a96fb6a2c99e501c1888509d))
* describe manifest files in three serializations and list inherited file methods ([08f0dcd](https://github.com/TimeToAlign/TimeToAlign/commit/08f0dcd134eb085cc904dcf94a1b89bbd71958da))
* describe named-mdiv movement children and the per-property claim table ([07c6c5a](https://github.com/TimeToAlign/TimeToAlign/commit/07c6c5aac10cfdd4ebe83e89a0d73e54c5437a55))
* describe the engraved score's tuplet positions as the output shows them ([aa9f220](https://github.com/TimeToAlign/TimeToAlign/commit/aa9f2205919f47c6ac65c17731d6add87bf3d52b))
* document SupportPolicy and the cross-group matchstamp union ([62f9073](https://github.com/TimeToAlign/TimeToAlign/commit/62f9073baccf917220c1101b9c552e9c34b00f35))
* drop API reference entries for objects that no longer exist ([5c2555f](https://github.com/TimeToAlign/TimeToAlign/commit/5c2555f7739fee1ba62392aef9d76caf070e81e9))
* drops the redundant calls to .diagram() ([b927a26](https://github.com/TimeToAlign/TimeToAlign/commit/b927a2649d5ba05b25a705331650559ea042b434))
* fix quartodoc crash on lazy-exported types, document full API surface ([19608ec](https://github.com/TimeToAlign/TimeToAlign/commit/19608eceed0be22eb17719e836b8d20c6cc949ba))
* full site render with re-executed notebooks ([41d0c2e](https://github.com/TimeToAlign/TimeToAlign/commit/41d0c2e04076ed4c160668d637a8a018881c5a95))
* how-to on aligning an engraved score with a recording and a drawn time axis ([c868824](https://github.com/TimeToAlign/TimeToAlign/commit/c868824c9ccb3fb48504b2a7f32626f7f3fdff4f))
* how-to on MatchClaims, Projections and beliefs in a package; reference pages for provenance and Projections ([aa4a382](https://github.com/TimeToAlign/TimeToAlign/commit/aa4a382a1f09cc4ff6689129e067c5951040b5e3))
* **howto:** demonstrate traversal diagnostics on the Choros flows notebook ([3465a00](https://github.com/TimeToAlign/TimeToAlign/commit/3465a006774479ebb09e35488a20413f7e28f58f))
* **howto:** tabulate only unambiguous positions in the match-file walkthrough ([f43b9d1](https://github.com/TimeToAlign/TimeToAlign/commit/f43b9d1716eb4ac9e809b2851539119762fd2e86))
* IEEE 1599 how-to notebook and slim to_dict updates ([0fce27a](https://github.com/TimeToAlign/TimeToAlign/commit/0fce27a0b4bf34dbf53d60c82d5d0d257eb06c4a))
* improves project links ([949a8c7](https://github.com/TimeToAlign/TimeToAlign/commit/949a8c7df26b0b570d126eec320dfd8fa76a5eec))
* link the published TISMIR article ([ea31e2e](https://github.com/TimeToAlign/TimeToAlign/commit/ea31e2efa21e0f1cb98dcd41005a97909afb18ec))
* link the published TISMIR article and repair stale cross-references ([ae619cf](https://github.com/TimeToAlign/TimeToAlign/commit/ae619cfbbba59175b9116b2314cd4c6933ed8033))
* list OutputNumberTypeError in the conversion-map API reference ([77ff54c](https://github.com/TimeToAlign/TimeToAlign/commit/77ff54c36ec92081371b1e6b1c5a2a07027a23e2))
* make the MatchGraph and MatchStamp.__str__ examples runnable ([90fa070](https://github.com/TimeToAlign/TimeToAlign/commit/90fa0706411cd481470fc17e78627f42c0655611))
* **manifest:** say a file form records whether any timeline holds the table keys ([99a155d](https://github.com/TimeToAlign/TimeToAlign/commit/99a155deb31d7c8aeb62ae091eb9ee723123b59e))
* migrate notebooks and pages to the consolidated loader, ID, and map APIs ([4961ce6](https://github.com/TimeToAlign/TimeToAlign/commit/4961ce6d9135b48b651706bc698eb8751b9137b9))
* migrate notebooks and site to the typed retrieval surface ([5bb8ec8](https://github.com/TimeToAlign/TimeToAlign/commit/5bb8ec87fcaf82292661f36ef2370d5bbfe4a9aa))
* migrate notebooks to the consolidated query and loader API ([e8a8062](https://github.com/TimeToAlign/TimeToAlign/commit/e8a8062445db75d7d16d93c9647b93305153081f))
* modifies notebook for producing the flow diagrams for the three scores of Beethoven op. 35 (Eroica variations) ([d1aa553](https://github.com/TimeToAlign/TimeToAlign/commit/d1aa5532b5e81bffc3cb848ef90d9393c9c92b95))
* name both test oracles of the conformance group as git references ([acef1a2](https://github.com/TimeToAlign/TimeToAlign/commit/acef1a2c9ffbc20a36c65a5479ae3c73ed3f65be))
* name flow-control ids by element type in the flow tutorial ([4e5a862](https://github.com/TimeToAlign/TimeToAlign/commit/4e5a862972b5cba2aebf0edaad1e6ea156f5f9b6))
* name the manifest format TimeToShare throughout ([7a20524](https://github.com/TimeToAlign/TimeToAlign/commit/7a205244731171d3db19f8875183e1f978d7c372))
* name the manifest's top-level measure maps ([187eafc](https://github.com/TimeToAlign/TimeToAlign/commit/187eafc404c14e4bd63818d654d4b820655a6dc0))
* note that only pymeasuremap is still a git reference ([2549cbe](https://github.com/TimeToAlign/TimeToAlign/commit/2549cbe448dcac1472d6c637dae1b1f929fb9f3b))
* open the landing page with the article's lead figure ([699046e](https://github.com/TimeToAlign/TimeToAlign/commit/699046ee90b65811c43b7a9210e7094e06fd7a89))
* promote DataFields and EventData to a foundational tutorial ([825c272](https://github.com/TimeToAlign/TimeToAlign/commit/825c272815731852e7ba468d6a12d57e4aaaf2e4))
* promote DataFields and EventData to a foundational tutorial ([79182ff](https://github.com/TimeToAlign/TimeToAlign/commit/79182ff8b84e23e62c27109586ec38dc32f11bd4))
* promote Pitch and Harmony to a cross-format tutorial ([35aa953](https://github.com/TimeToAlign/TimeToAlign/commit/35aa953779c2967c9bfe8991abbfe4ac63f9975a))
* promote Pitch and Harmony to a cross-format tutorial ([31b9908](https://github.com/TimeToAlign/TimeToAlign/commit/31b99083a0a075e2cc3cbd88c0f10530ef838559))
* publish the IEEE 1599 how-to and rebuild the reference site ([c213ff5](https://github.com/TimeToAlign/TimeToAlign/commit/c213ff5923497f867e23efdc00e1c5875d8e7e36))
* publish the loader walkthrough and fix its heading order ([6b0f428](https://github.com/TimeToAlign/TimeToAlign/commit/6b0f428506e1bdd3f124466eef7e229f4d97e970))
* re-execute the Beethoven multimodal how-to on tie-aware note matching ([54db243](https://github.com/TimeToAlign/TimeToAlign/commit/54db2430425535563b7874254f503a8bb0c53fe2))
* re-execute the how-tos and tutorial whose output the library changed ([c38df4e](https://github.com/TimeToAlign/TimeToAlign/commit/c38df4ea2c9730a87cf9937d5878da6a43b7b767))
* re-execute the notebooks whose output drifted ([29ed486](https://github.com/TimeToAlign/TimeToAlign/commit/29ed486945c97dfa6e4032f13e73794ec687b753))
* rebuild the how-to pages on files, piano rolls, work descriptions and loading ([41975d0](https://github.com/TimeToAlign/TimeToAlign/commit/41975d0e64bda8cf7f1b2ad0199f88e5c3857b3f))
* recompile the MPM toolbox how-to, which now shows the exact pixels-to-seconds scalar ([5109393](https://github.com/TimeToAlign/TimeToAlign/commit/51093934c2327757b7ac0d107117bbaa5c0b89b4))
* reconcile the MetricalPositionMap entry and the to_coda rule with the revised manifest ([cbde55f](https://github.com/TimeToAlign/TimeToAlign/commit/cbde55f8c04eaeba24302c6eafe468fe87e9ab2f))
* reference the real midi column in the DataFields tutorial ([9be45f1](https://github.com/TimeToAlign/TimeToAlign/commit/9be45f1eb6e0be5a564a225f5d463bc67c496175))
* regenerate API reference and MPM how-to for the merged library ([9eed89a](https://github.com/TimeToAlign/TimeToAlign/commit/9eed89ab422cfa5b1c5d379c455ad9995528ab6f))
* regenerate MPM how-to outputs for the represent-once pitch surface ([4c49df6](https://github.com/TimeToAlign/TimeToAlign/commit/4c49df6297038b7d0bdcb065693b6f22b168955c))
* regenerate reference and site after the docstring restatement ([56c4c6c](https://github.com/TimeToAlign/TimeToAlign/commit/56c4c6c3ede366099e305b475dbcf4fa4cc61746))
* regenerate the API reference with Google-style docstring parsing ([8bc6d6d](https://github.com/TimeToAlign/TimeToAlign/commit/8bc6d6d5eaa5a957a4121d72859192d5defd4000))
* register SegmentLine and inherited timeline methods in the reference ([0533ab3](https://github.com/TimeToAlign/TimeToAlign/commit/0533ab3fd7cd7a9853a550948ea2665dfacd6518))
* remove orphaned PerfectAlignment API-reference stub ([17fb417](https://github.com/TimeToAlign/TimeToAlign/commit/17fb417caef43bb15cae63427d10af33047a3736))
* remove orphaned PerfectAlignment API-reference stub ([7f59751](https://github.com/TimeToAlign/TimeToAlign/commit/7f5975140310c636d40256b4a9ebd86ace328e0e))
* render Google-style docstrings with a Yields-capable renderer ([62500b0](https://github.com/TimeToAlign/TimeToAlign/commit/62500b07ab2a2638373b5b136a890e95ac631dcb))
* replaces invariants table with CSV export ([a096ae4](https://github.com/TimeToAlign/TimeToAlign/commit/a096ae4adf4464898918eb7e3a1113c4ec55370c))
* replaces invariants table with CSV export ([93af713](https://github.com/TimeToAlign/TimeToAlign/commit/93af713751f6538387f45aa278c81bd93e71d4a4))
* republish bundle pages with true MatchClaim counts and fix the beat-grid how-to ([0d46c9b](https://github.com/TimeToAlign/TimeToAlign/commit/0d46c9bbdea5cb84a21d49b82ca54d8ba40b8dcc))
* route cross-group queries through get_matchstamp_at and provision tutorial data on demand ([466e098](https://github.com/TimeToAlign/TimeToAlign/commit/466e09822ce20d83fe9457a231d498480addd808))
* save, load and edit manifest files in the how-to guides ([a346204](https://github.com/TimeToAlign/TimeToAlign/commit/a346204784d1455a78cde16d5efef4c9c4a97cdf))
* show element ids and names, and follow the id scheme in every notebook ([20475c1](https://github.com/TimeToAlign/TimeToAlign/commit/20475c132532eaeb28fd52f7472c454b0fc58cb0))
* show placing flows and inverted cuts in the FlowMap how-to ([4c274bf](https://github.com/TimeToAlign/TimeToAlign/commit/4c274bf13f724e99647869e478f90732dbd5cab0))
* **site:** compile the tutorial series into the published notebooks ([ecb0826](https://github.com/TimeToAlign/TimeToAlign/commit/ecb0826ce73fa95530e95a76e48866e80b65d833))
* **site:** rebuild the published notebooks ([4926205](https://github.com/TimeToAlign/TimeToAlign/commit/4926205f37199916e0d37db9e99ea2d681d8bd5f))
* **site:** rebuild the published notebooks after the consistency pass ([0042cfe](https://github.com/TimeToAlign/TimeToAlign/commit/0042cfe9d90d96bbcd1bc92c4421537f93eb8ecc))
* **site:** rebuild the published notebooks after the final corrections ([07d7dba](https://github.com/TimeToAlign/TimeToAlign/commit/07d7dbadeb84191bc4b0ffe73fd044edc445f2ec))
* **site:** rebuild the published notebooks with exact claim rendering ([192b43f](https://github.com/TimeToAlign/TimeToAlign/commit/192b43ffa29819cc733fbe8a48999f509e7bb795))
* state free, manifest-wide unique ids in the glossary and the flow tutorial ([c2880b0](https://github.com/TimeToAlign/TimeToAlign/commit/c2880b0b297a98625bf51cdc59fd549c8fe1636e))
* state model definitions in the library's own voice ([75a12cf](https://github.com/TimeToAlign/TimeToAlign/commit/75a12cf3d107bf92550781859a6e71b24a598db2))
* state that a root id without a number keeps the movement ids of two_movements.mei ([2382d35](https://github.com/TimeToAlign/TimeToAlign/commit/2382d355b73f6abe09a27128d9749e31445e43dd))
* state that an exact anchor resolves a timeline interval claims withhold ([e1086e7](https://github.com/TimeToAlign/TimeToAlign/commit/e1086e7351c4fd291d2479ff5cff2a4b3cca93df))
* state the containment rule for children in the nesting tutorial ([6ba99a1](https://github.com/TimeToAlign/TimeToAlign/commit/6ba99a1ada0c71336df86a93443b88cc80a0a7f4))
* state the flow-control id prefixes without the exchange format ([63f807b](https://github.com/TimeToAlign/TimeToAlign/commit/63f807bf189ff0c3c82405a344dbe3789ec17925))
* state timetoshare as a runtime dependency ([774e679](https://github.com/TimeToAlign/TimeToAlign/commit/774e67990ad92a94240f8ffeea6282d052aa3ee1))
* teach add_belief and actor=, write packages without event rows, reference Resource and Integrity ([22e135a](https://github.com/TimeToAlign/TimeToAlign/commit/22e135a19387488888c925aea129a5199efbcae5))
* teach groups as shared-extent arrays and add a group and bundle file how-to ([98a6bb3](https://github.com/TimeToAlign/TimeToAlign/commit/98a6bb3612bc92af85da41a22e85ee381cc17798))
* teach MatchClaims between events, Projections and beliefs ([2f6cae5](https://github.com/TimeToAlign/TimeToAlign/commit/2f6cae526a518250786bb14e2a4a13141ffaea2c))
* teach the position/span operator rules in the coordinate how-to ([a4be4bd](https://github.com/TimeToAlign/TimeToAlign/commit/a4be4bdeaf581b8637c14b6d6066e87a41c78d9f))
* teach the position/span operator rules in the coordinate how-to ([9fae258](https://github.com/TimeToAlign/TimeToAlign/commit/9fae258b5c1f7c163aafa51a1078db8767aa7de0))
* teach the tempo-anchored beat grids ([7fcfdad](https://github.com/TimeToAlign/TimeToAlign/commit/7fcfdad649dbca7127161676c986a30f189e396a))
* tell the MPM bundle's group names from their minted IDs ([940c3ec](https://github.com/TimeToAlign/TimeToAlign/commit/940c3ec45fe15c1c603f0791b950e7320f8e0ae2))
* the containment rule for children holds along the path ([3ec16e0](https://github.com/TimeToAlign/TimeToAlign/commit/3ec16e09edc3ea6a13eedca0c709cf120ce5ab4b))
* the library carries statements, no link model ([2927ac2](https://github.com/TimeToAlign/TimeToAlign/commit/2927ac2cbcf50d27fb1a2d3a5031782e91485bb8))
* tour accolade segment lines in the IEEE 1599 how-to ([577197a](https://github.com/TimeToAlign/TimeToAlign/commit/577197a84a2f5eae121b294d192f27c392fe7a4f))
* **tutorials:** close the last gaps between prose and rendered output ([b48d100](https://github.com/TimeToAlign/TimeToAlign/commit/b48d100ae5edf771ddecca81fba8cc3ab7d93820))
* **tutorials:** make the ten notebooks read as one series ([dc7f5bd](https://github.com/TimeToAlign/TimeToAlign/commit/dc7f5bdb3c7b9da3552cc8f5001cce5bf64e205a))
* **tutorials:** match every claim in the prose to what the page shows ([9a53baa](https://github.com/TimeToAlign/TimeToAlign/commit/9a53baa111d7d9c897f0232ada8a17cdf3fa438a))
* **tutorials:** rebuild the tutorial series around a single learning curve ([ac37610](https://github.com/TimeToAlign/TimeToAlign/commit/ac37610208eb7af585ad02cec08e2fe0ce116c45))
* **tutorials:** show what the library returns, and one idea per cell ([c2a3238](https://github.com/TimeToAlign/TimeToAlign/commit/c2a32380c6421654b3f2bb1e4a03aa626dd0386f))


### Miscellaneous Chores

* release 1.0.1 ([50af5f8](https://github.com/TimeToAlign/TimeToAlign/commit/50af5f8e892cee5bf3988701794fc9f12cdafe3d))


### Code Refactoring

* **alignment:** name the bundle grouping parameter grouped_with ([fdcf269](https://github.com/TimeToAlign/TimeToAlign/commit/fdcf269c3aa097899cf606ba608ed5fe2f2d60a0))

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
