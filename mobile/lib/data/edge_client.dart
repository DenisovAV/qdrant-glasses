import 'dart:convert';
import 'dart:typed_data';

import 'package:qdrant_edge/qdrant_edge.dart' as qe;

import '../logging.dart';

/// One `type=frame` hit from the fleet-curated shard: mirrors the glasses'
/// `MomentStore.MomentHit`, trimmed to what the phone's read-only chat UI
/// needs, plus [thumbB64] (Phase 4: the glasses will start stamping a small
/// base64 thumbnail into the payload; a corpus pulled before that change
/// simply has no `thumb_b64` key, so this is always nullable, never a crash).
class MomentHit {
  final String id;
  final double score;
  final String momentId;
  final int timestampMs;
  final String label;
  final String? thumbB64;

  const MomentHit({
    required this.id,
    required this.score,
    required this.momentId,
    required this.timestampMs,
    required this.label,
    this.thumbB64,
  });

  @override
  String toString() =>
      'MomentHit(id: $id, score: $score, momentId: $momentId, '
      'timestampMs: $timestampMs, label: $label, '
      'hasThumb: ${thumbB64 != null})';
}

/// Thin, fail-soft Dart client over `package:qdrant_edge`'s [qe.EdgeShard] —
/// the phone's read-only view of a pulled `fleet_curated` shard (or any shard
/// opened the same way, e.g. a test fixture built with the SDK directly).
///
/// Every public method degrades to an empty result rather than throwing when
/// no shard is loaded or a native call fails: the chat UI must never crash on
/// a missing/corrupt/unreachable corpus (project rule: fail-soft everywhere).
class EdgeClient {
  /// Named-vector field holding the SigLIP/CLIP-space image embedding.
  static const clipField = 'clip';

  /// Named-vector field holding the OCR text-space embedding (Stage 3, glasses
  /// side). Declared here only so [loadFromDir]'s schema matches the fleet
  /// collection — Phase 1 never queries it.
  static const textField = 'text';

  static const _clipDim = 768;
  static const _textDim = 384;

  // A defensive cap on how many scroll pages/records [_timelineFallback]
  // will walk client-side before giving up — the pre-fix-B approach,
  // reinstated ONLY for the path taken when the server-side `orderBy` in
  // [_timelineOrdered] throws (round-2 review fix #2). Fine at
  // fleet-curated scale; revisit if that corpus grows past this cap.
  static const _scrollPageSize = 500;
  static const _maxScrolledRecords = 5000;

  qe.EdgeShard? _shard;

  /// Test seam ONLY — production code always uses the default `true`.
  /// When `false`, [loadFromDir] skips [_ensureTimestampRangeIndex]
  /// entirely, standing in for "the index (re)creation attempt failed"
  /// without needing to force an actual native failure: from [timeline]'s
  /// point of view the two are indistinguishable (no range index present
  /// when `orderBy` runs, for whatever reason) — exactly the resilience
  /// round-2 review fix #2's guarding test proves.
  final bool _createTimestampIndexOnLoad;

  EdgeClient({bool createTimestampIndexOnLoad = true})
    : _createTimestampIndexOnLoad = createTimestampIndexOnLoad; // ignore: prefer_initializing_formals (renamed for a self-documenting public param name)

  /// True once [loadFromDir] has opened a shard.
  bool get isLoaded => _shard != null;

  /// Opens the shard already on disk at [dir] (e.g. after `FleetPull` has run
  /// `unpackSnapshot`, or a shard built directly via the SDK in a test).
  /// Declares BOTH named vectors ('clip' 768-d + 'text' 384-d, cosine) to
  /// match the fleet schema — same config the S1 spike and the glasses'
  /// `QdrantEdgeMomentStore` use.
  ///
  /// **Unloads whatever is currently loaded FIRST (Phase 1 review fix C).**
  /// The old body overwrote `_shard` directly — the previous native handle's
  /// WAL lock then outlived it for the rest of the process, because the
  /// AAR's GC finalizer only frees the Dart-side pointer, never calls the
  /// native `unload()` (see [close]'s own KDoc). A repeated `loadFromDir`
  /// (every [FleetPull.pull]) used to leak one lock per call.
  Future<void> loadFromDir(String dir) async {
    await close();
    final config = qe.EdgeConfig(
      vectorData: {
        clipField: qe.VectorDataConfig(
          size: _clipDim,
          distance: qe.Distance.cosine,
        ),
        textField: qe.VectorDataConfig(
          size: _textDim,
          distance: qe.Distance.cosine,
        ),
      },
    );
    _shard = qe.EdgeShard.load(path: dir, config: config);
    if (_createTimestampIndexOnLoad) _ensureTimestampRangeIndex();
  }

  /// (Re)creates the `timestamp_ms` RANGE payload index [timeline] needs to
  /// `orderBy` server-side (Phase 1 review fix B) — mirrors the glasses'
  /// `QdrantEdgeMomentStore.ensurePayloadIndexes`. Must run on every fresh
  /// shard handle, not just the first: `EdgeShard.load` does not carry
  /// indexes forward from a previous handle on the same directory, and a
  /// freshly-pulled snapshot is not guaranteed to already have one either.
  /// `runCatching`-style (matches the Kotlin reference): creating an index
  /// that already exists throws — swallow it, just log.
  ///
  /// **A creation failure is logged LOUDLY, not as "harmless" (round-2
  /// review fix #2, codex HIGH).** It used to read that way because
  /// [timeline] silently swallowed the resulting `orderBy` failure into an
  /// EMPTY result — the whole memory looked empty. Now that [timeline]
  /// itself degrades to [_timelineFallback] instead, a creation failure here
  /// is genuinely lower-stakes than before, but still worth a loud log: it
  /// means every future [timeline] call on this shard pays the slower
  /// client-side-sort path until the index is retried on a later
  /// [loadFromDir].
  void _ensureTimestampRangeIndex() {
    final shard = _shard;
    if (shard == null) return;
    try {
      shard.update(
        operation: qe.UpdateOperation.createFieldIndexWithParams(
          fieldName: 'timestamp_ms',
          params: qe.IntegerPayloadIndexParams(
            qe.IntegerIndexParams(range: true),
          ),
        ),
      );
      shard.flush();
    } catch (e) {
      fleetLog(
        "EdgeClient.loadFromDir: 'timestamp_ms' range index not (re)created "
        '— timeline() will fall back to a slower, unordered scroll + '
        'client-side sort until a later loadFromDir retries this: $e',
        level: 900,
      );
    }
  }

  /// Exact point count in the loaded shard (every channel, not just frames) —
  /// mainly a pull-succeeded sanity check (`count() > 0`). 0 when nothing is
  /// loaded (a genuine, known answer: there is nothing to count); **`null`
  /// when the native call itself fails (round-2 review fix #4, silent-
  /// failure)** — a real, distinct outcome from a real 0. The old body
  /// coalesced BOTH to `0`, which made [FleetPull.pull] unable to tell "the
  /// hub answered with a genuinely empty shard" apart from "count() errored
  /// right after a good load" — the former is [PullEmpty], the latter must
  /// be [PullUnreachable] (an unknown state, not a known-empty one).
  Future<int?> count() async {
    final shard = _shard;
    if (shard == null) return 0;
    try {
      return shard.count(request: qe.CountRequest());
    } catch (e) {
      fleetLog('EdgeClient.count: native call failed: $e', level: 900);
      return null;
    }
  }

  /// Nearest-neighbour search over the `clip` vector, restricted to
  /// `type=frame` points, optionally further filtered by a `timestamp_ms`
  /// window and/or an exact `label` match. Empty (never throws) if no shard
  /// is loaded or the native call fails.
  Future<List<MomentHit>> searchFrames({
    required Float32List clip,
    int topK = 20,
    int? sinceMs,
    int? untilMs,
    String? label,
  }) async {
    final shard = _shard;
    if (shard == null) return const [];
    try {
      final results = shard.query(
        request: qe.QueryRequest(
          limit: topK,
          query: qe.VectorScoringQuery(
            qe.NearestQuery(
              vector: qe.DenseNamedVector(clip.toList()),
              using: clipField,
            ),
          ),
          filter: _frameFilter(sinceMs: sinceMs, untilMs: untilMs, label: label),
          withPayload: qe.BoolWithPayload(true),
        ),
      );
      final hits = <MomentHit>[];
      for (final sp in results) {
        final hit = _hitFromPayload(_idString(sp.id), sp.score, sp.payload);
        if (hit != null) hits.add(hit);
      }
      return hits;
    } catch (e) {
      fleetLog('EdgeClient.searchFrames: native call failed, reporting no hits: $e', level: 900);
      return const [];
    }
  }

  /// The `type=frame` points in `[sinceMs, untilMs]` (either bound optional)
  /// AND matching [label] (exact match; Phase 3: normalize case — see
  /// [MemoryRepository.search]'s belt-and-suspenders re-filter for why this
  /// stays case-sensitive for now), newest-first, up to [limit]. No query
  /// vector — the pure-time/browse/label path. Empty (never throws) if no
  /// shard is loaded or the native call fails.
  ///
  /// **[label] is applied server-side, in the SAME `scroll()` request that
  /// enforces [limit] (Phase 1 review fix A).** The old
  /// `MemoryRepository.search` fetched the newest [limit] frames first and
  /// filtered by label AFTER — so a label match older than the newest
  /// [limit] frames was silently dropped. Filtering here, before the shard
  /// ever truncates to [limit], is what fixes that.
  ///
  /// **Server-side order + limit (Phase 1 review fix B).** [loadFromDir]
  /// (re)creates a `timestamp_ms` RANGE payload index on every fresh shard
  /// handle, which lets this ask the shard itself to `orderBy(timestamp_ms
  /// DESC)` and `limit` in one `scroll()` call — no more paging the WHOLE
  /// matching set client-side and sorting/truncating in Dart. The old body's
  /// `_maxScrolledRecords = 5000` walk cap meant a corpus past that size
  /// could silently omit newer frames from the result; this has no such cap
  /// because the shard does the ordering, not this client.
  ///
  /// **Degrades, never silently empties, if the `timestamp_ms` range index
  /// is missing (round-2 review fix #2, codex HIGH).** [loadFromDir]
  /// (re)creates that index on every fresh shard handle, but if that attempt
  /// itself ever fails, the server-side `orderBy` this method relies on
  /// throws "No range index" — which used to be caught here and reported as
  /// an EMPTY timeline, making the whole memory look empty. This now falls
  /// back to [_timelineFallback] (an unordered scroll + client-side sort —
  /// the pre-fix-B approach) whenever the ordered path throws, for ANY
  /// reason: correct-but-slower, never silently-empty.
  Future<List<MomentHit>> timeline({
    int? sinceMs,
    int? untilMs,
    String? label,
    int limit = 50,
  }) async {
    final shard = _shard;
    if (shard == null) return const [];
    try {
      return _timelineOrdered(shard, sinceMs: sinceMs, untilMs: untilMs, label: label, limit: limit);
    } catch (e) {
      fleetLog(
        'EdgeClient.timeline: server-side orderBy(timestamp_ms) failed (a '
        "missing 'timestamp_ms' range index is the likely cause) — "
        'degrading to an unordered scroll + client-side sort instead of an '
        'empty timeline: $e',
        level: 900,
      );
      try {
        return await _timelineFallback(shard, sinceMs: sinceMs, untilMs: untilMs, label: label, limit: limit);
      } catch (e2) {
        fleetLog('EdgeClient.timeline: fallback scroll also failed, reporting no hits: $e2', level: 900);
        return const [];
      }
    }
  }

  /// The fast path: asks the shard to `orderBy(timestamp_ms DESC)` and
  /// `limit` server-side in one `scroll()` call. Throws (does not catch)
  /// when the `timestamp_ms` range index is missing — [timeline] is what
  /// catches that and falls back to [_timelineFallback].
  List<MomentHit> _timelineOrdered(
    qe.EdgeShard shard, {
    required int? sinceMs,
    required int? untilMs,
    required String? label,
    required int limit,
  }) {
    final resp = shard.scroll(
      request: qe.ScrollRequest(
        limit: limit,
        filter: _frameFilter(sinceMs: sinceMs, untilMs: untilMs, label: label),
        withPayload: qe.BoolWithPayload(true),
        orderBy: qe.OrderBy(key: 'timestamp_ms', direction: qe.Direction.desc),
      ),
    );
    final hits = <MomentHit>[];
    for (final record in resp.records) {
      final hit = _hitFromPayload(_idString(record.id), 0, record.payload);
      if (hit != null) hits.add(hit);
    }
    return hits;
  }

  /// The pre-fix-B approach: walks the shard's scroll cursor to completion
  /// (bounded by [_maxScrolledRecords]), decoding every matching `type=frame`
  /// record, then sorts + truncates in Dart. Correct regardless of index
  /// state — only reached when [_timelineOrdered]'s server-side `orderBy`
  /// throws (round-2 review fix #2), so the extra client-side work only ever
  /// happens on the degraded path, not the common one.
  Future<List<MomentHit>> _timelineFallback(
    qe.EdgeShard shard, {
    required int? sinceMs,
    required int? untilMs,
    required String? label,
    required int limit,
  }) async {
    final hits = <MomentHit>[];
    qe.PointId? cursor;
    var scanned = 0;
    while (scanned < _maxScrolledRecords) {
      final resp = shard.scroll(
        request: qe.ScrollRequest(
          offset: cursor,
          limit: _scrollPageSize,
          filter: _frameFilter(sinceMs: sinceMs, untilMs: untilMs, label: label),
          withPayload: qe.BoolWithPayload(true),
        ),
      );
      for (final record in resp.records) {
        final hit = _hitFromPayload(_idString(record.id), 0, record.payload);
        if (hit != null) hits.add(hit);
      }
      scanned += resp.records.length;
      cursor = resp.nextOffset;
      if (cursor == null || resp.records.isEmpty) break;
    }
    hits.sort((a, b) => b.timestampMs.compareTo(a.timestampMs));
    if (hits.length <= limit) return hits;
    return hits.sublist(0, limit);
  }

  qe.Filter _frameFilter({int? sinceMs, int? untilMs, String? label}) {
    final must = <qe.Condition>[
      qe.FieldConditionVariant(
        qe.FieldCondition(
          key: 'type',
          match: qe.ValueMatch(qe.StringValueVariants('frame')),
        ),
      ),
    ];
    if (sinceMs != null || untilMs != null) {
      must.add(
        qe.FieldConditionVariant(
          qe.FieldCondition(
            key: 'timestamp_ms',
            range: qe.RangeFloat(
              gte: sinceMs?.toDouble(),
              lte: untilMs?.toDouble(),
            ),
          ),
        ),
      );
    }
    if (label != null) {
      must.add(
        qe.FieldConditionVariant(
          qe.FieldCondition(
            key: 'label',
            match: qe.ValueMatch(qe.StringValueVariants(label)),
          ),
        ),
      );
    }
    return qe.Filter(must: must);
  }

  /// Decodes a raw payload JSON [String] into a [MomentHit]. Payload comes
  /// back as a FLAT, un-tagged JSON object (verified in S1 — distinct from
  /// the vector readback's externally-tagged shape,
  /// [[edge-ffi-vector-readback-tagged]]). Returns null (never throws) for a
  /// missing/malformed payload — the caller skips it, the same fail-soft rule
  /// the glasses' `recordToFleetPoint` applies on its read side.
  MomentHit? _hitFromPayload(String id, double score, String? payload) {
    if (payload == null) return null;
    try {
      final decoded = jsonDecode(payload);
      if (decoded is! Map<String, dynamic>) return null;
      // The `label` key is expected to be ABSENT on plenty of legitimate
      // points (e.g. an OCR/region point that never set one) — that's not
      // worth a log line. A key that IS present but holds the wrong JSON
      // type is a genuine anomaly, so only THAT case logs before the
      // `?? ''` coalesce silently papers over it.
      final rawLabel = decoded['label'];
      if (decoded.containsKey('label') && rawLabel is! String) {
        fleetLog(
          "EdgeClient._hitFromPayload: id=$id payload has a non-string "
          "'label' ($rawLabel) — coalesced to ''",
        );
      }
      return MomentHit(
        id: id,
        score: score,
        momentId: decoded['moment_id'] as String? ?? '',
        timestampMs: (decoded['timestamp_ms'] as num?)?.toInt() ?? 0,
        label: rawLabel is String ? rawLabel : '',
        thumbB64: decoded['thumb_b64'] as String?,
      );
    } catch (e) {
      fleetLog('EdgeClient._hitFromPayload: id=$id payload decode failed, skipping: $e', level: 900);
      return null;
    }
  }

  static String _idString(qe.PointId id) => switch (id) {
    qe.UuidPointId(:final value) => value,
    qe.NumIdPointId(:final value) => value.toString(),
    _ => id.toString(),
  };

  /// Releases the underlying native shard handle, if any. Idempotent.
  Future<void> close() async {
    try {
      _shard?.unload();
    } catch (e) {
      // Best-effort, mirrors flutter_gemma_rag_qdrant's QdrantEdgeClient.close():
      // a shard that won't unload keeps its WAL lock for the process lifetime,
      // which the next open reports as a locked shard — a better place to
      // surface it than an exception out of a close the caller can't act on.
      // Still logged (Phase 1 review fix F): a leaked WAL lock used to fail
      // completely silently.
      fleetLog('EdgeClient.close: native unload failed, WAL lock may persist: $e', level: 900);
    }
    _shard = null;
  }
}
