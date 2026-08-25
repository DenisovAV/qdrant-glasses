import 'dart:convert';
import 'dart:typed_data';

import 'package:qdrant_edge/qdrant_edge.dart' as qe;

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

  // A defensive cap on how many scroll pages / records timeline() and its
  // time-filtered callers will walk client-side before giving up. This client
  // sorts by `timestamp_ms` itself in Dart (see _pageThroughFrames) rather
  // than asking the shard to orderBy() server-side: server-side ordering
  // needs a `timestamp_ms` RANGE payload index (the glasses' own
  // QdrantEdgeMomentStore creates one explicitly on every fresh shard handle
  // — ensurePayloadIndexes), and a shard freshly unpacked from a pulled
  // snapshot is not guaranteed to carry one. Client-side sort is correct
  // regardless of index state; fine at fleet-curated scale (a curated pull,
  // not the raw on-device store). Revisit if that corpus grows past this cap.
  static const _scrollPageSize = 500;
  static const _maxScrolledRecords = 5000;

  qe.EdgeShard? _shard;

  /// True once [loadFromDir] has opened a shard.
  bool get isLoaded => _shard != null;

  /// Opens the shard already on disk at [dir] (e.g. after `FleetPull` has run
  /// `unpackSnapshot`, or a shard built directly via the SDK in a test).
  /// Declares BOTH named vectors ('clip' 768-d + 'text' 384-d, cosine) to
  /// match the fleet schema — same config the S1 spike and the glasses'
  /// `QdrantEdgeMomentStore` use.
  Future<void> loadFromDir(String dir) async {
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
  }

  /// Exact point count in the loaded shard (every channel, not just frames) —
  /// mainly a pull-succeeded sanity check (`count() > 0`). 0 if nothing is
  /// loaded or the native call fails.
  Future<int> count() async {
    final shard = _shard;
    if (shard == null) return 0;
    try {
      return shard.count(request: qe.CountRequest());
    } catch (_) {
      return 0;
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
    } catch (_) {
      return const [];
    }
  }

  /// The `type=frame` points in `[sinceMs, untilMs]` (either bound optional),
  /// newest-first, up to [limit]. No query vector — the pure-time/browse
  /// path. Empty (never throws) if no shard is loaded or the native call
  /// fails.
  Future<List<MomentHit>> timeline({
    int? sinceMs,
    int? untilMs,
    int limit = 50,
  }) async {
    final shard = _shard;
    if (shard == null) return const [];
    try {
      final hits = await _pageThroughFrames(
        shard,
        sinceMs: sinceMs,
        untilMs: untilMs,
      );
      hits.sort((a, b) => b.timestampMs.compareTo(a.timestampMs));
      if (hits.length <= limit) return hits;
      return hits.sublist(0, limit);
    } catch (_) {
      return const [];
    }
  }

  /// Walks the shard's scroll cursor to completion (bounded by
  /// [_maxScrolledRecords]), decoding every `type=frame` record it finds.
  Future<List<MomentHit>> _pageThroughFrames(
    qe.EdgeShard shard, {
    int? sinceMs,
    int? untilMs,
  }) async {
    final hits = <MomentHit>[];
    qe.PointId? cursor;
    var scanned = 0;
    while (scanned < _maxScrolledRecords) {
      final resp = shard.scroll(
        request: qe.ScrollRequest(
          offset: cursor,
          limit: _scrollPageSize,
          filter: _frameFilter(sinceMs: sinceMs, untilMs: untilMs),
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
    return hits;
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
      return MomentHit(
        id: id,
        score: score,
        momentId: decoded['moment_id'] as String? ?? '',
        timestampMs: (decoded['timestamp_ms'] as num?)?.toInt() ?? 0,
        label: decoded['label'] as String? ?? '',
        thumbB64: decoded['thumb_b64'] as String?,
      );
    } catch (_) {
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
    } catch (_) {
      // Best-effort, mirrors flutter_gemma_rag_qdrant's QdrantEdgeClient.close():
      // a shard that won't unload keeps its WAL lock for the process lifetime,
      // which the next open reports as a locked shard — a better place to
      // surface it than an exception out of a close the caller can't act on.
    }
    _shard = null;
  }
}
