import '../query/parsed_query.dart';
import 'edge_client.dart';

/// The chat UI's one seam onto memory: list the recent timeline, or search it
/// with a [ParsedQuery]. Phase 1 has no embedder yet, so [search] runs a
/// pure time/label path over [EdgeClient.timeline] — Phase 2 (Task 7) adds a
/// `clip`-space vector branch behind this same method without changing the
/// signature or [ChatScreen]'s call site.
class MemoryRepository {
  MemoryRepository({required this.edgeClient});

  final EdgeClient edgeClient;

  static const _defaultLimit = 50;

  /// The most recent [limit] `type=frame` moments, newest-first — the plain
  /// browse path (no filter at all).
  Future<List<MomentHit>> list({int limit = _defaultLimit}) {
    return edgeClient.timeline(limit: limit);
  }

  /// Phase 1: [ParsedQuery.phrase] is not embedded yet (Phase 2/Task 7 adds
  /// that branch) — only the time window and label narrow the result. Pulls
  /// the `[sinceMs, untilMs]` window from [EdgeClient.timeline] (which has no
  /// label parameter of its own) and applies an exact, case-insensitive
  /// [ParsedQuery.label] match client-side.
  Future<List<MomentHit>> search(ParsedQuery query, {int limit = _defaultLimit}) async {
    final hits = await edgeClient.timeline(
      sinceMs: query.sinceMs,
      untilMs: query.untilMs,
      limit: limit,
    );
    final label = query.label;
    if (label == null || label.isEmpty) return hits;
    final wanted = label.toLowerCase();
    return hits.where((h) => h.label.toLowerCase() == wanted).toList();
  }
}
