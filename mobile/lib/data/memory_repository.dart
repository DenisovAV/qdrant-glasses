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
  /// that branch) — only the time window and label narrow the result.
  ///
  /// **[label] is passed DOWN into [EdgeClient.timeline], not just applied
  /// here (Phase 1 review fix A).** The old body fetched the newest [limit]
  /// frames from [EdgeClient.timeline] first and filtered by label
  /// afterwards — so a label match older than the newest [limit] frames was
  /// silently dropped before it ever got a chance to match. [EdgeClient]'s
  /// own `label` filter now runs server-side, inside the SAME request that
  /// enforces [limit], so it is the actual source of truth.
  ///
  /// The re-filter below is belt-and-suspenders ONLY (defends a caller whose
  /// [EdgeClient] doesn't enforce its own filter, e.g. a test fake) — it
  /// must never be the reason a real match is found; case-insensitive to
  /// match the previous behavior for anything that reaches it. Phase 3:
  /// normalize case at the storage layer instead of here (and in
  /// [EdgeClient]'s exact, case-sensitive server-side match).
  Future<List<MomentHit>> search(ParsedQuery query, {int limit = _defaultLimit}) async {
    final label = query.label;
    final hits = await edgeClient.timeline(
      sinceMs: query.sinceMs,
      untilMs: query.untilMs,
      label: label,
      limit: limit,
    );
    if (label == null || label.isEmpty) return hits;
    final wanted = label.toLowerCase();
    return hits.where((h) => h.label.toLowerCase() == wanted).toList();
  }
}
