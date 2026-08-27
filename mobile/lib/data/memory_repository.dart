import '../embed/siglip_text.dart';
import '../query/parsed_query.dart';
import 'edge_client.dart';

/// The chat UI's one seam onto memory: list the recent timeline, or search it
/// with a [ParsedQuery]. [search] has two branches behind one signature:
///  - a non-empty [ParsedQuery.phrase] AND an [embedder] present → the Phase 2
///    `clip`-space vector path ([SiglipText.encode] → [EdgeClient.searchFrames],
///    time/label-filtered);
///  - otherwise → the pure time/label path over [EdgeClient.timeline].
/// [ChatScreen]'s call site never changes across the two.
class MemoryRepository {
  MemoryRepository({required this.edgeClient, this.embedder});

  final EdgeClient edgeClient;

  /// The query embedder for the semantic path. `null` (Phase 1 / a
  /// no-embedder build) keeps [search] on the pure time/label path even for a
  /// non-empty phrase — the vector branch is only taken when both a phrase and
  /// an embedder are present.
  final SiglipText? embedder;

  static const _defaultLimit = 50;

  /// The most recent [limit] `type=frame` moments, newest-first — the plain
  /// browse path (no filter at all).
  Future<List<MomentHit>> list({int limit = _defaultLimit}) {
    return edgeClient.timeline(limit: limit);
  }

  /// Two paths behind one signature:
  ///  - **Vector path** (a non-empty [ParsedQuery.phrase] AND [embedder] set):
  ///    embed the phrase into the `clip` space and kNN over it via
  ///    [EdgeClient.searchFrames], restricted to the same time window and
  ///    label. Results come back semantically ranked (by cosine score).
  ///  - **Pure time/label path** (blank phrase, or no embedder): the Phase 1
  ///    [EdgeClient.timeline] browse, newest-first.
  ///
  /// **[label] is passed DOWN into the EdgeClient call, not just applied here
  /// (Phase 1 review fix A).** Both `timeline` and `searchFrames` enforce the
  /// `label` filter server-side, inside the SAME request that enforces the
  /// limit/topK — so the filter is the actual source of truth, not a
  /// client-side afterthought that could drop a match beyond the first page.
  ///
  /// The re-filter below is belt-and-suspenders ONLY (defends a caller whose
  /// [EdgeClient] doesn't enforce its own filter, e.g. a test fake) — it must
  /// never be the reason a real match is found; case-insensitive to match the
  /// previous behavior for anything that reaches it. Phase 3: normalize case at
  /// the storage layer instead of here.
  Future<List<MomentHit>> search(ParsedQuery query, {int limit = _defaultLimit}) async {
    final label = query.label;
    final phrase = query.phrase.trim();
    final embedder = this.embedder;

    final List<MomentHit> hits;
    if (phrase.isEmpty || embedder == null) {
      hits = await edgeClient.timeline(
        sinceMs: query.sinceMs,
        untilMs: query.untilMs,
        label: label,
        limit: limit,
      );
    } else {
      final clip = await embedder.encode(phrase);
      hits = await edgeClient.searchFrames(
        clip: clip,
        topK: limit,
        sinceMs: query.sinceMs,
        untilMs: query.untilMs,
        label: label,
      );
    }

    if (label == null || label.isEmpty) return hits;
    final wanted = label.toLowerCase();
    return hits.where((h) => h.label.toLowerCase() == wanted).toList();
  }
}
