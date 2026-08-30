/// The extracted intent behind one chat turn's raw text: a free-text
/// [phrase] plus an optional `[sinceMs, untilMs]` time window and an optional
/// exact [label] match.
///
/// Phase 1 constructs this directly from raw input (`ParsedQuery(phrase:
/// raw)`, no filter) — [MemoryRepository.search] runs the pure-time/label
/// path. Phase 3's `QueryParser`/`GemmaQueryParser` populate the filter
/// fields from natural language; this type is defined now so that later
/// phase slots in without changing [MemoryRepository]'s signature.
class ParsedQuery {
  const ParsedQuery({
    required this.phrase,
    this.sinceMs,
    this.untilMs,
    this.label,
  });

  /// The raw (or LLM-normalized) search text. Phase 2 embeds this into a
  /// `clip`-space vector; Phase 1 does not use it for retrieval yet.
  final String phrase;

  /// Inclusive lower bound on `timestamp_ms`, or null for no lower bound.
  final int? sinceMs;

  /// Inclusive upper bound on `timestamp_ms`, or null for no upper bound.
  final int? untilMs;

  /// An exact label to filter to (e.g. "cup"), or null for no label filter.
  final String? label;

  @override
  String toString() =>
      'ParsedQuery(phrase: "$phrase", sinceMs: $sinceMs, untilMs: $untilMs, '
      'label: $label)';
}
