import 'parsed_query.dart';

/// Turns one chat turn's raw natural language into a [ParsedQuery] — a search
/// [ParsedQuery.phrase] plus any time window / label the user implied.
///
/// [now] anchors relative dates ("last week", "yesterday"); implementations
/// default it to the wall clock when omitted, while tests pass a fixed instant
/// so date resolution is deterministic. Implementations MUST be fail-soft: on
/// any parse/model failure, return `ParsedQuery(phrase: nl)` (the raw text, no
/// filter) rather than throwing — the search must never be blocked (the plan's
/// optional/fail-soft contract; mirrors the glasses B1 rules-first seam).
abstract class QueryParser {
  Future<ParsedQuery> parse(String nl, {DateTime? now});
}

/// Test double for [ChatAgent]/repository tests. Returns [result] for every
/// call when set; otherwise echoes the raw text as an unfiltered [ParsedQuery]
/// — the exact shape a real parser degrades to on failure. [byInput] maps
/// specific inputs first, falling through to [result]/echo for anything else.
class FakeQueryParser implements QueryParser {
  const FakeQueryParser({this.result, this.byInput});

  final ParsedQuery? result;
  final Map<String, ParsedQuery>? byInput;

  @override
  Future<ParsedQuery> parse(String nl, {DateTime? now}) async {
    final mapped = byInput?[nl];
    if (mapped != null) return mapped;
    return result ?? ParsedQuery(phrase: nl);
  }
}
