import 'parsed_query.dart';

/// Maps a `search_memory` tool-call's [args] (what Gemma 4 emits when it
/// function-calls) into a [ParsedQuery]. Pure + total: the live
/// `GemmaQueryParser` (Task 9 Step 3) feeds it `FunctionCallResponse.args`,
/// while tests feed a recorded arg map — no live model needed either way.
///
/// Contract (fail-soft, the plan's optional/offline rule):
///  - `phrase` — the LLM's cleaned search text; blank/absent → fall back to the
///    user's [rawText] so retrieval is never blocked.
///  - `since` / `until` — an ISO date (`YYYY-MM-DD`, resolved to the local
///    start/end of that day) OR a raw epoch-ms number/string; unparseable →
///    that bound is simply dropped, never an exception.
///  - `label` — an exact label; blank/absent → no label filter.
///
/// The LLM resolves RELATIVE dates ("last week") itself, because
/// `GemmaQueryParser` puts today's date in the system instruction — so by the
/// time args arrive here the dates are already absolute. A null [args] (no
/// tool-call at all, or a malformed one) yields the raw-text fallback.
ParsedQuery parsedQueryFromToolArgs(
  Map<String, dynamic>? args, {
  required String rawText,
}) {
  if (args == null) return ParsedQuery(phrase: rawText);

  final phrase = _asNonEmptyString(args['phrase']);
  final label = _asNonEmptyString(args['label']);
  return ParsedQuery(
    phrase: phrase ?? rawText,
    sinceMs: _boundToMs(args['since'], endOfDay: false),
    untilMs: _boundToMs(args['until'], endOfDay: true),
    label: label,
  );
}

String? _asNonEmptyString(dynamic v) {
  if (v is! String) return null;
  final s = v.trim();
  return s.isEmpty ? null : s;
}

/// Resolves a time-window bound to epoch millis. Accepts a raw epoch-ms
/// number/numeric-string, or an ISO date/datetime string. For a bare
/// `YYYY-MM-DD`, [endOfDay]=false gives that day's local 00:00:00.000 and
/// [endOfDay]=true gives its local 23:59:59.999, so `[since, until]` covers the
/// whole day inclusively. Anything unparseable returns null (drop the bound).
int? _boundToMs(dynamic v, {required bool endOfDay}) {
  if (v == null) return null;
  if (v is num) return v.toInt();
  if (v is! String) return null;
  final s = v.trim();
  if (s.isEmpty) return null;

  final asMs = int.tryParse(s);
  if (asMs != null) return asMs;

  final parsed = DateTime.tryParse(s);
  if (parsed == null) return null;
  final day = DateTime(parsed.year, parsed.month, parsed.day);
  if (!endOfDay) return day.millisecondsSinceEpoch;
  return day
      .add(const Duration(days: 1))
      .subtract(const Duration(milliseconds: 1))
      .millisecondsSinceEpoch;
}
