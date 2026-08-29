import 'dart:developer' as developer;

import 'package:flutter_gemma/flutter_gemma.dart';

import 'parsed_query.dart';
import 'query_parser.dart';

/// The live [QueryParser]: Gemma 4 E2B function-calls a `search_memory` tool to
/// turn a chat turn into a [ParsedQuery]. Long-lived [InferenceModel] (loaded
/// once), short-lived session per turn (the `rag_demo` pattern). Fail-soft: any
/// model error/timeout, or no tool-call, degrades to `ParsedQuery(phrase: nl)`
/// so retrieval is never blocked. The pure [parsedQueryFromToolArgs] below does
/// the args → [ParsedQuery] mapping (host-tested); this class only drives the model.
class GemmaQueryParser implements QueryParser {
  GemmaQueryParser(
    this._model, {
    Duration timeout = const Duration(seconds: 12),
  }) : _timeout = timeout; // ignore: prefer_initializing_formals (public param name)

  final InferenceModel _model;
  final Duration _timeout;

  /// The tool Gemma 4 is asked to call. Params are a JSON-schema object; `since`
  /// / `until` are absolute ISO dates the model resolves from the system
  /// instruction's "today", so [parsedQueryFromToolArgs] only ever parses absolutes.
  static const _searchMemoryTool = Tool(
    name: 'search_memory',
    description:
        "Search the wearer's visual memory for moments that match what they're "
        'asking about. Always call this for a memory question.',
    parameters: {
      'type': 'object',
      'properties': {
        'phrase': {
          'type': 'string',
          'description':
              'the visual thing to look for, cleaned, e.g. "red coffee mug"',
        },
        'since': {
          'type': 'string',
          'description': 'inclusive start date YYYY-MM-DD, or omit if no time range',
        },
        'until': {
          'type': 'string',
          'description': 'inclusive end date YYYY-MM-DD, or omit if no time range',
        },
        'label': {
          'type': 'string',
          'description': 'an exact object label to filter to, or omit',
        },
      },
      'required': ['phrase'],
    },
  );

  @override
  Future<ParsedQuery> parse(String nl, {DateTime? now}) async {
    final today = now ?? DateTime.now();
    try {
      // Tool-calling surfaces through the CHAT path: generateChatResponse()
      // returns a structured ModelResponse — a FunctionCallResponse when the
      // model calls a tool — whereas the low-level session.getResponse() only
      // yields text. createChat(supportsFunctionCalls: true) is the documented
      // way to get Gemma 4's native function-call.
      final chat = await _model.createChat(
        temperature: 0,
        tools: const [_searchMemoryTool],
        supportsFunctionCalls: true,
        toolChoice: ToolChoice.auto,
        modelType: ModelType.gemma4,
        maxOutputTokens: 256,
        systemInstruction:
            'Today is ${_isoDate(today)}. The user asks about their own visual '
            'memory. Call search_memory with a cleaned phrase; if they imply a '
            'time range ("yesterday", "last week"), resolve it to absolute '
            'YYYY-MM-DD dates relative to today.',
      );
      await chat.addQuery(Message.text(text: nl, isUser: true));
      final response = await chat.generateChatResponse().timeout(_timeout);
      final args =
          (response is FunctionCallResponse && response.name == 'search_memory')
              ? response.args
              : null;
      return parsedQueryFromToolArgs(args, rawText: nl);
    } catch (e) {
      developer.log(
        'GemmaQueryParser: parse failed, using raw text unfiltered: $e',
        name: 'fleet',
        level: 900,
      );
      return ParsedQuery(phrase: nl);
    }
  }

  static String _isoDate(DateTime d) =>
      '${d.year.toString().padLeft(4, '0')}-'
      '${d.month.toString().padLeft(2, '0')}-'
      '${d.day.toString().padLeft(2, '0')}';
}

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
