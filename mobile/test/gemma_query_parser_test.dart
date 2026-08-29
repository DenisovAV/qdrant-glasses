import 'package:fleet_node/query/gemma_query_parser.dart';
import 'package:fleet_node/query/parsed_query.dart';
import 'package:flutter_test/flutter_test.dart';

// Local start/end-of-day epoch-ms for the assertions (mirrors the mapper).
int _dayStart(int y, int m, int d) => DateTime(y, m, d).millisecondsSinceEpoch;
int _dayEnd(int y, int m, int d) => DateTime(y, m, d)
    .add(const Duration(days: 1))
    .subtract(const Duration(milliseconds: 1))
    .millisecondsSinceEpoch;

void main() {
  group('parsedQueryFromToolArgs', () {
    test('maps phrase + ISO date window + label (a full search_memory call)', () {
      final pq = parsedQueryFromToolArgs(
        {
          'phrase': 'red coffee mug',
          'since': '2026-09-05',
          'until': '2026-09-05',
          'label': 'cup',
        },
        rawText: 'the red mug I saw on the 5th',
      );
      expect(pq.phrase, 'red coffee mug');
      expect(pq.sinceMs, _dayStart(2026, 9, 5));
      expect(pq.untilMs, _dayEnd(2026, 9, 5)); // whole day inclusive
      expect(pq.label, 'cup');
    });

    test('accepts raw epoch-ms bounds (number or numeric string)', () {
      final pq = parsedQueryFromToolArgs(
        {'phrase': 'x', 'since': 1000, 'until': '2000'},
        rawText: 'x',
      );
      expect(pq.sinceMs, 1000);
      expect(pq.untilMs, 2000);
    });

    test('blank/absent phrase falls back to the raw text', () {
      expect(
        parsedQueryFromToolArgs({'phrase': '   '}, rawText: 'wallet').phrase,
        'wallet',
      );
      expect(
        parsedQueryFromToolArgs({'label': 'cup'}, rawText: 'cups').phrase,
        'cups',
      );
    });

    test('blank label → no label filter; missing bounds → null', () {
      final pq = parsedQueryFromToolArgs(
        {'phrase': 'bike', 'label': ''},
        rawText: 'bike',
      );
      expect(pq.label, isNull);
      expect(pq.sinceMs, isNull);
      expect(pq.untilMs, isNull);
    });

    test('unparseable date bound is dropped, never throws', () {
      final pq = parsedQueryFromToolArgs(
        {'phrase': 'q', 'since': 'last week', 'until': 'garbage'},
        rawText: 'q',
      );
      expect(pq.sinceMs, isNull);
      expect(pq.untilMs, isNull);
      expect(pq.phrase, 'q');
    });

    test('null args (no tool-call / malformed) → raw-text fallback, no filter', () {
      final pq = parsedQueryFromToolArgs(null, rawText: 'что я видел вчера');
      expect(pq, isA<ParsedQuery>());
      expect(pq.phrase, 'что я видел вчера');
      expect(pq.sinceMs, isNull);
      expect(pq.untilMs, isNull);
      expect(pq.label, isNull);
    });
  });
}
