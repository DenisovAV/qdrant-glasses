import 'package:fleet_node/query/parsed_query.dart';
import 'package:fleet_node/query/query_parser.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('FakeQueryParser', () {
    test('echoes raw text as an unfiltered ParsedQuery by default', () async {
      const parser = FakeQueryParser();
      final pq = await parser.parse('red cup yesterday');
      expect(pq.phrase, 'red cup yesterday');
      expect(pq.sinceMs, isNull);
      expect(pq.untilMs, isNull);
      expect(pq.label, isNull);
    });

    test('returns the preset result for every input', () async {
      const preset = ParsedQuery(phrase: 'cup', label: 'cup', sinceMs: 1000);
      const parser = FakeQueryParser(result: preset);
      final pq = await parser.parse('anything at all');
      expect(pq, same(preset));
    });

    test('byInput maps specific inputs, else falls through to echo', () async {
      const parser = FakeQueryParser(
        byInput: {'cups': ParsedQuery(phrase: 'cup', label: 'cup')},
      );
      expect((await parser.parse('cups')).label, 'cup');
      expect((await parser.parse('something else')).phrase, 'something else');
      expect((await parser.parse('something else')).label, isNull);
    });

    test('implements QueryParser', () {
      const QueryParser p = FakeQueryParser();
      expect(p, isA<QueryParser>());
    });
  });
}
