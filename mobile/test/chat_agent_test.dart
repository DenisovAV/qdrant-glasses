import 'package:fleet_node/chat/answerer.dart';
import 'package:fleet_node/chat/chat_agent.dart';
import 'package:fleet_node/data/edge_client.dart';
import 'package:fleet_node/data/memory_repository.dart';
import 'package:fleet_node/query/parsed_query.dart';
import 'package:fleet_node/query/query_parser.dart';
import 'package:fleet_node/ui/chat_message.dart';
import 'package:flutter_test/flutter_test.dart';

MomentHit _hit(String id, String label) => MomentHit(
      id: id,
      score: 1,
      momentId: 'm_$id',
      timestampMs: 1000,
      label: label,
    );

/// Returns preset hits from the pure-time path ([MemoryRepository] with no
/// embedder → `edgeClient.timeline`).
class _FakeEdge extends EdgeClient {
  _FakeEdge(this.hits);
  final List<MomentHit> hits;
  @override
  Future<List<MomentHit>> timeline({
    int? sinceMs,
    int? untilMs,
    String? label,
    int limit = 50,
  }) async =>
      hits;
}

class _FakeAnswerer implements Answerer {
  _FakeAnswerer(this.chunks);
  final List<String> chunks;
  @override
  Stream<String> answer(String query, List<MomentHit> hits) async* {
    for (final c in chunks) {
      yield c;
    }
  }
}

class _BoomAnswerer implements Answerer {
  @override
  Stream<String> answer(String query, List<MomentHit> hits) async* {
    throw StateError('LLM down');
  }
}

ChatAgent _agent({
  required List<MomentHit> hits,
  required Answerer answerer,
  QueryParser parser = const FakeQueryParser(),
}) =>
    ChatAgent(
      parser: parser,
      repository: MemoryRepository(edgeClient: _FakeEdge(hits)),
      answerer: answerer,
    );

void main() {
  group('ChatAgent.ask', () {
    test('parse → search → answer: assistant turn carries the answer + hits', () async {
      final agent = _agent(
        hits: [_hit('1', 'cup'), _hit('2', 'mug')],
        answerer: _FakeAnswerer(['Нашёл ', 'кружку.']),
      );

      final turns = await agent.ask('красная кружка').toList();

      expect(turns, isNotEmpty);
      expect(turns.every((t) => t.role == ChatRole.assistant), isTrue);
      final last = turns.last;
      expect(last.text, 'Нашёл кружку.'); // streamed tokens accumulated
      expect(last.hits, hasLength(2)); // hits carried on the turn
    });

    test('streams progressively — text grows, hits present from the first emit', () async {
      final agent = _agent(
        hits: [_hit('1', 'cup')],
        answerer: _FakeAnswerer(['A', 'B', 'C']),
      );

      final texts = [await for (final t in agent.ask('q')) t.text];

      expect(texts, ['A', 'AB', 'ABC']);
    });

    test('LLM failure → hits-with-stub-text, never throws', () async {
      final agent = _agent(
        hits: [_hit('1', 'cup'), _hit('2', 'mug')],
        answerer: _BoomAnswerer(),
      );

      final turns = await agent.ask('q').toList(); // must not throw

      final last = turns.last;
      expect(last.role, ChatRole.assistant);
      expect(last.text, 'Нашёл 2.'); // stub summary over the hits
      expect(last.hits, hasLength(2));
    });

    test('empty LLM output → stub over the hits', () async {
      final agent = _agent(hits: [_hit('1', 'cup')], answerer: _FakeAnswerer([]));

      final last = (await agent.ask('q').toList()).last;

      expect(last.text, 'Нашёл 1.');
      expect(last.hits, hasLength(1));
    });

    test('no hits → "nothing found" stub', () async {
      final agent = _agent(hits: const [], answerer: _FakeAnswerer([]));

      final last = (await agent.ask('q').toList()).last;

      expect(last.text, 'Ничего не нашёл.');
      expect(last.hits, isEmpty);
    });

    test('parser throwing degrades to the raw phrase, still answers', () async {
      final agent = _agent(
        hits: [_hit('1', 'cup')],
        answerer: _FakeAnswerer(['ok']),
        parser: _ThrowingParser(),
      );

      final last = (await agent.ask('anything').toList()).last;

      expect(last.text, 'ok');
      expect(last.hits, hasLength(1));
    });
  });
}

class _ThrowingParser implements QueryParser {
  @override
  Future<ParsedQuery> parse(String nl, {DateTime? now}) async =>
      throw StateError('parser down');
}
