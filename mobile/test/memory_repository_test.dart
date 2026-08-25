// Task 4: MemoryRepository — list()/search() against a fake EdgeClient (no
// native shard involved at all).
import 'package:fleet_node/data/edge_client.dart';
import 'package:fleet_node/data/memory_repository.dart';
import 'package:fleet_node/query/parsed_query.dart';
import 'package:flutter_test/flutter_test.dart';

/// Records every call it receives and answers with a canned, unfiltered
/// hit list — MemoryRepository is the one responsible for doing the actual
/// time/label narrowing on top of it (Phase 1: no vector search yet).
class FakeEdgeClient extends EdgeClient {
  final List<({int? sinceMs, int? untilMs, String? label, int limit})> timelineCalls = [];
  List<MomentHit> timelineResult = const [];

  @override
  Future<List<MomentHit>> timeline({
    int? sinceMs,
    int? untilMs,
    String? label,
    int limit = 50,
  }) async {
    timelineCalls.add((sinceMs: sinceMs, untilMs: untilMs, label: label, limit: limit));
    // Mirror the real EdgeClient's own time-window contract, so a test that
    // combines a window with a label sees the same AND-of-both-filters shape
    // MemoryRepository actually gets from the real thing.
    return timelineResult
        .where((h) => sinceMs == null || h.timestampMs >= sinceMs)
        .where((h) => untilMs == null || h.timestampMs <= untilMs)
        .toList();
  }
}

MomentHit _hit(String momentId, int ts, String label) => MomentHit(
  id: momentId,
  score: 0,
  momentId: momentId,
  timestampMs: ts,
  label: label,
);

void main() {
  group('list', () {
    test('delegates to EdgeClient.timeline with the given limit', () async {
      final fake = FakeEdgeClient()
        ..timelineResult = [_hit('m1', 1000, 'cup')];
      final repo = MemoryRepository(edgeClient: fake);

      final hits = await repo.list(limit: 7);

      expect(fake.timelineCalls.single.limit, 7);
      expect(fake.timelineCalls.single.sinceMs, isNull);
      expect(fake.timelineCalls.single.untilMs, isNull);
      expect(hits, hasLength(1));
      expect(hits.single.momentId, 'm1');
    });

    test('defaults to a reasonable limit', () async {
      final fake = FakeEdgeClient();
      final repo = MemoryRepository(edgeClient: fake);

      await repo.list();

      expect(fake.timelineCalls.single.limit, greaterThan(0));
    });
  });

  group('search', () {
    test('a time window with no label calls timeline with that window, '
        'no filtering', () async {
      final fake = FakeEdgeClient()
        ..timelineResult = [
          _hit('m1', 1000, 'cup'),
          _hit('m2', 2000, 'plant'),
        ];
      final repo = MemoryRepository(edgeClient: fake);

      final hits = await repo.search(
        const ParsedQuery(phrase: '', sinceMs: 500, untilMs: 2500),
      );

      expect(fake.timelineCalls.single.sinceMs, 500);
      expect(fake.timelineCalls.single.untilMs, 2500);
      expect(hits.map((h) => h.momentId), ['m1', 'm2']);
    });

    test(
      'a label is passed DOWN into EdgeClient.timeline — the real source of '
      'truth for filtering runs there, before EdgeClient enforces `limit` '
      '(not just applied client-side after the fact)',
      () async {
        final fake = FakeEdgeClient()..timelineResult = [_hit('m1', 1000, 'cup')];
        final repo = MemoryRepository(edgeClient: fake);

        await repo.search(const ParsedQuery(phrase: '', label: 'cup'));

        expect(fake.timelineCalls.single.label, 'cup');
      },
    );

    test('a label filters the timeline results client-side', () async {
      final fake = FakeEdgeClient()
        ..timelineResult = [
          _hit('m1', 1000, 'cup'),
          _hit('m2', 2000, 'plant'),
          _hit('m3', 3000, 'cup'),
        ];
      final repo = MemoryRepository(edgeClient: fake);

      final hits = await repo.search(const ParsedQuery(phrase: '', label: 'cup'));

      expect(hits.map((h) => h.momentId), ['m1', 'm3']);
    });

    test('label match is case-insensitive', () async {
      final fake = FakeEdgeClient()
        ..timelineResult = [_hit('m1', 1000, 'Cup')];
      final repo = MemoryRepository(edgeClient: fake);

      final hits = await repo.search(const ParsedQuery(phrase: '', label: 'cup'));

      expect(hits, hasLength(1));
    });

    test('a label that matches nothing returns empty, not an error', () async {
      final fake = FakeEdgeClient()
        ..timelineResult = [_hit('m1', 1000, 'cup')];
      final repo = MemoryRepository(edgeClient: fake);

      final hits = await repo.search(const ParsedQuery(phrase: '', label: 'dinosaur'));

      expect(hits, isEmpty);
    });

    test('time window + label combine (AND)', () async {
      final fake = FakeEdgeClient()
        ..timelineResult = [
          _hit('m1', 1000, 'cup'),
          _hit('m2', 2000, 'cup'),
        ];
      final repo = MemoryRepository(edgeClient: fake);

      final hits = await repo.search(
        const ParsedQuery(phrase: '', sinceMs: 1500, label: 'cup'),
      );

      expect(fake.timelineCalls.single.sinceMs, 1500);
      expect(hits.map((h) => h.momentId), ['m2']);
    });

    test('no filters at all -> everything timeline returns, unfiltered', () async {
      final fake = FakeEdgeClient()
        ..timelineResult = [_hit('m1', 1000, 'cup'), _hit('m2', 2000, 'plant')];
      final repo = MemoryRepository(edgeClient: fake);

      final hits = await repo.search(const ParsedQuery(phrase: 'anything'));

      expect(hits, hasLength(2));
    });
  });
}
