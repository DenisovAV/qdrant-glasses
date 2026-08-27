// Task 4 + Task 7: MemoryRepository — list()/search() against a fake EdgeClient
// (no native shard involved at all). Task 7 adds the vector-search path.
import 'dart:typed_data';

import 'package:fleet_node/data/edge_client.dart';
import 'package:fleet_node/data/memory_repository.dart';
import 'package:fleet_node/embed/siglip_text.dart';
import 'package:fleet_node/query/parsed_query.dart';
import 'package:flutter_test/flutter_test.dart';

/// Records every call it receives and answers with a canned, unfiltered
/// hit list — MemoryRepository is the one responsible for doing the actual
/// time/label narrowing on top of it. Overrides BOTH the pure-time
/// [timeline] path and the Phase 2 vector [searchFrames] path so a test can
/// assert which branch [MemoryRepository.search] took.
class FakeEdgeClient extends EdgeClient {
  final List<({int? sinceMs, int? untilMs, String? label, int limit})> timelineCalls = [];
  final List<({Float32List clip, int topK, int? sinceMs, int? untilMs, String? label})> searchCalls = [];
  List<MomentHit> timelineResult = const [];
  List<MomentHit> searchResult = const [];

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

  @override
  Future<List<MomentHit>> searchFrames({
    required Float32List clip,
    int topK = 20,
    int? sinceMs,
    int? untilMs,
    String? label,
  }) async {
    searchCalls.add((clip: clip, topK: topK, sinceMs: sinceMs, untilMs: untilMs, label: label));
    // The real searchFrames applies the time window server-side too.
    return searchResult
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

  group('search — vector path (Task 7)', () {
    test('a non-empty phrase WITH an embedder takes the vector path: embeds '
        'and calls searchFrames with the clip vector + time/label filters', () async {
      final fake = FakeEdgeClient()
        ..searchResult = [_hit('m1', 1000, 'cup'), _hit('m2', 2000, 'cup')];
      final repo = MemoryRepository(
        edgeClient: fake,
        embedder: const FakeSiglipText(),
      );

      final hits = await repo.search(
        const ParsedQuery(phrase: 'a cup', sinceMs: 500, untilMs: 2500, label: 'cup'),
        limit: 12,
      );

      // vector path, not the timeline path.
      expect(fake.searchCalls, hasLength(1));
      expect(fake.timelineCalls, isEmpty);
      final call = fake.searchCalls.single;
      expect(call.clip, hasLength(768));
      expect(call.clip[0], 1.0); // the FakeSiglipText e_0 vector reached searchFrames
      expect(call.topK, 12);
      expect(call.sinceMs, 500);
      expect(call.untilMs, 2500);
      expect(call.label, 'cup');
      expect(hits.map((h) => h.momentId), ['m1', 'm2']);
    });

    test('a blank phrase WITH an embedder stays on the pure-time path', () async {
      final fake = FakeEdgeClient()..timelineResult = [_hit('m1', 1000, 'cup')];
      final repo = MemoryRepository(
        edgeClient: fake,
        embedder: const FakeSiglipText(),
      );

      await repo.search(const ParsedQuery(phrase: '   ', sinceMs: 500));

      expect(fake.timelineCalls, hasLength(1));
      expect(fake.searchCalls, isEmpty);
    });

    test('a non-empty phrase WITHOUT an embedder stays on the pure-time path', () async {
      final fake = FakeEdgeClient()..timelineResult = [_hit('m1', 1000, 'cup')];
      final repo = MemoryRepository(edgeClient: fake); // no embedder

      await repo.search(const ParsedQuery(phrase: 'a cup'));

      expect(fake.timelineCalls, hasLength(1));
      expect(fake.searchCalls, isEmpty);
    });

    test('the belt-and-suspenders label re-filter also guards the vector path', () async {
      final fake = FakeEdgeClient()
        ..searchResult = [_hit('m1', 1000, 'cup'), _hit('m2', 2000, 'plant')];
      final repo = MemoryRepository(
        edgeClient: fake,
        embedder: const FakeSiglipText(),
      );

      // A fake that does NOT enforce the label filter itself still gets
      // narrowed to the wanted label by the repository.
      final hits = await repo.search(const ParsedQuery(phrase: 'a cup', label: 'cup'));

      expect(hits.map((h) => h.momentId), ['m1']);
    });
  });
}
