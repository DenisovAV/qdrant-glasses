// Task 2: EdgeClient over package:qdrant_edge. Desktop-runnable — builds its
// own tiny shard on disk via the SDK directly (EdgeShard.create + upsert),
// then exercises EdgeClient's public surface (loadFromDir/timeline/
// searchFrames) against it. No phone needed.
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:fleet_node/data/edge_client.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:qdrant_edge/qdrant_edge.dart' as qe;

const _clipDim = 768;
const _textDim = 384;

void main() {
  late Directory tmp;
  late String shardDir;

  setUp(() {
    tmp = Directory.systemTemp.createTempSync('edge_client_test_');
    shardDir = '${tmp.path}/shard';
    Directory(shardDir).createSync(recursive: true);
    _seedShard(shardDir);
  });

  tearDown(() {
    if (tmp.existsSync()) tmp.deleteSync(recursive: true);
  });

  group('timeline', () {
    test('returns type=frame points newest-first', () async {
      final client = EdgeClient();
      await client.loadFromDir(shardDir);

      final hits = await client.timeline();

      expect(
        hits.map((h) => h.momentId).toList(),
        ['m3', 'm2', 'm1'],
        reason: 'newest timestamp_ms first',
      );
    });

    test('respects the limit', () async {
      final client = EdgeClient();
      await client.loadFromDir(shardDir);

      final hits = await client.timeline(limit: 2);

      expect(hits, hasLength(2));
      expect(hits.map((h) => h.momentId).toList(), ['m3', 'm2']);
    });

    test('filters by sinceMs/untilMs window', () async {
      final client = EdgeClient();
      await client.loadFromDir(shardDir);

      final hits = await client.timeline(sinceMs: 1500, untilMs: 2500);

      expect(hits, hasLength(1));
      expect(hits.single.momentId, 'm2');
    });

    test('excludes non-frame points', () async {
      final client = EdgeClient();
      await client.loadFromDir(shardDir);

      final hits = await client.timeline(limit: 100);

      expect(hits.any((h) => h.momentId == 'region-only'), isFalse);
    });

    test('no shard loaded → empty, never throws', () async {
      final client = EdgeClient();

      final hits = await client.timeline();

      expect(hits, isEmpty);
    });
  });

  group('searchFrames', () {
    test('filters by label', () async {
      final client = EdgeClient();
      await client.loadFromDir(shardDir);

      final hits = await client.searchFrames(
        clip: _unitVector(_clipDim),
        label: 'cup',
      );

      expect(hits, hasLength(1));
      expect(hits.single.label, 'cup');
      expect(hits.single.momentId, 'm1');
    });

    test('filters by time window combined with label', () async {
      final client = EdgeClient();
      await client.loadFromDir(shardDir);

      final hits = await client.searchFrames(
        clip: _unitVector(_clipDim),
        label: 'plant',
        sinceMs: 2500,
      );

      expect(hits, hasLength(1));
      expect(hits.single.momentId, 'm3');
    });

    test('unfiltered returns every frame point', () async {
      final client = EdgeClient();
      await client.loadFromDir(shardDir);

      final hits = await client.searchFrames(clip: _unitVector(_clipDim));

      expect(hits, hasLength(3));
    });

    test('no shard loaded → empty, never throws', () async {
      final client = EdgeClient();

      final hits = await client.searchFrames(clip: _unitVector(_clipDim));

      expect(hits, isEmpty);
    });
  });

  test('count reflects every point in the shard', () async {
    final client = EdgeClient();
    await client.loadFromDir(shardDir);

    // 3 frame points + 1 region-only point seeded below.
    expect(await client.count(), 4);
  });
}

Float32List _unitVector(int dim) {
  final v = Float32List(dim);
  final each = 1 / math.sqrt(dim);
  for (var i = 0; i < dim; i++) {
    v[i] = each;
  }
  return v;
}

void _seedShard(String dir) {
  final config = qe.EdgeConfig(
    vectorData: {
      'clip': qe.VectorDataConfig(size: _clipDim, distance: qe.Distance.cosine),
      'text': qe.VectorDataConfig(size: _textDim, distance: qe.Distance.cosine),
    },
  );
  final shard = qe.EdgeShard.create(path: dir, config: config);
  try {
    final points = [
      _framePoint(
        id: '11111111-1111-4111-8111-111111111111',
        momentId: 'm1',
        timestampMs: 1000,
        label: 'cup',
      ),
      _framePoint(
        id: '22222222-2222-4222-8222-222222222222',
        momentId: 'm2',
        timestampMs: 2000,
        label: 'plant',
      ),
      _framePoint(
        id: '33333333-3333-4333-8333-333333333333',
        momentId: 'm3',
        timestampMs: 3000,
        label: 'plant',
      ),
      // Not type=frame — must never surface from timeline()/searchFrames().
      qe.Point(
        id: qe.UuidPointId('44444444-4444-4444-8444-444444444444'),
        vector: qe.NamedVectorVariant({
          'clip': qe.DenseNamedVector(List<double>.filled(_clipDim, 0.02)),
        }),
        payload: jsonEncode({
          'type': 'region',
          'moment_id': 'region-only',
          'timestamp_ms': 4000,
          'label': 'mug',
        }),
      ),
    ];
    shard.update(operation: qe.UpdateOperation.upsertPoints(points: points));
    shard.flush();
  } finally {
    shard.unload();
  }
}

qe.Point _framePoint({
  required String id,
  required String momentId,
  required int timestampMs,
  required String label,
}) {
  return qe.Point(
    id: qe.UuidPointId(id),
    vector: qe.NamedVectorVariant({
      'clip': qe.DenseNamedVector(List<double>.filled(_clipDim, 0.01)),
    }),
    payload: jsonEncode({
      'type': 'frame',
      'moment_id': momentId,
      'timestamp_ms': timestampMs,
      'label': label,
    }),
  );
}
