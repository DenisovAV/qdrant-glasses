// Spike S1 (throwaway): proves the official qdrant_edge Dart/UniFFI SDK, on the
// arm64 Android emulator, can pull a real `fleet_curated` shard snapshot from a
// local Qdrant hub over REST, unpack + load it with a NAMED "clip" (768-d) +
// "text" (384-d) vector config, and run a Nearest `query()` + a filtered
// `scroll()` against it.
//
// Prerequisite (NOT done by this test): `adb -s emulator-5554 reverse tcp:6333
// tcp:6333` so `http://localhost:6333` on-device reaches the host Qdrant hub.
//
// Run: fvm flutter test integration_test/s1_fleet_pull_test.dart -d emulator-5554
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'package:http/http.dart' as http;
import 'package:integration_test/integration_test.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:qdrant_edge/qdrant_edge.dart';
import 'package:flutter_test/flutter_test.dart';

const _baseUrl = 'http://localhost:6333';
const _collection = 'fleet_curated';
const _clipDim = 768;
const _textDim = 384;

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('S1: pull fleet_curated snapshot, load, query, scroll', (
    tester,
  ) async {
    final appDir = await getApplicationSupportDirectory();
    final snapFile = File(p.join(appDir.path, 's1_fleet_snap.bin'));
    final shardDir = Directory(p.join(appDir.path, 's1_fleet_shard'));

    String? snapshotName;
    EdgeShard? shard;
    try {
      // 1. Create + download a fresh shard-0 snapshot over REST (same
      // endpoints the glasses' Kotlin FleetQdrantClient uses).
      final createResp = await http.post(
        Uri.parse('$_baseUrl/collections/$_collection/shards/0/snapshots'),
      );
      expect(
        createResp.statusCode,
        200,
        reason: 'snapshot create: ${createResp.body}',
      );
      final createBody = jsonDecode(createResp.body) as Map<String, dynamic>;
      snapshotName =
          (createBody['result'] as Map<String, dynamic>)['name'] as String;
      // ignore: avoid_print
      print('S1: created snapshot $snapshotName');

      final downloadResp = await http.get(
        Uri.parse(
          '$_baseUrl/collections/$_collection/shards/0/snapshots/$snapshotName',
        ),
      );
      expect(
        downloadResp.statusCode,
        200,
        reason: 'snapshot download: ${downloadResp.statusCode}',
      );
      if (snapFile.existsSync()) snapFile.deleteSync();
      snapFile.writeAsBytesSync(downloadResp.bodyBytes);
      // ignore: avoid_print
      print(
        'S1: downloaded snapshot ${downloadResp.bodyBytes.length} bytes to ${snapFile.path}',
      );

      // 2. Unpack. EdgeShard/unpack need the target dir to EXIST (and be
      // empty) — unpackSnapshot does not create parents.
      if (shardDir.existsSync()) shardDir.deleteSync(recursive: true);
      shardDir.createSync(recursive: true);
      unpackSnapshot(
        snapshotPath: snapFile.path,
        targetPath: shardDir.path,
      );
      // ignore: avoid_print
      print('S1: unpacked snapshot into ${shardDir.path}');

      // 3. Load with a NAMED-vector config matching the collection schema
      // (clip 768-d + text 384-d, both Cosine) — must match what's on disk.
      final config = EdgeConfig(
        vectorData: {
          'clip': VectorDataConfig(size: _clipDim, distance: Distance.cosine),
          'text': VectorDataConfig(size: _textDim, distance: Distance.cosine),
        },
      );
      shard = EdgeShard.load(path: shardDir.path, config: config);

      final count = shard.count(request: CountRequest());
      // ignore: avoid_print
      print('S1: loaded shard, count=$count');
      expect(count, 8);

      // 4. Nearest query on the "clip" named vector.
      final queryVec = List<double>.filled(_clipDim, 1 / math.sqrt(_clipDim));
      final hits = shard.query(
        request: QueryRequest(
          limit: 5,
          query: VectorScoringQuery(
            NearestQuery(vector: DenseNamedVector(queryVec), using: 'clip'),
          ),
          withPayload: BoolWithPayload(true),
        ),
      );
      // ignore: avoid_print
      print('S1: query returned ${hits.length} hit(s)');
      expect(hits, isNotEmpty);
      for (final hit in hits) {
        expect(hit.payload, isNotNull);
        final payload = jsonDecode(hit.payload!) as Map<String, dynamic>;
        // ignore: avoid_print
        print(
          '  hit id=${_idString(hit.id)} score=${hit.score} '
          'label=${payload['label']}',
        );
      }

      // 5. Scroll filtered by type=frame; assert payloads carry the fields
      // the task cares about.
      final scrollResp = shard.scroll(
        request: ScrollRequest(
          filter: Filter(
            must: [
              FieldConditionVariant(
                FieldCondition(
                  key: 'type',
                  match: ValueMatch(StringValueVariants('frame')),
                ),
              ),
            ],
          ),
          withPayload: BoolWithPayload(true),
          limit: 20,
        ),
      );
      // ignore: avoid_print
      print('S1: scroll returned ${scrollResp.records.length} record(s)');
      expect(scrollResp.records, isNotEmpty);
      for (final record in scrollResp.records) {
        expect(record.payload, isNotNull);
        final payload = jsonDecode(record.payload!) as Map<String, dynamic>;
        expect(payload['timestamp_ms'], isNotNull);
        expect(payload.containsKey('label'), isTrue);
      }
    } finally {
      // Best-effort cleanup, mirrors FleetSync.pull's finally block: the
      // server-side snapshot must not accumulate across test runs.
      shard?.unload();
      if (snapshotName != null) {
        try {
          final delResp = await http.delete(
            Uri.parse(
              '$_baseUrl/collections/$_collection/shards/0/snapshots/$snapshotName',
            ),
          );
          // ignore: avoid_print
          print('S1: deleted server snapshot, status=${delResp.statusCode}');
        } catch (e) {
          // ignore: avoid_print
          print('S1: snapshot delete failed (non-fatal): $e');
        }
      }
      if (snapFile.existsSync()) snapFile.deleteSync();
      if (shardDir.existsSync()) shardDir.deleteSync(recursive: true);
    }
  });
}

String _idString(PointId id) => switch (id) {
  UuidPointId(:final value) => value,
  NumIdPointId(:final value) => value.toString(),
  _ => id.toString(),
};
