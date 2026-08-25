// Task 3, extended by Phase 1 review fixes D+E: FleetPull's stage/validate/
// promote orchestration (create -> download -> unpack into a STAGING dir ->
// load+validate -> only then promote over the live corpus; delete the
// server-side snapshot in a finally). Host-runnable: exercises real failure
// paths (a mock HTTP layer, real qdrant_edge native calls for unpack) without
// needing a live fleet hub — the true success path (a real fleet_curated
// snapshot) is covered on-device by integration_test/fleet_pull_test.dart.
import 'dart:convert';
import 'dart:io';

import 'package:fleet_node/data/edge_client.dart';
import 'package:fleet_node/data/fleet_http.dart';
import 'package:fleet_node/data/fleet_pull.dart';
import 'package:fleet_node/data/pull_result.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:qdrant_edge/qdrant_edge.dart' as qe;

const _clipDim = 768;
const _textDim = 384;

void main() {
  late Directory workDir;

  setUp(() {
    workDir = Directory.systemTemp.createTempSync('fleet_pull_test_');
  });

  tearDown(() {
    if (workDir.existsSync()) workDir.deleteSync(recursive: true);
  });

  test('unreachable hub (create fails) -> PullUnreachable, never throws', () async {
    final client = MockClient((request) async {
      throw const SocketException('connection refused');
    });
    final fleetHttp = FleetHttp(baseUrl: 'http://localhost:6333', client: client);
    final edgeClient = EdgeClient();
    final fleetPull = FleetPull(
      http: fleetHttp,
      edgeClient: edgeClient,
      workDir: workDir.path,
    );

    final result = await fleetPull.pull();

    expect(result, isA<PullUnreachable>());
    expect(edgeClient.isLoaded, isFalse);
    // No leftover work-dir artifacts on a create-time failure.
    expect(File('${workDir.path}/fleet_snap.bin').existsSync(), isFalse);
    expect(Directory('${workDir.path}/fleet_shard').existsSync(), isFalse);
    expect(Directory('${workDir.path}/fleet_shard_staging').existsSync(), isFalse);
  });

  test('create succeeds but download 404s -> PullUnreachable; server '
      'snapshot still deleted', () async {
    final deleteRequests = <http.Request>[];
    final client = MockClient((request) async {
      if (request.method == 'POST') {
        return http.Response('{"result":{"name":"snap-1.snapshot"}}', 200);
      }
      if (request.method == 'GET') {
        return http.Response('not found', 404);
      }
      if (request.method == 'DELETE') {
        deleteRequests.add(request);
        return http.Response('', 200);
      }
      return http.Response('unexpected', 500);
    });
    final fleetHttp = FleetHttp(baseUrl: 'http://localhost:6333', client: client);
    final edgeClient = EdgeClient();
    final fleetPull = FleetPull(
      http: fleetHttp,
      edgeClient: edgeClient,
      workDir: workDir.path,
    );

    final result = await fleetPull.pull(collection: 'fleet_curated');

    expect(result, isA<PullUnreachable>());
    expect(edgeClient.isLoaded, isFalse);
    expect(deleteRequests, hasLength(1));
    expect(deleteRequests.single.url.toString(), contains('snap-1.snapshot'));
    // The intermediate snapshot file must not linger.
    expect(File('${workDir.path}/fleet_snap.bin').existsSync(), isFalse);
  });

  test('download succeeds but the bytes are not a real snapshot -> unpack '
      'fails, PullUnreachable, no orphan staging dir', () async {
    final client = MockClient((request) async {
      if (request.method == 'POST') {
        return http.Response('{"result":{"name":"snap-2.snapshot"}}', 200);
      }
      if (request.method == 'GET') {
        return http.Response.bytes([1, 2, 3, 4, 5], 200);
      }
      if (request.method == 'DELETE') {
        return http.Response('', 200);
      }
      return http.Response('unexpected', 500);
    });
    final fleetHttp = FleetHttp(baseUrl: 'http://localhost:6333', client: client);
    final edgeClient = EdgeClient();
    final fleetPull = FleetPull(
      http: fleetHttp,
      edgeClient: edgeClient,
      workDir: workDir.path,
    );

    final result = await fleetPull.pull();

    expect(result, isA<PullUnreachable>());
    expect(edgeClient.isLoaded, isFalse);
    expect(Directory('${workDir.path}/fleet_shard').existsSync(), isFalse);
    expect(Directory('${workDir.path}/fleet_shard_staging').existsSync(), isFalse);
    expect(File('${workDir.path}/fleet_snap.bin').existsSync(), isFalse);
  });

  test('server-snapshot delete failure never surfaces (best-effort)', () async {
    final client = MockClient((request) async {
      if (request.method == 'POST') {
        return http.Response('{"result":{"name":"snap-3.snapshot"}}', 200);
      }
      if (request.method == 'GET') {
        return http.Response('gone', 404);
      }
      if (request.method == 'DELETE') {
        return http.Response('boom', 500);
      }
      return http.Response('unexpected', 500);
    });
    final fleetHttp = FleetHttp(baseUrl: 'http://localhost:6333', client: client);
    final edgeClient = EdgeClient();
    final fleetPull = FleetPull(
      http: fleetHttp,
      edgeClient: edgeClient,
      workDir: workDir.path,
    );

    final result = await fleetPull.pull();

    expect(result, isA<PullUnreachable>());
  });

  // Phase 1 review fix D (codex HIGH): the old body deleted the LIVE
  // `fleet_shard` directory BEFORE the replacement was validated (unpack
  // needs an empty target dir) — so a failed re-pull destroyed a perfectly
  // good previous corpus. This seeds a real "already successfully pulled"
  // corpus at the exact `fleet_shard` path FleetPull.pull uses, loads it
  // (mirrors app startup after a prior good pull), then runs a re-pull that
  // fails at the unpack step (bad snapshot bytes) and asserts the old
  // corpus is untouched AND still queryable.
  test(
    'an already-loaded corpus survives (and stays queryable through) a '
    'failing re-pull with bad snapshot bytes',
    () async {
      final liveDir = Directory('${workDir.path}/fleet_shard');
      liveDir.createSync(recursive: true);
      _seedShard(liveDir.path, momentId: 'old-good-moment');

      final edgeClient = EdgeClient();
      await edgeClient.loadFromDir(liveDir.path);
      expect(await edgeClient.count(), 1);

      final client = MockClient((request) async {
        if (request.method == 'POST') {
          return http.Response('{"result":{"name":"snap-bad.snapshot"}}', 200);
        }
        if (request.method == 'GET') {
          return http.Response.bytes([9, 9, 9, 9, 9], 200); // not a real snapshot
        }
        if (request.method == 'DELETE') {
          return http.Response('', 200);
        }
        return http.Response('unexpected', 500);
      });
      final fleetHttp = FleetHttp(baseUrl: 'http://localhost:6333', client: client);
      final fleetPull = FleetPull(
        http: fleetHttp,
        edgeClient: edgeClient,
        workDir: workDir.path,
      );

      final result = await fleetPull.pull(collection: 'fleet_curated');

      expect(result, isA<PullUnreachable>());
      expect(edgeClient.isLoaded, isTrue);
      expect(await edgeClient.count(), 1, reason: 'the old corpus must survive a failed re-pull');
      final hits = await edgeClient.timeline();
      expect(
        hits.map((h) => h.momentId),
        contains('old-good-moment'),
        reason: 'the old corpus must still be queryable, not just "loaded"',
      );

      await edgeClient.close();
    },
  );

  // Phase 1 review fix E (silent-failure H1/M3): a hub that answers with a
  // genuinely empty shard must be reported distinctly from an unreachable
  // hub, and must NOT be promoted over a good previous corpus. Real snapshot
  // bytes can't be fabricated host-side (package:qdrant_edge exposes no
  // snapshot-CREATE API) — so this drives FleetPull through its injectable
  // `unpackSnapshotFn` seam with a fake EdgeClient standing in for "the
  // staged shard loaded fine but has 0 points".
  test(
    'a validated-but-empty staged shard -> PullEmpty, old corpus (if any) '
    'restored, never promoted',
    () async {
      final liveDir = Directory('${workDir.path}/fleet_shard');
      liveDir.createSync(recursive: true);
      _seedShard(liveDir.path, momentId: 'old-good-moment');

      final fakeEdgeClient = _AlwaysEmptyFakeEdgeClient();
      final client = MockClient((request) async {
        if (request.method == 'POST') {
          return http.Response('{"result":{"name":"snap-empty.snapshot"}}', 200);
        }
        if (request.method == 'GET') {
          return http.Response.bytes([1], 200);
        }
        if (request.method == 'DELETE') {
          return http.Response('', 200);
        }
        return http.Response('unexpected', 500);
      });
      final fleetHttp = FleetHttp(baseUrl: 'http://localhost:6333', client: client);
      final fleetPull = FleetPull(
        http: fleetHttp,
        edgeClient: fakeEdgeClient,
        workDir: workDir.path,
        unpackSnapshotFn: ({required snapshotPath, required targetPath}) {
          // No-op: simulates a successful unpack of a (fictitious) snapshot
          // whose shard has zero points — fakeEdgeClient.count() always
          // reports 0 regardless of what's on disk at targetPath.
        },
      );

      final result = await fleetPull.pull(collection: 'fleet_curated');

      expect(result, isA<PullEmpty>());
      expect(
        fakeEdgeClient.loadedPaths.first,
        contains('fleet_shard_staging'),
        reason: 'must validate the STAGING dir FIRST, before ever touching the live one',
      );
      // The live directory on disk was never promoted-over: still there.
      expect(liveDir.existsSync(), isTrue);
      expect(
        Directory('${workDir.path}/fleet_shard_staging').existsSync(),
        isFalse,
        reason: 'a rejected staging dir must not linger',
      );
    },
  );
}

/// Reports `loadFromDir` was called (recording every path) but always
/// answers `count()` with 0 — stands in for "the staged shard loaded fine
/// but is empty" without needing a real 0-point snapshot archive (see the
/// "validated-but-empty" test's own comment for why that can't be
/// fabricated host-side).
class _AlwaysEmptyFakeEdgeClient extends EdgeClient {
  final List<String> loadedPaths = [];

  @override
  Future<void> loadFromDir(String dir) async {
    loadedPaths.add(dir);
  }

  @override
  Future<int> count() async => 0;

  @override
  Future<void> close() async {}
}

void _seedShard(String dir, {required String momentId}) {
  final config = qe.EdgeConfig(
    vectorData: {
      'clip': qe.VectorDataConfig(size: _clipDim, distance: qe.Distance.cosine),
      'text': qe.VectorDataConfig(size: _textDim, distance: qe.Distance.cosine),
    },
  );
  final shard = qe.EdgeShard.create(path: dir, config: config);
  try {
    shard.update(
      operation: qe.UpdateOperation.upsertPoints(
        points: [
          qe.Point(
            id: qe.UuidPointId('11111111-1111-4111-8111-111111111111'),
            vector: qe.NamedVectorVariant({
              'clip': qe.DenseNamedVector(List<double>.filled(_clipDim, 0.01)),
            }),
            payload: jsonEncode({
              'type': 'frame',
              'moment_id': momentId,
              'timestamp_ms': 1000,
              'label': 'cup',
            }),
          ),
        ],
      ),
    );
    shard.flush();
  } finally {
    shard.unload();
  }
}
