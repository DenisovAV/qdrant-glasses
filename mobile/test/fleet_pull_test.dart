// Task 3: FleetPull's fail-soft orchestration (create -> download -> unpack
// -> load, delete the server-side snapshot in a finally). Host-runnable:
// exercises real failure paths (a mock HTTP layer, real qdrant_edge native
// calls for unpack) without needing a live fleet hub — the true success path
// (a real fleet_curated snapshot) is covered on-device by
// integration_test/fleet_pull_test.dart.
import 'dart:io';

import 'package:fleet_node/data/edge_client.dart';
import 'package:fleet_node/data/fleet_http.dart';
import 'package:fleet_node/data/fleet_pull.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';

void main() {
  late Directory workDir;

  setUp(() {
    workDir = Directory.systemTemp.createTempSync('fleet_pull_test_');
  });

  tearDown(() {
    if (workDir.existsSync()) workDir.deleteSync(recursive: true);
  });

  test('unreachable hub (create fails) -> pull() returns null, never throws', () async {
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

    expect(result, isNull);
    expect(edgeClient.isLoaded, isFalse);
    // No leftover work-dir artifacts on a create-time failure.
    expect(File('${workDir.path}/fleet_snap.bin').existsSync(), isFalse);
    expect(Directory('${workDir.path}/fleet_shard').existsSync(), isFalse);
  });

  test('create succeeds but download 404s -> pull() returns null; server '
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

    expect(result, isNull);
    expect(edgeClient.isLoaded, isFalse);
    expect(deleteRequests, hasLength(1));
    expect(deleteRequests.single.url.toString(), contains('snap-1.snapshot'));
    // The intermediate snapshot file must not linger.
    expect(File('${workDir.path}/fleet_snap.bin').existsSync(), isFalse);
  });

  test('download succeeds but the bytes are not a real snapshot -> unpack '
      'fails, pull() returns null, no orphan shard dir', () async {
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

    expect(result, isNull);
    expect(edgeClient.isLoaded, isFalse);
    expect(Directory('${workDir.path}/fleet_shard').existsSync(), isFalse);
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

    expect(result, isNull);
  });
}
