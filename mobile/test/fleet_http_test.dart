// Task 3: FleetHttp — REST to the fleet hub, mirrors the glasses'
// FleetQdrantClient.kt (create/download/delete a shard snapshot). Unit-tested
// against a mock server; no live hub, no native calls.
import 'dart:convert';
import 'dart:io';

import 'package:fleet_node/data/fleet_http.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';

void main() {
  group('createShardSnapshot', () {
    test('POSTs the snapshot endpoint and parses result.name', () async {
      http.Request? seen;
      final client = MockClient((request) async {
        seen = request;
        return http.Response(
          jsonEncode({
            'result': {'name': 'snap-123.snapshot'},
          }),
          200,
        );
      });
      final fleetHttp = FleetHttp(baseUrl: 'http://localhost:6333', client: client);

      final name = await fleetHttp.createShardSnapshot('fleet_curated');

      expect(name, 'snap-123.snapshot');
      expect(seen!.method, 'POST');
      expect(
        seen!.url.toString(),
        'http://localhost:6333/collections/fleet_curated/shards/0/snapshots',
      );
    });

    test('non-200 throws FleetHttpException', () async {
      final client = MockClient((request) async => http.Response('boom', 500));
      final fleetHttp = FleetHttp(baseUrl: 'http://localhost:6333', client: client);

      expect(
        () => fleetHttp.createShardSnapshot('fleet_curated'),
        throwsA(isA<FleetHttpException>()),
      );
    });

    test('malformed body throws FleetHttpException', () async {
      final client = MockClient((request) async => http.Response('not json', 200));
      final fleetHttp = FleetHttp(baseUrl: 'http://localhost:6333', client: client);

      expect(
        () => fleetHttp.createShardSnapshot('fleet_curated'),
        throwsA(isA<FleetHttpException>()),
      );
    });
  });

  group('downloadSnapshot', () {
    test('GETs the named snapshot and writes its bytes to dest', () async {
      final bytes = [1, 2, 3, 4, 5];
      http.Request? seen;
      final client = MockClient((request) async {
        seen = request;
        return http.Response.bytes(bytes, 200);
      });
      final fleetHttp = FleetHttp(baseUrl: 'http://localhost:6333', client: client);
      final dest = File(
        '${Directory.systemTemp.createTempSync('fleet_http_test_').path}/snap.bin',
      );

      await fleetHttp.downloadSnapshot('fleet_curated', 0, 'snap-123.snapshot', dest);

      expect(dest.readAsBytesSync(), bytes);
      expect(
        seen!.url.toString(),
        'http://localhost:6333/collections/fleet_curated/shards/0/snapshots/snap-123.snapshot',
      );
    });

    test('non-200 throws FleetHttpException, no file written', () async {
      final client = MockClient((request) async => http.Response('nope', 404));
      final fleetHttp = FleetHttp(baseUrl: 'http://localhost:6333', client: client);
      final dest = File(
        '${Directory.systemTemp.createTempSync('fleet_http_test_').path}/snap.bin',
      );

      await expectLater(
        fleetHttp.downloadSnapshot('fleet_curated', 0, 'missing', dest),
        throwsA(isA<FleetHttpException>()),
      );
      expect(dest.existsSync(), isFalse);
    });
  });

  group('deleteSnapshot', () {
    test('DELETEs the named snapshot', () async {
      http.Request? seen;
      final client = MockClient((request) async {
        seen = request;
        return http.Response('', 200);
      });
      final fleetHttp = FleetHttp(baseUrl: 'http://localhost:6333', client: client);

      await fleetHttp.deleteSnapshot('fleet_curated', 0, 'snap-123.snapshot');

      expect(seen!.method, 'DELETE');
      expect(
        seen!.url.toString(),
        'http://localhost:6333/collections/fleet_curated/shards/0/snapshots/snap-123.snapshot',
      );
    });

    test('non-200 throws FleetHttpException', () async {
      final client = MockClient((request) async => http.Response('boom', 500));
      final fleetHttp = FleetHttp(baseUrl: 'http://localhost:6333', client: client);

      expect(
        () => fleetHttp.deleteSnapshot('fleet_curated', 0, 'snap-123.snapshot'),
        throwsA(isA<FleetHttpException>()),
      );
    });
  });
}
