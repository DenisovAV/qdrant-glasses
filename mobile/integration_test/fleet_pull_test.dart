// Task 3, on-device: FleetPull.pull() against the REAL fleet hub (mirrors
// S1's spike, but through the production FleetHttp/FleetPull/EdgeClient
// classes instead of ad hoc test code).
//
// Prerequisite (NOT done by this test): `adb -s emulator-5554 reverse
// tcp:6333 tcp:6333` so `http://localhost:6333` on-device reaches the host
// Qdrant hub, and the hub's `fleet_curated` collection has points (8, per
// the current dev stand).
//
// Run: fvm flutter test integration_test/fleet_pull_test.dart -d emulator-5554
import 'package:fleet_node/data/edge_client.dart';
import 'package:fleet_node/data/fleet_http.dart';
import 'package:fleet_node/data/fleet_pull.dart';
import 'package:fleet_node/data/pull_result.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:path_provider/path_provider.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('pull() downloads+unpacks+loads fleet_curated; EdgeClient count > 0', (
    tester,
  ) async {
    final appDir = await getApplicationSupportDirectory();
    final fleetHttp = FleetHttp(baseUrl: 'http://localhost:6333');
    final edgeClient = EdgeClient();
    final fleetPull = FleetPull(
      http: fleetHttp,
      edgeClient: edgeClient,
      workDir: appDir.path,
    );

    final result = await fleetPull.pull(collection: 'fleet_curated');

    expect(result, isA<PullLoaded>(), reason: 'pull() should succeed against a reachable hub');
    expect(edgeClient.isLoaded, isTrue);
    final count = await edgeClient.count();
    // ignore: avoid_print
    print('fleet_pull integration: $result, count=$count');
    expect(count, greaterThan(0));

    final hits = await edgeClient.timeline(limit: 20);
    // ignore: avoid_print
    print('fleet_pull integration: timeline() returned ${hits.length} frame(s)');
    for (final hit in hits) {
      // ignore: avoid_print
      print('  $hit');
    }
  });

  testWidgets('unreachable hub -> PullUnreachable, EdgeClient stays unloaded', (
    tester,
  ) async {
    final appDir = await getApplicationSupportDirectory();
    // Port 1 is not the fleet hub — connection should fail fast.
    final fleetHttp = FleetHttp(baseUrl: 'http://localhost:1');
    final edgeClient = EdgeClient();
    final fleetPull = FleetPull(
      http: fleetHttp,
      edgeClient: edgeClient,
      workDir: appDir.path,
    );

    final result = await fleetPull.pull(collection: 'fleet_curated');

    expect(result, isA<PullUnreachable>());
    expect(edgeClient.isLoaded, isFalse);
  });
}
