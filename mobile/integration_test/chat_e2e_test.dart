// PHASE 1 GATE, on-device: pull fleet_curated for real, drive ChatScreen
// through a real send, and assert the assistant turn renders inline
// MomentCards sourced from the pulled corpus.
//
// Prerequisite (NOT done by this test): `adb -s emulator-5554 reverse
// tcp:6333 tcp:6333` so `http://localhost:6333` on-device reaches the host
// Qdrant hub, with `fleet_curated` populated (8 points, per the current dev
// stand).
//
// Run: fvm flutter test integration_test/chat_e2e_test.dart -d emulator-5554
import 'package:fleet_node/data/edge_client.dart';
import 'package:fleet_node/data/fleet_http.dart';
import 'package:fleet_node/data/fleet_pull.dart';
import 'package:fleet_node/data/memory_repository.dart';
import 'package:fleet_node/ui/chat_screen.dart';
import 'package:fleet_node/ui/moment_card.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:path_provider/path_provider.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('a chat send returns pulled fleet_curated moments as inline cards', (
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

    final shardDir = await fleetPull.pull(collection: 'fleet_curated');
    expect(shardDir, isNotNull, reason: 'pull() should succeed against a reachable hub');
    final corpusCount = await edgeClient.count();
    // ignore: avoid_print
    print('chat_e2e: pulled corpus count=$corpusCount');
    expect(corpusCount, greaterThan(0));

    final repository = MemoryRepository(edgeClient: edgeClient);

    await tester.pumpWidget(
      MaterialApp(home: ChatScreen(repository: repository)),
    );
    expect(find.byKey(const Key('empty_state')), findsOneWidget);

    await tester.enterText(find.byKey(const Key('chat_input')), 'что я видел?');
    await tester.tap(find.byKey(const Key('send_button')));
    await tester.pump();
    // The search is a real async EdgeClient scroll — settle until it resolves.
    await tester.pumpAndSettle();

    expect(find.byKey(const Key('empty_state')), findsNothing);
    expect(find.text('что я видел?'), findsOneWidget);
    expect(find.byKey(const Key('search_spinner')), findsNothing);

    final cards = find.byType(MomentCard);
    // ignore: avoid_print
    print('chat_e2e: rendered ${tester.widgetList(cards).length} inline MomentCard(s)');
    expect(cards, findsWidgets);
  });
}
