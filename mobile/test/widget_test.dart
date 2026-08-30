// Smoke test: the app composes and renders its first frame without
// crashing. Deliberately does not await the fleet pull's real network
// attempt (getApplicationSupportDirectory + HTTP to localhost:6333) —
// _AppRoot shows a loading spinner until that resolves either way, and this
// test only needs to prove that first frame is reachable.
import 'package:fleet_node/main.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  testWidgets('FleetNodeApp builds its first frame without crashing', (
    tester,
  ) async {
    await tester.pumpWidget(const FleetNodeApp());
    await tester.pump();

    expect(find.byType(MaterialApp), findsOneWidget);
    expect(tester.takeException(), isNull);
  });
}
