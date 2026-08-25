import 'dart:convert';

import 'package:fleet_node/data/edge_client.dart';
import 'package:fleet_node/ui/moment_card.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

Future<void> _pump(WidgetTester tester, Widget child) {
  return tester.pumpWidget(MaterialApp(home: Scaffold(body: child)));
}

// A minimal, valid 1x1 PNG — enough for Image.memory to decode without a
// real asset bundle.
const _tinyPngBase64 =
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=';

void main() {
  testWidgets('renders a decoded image when thumbB64 is present', (tester) async {
    const hit = MomentHit(
      id: 'id1',
      score: 0,
      momentId: 'm1',
      timestampMs: 1700000000000,
      label: 'cup',
      thumbB64: _tinyPngBase64,
    );

    await _pump(tester, const MomentCard(hit: hit));

    expect(find.byType(Image), findsOneWidget);
    final image = tester.widget<Image>(find.byType(Image));
    expect(image.image, isA<MemoryImage>());
    expect((image.image as MemoryImage).bytes, base64Decode(_tinyPngBase64));
  });

  testWidgets('falls back to label + time text when thumbB64 is absent', (tester) async {
    const hit = MomentHit(
      id: 'id1',
      score: 0,
      momentId: 'm1',
      timestampMs: 1700000000000,
      label: 'cup',
    );

    await _pump(tester, const MomentCard(hit: hit));

    expect(find.byType(Image), findsNothing);
    expect(find.text('cup'), findsOneWidget);
  });

  testWidgets('malformed base64 never throws — falls back to text', (tester) async {
    const hit = MomentHit(
      id: 'id1',
      score: 0,
      momentId: 'm1',
      timestampMs: 1700000000000,
      label: 'plant',
      thumbB64: 'not-valid-base64!!!',
    );

    await _pump(tester, const MomentCard(hit: hit));

    expect(find.byType(Image), findsNothing);
    expect(find.text('plant'), findsOneWidget);
  });

  testWidgets('an empty label shows a placeholder, not a blank card', (tester) async {
    const hit = MomentHit(
      id: 'id1',
      score: 0,
      momentId: 'm1',
      timestampMs: 1700000000000,
      label: '',
    );

    await _pump(tester, const MomentCard(hit: hit));

    expect(find.text('—'), findsOneWidget);
  });
}
