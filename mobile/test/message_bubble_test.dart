import 'package:fleet_node/data/edge_client.dart';
import 'package:fleet_node/ui/chat_message.dart';
import 'package:fleet_node/ui/message_bubble.dart';
import 'package:fleet_node/ui/moment_card.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

Future<void> _pump(WidgetTester tester, Widget child) {
  return tester.pumpWidget(MaterialApp(home: Scaffold(body: child)));
}

MomentHit _hit(String id) =>
    MomentHit(id: id, score: 0, momentId: id, timestampMs: 1700000000000, label: 'cup');

void main() {
  testWidgets('a user turn renders its text, no cards', (tester) async {
    await _pump(
      tester,
      const MessageBubble(message: ChatMessage(role: ChatRole.user, text: 'привет')),
    );

    expect(find.text('привет'), findsOneWidget);
    expect(find.byType(MomentCard), findsNothing);
  });

  testWidgets('an assistant turn renders its text and N inline MomentCards', (tester) async {
    await _pump(
      tester,
      MessageBubble(
        message: ChatMessage(
          role: ChatRole.assistant,
          text: 'нашёл 3',
          hits: [_hit('a'), _hit('b'), _hit('c')],
        ),
      ),
    );

    expect(find.text('нашёл 3'), findsOneWidget);
    expect(find.byType(MomentCard), findsNWidgets(3));
  });

  testWidgets('an assistant turn with zero hits renders just the text', (tester) async {
    await _pump(
      tester,
      const MessageBubble(
        message: ChatMessage(role: ChatRole.assistant, text: 'ничего не нашёл'),
      ),
    );

    expect(find.text('ничего не нашёл'), findsOneWidget);
    expect(find.byType(MomentCard), findsNothing);
  });
}
