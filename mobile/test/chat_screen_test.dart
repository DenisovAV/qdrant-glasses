import 'dart:async';

import 'package:fleet_node/data/edge_client.dart';
import 'package:fleet_node/data/memory_repository.dart';
import 'package:fleet_node/query/parsed_query.dart';
import 'package:fleet_node/ui/chat_screen.dart';
import 'package:fleet_node/ui/moment_card.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

/// A MemoryRepository whose search() only resolves when the test tells it
/// to — lets a test observe the in-flight (spinner) state deterministically.
class FakeMemoryRepository extends MemoryRepository {
  FakeMemoryRepository() : super(edgeClient: EdgeClient());

  Completer<List<MomentHit>>? pendingSearch;
  ParsedQuery? lastQuery;
  int searchCallCount = 0;

  @override
  Future<List<MomentHit>> search(ParsedQuery query, {int limit = 50}) {
    lastQuery = query;
    searchCallCount++;
    final completer = Completer<List<MomentHit>>();
    pendingSearch = completer;
    return completer.future;
  }
}

MomentHit _hit(String id) =>
    MomentHit(id: id, score: 0, momentId: id, timestampMs: 1700000000000, label: 'cup');

Future<void> _typeAndSend(WidgetTester tester, String text) async {
  await tester.enterText(find.byKey(const Key('chat_input')), text);
  await tester.tap(find.byKey(const Key('send_button')));
  await tester.pump();
}

void main() {
  testWidgets('empty state renders before any turn', (tester) async {
    final repo = FakeMemoryRepository();

    await tester.pumpWidget(MaterialApp(home: ChatScreen(repository: repo)));

    expect(find.byKey(const Key('empty_state')), findsOneWidget);
    expect(find.byKey(const Key('message_list')), findsNothing);
  });

  testWidgets('sending appends the user turn immediately and shows a spinner '
      'while the search is in flight', (tester) async {
    final repo = FakeMemoryRepository();
    await tester.pumpWidget(MaterialApp(home: ChatScreen(repository: repo)));

    await _typeAndSend(tester, 'чашка');

    expect(find.text('чашка'), findsOneWidget);
    expect(find.byKey(const Key('search_spinner')), findsOneWidget);
    expect(repo.lastQuery?.phrase, 'чашка');

    // Resolve so the pending timer/future doesn't leak into the next test.
    repo.pendingSearch!.complete(const []);
    await tester.pumpAndSettle();
  });

  testWidgets('a resolved search with hits appends an assistant turn with '
      'inline MomentCards and hides the spinner', (tester) async {
    final repo = FakeMemoryRepository();
    await tester.pumpWidget(MaterialApp(home: ChatScreen(repository: repo)));

    await _typeAndSend(tester, 'чашка');
    repo.pendingSearch!.complete([_hit('a'), _hit('b')]);
    await tester.pumpAndSettle();

    expect(find.byKey(const Key('search_spinner')), findsNothing);
    expect(find.byType(MomentCard), findsNWidgets(2));
  });

  testWidgets('a resolved search with zero hits appends an assistant turn '
      'with no cards, not an error', (tester) async {
    final repo = FakeMemoryRepository();
    await tester.pumpWidget(MaterialApp(home: ChatScreen(repository: repo)));

    await _typeAndSend(tester, 'зонтик');
    repo.pendingSearch!.complete(const []);
    await tester.pumpAndSettle();

    expect(find.byKey(const Key('search_spinner')), findsNothing);
    expect(find.byType(MomentCard), findsNothing);
    // The user turn plus one assistant turn — thread is non-empty now.
    expect(find.byKey(const Key('message_list')), findsOneWidget);
  });

  testWidgets(
    'the thread auto-scrolls to the bottom as turns are appended, so new '
    'turns are not stranded below the fold',
    (tester) async {
      await tester.binding.setSurfaceSize(const Size(400, 300));
      addTearDown(() => tester.binding.setSurfaceSize(null));
      final repo = FakeMemoryRepository();
      await tester.pumpWidget(MaterialApp(home: ChatScreen(repository: repo)));

      // Enough turns (each with 2 inline MomentCards) to overflow the small
      // viewport above.
      for (var i = 0; i < 6; i++) {
        await _typeAndSend(tester, 'вопрос $i');
        repo.pendingSearch!.complete([_hit('a$i'), _hit('b$i')]);
        await tester.pumpAndSettle();
      }

      final listView = tester.widget<ListView>(find.byKey(const Key('message_list')));
      final controller = listView.controller!;
      expect(
        controller.position.maxScrollExtent,
        greaterThan(0),
        reason: 'the test setup must actually overflow the viewport, or this assertion is vacuous',
      );
      expect(controller.offset, moreOrLessEquals(controller.position.maxScrollExtent, epsilon: 1.0));
    },
  );

  testWidgets('blank input is not sent', (tester) async {
    final repo = FakeMemoryRepository();
    await tester.pumpWidget(MaterialApp(home: ChatScreen(repository: repo)));

    await _typeAndSend(tester, '   ');

    expect(repo.searchCallCount, 0);
    expect(find.byKey(const Key('empty_state')), findsOneWidget);
  });
}
