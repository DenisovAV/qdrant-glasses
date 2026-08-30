// Phase 1 review fixes E (honest pull state -> a banner distinct from
// "empty") + I (AppRoot disposes EdgeClient; runFleetPull closes FleetHttp).
//
// AppRoot's `pullOverride` constructor param is a test-only seam (production
// `main()` never sets it) — it stands in for the real
// getApplicationSupportDirectory() + runFleetPull() chain so these tests are
// deterministic and don't touch the filesystem/network.
import 'dart:io';

import 'package:fleet_node/data/edge_client.dart';
import 'package:fleet_node/data/pull_result.dart';
import 'package:fleet_node/main.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:qdrant_edge/qdrant_edge.dart' as qe;

const _clipDim = 768;
const _textDim = 384;

/// AppRoot is always hosted inside FleetNodeApp's MaterialApp in production
/// (which is what supplies Directionality/Navigator/etc.) — mirrors that
/// here instead of pumping it bare.
Future<void> _pump(WidgetTester tester, Widget appRoot) {
  return tester.pumpWidget(MaterialApp(home: appRoot));
}

void main() {
  testWidgets(
    'an unreachable-hub pull shows a dismissible banner naming the failure',
    (tester) async {
      await _pump(
        tester,
        AppRoot(pullOverride: (edgeClient) async => const PullUnreachable('connection refused')),
      );
      await tester.pumpAndSettle();

      expect(find.byKey(const Key('pull_status_banner')), findsOneWidget);
      expect(find.textContaining('connection refused'), findsOneWidget);

      await tester.tap(find.byKey(const Key('dismiss_pull_banner')));
      await tester.pump();

      expect(find.byKey(const Key('pull_status_banner')), findsNothing);
    },
  );

  testWidgets(
    'a validated-but-empty pull shows a banner, with DIFFERENT text than '
    'unreachable — "hub down" must never look identical to "memory is empty"',
    (tester) async {
      await _pump(tester, AppRoot(pullOverride: (edgeClient) async => const PullEmpty()));
      await tester.pumpAndSettle();

      expect(find.byKey(const Key('pull_status_banner')), findsOneWidget);
      final bannerText = tester
          .widget<Text>(find.byKey(const Key('pull_status_banner_text')))
          .data;
      expect(bannerText, isNot(contains('connection refused')));
      expect(bannerText, isNot(contains('недоступен')), reason: 'must not read as "unreachable"');
    },
  );

  testWidgets('a successful non-empty pull shows no banner at all', (tester) async {
    await _pump(
      tester,
      AppRoot(pullOverride: (edgeClient) async => const PullLoaded(count: 3, dir: '/tmp/x')),
    );
    await tester.pumpAndSettle();

    expect(find.byKey(const Key('pull_status_banner')), findsNothing);
  });

  // Round-2 review fix #8 (silent-failure, low): `pullOverride`/
  // `runFleetPull` THROWING (as opposed to resolving to a `PullResult`) is a
  // real, if rare, path — e.g. `getApplicationSupportDirectory()` itself
  // throwing on an unsupported platform. The old catch just logged and left
  // `_pullResult == null`, which `_bannerMessage` treats identically to
  // "never attempted a pull" — no banner at all, silently hiding a genuine
  // startup failure.
  testWidgets(
    'the pull throwing outright (not resolving to a PullResult) still shows '
    'the unreachable banner, not silence (round-2 review fix #8)',
    (tester) async {
      await _pump(
        tester,
        AppRoot(pullOverride: (edgeClient) async => throw Exception('boom-directory-failure')),
      );
      await tester.pumpAndSettle();

      expect(find.byKey(const Key('pull_status_banner')), findsOneWidget);
      expect(find.textContaining('boom-directory-failure'), findsOneWidget);
    },
  );

  testWidgets(
    'disposing AppRoot closes its EdgeClient (fix I: no leaked native shard)',
    (tester) async {
      final tmp = Directory.systemTemp.createTempSync('app_root_dispose_test_');
      final shardDir = '${tmp.path}/shard';
      Directory(shardDir).createSync(recursive: true);
      _seedShard(shardDir);

      late EdgeClient captured;
      await _pump(
        tester,
        AppRoot(
          pullOverride: (edgeClient) async {
            captured = edgeClient;
            await edgeClient.loadFromDir(shardDir);
            return PullLoaded(count: await edgeClient.count() ?? 0, dir: shardDir);
          },
        ),
      );
      await tester.pumpAndSettle();
      expect(captured.isLoaded, isTrue);

      // Replace the whole widget tree -> disposes AppRoot's State.
      await tester.pumpWidget(const SizedBox.shrink());
      await tester.pump();

      expect(captured.isLoaded, isFalse, reason: 'dispose() must have called EdgeClient.close()');

      tmp.deleteSync(recursive: true);
    },
  );

  test('runFleetPull closes the FleetHttp client even when the pull fails', () async {
    final mock = MockClient((request) async => http.Response('boom', 500));
    final tracking = _CloseTrackingClient(mock);
    final edgeClient = EdgeClient();
    final workDir = Directory.systemTemp.createTempSync('run_fleet_pull_test_');

    final result = await runFleetPull(
      edgeClient: edgeClient,
      workDir: workDir.path,
      httpClient: tracking,
    );

    expect(result, isA<PullUnreachable>());
    expect(tracking.closed, isTrue, reason: 'fix I: FleetHttp/its http.Client must be closed after use');

    workDir.deleteSync(recursive: true);
  });
}

/// Wraps a [http.Client] to observe whether [close] was ever called —
/// `http.testing.MockClient`'s own `close()` is a silent no-op, so it can't
/// tell this test whether [runFleetPull] actually closed the `FleetHttp`
/// it built internally.
class _CloseTrackingClient extends http.BaseClient {
  _CloseTrackingClient(this._inner);

  final http.Client _inner;
  bool closed = false;

  @override
  Future<http.StreamedResponse> send(http.BaseRequest request) => _inner.send(request);

  @override
  void close() {
    closed = true;
    _inner.close();
  }
}

void _seedShard(String dir) {
  final config = qe.EdgeConfig(
    vectorData: {
      'clip': qe.VectorDataConfig(size: _clipDim, distance: qe.Distance.cosine),
      'text': qe.VectorDataConfig(size: _textDim, distance: qe.Distance.cosine),
    },
  );
  final shard = qe.EdgeShard.create(path: dir, config: config);
  shard.flush();
  shard.unload();
}
