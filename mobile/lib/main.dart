import 'dart:async';
import 'dart:developer' as developer;

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';

import 'data/edge_client.dart';
import 'data/fleet_http.dart';
import 'data/fleet_pull.dart';
import 'data/memory_repository.dart';
import 'data/pull_result.dart';
import 'ui/chat_screen.dart';

void main() {
  runApp(const FleetNodeApp());
}

class FleetNodeApp extends StatelessWidget {
  const FleetNodeApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Qdrant Fleet Node',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
      ),
      home: const AppRoot(),
    );
  }
}

/// One fleet pull against [baseUrl], via [edgeClient], into [workDir] — the
/// concrete production path behind [AppRoot.pullOverride]'s test seam.
/// Extracted to a top-level function (rather than inlined in
/// [_AppRootState._pullOnStart]) so the FleetHttp-close half of fix I's
/// fail-soft contract is unit-testable without a full widget pump — see
/// `test/main_test.dart`.
Future<PullResult> runFleetPull({
  required EdgeClient edgeClient,
  required String workDir,
  String baseUrl = 'http://localhost:6333',
  http.Client? httpClient,
}) async {
  final fleetHttp = FleetHttp(baseUrl: baseUrl, client: httpClient);
  try {
    final fleetPull = FleetPull(http: fleetHttp, edgeClient: edgeClient, workDir: workDir);
    return await fleetPull.pull();
  } finally {
    // Fix I (all reviewers): FleetHttp holds an http.Client (a keep-alive
    // connection pool) that nothing ever closed — every startup pull leaked
    // one. FleetPull.pull()'s own return paths (including exceptions it
    // catches internally) all funnel through this finally either way.
    fleetHttp.close();
  }
}

/// Pulls the fleet corpus once on start and hands a ready [MemoryRepository]
/// to [ChatScreen]. Fail-soft by construction: [runFleetPull] never throws
/// (mirrors [FleetPull.pull]'s own contract) — an unreachable hub (no dev
/// tunnel, hub down, offline) or an empty pulled corpus just means the chat
/// opens with an empty timeline, surfaced via a dismissible banner (fix E)
/// instead of looking identical to "there's simply nothing to see yet".
class AppRoot extends StatefulWidget {
  const AppRoot({super.key, this.pullOverride});

  /// Test seam: when set, called INSTEAD of the real
  /// `getApplicationSupportDirectory` + [runFleetPull] chain, so a widget
  /// test can drive [_AppRootState]'s ready/banner/dispose logic
  /// deterministically without a real filesystem or network call.
  /// Production (`main()`) never sets this.
  final Future<PullResult> Function(EdgeClient edgeClient)? pullOverride;

  @override
  State<AppRoot> createState() => _AppRootState();
}

class _AppRootState extends State<AppRoot> {
  final EdgeClient _edgeClient = EdgeClient();
  late final MemoryRepository _repository = MemoryRepository(edgeClient: _edgeClient);
  bool _ready = false;
  bool _bannerDismissed = false;
  PullResult? _pullResult;

  @override
  void initState() {
    super.initState();
    _pullOnStart();
  }

  @override
  void dispose() {
    // Fix I (architect + codex): EdgeClient.close() releases the native
    // shard's WAL lock — the AAR's GC finalizer only frees the Dart-side
    // pointer, never calls the native unload() (see EdgeClient.close's own
    // KDoc) — so never calling this meant every AppRoot teardown leaked one.
    // close() is async; fire-and-forget is correct in a synchronous
    // dispose() (there is no "after" to await into once the widget is gone).
    unawaited(_edgeClient.close());
    super.dispose();
  }

  Future<void> _pullOnStart() async {
    try {
      final PullResult result;
      final override = widget.pullOverride;
      if (override != null) {
        result = await override(_edgeClient);
      } else {
        final appDir = await getApplicationSupportDirectory();
        result = await runFleetPull(edgeClient: _edgeClient, workDir: appDir.path);
      }
      if (mounted) setState(() => _pullResult = result);
    } catch (e) {
      // Fail-soft: the chat still opens, just with whatever (nothing) is
      // loaded — getApplicationSupportDirectory() itself could in principle
      // throw on an unsupported platform, which must not crash startup.
      // Fix F: still logged, not silent.
      developer.log('AppRoot: startup pull failed: $e', name: 'fleet', level: 900);
      // Round-2 review fix #8 (silent-failure, low): the old body left
      // `_pullResult == null` here, which `_bannerMessage` treats
      // identically to "never attempted a pull" — no banner at all, hiding
      // a genuine startup failure behind silence. Surfacing it as
      // `PullUnreachable` reuses the SAME banner an unreachable hub already
      // gets (this failure mode never even reached the hub, but "the fleet
      // corpus isn't loaded, here's why" is the right message either way).
      if (mounted) setState(() => _pullResult = PullUnreachable('$e'));
    } finally {
      if (mounted) setState(() => _ready = true);
    }
  }

  /// Fix E (silent-failure H1/M3): distinguishes "hub down" from "hub
  /// reachable but the corpus is empty" — two outcomes that must never look
  /// identical to the user. Null when the pull loaded a non-empty corpus,
  /// or the banner was dismissed.
  String? get _bannerMessage {
    if (_bannerDismissed) return null;
    return switch (_pullResult) {
      PullUnreachable(:final message) => 'Хаб флота недоступен: $message',
      PullEmpty() => 'Хаб флота доступен, но память пока пуста — ни одного момента.',
      PullLoaded() || null => null,
    };
  }

  @override
  Widget build(BuildContext context) {
    if (!_ready) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }
    final message = _bannerMessage;
    return Column(
      children: [
        if (message != null)
          MaterialBanner(
            key: const Key('pull_status_banner'),
            content: Text(message, key: const Key('pull_status_banner_text')),
            actions: [
              TextButton(
                key: const Key('dismiss_pull_banner'),
                onPressed: () => setState(() => _bannerDismissed = true),
                child: const Text('OK'),
              ),
            ],
          ),
        Expanded(child: ChatScreen(repository: _repository)),
      ],
    );
  }
}
