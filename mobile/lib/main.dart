import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';

import 'data/edge_client.dart';
import 'data/fleet_http.dart';
import 'data/fleet_pull.dart';
import 'data/memory_repository.dart';
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
      home: const _AppRoot(),
    );
  }
}

/// Pulls the fleet corpus once on start and hands a ready [MemoryRepository]
/// to [ChatScreen]. Fail-soft by construction: [FleetPull.pull] never
/// throws, so an unreachable hub (no dev tunnel, hub down, offline) just
/// means the chat opens with an empty timeline — never a crash.
class _AppRoot extends StatefulWidget {
  const _AppRoot();

  @override
  State<_AppRoot> createState() => _AppRootState();
}

class _AppRootState extends State<_AppRoot> {
  final EdgeClient _edgeClient = EdgeClient();
  late final MemoryRepository _repository = MemoryRepository(
    edgeClient: _edgeClient,
  );
  bool _ready = false;

  @override
  void initState() {
    super.initState();
    _pullOnStart();
  }

  Future<void> _pullOnStart() async {
    try {
      final appDir = await getApplicationSupportDirectory();
      final fleetHttp = FleetHttp(baseUrl: 'http://localhost:6333');
      final fleetPull = FleetPull(
        http: fleetHttp,
        edgeClient: _edgeClient,
        workDir: appDir.path,
      );
      await fleetPull.pull();
    } catch (_) {
      // Fail-soft: the chat still opens, just with whatever (nothing) is
      // loaded — getApplicationSupportDirectory() itself could in principle
      // throw on an unsupported platform, which must not crash startup.
    } finally {
      if (mounted) setState(() => _ready = true);
    }
  }

  @override
  Widget build(BuildContext context) {
    if (!_ready) {
      return const Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
    }
    return ChatScreen(repository: _repository);
  }
}
