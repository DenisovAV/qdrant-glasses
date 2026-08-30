import 'dart:async';
import 'dart:developer' as developer;
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_gemma/flutter_gemma.dart';
import 'package:flutter_gemma_litertlm/flutter_gemma_litertlm.dart';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';

import 'chat/answerer.dart';
import 'chat/chat_agent.dart';
import 'chat/gemma_answerer.dart';
import 'data/edge_client.dart';
import 'data/fleet_http.dart';
import 'data/fleet_pull.dart';
import 'data/memory_repository.dart';
import 'data/pull_result.dart';
import 'embed/gemma_embeddings_siglip_text.dart';
import 'embed/siglip_text.dart';
import 'query/gemma_query_parser.dart';
import 'query/query_parser.dart';
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
  // Built once the startup pull resolves (in _pullOnStart's finally): the
  // embedder + Gemma loads are async + filesystem-dependent, so the agent can
  // only be assembled after we know which models came up — full agentic RAG
  // (SigLIP + Gemma 4), or a degraded path (FakeQueryParser + StubAnswerer:
  // still searches + shows cards). Null only during the initial spinner.
  ChatAgent? _agent;
  // Held for dispose(): both native sessions must be released like the shard.
  GemmaEmbeddingsSiglipText? _embedder;
  InferenceModel? _gemma;
  // Set the instant dispose() runs. The model loads are async and can finish
  // AFTER a dispose during startup — dispose() only closes what exists at that
  // instant, so each loader re-checks this after its load and closes the just-
  // loaded native resource itself if the widget is already gone (Codex #2).
  bool _disposed = false;
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
    _disposed = true;
    // Fix I (architect + codex): EdgeClient.close() releases the native
    // shard's WAL lock — the AAR's GC finalizer only frees the Dart-side
    // pointer, never calls the native unload() (see EdgeClient.close's own
    // KDoc) — so never calling this meant every AppRoot teardown leaked one.
    // close() is async; fire-and-forget is correct in a synchronous
    // dispose() (there is no "after" to await into once the widget is gone).
    unawaited(_edgeClient.close());
    // The SigLIP ONNX session + the Gemma LiteRT-LM model are the same kind of
    // native resource — release them.
    unawaited(_embedder?.close());
    unawaited(_gemma?.close());
    super.dispose();
  }

  Future<void> _pullOnStart() async {
    SiglipText? embedder;
    try {
      final PullResult result;
      final override = widget.pullOverride;
      if (override != null) {
        result = await override(_edgeClient);
      } else {
        final appDir = await getApplicationSupportDirectory();
        // Bring up the models BEFORE the pull so the chat is fully ready when
        // _ready flips. Both fail-soft: a missing/broken SigLIP model leaves
        // search on the pure-time path; a missing/broken Gemma leaves the chat
        // on the degraded (search + cards + stub) path.
        embedder = await _loadEmbedder(appDir.path);
        await _loadGemma(appDir.path);
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
      if (_disposed) {
        // Disposed mid-startup: the fleet pull may have loaded a shard into
        // _edgeClient AFTER dispose()'s close() already ran. Close it here
        // (idempotent) so the WAL lock never leaks. The embedder/Gemma are
        // handled by their loaders' own _disposed guards above.
        unawaited(_edgeClient.close());
      } else if (mounted) {
        // Assemble the agent now that we know which models came up. A null
        // embedder keeps search on the pure-time branch; a null Gemma (no model,
        // load failure, or the pullOverride test path) uses FakeQueryParser (raw
        // phrase, no LLM filter) + StubAnswerer (no conversational answer) — the
        // node still parses/retrieves and shows cards, just degraded.
        final gemma = _gemma;
        final repository =
            MemoryRepository(edgeClient: _edgeClient, embedder: embedder);
        final QueryParser parser =
            gemma != null ? GemmaQueryParser(gemma) : const FakeQueryParser();
        final Answerer answerer =
            gemma != null ? GemmaAnswerer(gemma) : const StubAnswerer();
        setState(() {
          _agent = ChatAgent(
            parser: parser,
            repository: repository,
            answerer: answerer,
          );
          _ready = true;
        });
      }
    }
  }

  /// Bring up Gemma 4 E2B (LiteRT-LM) from app storage, fail-soft. Null (→ the
  /// degraded FakeQueryParser + StubAnswerer path) when the model isn't present
  /// or won't load. The `.litertlm` lives under `<appSupport>/models/`
  /// (first-run-download in production; dev-pushed — see the mobile-fleet-node
  /// plan). Vision is enabled so the answer can be grounded in thumbnails.
  Future<InferenceModel?> _loadGemma(String appDir) async {
    final modelPath = '$appDir/models/gemma-4-E2B-it.litertlm';
    if (!File(modelPath).existsSync()) {
      developer.log(
        'AppRoot: Gemma model not found at $modelPath — agentic RAG off '
        '(search + cards only)',
        name: 'fleet',
        level: 900,
      );
      return null;
    }
    try {
      await FlutterGemma.initialize(inferenceEngines: const [LiteRtLmEngine()]);
      await FlutterGemma.installModel(
        modelType: ModelType.gemma4,
        fileType: ModelFileType.litertlm,
      ).fromFile(modelPath).install();
      final model = await FlutterGemma.getActiveModel(
        maxTokens: 4096,
        supportImage: true,
      );
      if (_disposed) {
        await model.close(); // disposed mid-load — release, don't retain.
        return null;
      }
      _gemma = model;
      developer.log('AppRoot: Gemma 4 loaded — agentic RAG on', name: 'fleet');
      return model;
    } catch (e) {
      developer.log(
        'AppRoot: Gemma failed to load, falling back to search-only: $e',
        name: 'fleet',
        level: 900,
      );
      return null;
    }
  }

  /// Bring up the real SigLIP-text embedder from app storage, fail-soft.
  /// Returns null (→ the repository's pure-time path) when the model isn't
  /// present or won't load — the phone still browses/time-filters, it just
  /// can't rank semantically. The 270 MB `siglip-text-int8.onnx` + its
  /// tokenizer live under `<appSupport>/models/` (first-run-download in
  /// production; `adb push`ed there for dev — see the mobile-fleet-node plan).
  Future<SiglipText?> _loadEmbedder(String appDir) async {
    final modelPath = '$appDir/models/siglip-text-int8.onnx';
    final tokenizerPath = '$appDir/models/siglip-tokenizer.json';
    if (!File(modelPath).existsSync() || !File(tokenizerPath).existsSync()) {
      developer.log(
        'AppRoot: SigLIP model not found at $modelPath — semantic search off, '
        'pure-time only',
        name: 'fleet',
        level: 900,
      );
      return null;
    }
    try {
      final embedder = GemmaEmbeddingsSiglipText(
        modelPath: modelPath,
        tokenizerPath: tokenizerPath,
      );
      await embedder.load();
      if (_disposed) {
        await embedder.close(); // disposed mid-load — release, don't retain.
        return null;
      }
      _embedder = embedder;
      developer.log('AppRoot: SigLIP embedder loaded — semantic search on',
          name: 'fleet');
      return embedder;
    } catch (e) {
      developer.log(
        'AppRoot: SigLIP embedder failed to load, falling back to pure-time: $e',
        name: 'fleet',
        level: 900,
      );
      return null;
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
    final agent = _agent;
    if (!_ready || agent == null) {
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
        Expanded(child: ChatScreen(agent: agent)),
      ],
    );
  }
}
