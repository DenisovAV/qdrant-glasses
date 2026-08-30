// Phase 3 Task 10 GATE on macOS: the full agentic RAG turn end-to-end with a
// REAL Gemma 4 (query parse + conversational answer) over a REAL Edge corpus.
// Retrieval uses the pure-time path (no embedder), so this needs only Gemma —
// the focus is ChatAgent orchestration + Gemma answering over the hits.
//   flutter test integration_test/chat_agent_macos_it.dart -d macos
import 'dart:convert';
import 'dart:io';

import 'package:fleet_node/chat/chat_agent.dart';
import 'package:fleet_node/chat/gemma_answerer.dart';
import 'package:fleet_node/data/edge_client.dart';
import 'package:fleet_node/data/memory_repository.dart';
import 'package:fleet_node/query/gemma_query_parser.dart';
import 'package:fleet_node/ui/chat_message.dart';
import 'package:flutter_gemma/flutter_gemma.dart';
import 'package:flutter_gemma_litertlm/flutter_gemma_litertlm.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:qdrant_edge/qdrant_edge.dart' as qe;

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('ChatAgent → Gemma answers conversationally over retrieved moments',
      (tester) async {
    const modelPath = '/Users/sashadenisov/Downloads/gemma-4-E2B-it.litertlm';
    if (!File(modelPath).existsSync()) {
      markTestSkipped('Gemma 4 model not present — skipped');
      return;
    }

    await FlutterGemma.initialize(inferenceEngines: const [LiteRtLmEngine()]);
    await FlutterGemma.installModel(
      modelType: ModelType.gemma4,
      fileType: ModelFileType.litertlm,
    ).fromFile(modelPath).install();
    final model = await FlutterGemma.getActiveModel(maxTokens: 4096, supportImage: true);

    // A tiny recent corpus (dummy clip vectors — retrieval is pure-time here).
    final base = DateTime(2026, 9, 5, 12).millisecondsSinceEpoch;
    final frames = <(String, int)>[
      ('red coffee mug', base),
      ('blue bicycle', base + 3600000),
      ('golden retriever', base + 7200000),
    ];
    final tmp = Directory.systemTemp.createTempSync('chat_e2e_');
    addTearDown(() => tmp.deleteSync(recursive: true));
    final shard = qe.EdgeShard.load(
      path: tmp.path,
      config: qe.EdgeConfig(vectorData: {
        'clip': qe.VectorDataConfig(size: 768, distance: qe.Distance.cosine),
        'text': qe.VectorDataConfig(size: 384, distance: qe.Distance.cosine),
      }),
    );
    var i = 0;
    for (final (label, ts) in frames) {
      i++;
      shard.update(
        operation: qe.UpdateOperation.upsertPoints(points: [
          qe.Point(
            id: qe.NumIdPointId(i),
            vector: qe.NamedVectorVariant(
                {'clip': qe.DenseNamedVector(List<double>.filled(768, 0))}),
            payload: jsonEncode({
              'type': 'frame',
              'moment_id': 'm$i',
              'timestamp_ms': ts,
              'label': label,
            }),
          ),
        ]),
      );
    }
    shard.flush();
    shard.unload();

    final edge = EdgeClient();
    await edge.loadFromDir(tmp.path);
    final agent = ChatAgent(
      parser: GemmaQueryParser(model),
      repository: MemoryRepository(edgeClient: edge), // pure-time (no embedder)
      answerer: GemmaAnswerer(model),
    );

    final turns = await agent.ask('what did I see?').toList();

    await edge.close();
    await model.close();

    expect(turns, isNotEmpty);
    final last = turns.last;
    expect(last.role, ChatRole.assistant);
    expect((last.text ?? '').trim(), isNotEmpty,
        reason: 'Gemma produced a conversational answer');
    // ignore: avoid_print
    print('RAG answer: ${last.text}');
    // ignore: avoid_print
    print('RAG hits: ${last.hits.map((h) => h.label).toList()}');
  }, timeout: const Timeout(Duration(minutes: 8)));
}
