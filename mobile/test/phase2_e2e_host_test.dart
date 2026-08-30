// Phase 2 GATE on the HOST (macOS) — no phone. Proves the whole vector-search
// path: real SigLIP embedder → clip-space corpus in a real EdgeShard →
// EdgeClient.searchFrames ranks + time-filters. Skips if the model is absent.
import 'dart:convert';
import 'dart:io';

import 'package:fleet_node/data/edge_client.dart';
import 'package:fleet_node/embed/gemma_embeddings_siglip_text.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:qdrant_edge/qdrant_edge.dart' as qe;

void main() {
  const modelPath = '../app/src/main/assets/siglip-text-int8.onnx';
  const tokenizerPath = '../app/src/main/assets/siglip-tokenizer.json';

  test('real embed → Edge search → ranked + time-filtered, on host', () async {
    if (!File(modelPath).existsSync() || !File(tokenizerPath).existsSync()) {
      markTestSkipped('SigLIP model not present locally — skipped');
      return;
    }

    final tmp = Directory.systemTemp.createTempSync('fleet_e2e_');
    addTearDown(() => tmp.deleteSync(recursive: true));
    final dir = tmp.path;

    final embedder = GemmaEmbeddingsSiglipText(
      modelPath: modelPath,
      tokenizerPath: tokenizerPath,
    );
    try {
      await embedder.load();
    } catch (e) {
      markTestSkipped(
        'ONNX Runtime not available on this host ($e) — set '
        'FLUTTER_GEMMA_ORT_LIBRARY to a libonnxruntime.dylib to run; skipped',
      );
      return;
    }

    // Build a tiny clip-space corpus directly via the SDK (EdgeClient is read-only).
    final corpus = <String, int>{
      'a red coffee mug on a desk': 1000,
      'a blue bicycle leaning on a wall': 2000,
      'a golden retriever in a park': 3000,
      'a laptop with code on the screen': 4000,
    };
    final config = qe.EdgeConfig(
      vectorData: {
        'clip': qe.VectorDataConfig(size: 768, distance: qe.Distance.cosine),
        'text': qe.VectorDataConfig(size: 384, distance: qe.Distance.cosine),
      },
    );
    final shard = qe.EdgeShard.load(path: dir, config: config);
    var i = 0;
    for (final entry in corpus.entries) {
      final vec = await embedder.encode(entry.key);
      i++;
      shard.update(
        operation: qe.UpdateOperation.upsertPoints(
          points: [
            qe.Point(
              id: qe.NumIdPointId(i),
              vector: qe.NamedVectorVariant({
                'clip': qe.DenseNamedVector(vec.toList()),
              }),
              payload: jsonEncode({
                'type': 'frame',
                'moment_id': 'm$i',
                'timestamp_ms': entry.value,
                'label': entry.key,
              }),
            ),
          ],
        ),
      );
    }
    shard.flush();
    shard.unload();

    final client = EdgeClient();
    await client.loadFromDir(dir);
    expect(await client.count(), 4, reason: 'all 4 frames persisted + reopened');

    // Query with one stored phrase — identical embedding → cosine ~1 → must rank #1.
    const target = 'a golden retriever in a park';
    final q = await embedder.encode(target);
    final ranked = await client.searchFrames(clip: q, topK: 4);

    // Time window [2500,3500] keeps only the ts=3000 frame (the retriever).
    final windowed =
        await client.searchFrames(clip: q, topK: 4, sinceMs: 2500, untilMs: 3500);

    await embedder.close();
    await client.close();

    expect(ranked, isNotEmpty);
    expect(ranked.first.label, target,
        reason: 'exact-phrase query ranks its own frame #1');
    expect(ranked.first.score, greaterThan(0.99),
        reason: 'identical embedding → cosine ~1');

    expect(windowed.map((h) => h.label).toList(), [target],
        reason: 'time filter keeps only the in-window frame');

    // ignore: avoid_print
    print('E2E ranked: ${ranked.map((h) => "${h.label.split(' ').take(2).join('_')}=${h.score.toStringAsFixed(3)}").toList()}');
  }, timeout: const Timeout(Duration(minutes: 4)));
}
