// Runs the REAL SigLIP embedder against the local int8 model on the HOST (macOS) — no phone.
// Verifies flutter_gemma_onnx loads the model and produces a 768-d clip-space vector.
// Skips cleanly if the model file isn't present (it's gitignored, local-only), so CI stays green.
import 'dart:io';
import 'dart:math' as math;

import 'package:fleet_node/embed/gemma_embeddings_siglip_text.dart';
import 'package:flutter_test/flutter_test.dart';

double _l2(List<double> v) {
  var s = 0.0;
  for (final x in v) {
    s += x * x;
  }
  return math.sqrt(s);
}

void main() {
  // The 270 MB int8 model lives in the glasses app assets (gitignored). flutter test's CWD is the
  // package root (mobile/), so ../app/... reaches it.
  const modelPath = '../app/src/main/assets/siglip-text-int8.onnx';
  const tokenizerPath = '../app/src/main/assets/siglip-tokenizer.json';

  test('real SigLIP embedder loads + encodes on the host', () async {
    if (!File(modelPath).existsSync() || !File(tokenizerPath).existsSync()) {
      markTestSkipped('SigLIP model not present locally ($modelPath) — skipped');
      return;
    }
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
    final v = await embedder.encode('a red coffee mug on a wooden desk');
    await embedder.close();

    expect(v, hasLength(768), reason: 'SigLIP text tower is 768-d');
    expect(v.every((x) => x.isFinite), isTrue, reason: 'no NaN/Inf');
    final norm = _l2(v.toList());
    expect(norm, greaterThan(0.0));
    // ignore: avoid_print
    print('SigLIP encode OK: dim=${v.length}, L2=$norm, head=${v.sublist(0, 4)}');
  }, timeout: const Timeout(Duration(minutes: 3)));
}
