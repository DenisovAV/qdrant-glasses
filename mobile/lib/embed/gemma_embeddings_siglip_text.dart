import 'dart:typed_data';

import 'package:flutter_gemma_embeddings/embedding_tokenizer.dart';
import 'package:flutter_gemma_embeddings/flutter_gemma_embeddings.dart';
// flutter_gemma_onnx does not export OnnxEmbeddingForwardPass/OrtFfiClient publicly (only the
// backend facade), so we reach into src/ — the SAME path PR #467's own parity test uses to drive the
// forward pass directly, since the EmbeddingModel router has no SigLIP profile selector yet.
// ignore: implementation_imports
import 'package:flutter_gemma_onnx/src/embedding/onnx_embedding_forward_pass.dart';
// ignore: implementation_imports
import 'package:flutter_gemma_onnx/src/embedding/ort_ffi_client.dart';

import '../logging.dart';
import 'siglip_text.dart';

/// The real [SiglipText]: the SigLIP2 text tower, pure-Dart, over flutter_gemma's ONNX embedding
/// engine (PRs DenisovAV/flutter_gemma#467 + brody-0125/dart_sentencepiece_tokenizer#26; see memory
/// `siglip-embedding-prs`). No Kotlin bridge — every platform flutter_gemma_onnx supports.
///
/// It drives [OnnxEmbeddingForwardPass] directly with the SigLIP tokenizer profile, the SAME path the
/// PR's own parity test uses (the EmbeddingModel router does not yet carry a profile selector). That
/// path reproduced the glasses' native SigLIP output at cosine ≈0.97 on-device.
///
/// Three things make SigLIP work where the plain ONNX seam corrupted it:
///  - `loadSiglipSentencePieceEmbeddingTokenizer`: no BOS, single EOS, lowercased, right-padded to 64;
///  - the `pooler_output` picker fix: `pooledFinal` (copied verbatim), NOT mean-pooled hidden states;
///  - the dart_sentencepiece merges-format fix: parses SigLIP's `tokenizer.json`.
///
/// Warm singleton: [load] once (opens the 270 MB int8 session — seconds), then [encode] per query.
class GemmaEmbeddingsSiglipText implements SiglipText {
  GemmaEmbeddingsSiglipText({required this.modelPath, required this.tokenizerPath});

  /// On-disk path to `siglip-text-int8.onnx` (270 MB). First-run-downloaded to app storage in
  /// production ([[no-relay-fully-on-device]]); adb-pushed on the emulator for dev.
  final String modelPath;

  /// On-disk path to `siglip-tokenizer.json`.
  final String tokenizerPath;

  EmbeddingTokenizer? _tokenizer;
  OnnxEmbeddingForwardPass? _pass;
  bool _loaded = false;

  /// Open the ONNX session + tokenizer once. Idempotent. Throws if the model/tokenizer can't load —
  /// the caller decides whether to fall back to the pure-time path (a null embedder in the repository).
  Future<void> load() async {
    if (_loaded) return;
    _tokenizer = await loadSiglipSentencePieceEmbeddingTokenizer(tokenizerPath);
    final pass = OnnxEmbeddingForwardPass(modelPath, clientFactory: OrtFfiClient.new);
    await pass.load();
    if (pass.outputContract != EmbeddingOutputContract.pooledFinal) {
      // SigLIP's pooler_output MUST resolve to pooledFinal (copied verbatim). If the resolved package
      // is missing the picker fix, tokenLevel would silently mean-pool → a wrong vector. Fail loud.
      await pass.close();
      throw StateError(
        'SigLIP model resolved to ${pass.outputContract} (expected pooledFinal) — '
        'the pooler_output picker fix is missing from flutter_gemma_onnx.',
      );
    }
    _pass = pass;
    _loaded = true;
    fleetLog('SiglipText loaded: dim=${pass.outputDimension}, contract=${pass.outputContract}');
  }

  @override
  Future<Float32List> encode(String phrase) async {
    if (!_loaded) await load();
    final tokenizer = _tokenizer!;
    final pass = _pass!;
    final tokenized = tokenizer.encode('', phrase);
    final result = await pass.run(
      tokenIds: tokenized.ids,
      attentionMask: tokenized.attentionMask,
      tokenTypeIds: tokenized.tokenTypeIds,
    );
    // pooledFinal (asserted at load): the model's pooler_output IS the embedding — copy verbatim,
    // never re-pool. (The tokenLevel branch is here only to mirror the worker's real dispatch.)
    final List<double> vec = switch (pass.outputContract!) {
      EmbeddingOutputContract.pooledFinal => List<double>.of(result.values),
      EmbeddingOutputContract.tokenLevel => meanPoolAndNormalize(
          result,
          attentionMask: result.attentionMask ?? tokenized.attentionMask,
        ),
    };
    return Float32List.fromList(vec);
  }

  /// Release the native ONNX session. Safe to call more than once.
  Future<void> close() async {
    final pass = _pass;
    _pass = null;
    _tokenizer = null;
    _loaded = false;
    if (pass != null) await pass.close();
  }
}
