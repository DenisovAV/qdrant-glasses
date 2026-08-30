import 'dart:typed_data';

/// The query-embedding seam: turns a natural-language phrase into a 768-d,
/// L2-normalized vector in the SAME `clip` space as the pulled corpus (SigLIP2
/// text tower). [MemoryRepository]'s vector search path depends on THIS
/// interface, not a concrete engine — so the real on-device embedder
/// (`GemmaEmbeddingsSiglipText`, Phase 2 Step 2, over flutter_gemma's ONNX
/// SigLIP profile) drops in behind it without the repository or the chat UI
/// changing at all.
abstract class SiglipText {
  /// Embed [phrase] into a 768-d L2-normalized vector. MUST land in the same
  /// space as `EdgeClient.clipField` (768-d, cosine) or search is meaningless.
  Future<Float32List> encode(String phrase);
}

/// A deterministic, dependency-free fake for repository/UI tests: returns a
/// fixed 768-d unit vector (`e_0` = `[1, 0, 0, …]`) regardless of the phrase.
/// Tests that use it assert WIRING (was the phrase embedded, was the vector
/// passed to `EdgeClient.searchFrames` with the right filters?), not semantics
/// — so a constant, L2-normalized vector is exactly what they need. No model,
/// no I/O.
class FakeSiglipText implements SiglipText {
  const FakeSiglipText({this.dim = 768});

  /// Vector length. Defaults to the corpus's `clip` dimension (768).
  final int dim;

  @override
  Future<Float32List> encode(String phrase) async {
    final v = Float32List(dim);
    if (dim > 0) v[0] = 1.0; // e_0 — a fixed unit vector (‖v‖ = 1).
    return v;
  }
}
