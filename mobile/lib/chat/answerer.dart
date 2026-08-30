import '../data/edge_client.dart';

/// Produces the conversational answer for one RAG turn over the retrieved
/// [hits], streamed token-by-token. The real implementation ([GemmaAnswerer])
/// hands Gemma 4 the moments' thumbnails + a label/time summary and answers
/// over the actual photo content; a fake drives [ChatAgent] tests.
///
/// Implementations may throw/timeout — [ChatAgent] still guarantees an
/// assistant turn (a stub summary over the hits) if this fails, so an answerer
/// never has to be defensive about producing *something*.
abstract class Answerer {
  Stream<String> answer(String query, List<MomentHit> hits);
}

/// A no-LLM answerer: emits nothing, so [ChatAgent] falls back to its stub
/// summary over the hits. Used when no Gemma model is available — the node
/// still parses/retrieves and shows the moment cards, just without a
/// conversational answer.
class StubAnswerer implements Answerer {
  const StubAnswerer();

  @override
  Stream<String> answer(String query, List<MomentHit> hits) =>
      const Stream.empty();
}
