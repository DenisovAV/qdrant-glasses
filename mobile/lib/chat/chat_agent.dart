// Public constructor param names (parser/repository/answerer) map to private
// fields — a deliberate DI style, same as EdgeClient/GemmaQueryParser.
// ignore_for_file: prefer_initializing_formals
import '../data/edge_client.dart';
import '../data/memory_repository.dart';
import '../query/parsed_query.dart';
import '../query/query_parser.dart';
import '../ui/chat_message.dart';
import 'answerer.dart';

/// Orchestrates one agentic RAG turn: parse the natural-language question into a
/// [ParsedQuery] filter, retrieve matching moments, then stream a conversational
/// answer over them. Emits progressive assistant [ChatMessage]s (growing [text],
/// the same [hits] carried throughout so the cards render immediately).
///
/// Fail-soft by construction — the plan's optional/offline contract. A parser,
/// retriever, or LLM failure never propagates: the turn still yields at least
/// one assistant message carrying whatever hits were found (with a stub summary
/// when the LLM produced no text). `ask` never throws.
class ChatAgent {
  ChatAgent({
    required QueryParser parser,
    required MemoryRepository repository,
    required Answerer answerer,
    DateTime Function()? now,
  })  : _parser = parser,
        _repository = repository,
        _answerer = answerer,
        _now = now ?? DateTime.now;

  final QueryParser _parser;
  final MemoryRepository _repository;
  final Answerer _answerer;
  final DateTime Function() _now;

  Stream<ChatMessage> ask(String nl) async* {
    // 1 · understand — degrade to the raw phrase (no filter) on any failure.
    ParsedQuery pq;
    try {
      pq = await _parser.parse(nl, now: _now());
    } catch (_) {
      pq = ParsedQuery(phrase: nl);
    }

    // 2 · retrieve — the repository is itself fail-soft, but never let a stray
    // throw escape the turn.
    List<MomentHit> hits;
    try {
      hits = await _repository.search(pq);
    } catch (_) {
      hits = const [];
    }

    // 3 · answer — stream tokens, growing the assistant text; carry the hits on
    // every emission so cards show while the answer is still being written.
    var text = '';
    try {
      await for (final chunk in _answerer.answer(nl, hits)) {
        text += chunk;
        yield ChatMessage(role: ChatRole.assistant, text: text, hits: hits);
      }
    } catch (_) {
      // swallow — the guarantee below still produces an assistant turn.
    }

    // 4 · guarantee — if the LLM produced nothing usable, still answer with a
    // stub over the hits so the turn is never empty.
    if (text.trim().isEmpty) {
      yield ChatMessage(role: ChatRole.assistant, text: _stub(hits), hits: hits);
    }
  }

  String _stub(List<MomentHit> hits) =>
      hits.isEmpty ? 'Ничего не нашёл.' : 'Нашёл ${hits.length}.';
}
