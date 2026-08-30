import '../data/edge_client.dart';

/// Who spoke a chat turn.
enum ChatRole { user, assistant }

/// One turn in the chat thread. A user turn is text only; an assistant turn
/// carries [text] (Phase 1: a stub summary, e.g. "нашёл 3"; Phase 3: Gemma
/// 4's conversational answer) plus the retrieved [hits], rendered inline as
/// [MomentCard]s below the text.
class ChatMessage {
  const ChatMessage({required this.role, this.text, this.hits = const []});

  final ChatRole role;
  final String? text;
  final List<MomentHit> hits;
}
