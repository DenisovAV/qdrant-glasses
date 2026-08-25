import 'package:flutter/material.dart';

import '../data/memory_repository.dart';
import '../query/parsed_query.dart';
import 'chat_message.dart';
import 'message_bubble.dart';

/// The mobile fleet node's chat: a message thread plus an input box. Phase 1:
/// a send appends the user turn, calls [MemoryRepository.search] directly
/// with the raw phrase (no filter, no embedder yet), and appends an
/// assistant turn carrying a stub summary + the retrieved hits as inline
/// cards. Phase 3 (Task 10) inserts a `ChatAgent` between the input and the
/// repository without touching this widget's send-button wiring.
class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key, required this.repository});

  final MemoryRepository repository;

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final List<ChatMessage> _messages = [];
  final TextEditingController _controller = TextEditingController();
  final ScrollController _scrollController = ScrollController();
  bool _isSearching = false;

  @override
  void dispose() {
    _controller.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  Future<void> _send() async {
    final raw = _controller.text.trim();
    if (raw.isEmpty || _isSearching) return;
    _controller.clear();
    setState(() {
      _messages.add(ChatMessage(role: ChatRole.user, text: raw));
      _isSearching = true;
    });

    // Fail-soft by construction: MemoryRepository/EdgeClient never throw —
    // an unreachable hub or an empty corpus just means zero hits below, not
    // a crashed chat turn.
    final hits = await widget.repository.search(ParsedQuery(phrase: raw));

    if (!mounted) return;
    setState(() {
      _isSearching = false;
      _messages.add(
        ChatMessage(
          role: ChatRole.assistant,
          text: hits.isEmpty ? 'Ничего не нашёл.' : 'Нашёл ${hits.length}.',
          hits: hits,
        ),
      );
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Fleet Node')),
      body: SafeArea(
        child: Column(
          children: [
            Expanded(
              child: _messages.isEmpty
                  ? const Center(
                      key: Key('empty_state'),
                      child: Padding(
                        padding: EdgeInsets.all(24),
                        child: Text(
                          'Пока пусто — спросите что-нибудь про увиденное.',
                          textAlign: TextAlign.center,
                        ),
                      ),
                    )
                  : ListView.builder(
                      key: const Key('message_list'),
                      controller: _scrollController,
                      padding: const EdgeInsets.symmetric(vertical: 8),
                      itemCount: _messages.length,
                      itemBuilder: (context, i) =>
                          MessageBubble(message: _messages[i]),
                    ),
            ),
            if (_isSearching)
              const Padding(
                padding: EdgeInsets.symmetric(vertical: 8),
                child: SizedBox(
                  key: Key('search_spinner'),
                  height: 20,
                  width: 20,
                  child: CircularProgressIndicator(strokeWidth: 2),
                ),
              ),
            Padding(
              padding: const EdgeInsets.all(8),
              child: Row(
                children: [
                  Expanded(
                    child: TextField(
                      key: const Key('chat_input'),
                      controller: _controller,
                      decoration: const InputDecoration(
                        hintText: 'Спросить про память...',
                      ),
                      onSubmitted: (_) => _send(),
                    ),
                  ),
                  IconButton(
                    key: const Key('send_button'),
                    icon: const Icon(Icons.send),
                    onPressed: _send,
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
