import 'package:flutter/material.dart';

import '../chat/chat_agent.dart';
import 'chat_message.dart';
import 'message_bubble.dart';

/// The mobile fleet node's chat: a message thread plus an input box. A send
/// appends the user turn and drives one [ChatAgent.ask] turn — the agent
/// parses the filter, retrieves, and streams a conversational answer. The
/// assistant turn is appended once and then replaced in place as the stream
/// grows (its text fills in, its retrieved [ChatMessage.hits] render as inline
/// cards throughout). [ChatAgent.ask] is fail-soft and never throws.
class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key, required this.agent});

  final ChatAgent agent;

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
    _scrollToBottom();

    // One RAG turn, streamed. The agent is fail-soft (never throws); the first
    // emission appends the assistant turn, later emissions replace it in place
    // as the answer grows.
    int? assistantIndex;
    try {
      await for (final turn in widget.agent.ask(raw)) {
        if (!mounted) return;
        setState(() {
          if (assistantIndex == null) {
            _messages.add(turn);
            assistantIndex = _messages.length - 1;
          } else {
            _messages[assistantIndex!] = turn;
          }
        });
        _scrollToBottom();
      }
    } finally {
      if (mounted) setState(() => _isSearching = false);
    }
  }

  /// Fix H (code-reviewer): `_scrollController` was wired to the `ListView`
  /// but nothing ever drove it — new turns appended below the fold with no
  /// way to see them without a manual scroll. Runs after the frame that
  /// laid out the newly-appended turn (so `maxScrollExtent` already
  /// reflects it), not inside `setState` itself.
  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!_scrollController.hasClients) return;
      _scrollController.animateTo(
        _scrollController.position.maxScrollExtent,
        duration: const Duration(milliseconds: 200),
        curve: Curves.easeOut,
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
