import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/material.dart';

import '../data/edge_client.dart';

/// One retrieved moment, shown inline in the chat thread. Renders the
/// decoded [MomentHit.thumbB64] image when present (Phase 4 corpora); falls
/// back to a small text card (label + formatted local time) otherwise — a
/// pull from before Phase 4 has no thumbnails at all, and a malformed/
/// truncated base64 string must never crash the card either (fail-soft).
class MomentCard extends StatelessWidget {
  const MomentCard({super.key, required this.hit});

  final MomentHit hit;

  @override
  Widget build(BuildContext context) {
    final thumb = _decodeThumb(hit.thumbB64);
    final label = hit.label.isEmpty ? '—' : hit.label;
    return Container(
      key: Key('moment_card_${hit.id}'),
      width: 120,
      margin: const EdgeInsets.only(right: 8),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Theme.of(context).dividerColor),
      ),
      clipBehavior: Clip.antiAlias,
      child: thumb == null
          ? _TextCard(label: label, timestampMs: hit.timestampMs)
          : Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              mainAxisSize: MainAxisSize.min,
              children: [
                AspectRatio(
                  aspectRatio: 1,
                  child: Image.memory(thumb, fit: BoxFit.cover),
                ),
                Padding(
                  padding: const EdgeInsets.all(4),
                  child: Text(
                    label,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: Theme.of(context).textTheme.labelSmall,
                  ),
                ),
              ],
            ),
    );
  }
}

class _TextCard extends StatelessWidget {
  const _TextCard({required this.label, required this.timestampMs});

  final String label;
  final int timestampMs;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            label,
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
            style: Theme.of(context).textTheme.bodyMedium,
          ),
          const SizedBox(height: 4),
          Text(
            _formatTimestamp(timestampMs),
            style: Theme.of(context).textTheme.labelSmall,
          ),
        ],
      ),
    );
  }
}

/// Decodes a base64 thumbnail. Returns null (never throws) for a null,
/// empty, or malformed string — the caller falls back to the text card.
Uint8List? _decodeThumb(String? b64) {
  if (b64 == null || b64.isEmpty) return null;
  try {
    return base64Decode(b64);
  } catch (_) {
    return null;
  }
}

String _formatTimestamp(int timestampMs) {
  if (timestampMs <= 0) return '';
  final dt = DateTime.fromMillisecondsSinceEpoch(timestampMs);
  String two(int n) => n.toString().padLeft(2, '0');
  return '${dt.year}-${two(dt.month)}-${two(dt.day)} ${two(dt.hour)}:${two(dt.minute)}';
}
