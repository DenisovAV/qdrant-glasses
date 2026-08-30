// _maxImages: public-param → private-field DI style (see chat_agent.dart).
// ignore_for_file: prefer_initializing_formals
import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter_gemma/flutter_gemma.dart';

import '../data/edge_client.dart';
import 'answerer.dart';

/// The real [Answerer]: hands Gemma 4 the retrieved moments' thumbnails plus a
/// label/time summary and streams a conversational answer over the actual photo
/// content. Falls back to a text-only answer (labels/times, no images) when the
/// corpus carries no thumbnails yet (Phase 4 `thumb_b64` not pulled, or a
/// text-only moment), so the chat still works before thumbnails land.
///
/// One short-lived session per turn (the model is reused — see the desktop
/// model-lifecycle docs). Fail-soft is [ChatAgent]'s job: this may throw and the
/// turn still degrades to a stub over the hits.
class GemmaAnswerer implements Answerer {
  GemmaAnswerer(this._model, {int maxImages = 4}) : _maxImages = maxImages;

  final InferenceModel _model;
  final int _maxImages;

  @override
  Stream<String> answer(String query, List<MomentHit> hits) async* {
    final images = _thumbnails(hits);
    final session = await _model.createSession(
      temperature: 0.7,
      enableVisionModality: images.isNotEmpty,
      maxOutputTokens: 512,
      systemInstruction:
          "You are the wearer's memory assistant. Answer their question "
          'conversationally and briefly, in the same language they used, '
          'grounded in the retrieved moments'
          "${images.isNotEmpty ? ' and their photos' : ''}. "
          'If nothing relevant was found, say so plainly.',
    );
    try {
      final message = images.isEmpty
          ? Message.text(text: _prompt(query, hits, hasPhotos: false), isUser: true)
          : Message.withImages(
              text: _prompt(query, hits, hasPhotos: true),
              imageBytes: images,
              isUser: true,
            );
      await session.addQueryChunk(message);
      yield* session.getResponseAsync();
    } finally {
      await session.close();
    }
  }

  /// Decoded thumbnails for the first [_maxImages] hits that carry one. A
  /// corrupt base64 is skipped, never fatal.
  List<Uint8List> _thumbnails(List<MomentHit> hits) {
    final out = <Uint8List>[];
    for (final h in hits) {
      final b64 = h.thumbB64;
      if (b64 == null || b64.isEmpty) continue;
      try {
        out.add(base64Decode(b64));
      } catch (_) {
        continue;
      }
      if (out.length >= _maxImages) break;
    }
    return out;
  }

  String _prompt(String query, List<MomentHit> hits, {required bool hasPhotos}) {
    if (hits.isEmpty) {
      return 'The user asked: "$query". Nothing was found in their memory for '
          'this — tell them so, briefly.';
    }
    final lines = hits.take(8).map((h) {
      final when = DateTime.fromMillisecondsSinceEpoch(h.timestampMs);
      final label = h.label.isEmpty ? 'a moment' : h.label;
      return '- $label (${_date(when)})';
    }).join('\n');
    return 'The user asked: "$query".\n'
        'Here are ${hits.length} moment(s) from their visual memory'
        '${hasPhotos ? ', with photos attached' : ''}:\n$lines\n'
        'Answer their question conversationally, grounded in '
        '${hasPhotos ? 'the attached photos' : 'these moments'}.';
  }

  static String _date(DateTime d) =>
      '${d.year}-${d.month.toString().padLeft(2, '0')}-'
      '${d.day.toString().padLeft(2, '0')} '
      '${d.hour.toString().padLeft(2, '0')}:${d.minute.toString().padLeft(2, '0')}';
}
