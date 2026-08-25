import 'dart:convert';
import 'dart:io';

import 'package:http/http.dart' as http;

/// Raised by every [FleetHttp] method on a non-2xx response or an
/// unparseable body. Callers (chiefly [FleetPull]) catch this and degrade —
/// the fleet hub is optional, never a crash source.
class FleetHttpException implements Exception {
  final String message;
  const FleetHttpException(this.message);

  @override
  String toString() => 'FleetHttpException: $message';
}

/// Thin REST client to the private Qdrant fleet hub — the Dart mirror of the
/// glasses' `FleetQdrantClient.kt` (same three endpoints: create/download/
/// delete a shard snapshot). Read-only on the phone: no `upsertPoints`/upload
/// half — the phone VIEWS the fleet, it never contributes to it (Spec's
/// read-only constraint).
class FleetHttp {
  FleetHttp({required this.baseUrl, http.Client? client})
    : _client = client ?? http.Client();

  final String baseUrl;
  final http.Client _client;

  Uri _snapshotUri(String collection, int shard, [String? name]) {
    final path = 'collections/$collection/shards/$shard/snapshots'
        '${name == null ? '' : '/$name'}';
    return Uri.parse('$baseUrl/$path');
  }

  /// POST create a shard snapshot; returns the snapshot file name.
  Future<String> createShardSnapshot(String collection, {int shard = 0}) async {
    final resp = await _client.post(_snapshotUri(collection, shard));
    if (resp.statusCode != 200) {
      throw FleetHttpException(
        'snapshot create ${resp.statusCode}: ${resp.body}',
      );
    }
    final Object? decoded;
    try {
      decoded = jsonDecode(resp.body);
    } on FormatException catch (e) {
      throw FleetHttpException('snapshot create: malformed response body: $e');
    }
    final result = decoded is Map<String, dynamic> ? decoded['result'] : null;
    final name = result is Map<String, dynamic> ? result['name'] : null;
    if (name is! String || name.isEmpty) {
      throw FleetHttpException(
        'snapshot create: missing result.name in response: ${resp.body}',
      );
    }
    return name;
  }

  /// GET the snapshot bytes for [name] and writes them to [dest].
  Future<void> downloadSnapshot(
    String collection,
    int shard,
    String name,
    File dest,
  ) async {
    final resp = await _client.get(_snapshotUri(collection, shard, name));
    if (resp.statusCode != 200) {
      throw FleetHttpException('snapshot download ${resp.statusCode}');
    }
    await dest.writeAsBytes(resp.bodyBytes);
  }

  /// DELETE the server-side shard snapshot [name] — cleanup after
  /// [downloadSnapshot] pulls it locally, so snapshots don't accumulate on
  /// the fleet hub across every [FleetPull.pull]. Callers soft-fail this.
  Future<void> deleteSnapshot(String collection, int shard, String name) async {
    final resp = await _client.delete(_snapshotUri(collection, shard, name));
    if (resp.statusCode != 200) {
      throw FleetHttpException('snapshot delete ${resp.statusCode}');
    }
  }

  void close() => _client.close();
}
