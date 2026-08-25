import 'dart:async';
import 'dart:io';

import 'package:path/path.dart' as p;
import 'package:qdrant_edge/qdrant_edge.dart' show unpackSnapshot;

import 'edge_client.dart';
import 'fleet_http.dart';

/// Snapshot-pull orchestration for the phone's read-only fleet view — the
/// Dart mirror of the glasses' `FleetSync.pull` (Spec A): create a shard
/// snapshot on the hub, download it, unpack it, and open it with
/// [EdgeClient]. The server-side snapshot is deleted in a `finally` on every
/// exit path, mirroring the glasses (snapshots must not accumulate on the
/// hub across repeated pulls).
///
/// Never throws: any failure (unreachable hub, bad snapshot, unpack error)
/// returns null and leaves [workDir] exactly as it was found — the caller
/// falls back to whatever [EdgeClient] already has loaded (fail-soft, same
/// contract the glasses' Spec A enforces).
class FleetPull {
  FleetPull({
    required this.http,
    required this.edgeClient,
    required this.workDir,
  });

  final FleetHttp http;
  final EdgeClient edgeClient;

  /// Directory the intermediate snapshot file and unpacked shard live under
  /// (typically the app's support directory — injected, not looked up here,
  /// so this class stays fakeable/host-testable).
  final String workDir;

  /// Pulls [collection]'s shard [shard] down as a snapshot and opens it via
  /// [edgeClient]. Returns the loaded shard directory on success, or null on
  /// any failure — never throws.
  Future<String?> pull({String collection = 'fleet_curated', int shard = 0}) async {
    final snapFile = File(p.join(workDir, 'fleet_snap.bin'));
    final shardDir = Directory(p.join(workDir, 'fleet_shard'));
    String? snapshotName;
    try {
      snapshotName = await http.createShardSnapshot(collection, shard: shard);

      if (await snapFile.exists()) await snapFile.delete();
      await http.downloadSnapshot(collection, shard, snapshotName, snapFile);

      // unpackSnapshot needs the target dir to EXIST and be EMPTY — it does
      // not create parents (same as the glasses' Kotlin binding).
      if (await shardDir.exists()) await shardDir.delete(recursive: true);
      await shardDir.create(recursive: true);
      unpackSnapshot(snapshotPath: snapFile.path, targetPath: shardDir.path);

      await edgeClient.loadFromDir(shardDir.path);
      return shardDir.path;
    } catch (_) {
      // Clean up a possibly half-unpacked shard dir on every non-success
      // exit; on success this branch is never reached, so the loaded
      // store's on-disk backing survives.
      try {
        if (await shardDir.exists()) await shardDir.delete(recursive: true);
      } catch (_) {
        // best-effort
      }
      return null;
    } finally {
      try {
        if (await snapFile.exists()) await snapFile.delete();
      } catch (_) {
        // best-effort
      }
      // The server-side snapshot is only ever an intermediate — delete it on
      // every path (success, download failure, unpack failure). Best-effort:
      // a failed delete never affects pull()'s outcome.
      final name = snapshotName;
      if (name != null) {
        try {
          await http.deleteSnapshot(collection, shard, name);
        } catch (_) {
          // best-effort
        }
      }
    }
  }
}
