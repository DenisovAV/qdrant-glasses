import 'dart:async';
import 'dart:io';

import 'package:path/path.dart' as p;
import 'package:qdrant_edge/qdrant_edge.dart' as qe;

import '../logging.dart';
import 'edge_client.dart';
import 'fleet_http.dart';
import 'pull_result.dart';

/// The shape of `package:qdrant_edge`'s top-level `unpackSnapshot` —
/// factored to a typedef so [FleetPull] can take it as an injectable
/// constructor param (default: the real function). This is what lets a host
/// test drive [FleetPull.pull]'s stage/validate/promote logic (fixes D+E)
/// for the "validated but empty" branch: nothing on the Dart side can
/// fabricate a real snapshot archive's bytes (`package:qdrant_edge` exposes
/// no snapshot-CREATE API, only unpack), so a test that needs `unpackSnapshot`
/// to just "succeed" substitutes a no-op here instead.
typedef UnpackSnapshotFn = void Function({
  required String snapshotPath,
  required String targetPath,
});

/// Snapshot-pull orchestration for the phone's read-only fleet view — the
/// Dart mirror of the glasses' `FleetSync.pull` (Spec A): create a shard
/// snapshot on the hub, download it, unpack it into a STAGING directory,
/// validate it, and only then promote it over the live corpus.
///
/// **Never destroys a good corpus on a failed refresh (Phase 1 review fix
/// D, codex HIGH).** The old body downloaded straight into the LIVE
/// `fleet_shard` directory, deleting it first (`unpackSnapshot` requires an
/// empty target) — so a hub blip or a corrupt/empty snapshot wiped the
/// previous, perfectly good corpus before the replacement was ever
/// validated. This one stages into a SEPARATE `fleet_shard_staging`
/// directory, calls [EdgeClient.loadFromDir] + `count()` on THAT, and only
/// deletes/replaces `fleet_shard` once `count() > 0`. On any failure the
/// live directory is never touched by this class directly; if the live
/// handle WAS already swapped onto staging for validation (past the point
/// [EdgeClient.loadFromDir]'s own fix-C `close()` ran), [_restoreLiveOrClose]
/// reloads it from its still-intact on-disk copy.
///
/// Returns a [PullResult] (Phase 1 review fix E, silent-failure H1/M3), not
/// a bare `String?`: a caller that only checked "non-null" could not tell an
/// unreachable hub apart from a hub that answered with a genuinely empty
/// shard — two outcomes a user-facing banner must describe differently (see
/// `AppRoot` in `main.dart`).
///
/// Never throws: any failure (unreachable hub, bad snapshot, unpack error,
/// zero-point shard) resolves to [PullUnreachable]/[PullEmpty], not an
/// exception — the caller falls back to whatever [EdgeClient] already has
/// loaded (fail-soft, same contract the glasses' Spec A enforces).
class FleetPull {
  FleetPull({
    required this.http,
    required this.edgeClient,
    required this.workDir,
    UnpackSnapshotFn? unpackSnapshotFn,
  }) : _unpackSnapshot = unpackSnapshotFn ?? qe.unpackSnapshot;

  final FleetHttp http;
  final EdgeClient edgeClient;

  /// Directory the intermediate snapshot file, staging shard, and live shard
  /// live under (typically the app's support directory — injected, not
  /// looked up here, so this class stays fakeable/host-testable).
  final String workDir;

  final UnpackSnapshotFn _unpackSnapshot;

  /// Pulls [collection]'s shard [shard] down as a snapshot, stages +
  /// validates it, and — only on success — promotes it over the live
  /// corpus. See the class doc for the stage/validate/promote shape.
  Future<PullResult> pull({String collection = 'fleet_curated', int shard = 0}) async {
    final snapFile = File(p.join(workDir, 'fleet_snap.bin'));
    final liveDir = Directory(p.join(workDir, 'fleet_shard'));
    final stagingDir = Directory(p.join(workDir, 'fleet_shard_staging'));
    String? snapshotName;
    // Flips true only once EdgeClient's currently-loaded (good) shard has
    // actually been unloaded to validate staging (EdgeClient.loadFromDir's
    // own fix-C `close()`) — the exact point past which a failure must
    // restore the live corpus rather than just walk away (before that
    // point, the live handle was never touched, so there's nothing to
    // restore).
    var liveHandleDisturbed = false;
    try {
      snapshotName = await http.createShardSnapshot(collection, shard: shard);

      if (await snapFile.exists()) await snapFile.delete();
      await http.downloadSnapshot(collection, shard, snapshotName, snapFile);

      // unpackSnapshot needs the target dir to EXIST and be EMPTY — it does
      // not create parents (same as the glasses' Kotlin binding). Staging,
      // NOT `liveDir`: the live corpus must survive everything up to and
      // including a successful validation below.
      if (await stagingDir.exists()) await stagingDir.delete(recursive: true);
      await stagingDir.create(recursive: true);
      _unpackSnapshot(snapshotPath: snapFile.path, targetPath: stagingDir.path);

      liveHandleDisturbed = true;
      await edgeClient.loadFromDir(stagingDir.path);
      final count = await edgeClient.count();
      if (count <= 0) {
        fleetLog(
          'FleetPull.pull: staged shard at ${stagingDir.path} has 0 points — '
          'restoring the previous corpus (if any), NOT promoting',
          level: 900,
        );
        await _restoreLiveOrClose(liveDir);
        await _deleteDirQuietly(stagingDir);
        return const PullEmpty();
      }

      // Validated: promote staging -> live.
      await edgeClient.close();
      await _deleteDirQuietly(liveDir);
      await stagingDir.rename(liveDir.path);
      await edgeClient.loadFromDir(liveDir.path);
      return PullLoaded(count: count, dir: liveDir.path);
    } catch (e) {
      fleetLog(
        'FleetPull.pull: failed, leaving the previous corpus (if any) in place: $e',
        level: 900,
      );
      if (liveHandleDisturbed) await _restoreLiveOrClose(liveDir);
      await _deleteDirQuietly(stagingDir);
      return PullUnreachable('$e');
    } finally {
      await _deleteFileQuietly(snapFile);
      // The server-side snapshot is only ever an intermediate — delete it on
      // every path (success, download failure, unpack failure). Best-effort:
      // a failed delete never affects pull()'s outcome.
      final name = snapshotName;
      if (name != null) {
        try {
          await http.deleteSnapshot(collection, shard, name);
        } catch (e) {
          fleetLog(
            'FleetPull.pull: server-side snapshot delete failed (best-effort): $e',
            level: 900,
          );
        }
      }
    }
  }

  /// Reloads [liveDir] if it still has an on-disk shard (the normal restore
  /// case: staging failed validation, the live directory itself was never
  /// touched), or just closes [edgeClient] if there was never a live corpus
  /// to begin with (first-ever pull failing) or the reload itself throws.
  Future<void> _restoreLiveOrClose(Directory liveDir) async {
    if (await liveDir.exists()) {
      try {
        await edgeClient.loadFromDir(liveDir.path);
        return;
      } catch (e) {
        fleetLog(
          'FleetPull.pull: restoring the previous live shard at ${liveDir.path} failed: $e',
          level: 900,
        );
      }
    }
    await edgeClient.close();
  }

  Future<void> _deleteDirQuietly(Directory dir) async {
    try {
      if (await dir.exists()) await dir.delete(recursive: true);
    } catch (e) {
      fleetLog('FleetPull.pull: best-effort cleanup of ${dir.path} failed: $e', level: 900);
    }
  }

  Future<void> _deleteFileQuietly(File file) async {
    try {
      if (await file.exists()) await file.delete();
    } catch (e) {
      fleetLog('FleetPull.pull: best-effort cleanup of ${file.path} failed: $e', level: 900);
    }
  }
}
