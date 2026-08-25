/// The honest outcome of one [FleetPull.pull] (Phase 1 review fix E:
/// silent-failure H1/M3). Deliberately NOT a bare `String?`/`bool` — a
/// caller that only checks "non-null"/"true" cannot tell an unreachable hub
/// apart from a hub that answered with a genuinely empty shard, and
/// [AppRoot] must show a user a DIFFERENT message for each ("hub down"
/// must never look identical to "memory is empty").
sealed class PullResult {
  const PullResult();
}

/// The pull succeeded: the corpus at [dir] has [count] (> 0) points, and
/// [EdgeClient] now has it loaded.
final class PullLoaded extends PullResult {
  const PullLoaded({required this.count, required this.dir});

  final int count;
  final String dir;

  @override
  String toString() => 'PullLoaded(count: $count, dir: $dir)';
}

/// The hub could not be reached, or the create/download/unpack/validate
/// chain failed for any other reason — see [message]. [EdgeClient] still
/// has whatever corpus it had before this call: either untouched (the
/// failure happened before validation ever swapped the live handle) or
/// restored from disk (fix D's `_restoreLiveOrClose`).
final class PullUnreachable extends PullResult {
  const PullUnreachable(this.message);

  final String message;

  @override
  String toString() => 'PullUnreachable($message)';
}

/// The hub WAS reached and a shard WAS downloaded + unpacked, but it has
/// zero points — a real, distinct outcome from [PullUnreachable] (the crux
/// of fix E). The previous corpus (if any) survives, same as
/// [PullUnreachable].
final class PullEmpty extends PullResult {
  const PullEmpty();

  @override
  String toString() => 'PullEmpty()';
}
