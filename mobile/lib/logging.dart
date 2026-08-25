import 'dart:developer' as developer;

/// The mobile fleet node's one logger name. Phase 1 review fix (silent-
/// failure H2/L1): `mobile/lib` had ZERO logging — every fail-soft boundary
/// (an unreachable hub, a native call that failed, a malformed payload)
/// degraded to an empty result with no trace anywhere. The project's
/// fail-soft rule (never crash the chat) stays; this just stops it from
/// also being fail-SILENT — every swallow below now logs through here
/// first, observable via `flutter logs` / the VM service / `adb logcat`
/// (`dart:developer.log` also reaches logcat on Android).
///
/// [level] follows `dart:developer.log`'s convention (roughly
/// `java.util.logging` levels): 0 (default) for routine/expected fail-soft
/// notes, 900 for an actual anomaly worth someone's attention.
void fleetLog(String message, {int level = 0}) {
  developer.log(message, name: 'fleet', level: level);
}
