// Task 6b Step 1: the SiglipText seam + its test fake. No model, no I/O.
import 'dart:math' as math;

import 'package:fleet_node/embed/siglip_text.dart';
import 'package:flutter_test/flutter_test.dart';

double _l2(List<double> v) {
  var s = 0.0;
  for (final x in v) {
    s += x * x;
  }
  return math.sqrt(s);
}

void main() {
  group('FakeSiglipText', () {
    test('encodes to a 768-d unit vector by default', () async {
      final v = await const FakeSiglipText().encode('a potted plant');

      expect(v, hasLength(768));
      expect(_l2(v), closeTo(1.0, 1e-6));
      expect(v[0], 1.0);
      expect(v.skip(1).every((x) => x == 0.0), isTrue);
    });

    test('is deterministic and phrase-independent (wiring fake)', () async {
      final a = await const FakeSiglipText().encode('cup');
      final b = await const FakeSiglipText().encode('something else');

      expect(a, equals(b));
    });

    test('honours a custom dimension', () async {
      final v = await const FakeSiglipText(dim: 4).encode('x');

      expect(v, hasLength(4));
      expect(_l2(v), closeTo(1.0, 1e-6));
    });

    test('implements the SiglipText interface', () {
      const SiglipText embedder = FakeSiglipText();
      expect(embedder, isA<SiglipText>());
    });
  });
}
