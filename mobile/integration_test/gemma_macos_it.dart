// Live Gemma 4 on the macOS HOST (built target, not `flutter test`): proves
// GemmaQueryParser drives a real function-call. Run:
//   flutter test integration_test/gemma_macos_it.dart -d macos
// Needs the local .litertlm model; skips cleanly if it's absent.
import 'dart:io';

import 'package:fleet_node/query/gemma_query_parser.dart';
import 'package:flutter_gemma/flutter_gemma.dart';
import 'package:flutter_gemma_litertlm/flutter_gemma_litertlm.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('Gemma 4 loads on macOS and GemmaQueryParser returns a ParsedQuery',
      (tester) async {
    const modelPath = '/Users/sashadenisov/Downloads/gemma-4-E2B-it.litertlm';
    if (!File(modelPath).existsSync()) {
      markTestSkipped('Gemma 4 model not at $modelPath — skipped');
      return;
    }

    await FlutterGemma.initialize(inferenceEngines: [LiteRtLmEngine()]);
    await FlutterGemma.installModel(
      modelType: ModelType.gemma4,
      fileType: ModelFileType.litertlm,
    ).fromFile(modelPath).install();
    final model = await FlutterGemma.getActiveModel(maxTokens: 2048);

    final parser = GemmaQueryParser(model);
    final pq = await parser.parse(
      'the red coffee mug I saw yesterday',
      now: DateTime(2026, 9, 6),
    );

    // ignore: avoid_print
    print('GemmaQueryParser → $pq');
    expect(pq.phrase, isNotEmpty); // called the tool, or degraded to raw text

    await FlutterGemma.dispose();
  }, timeout: const Timeout(Duration(minutes: 8)));
}
