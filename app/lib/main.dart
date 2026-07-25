// Ledger OCR — Flutter Android app.
// BEFORE BUILDING: replace kApiBase below with your deployed Render URL.

import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';

// ─── Change this to your deployed API URL after pushing to Render. ───────────
const String kApiBase = 'https://YOUR-APP-NAME.onrender.com';
// ─────────────────────────────────────────────────────────────────────────────

void main() => runApp(const LedgerOcrApp());

class LedgerOcrApp extends StatelessWidget {
  const LedgerOcrApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Ledger OCR',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF3949AB)),
        useMaterial3: true,
      ),
      home: const HomeScreen(),
    );
  }
}

// ─── Result model ─────────────────────────────────────────────────────────────

class OcrResult {
  final String cleanedImageB64;
  final String tesseractText;
  final String easyOcrText;

  const OcrResult({
    required this.cleanedImageB64,
    required this.tesseractText,
    required this.easyOcrText,
  });

  factory OcrResult.fromJson(Map<String, dynamic> json) => OcrResult(
        cleanedImageB64: json['cleaned_image'] as String,
        tesseractText: json['tesseract_text'] as String? ?? '',
        easyOcrText: json['easyocr_text'] as String? ?? '',
      );
}

// ─── Home screen ──────────────────────────────────────────────────────────────

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final _picker = ImagePicker();

  File? _sourceFile;
  OcrResult? _result;
  String? _error;

  // Three-stage status shown while waiting.
  String? _statusMessage;

  Future<void> _pickAndProcess(ImageSource source) async {
    final picked = await _picker.pickImage(
      source: source,
      imageQuality: 85,
      maxWidth: 2048,
    );
    if (picked == null) return;

    setState(() {
      _sourceFile = File(picked.path);
      _result = null;
      _error = null;
      _statusMessage = 'Waking up server…';
    });

    // Warm-up ping — Render free tier sleeps after 15 min inactivity.
    try {
      await http
          .get(Uri.parse('$kApiBase/health'))
          .timeout(const Duration(seconds: 30));
    } catch (_) {
      // Ignore — /process will fail with a clear error if the server is down.
    }

    setState(() => _statusMessage = 'Processing image…');

    try {
      final request =
          http.MultipartRequest('POST', Uri.parse('$kApiBase/process'));
      request.files
          .add(await http.MultipartFile.fromPath('file', picked.path));

      final streamed = await request
          .send()
          .timeout(const Duration(seconds: 120));
      final body = await streamed.stream.bytesToString();

      if (streamed.statusCode == 200) {
        final data = jsonDecode(body) as Map<String, dynamic>;
        setState(() {
          _result = OcrResult.fromJson(data);
          _statusMessage = null;
        });
      } else {
        setState(() {
          _error = 'Server returned ${streamed.statusCode}:\n$body';
          _statusMessage = null;
        });
      }
    } on SocketException {
      setState(() {
        _error =
            'Cannot reach server.\nCheck your internet connection or that the API is deployed.';
        _statusMessage = null;
      });
    } catch (e) {
      setState(() {
        _error = 'Error: $e';
        _statusMessage = null;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final busy = _statusMessage != null;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Ledger OCR'),
        centerTitle: true,
        backgroundColor: Theme.of(context).colorScheme.primary,
        foregroundColor: Colors.white,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // ── Buttons ──────────────────────────────────────────────────
            Row(children: [
              Expanded(
                child: FilledButton.icon(
                  onPressed: busy ? null : () => _pickAndProcess(ImageSource.camera),
                  icon: const Icon(Icons.camera_alt),
                  label: const Text('Camera'),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: FilledButton.icon(
                  onPressed: busy ? null : () => _pickAndProcess(ImageSource.gallery),
                  icon: const Icon(Icons.photo_library),
                  label: const Text('Gallery'),
                ),
              ),
            ]),

            // ── Loading ───────────────────────────────────────────────────
            if (busy) ...[
              const SizedBox(height: 32),
              const Center(child: CircularProgressIndicator()),
              const SizedBox(height: 12),
              Center(
                child: Text(
                  _statusMessage!,
                  style: Theme.of(context).textTheme.bodyMedium,
                ),
              ),
            ],

            // ── Error ─────────────────────────────────────────────────────
            if (_error != null) ...[
              const SizedBox(height: 16),
              _ErrorCard(_error!),
            ],

            // ── Original image ────────────────────────────────────────────
            if (_sourceFile != null && !busy) ...[
              const SizedBox(height: 20),
              _SectionLabel('Original'),
              ClipRRect(
                borderRadius: BorderRadius.circular(8),
                child: Image.file(_sourceFile!,
                    height: 200, fit: BoxFit.contain),
              ),
            ],

            // ── Results ───────────────────────────────────────────────────
            if (_result != null) ...[
              const SizedBox(height: 20),
              _SectionLabel('Cleaned (after preprocessing)'),
              ClipRRect(
                borderRadius: BorderRadius.circular(8),
                child: Image.memory(
                  base64Decode(_result!.cleanedImageB64),
                  height: 200,
                  fit: BoxFit.contain,
                ),
              ),
              const SizedBox(height: 20),
              _SectionLabel('Tesseract'),
              _OcrTextCard(_result!.tesseractText),
              const SizedBox(height: 12),
              _SectionLabel('EasyOCR'),
              _OcrTextCard(_result!.easyOcrText),
              const SizedBox(height: 24),
            ],
          ],
        ),
      ),
    );
  }
}

// ─── Shared widgets ───────────────────────────────────────────────────────────

class _SectionLabel extends StatelessWidget {
  final String text;
  const _SectionLabel(this.text);

  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.only(bottom: 6),
        child: Text(
          text,
          style: Theme.of(context).textTheme.labelLarge?.copyWith(
                color: Theme.of(context).colorScheme.primary,
                fontWeight: FontWeight.bold,
              ),
        ),
      );
}

class _OcrTextCard extends StatelessWidget {
  final String text;
  const _OcrTextCard(this.text);

  @override
  Widget build(BuildContext context) => Card(
        child: Padding(
          padding: const EdgeInsets.all(12),
          child: SelectableText(
            text.isEmpty ? '(no text detected)' : text,
            style: const TextStyle(fontFamily: 'monospace', fontSize: 13),
          ),
        ),
      );
}

class _ErrorCard extends StatelessWidget {
  final String message;
  const _ErrorCard(this.message);

  @override
  Widget build(BuildContext context) => Card(
        color: Theme.of(context).colorScheme.errorContainer,
        child: Padding(
          padding: const EdgeInsets.all(12),
          child: Text(
            message,
            style: TextStyle(
                color: Theme.of(context).colorScheme.onErrorContainer),
          ),
        ),
      );
}
