// Ledger Digitisation — Flutter Android app.
// Calls the Flask backend's JSON API (web/app.py): /api/scan runs the real
// preprocessing + Tesseract/EasyOCR fusion pipeline, /api/save persists the
// (possibly corrected) rows to the same SQLite store the web UI uses.
// BEFORE BUILDING: replace kApiBase below with your deployed Space URL.

import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';

// ─── Change this to your deployed API URL after pushing to Hugging Face Spaces. ───
const String kApiBase = 'https://YOUR-USERNAME-YOUR-SPACE.hf.space';
// ─────────────────────────────────────────────────────────────────────────────────

void main() => runApp(const LedgerApp());

class LedgerApp extends StatelessWidget {
  const LedgerApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Ledger Digitisation',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF1C75BC)),
        useMaterial3: true,
      ),
      home: const HomeScreen(),
    );
  }
}

// ─── Row model ────────────────────────────────────────────────────────────
//
// One editable ledger row. Holds a TextEditingController per field (so the
// owner can correct OCR mistakes before saving) plus the fusion provenance
// for that row — which engine's value won each field, so the UI can flag
// what fusion auto-corrected.

class LedgerRow {
  final TextEditingController date;
  final TextEditingController item;
  final TextEditingController qty;
  final TextEditingController price;
  final TextEditingController total;
  final Map<String, String> provenance;

  LedgerRow({
    required String date,
    required String item,
    required String qty,
    required String price,
    required String total,
    required this.provenance,
  })  : date = TextEditingController(text: date),
        item = TextEditingController(text: item),
        qty = TextEditingController(text: qty),
        price = TextEditingController(text: price),
        total = TextEditingController(text: total);

  factory LedgerRow.fromJson(
    Map<String, dynamic> row,
    Map<String, dynamic> provenance,
  ) {
    return LedgerRow(
      date: row['date'] as String? ?? '',
      item: row['item'] as String? ?? '',
      qty: row['qty'] as String? ?? '',
      price: row['price'] as String? ?? '',
      total: row['total'] as String? ?? '',
      provenance: provenance.map((k, v) => MapEntry(k, v?.toString() ?? '')),
    );
  }

  Map<String, String> toJson() => {
        'date': date.text.trim(),
        'item': item.text.trim(),
        'qty': qty.text.trim(),
        'price': price.text.trim(),
        'total': total.text.trim(),
      };

  /// Human-readable summary of which fields fusion pulled from which OCR
  /// engine, or null when every field agreed (nothing to flag).
  String? autoCorrectedSummary() {
    final parts = <String>[];
    void check(String label, String key) {
      final winner = provenance[key];
      if (winner == null || winner.isEmpty || winner == 'agree') return;
      final engine = winner == 'easyocr'
          ? 'EasyOCR'
          : winner == 'tesseract'
              ? 'Tesseract'
              : winner;
      parts.add('$label ($engine)');
    }

    check('date', 'date_source');
    check('qty', 'qty');
    check('price', 'price');
    check('total', 'total');
    return parts.isEmpty ? null : 'Auto-corrected: ${parts.join(', ')}';
  }

  void dispose() {
    date.dispose();
    item.dispose();
    qty.dispose();
    price.dispose();
    total.dispose();
  }
}

// ─── Scan result ──────────────────────────────────────────────────────────

class ScanResult {
  final String sourceFile;
  final String rawUrl;
  final String processedUrl;
  final String usedEngine;
  final List<LedgerRow> rows;

  const ScanResult({
    required this.sourceFile,
    required this.rawUrl,
    required this.processedUrl,
    required this.usedEngine,
    required this.rows,
  });

  factory ScanResult.fromJson(Map<String, dynamic> json) {
    final rawRows = (json['rows'] as List).cast<Map<String, dynamic>>();
    final rawProvenance = (json['provenance'] as List).cast<Map<String, dynamic>>();
    return ScanResult(
      sourceFile: json['source_file'] as String,
      rawUrl: json['raw_url'] as String,
      processedUrl: json['processed_url'] as String,
      usedEngine: json['used_engine'] as String,
      rows: [
        for (var i = 0; i < rawRows.length; i++)
          LedgerRow.fromJson(rawRows[i], i < rawProvenance.length ? rawProvenance[i] : const {}),
      ],
    );
  }

  void dispose() {
    for (final row in rows) {
      row.dispose();
    }
  }
}

// ─── Home screen ──────────────────────────────────────────────────────────

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final _picker = ImagePicker();

  ScanResult? _result;
  String? _error;
  String? _statusMessage;
  String? _saveConfirmation;

  @override
  void dispose() {
    _result?.dispose();
    super.dispose();
  }

  Future<void> _pickAndScan(ImageSource source) async {
    final picked = await _picker.pickImage(
      source: source,
      imageQuality: 85,
      maxWidth: 2048,
    );
    if (picked == null) return;

    _result?.dispose();
    setState(() {
      _result = null;
      _error = null;
      _saveConfirmation = null;
      _statusMessage = 'Waking up server…';
    });

    // Warm-up ping — a sleeping Hugging Face Space can take a while to wake.
    try {
      await http
          .get(Uri.parse('$kApiBase/api/health'))
          .timeout(const Duration(seconds: 45));
    } catch (_) {
      // Ignore — /api/scan below will fail with a clear error if it's really down.
    }

    setState(() => _statusMessage = 'Running OCR + fusion…');

    try {
      final request = http.MultipartRequest('POST', Uri.parse('$kApiBase/api/scan'));
      request.files.add(await http.MultipartFile.fromPath('image', picked.path));

      final streamed = await request.send().timeout(const Duration(seconds: 120));
      final body = await streamed.stream.bytesToString();
      final data = jsonDecode(body) as Map<String, dynamic>;

      if (streamed.statusCode == 200) {
        setState(() {
          _result = ScanResult.fromJson(data);
          _statusMessage = null;
        });
      } else {
        setState(() {
          _error = data['error'] as String? ?? 'Server returned ${streamed.statusCode}.';
          _statusMessage = null;
        });
      }
    } on SocketException {
      setState(() {
        _error = 'Cannot reach server.\nCheck your internet connection or that the API is deployed.';
        _statusMessage = null;
      });
    } catch (e) {
      setState(() {
        _error = 'Error: $e';
        _statusMessage = null;
      });
    }
  }

  Future<void> _saveRows() async {
    final result = _result;
    if (result == null) return;

    setState(() {
      _statusMessage = 'Saving…';
      _error = null;
      _saveConfirmation = null;
    });

    try {
      final response = await http
          .post(
            Uri.parse('$kApiBase/api/save'),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode({
              'source_file': result.sourceFile,
              'rows': result.rows.map((r) => r.toJson()).toList(),
            }),
          )
          .timeout(const Duration(seconds: 30));

      final data = jsonDecode(response.body) as Map<String, dynamic>;
      if (response.statusCode == 200) {
        setState(() {
          _saveConfirmation = 'Saved ${data['saved']} record(s).';
          _statusMessage = null;
        });
      } else {
        setState(() {
          _error = data['error'] as String? ?? 'Save failed (${response.statusCode}).';
          _statusMessage = null;
        });
      }
    } catch (e) {
      setState(() {
        _error = 'Error saving: $e';
        _statusMessage = null;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final busy = _statusMessage != null;
    final result = _result;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Ledger Digitisation'),
        centerTitle: true,
        backgroundColor: Theme.of(context).colorScheme.primary,
        foregroundColor: Colors.white,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Row(children: [
              Expanded(
                child: FilledButton.icon(
                  onPressed: busy ? null : () => _pickAndScan(ImageSource.camera),
                  icon: const Icon(Icons.camera_alt),
                  label: const Text('Camera'),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: FilledButton.icon(
                  onPressed: busy ? null : () => _pickAndScan(ImageSource.gallery),
                  icon: const Icon(Icons.photo_library),
                  label: const Text('Gallery'),
                ),
              ),
            ]),

            if (busy) ...[
              const SizedBox(height: 32),
              const Center(child: CircularProgressIndicator()),
              const SizedBox(height: 12),
              Center(child: Text(_statusMessage!, style: Theme.of(context).textTheme.bodyMedium)),
            ],

            if (_error != null) ...[
              const SizedBox(height: 16),
              _MessageCard(_error!, isError: true),
            ],

            if (_saveConfirmation != null) ...[
              const SizedBox(height: 16),
              _MessageCard(_saveConfirmation!, isError: false),
            ],

            if (result != null && !busy) ...[
              const SizedBox(height: 20),
              _SectionLabel('Preview'),
              Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Expanded(child: _NetworkPreview(label: 'Raw', url: '$kApiBase${result.rawUrl}')),
                  const SizedBox(width: 12),
                  Expanded(child: _NetworkPreview(label: 'Preprocessed', url: '$kApiBase${result.processedUrl}')),
                ],
              ),
              const SizedBox(height: 8),
              Text('Extracted via: ${result.usedEngine}', style: Theme.of(context).textTheme.bodySmall),
              const SizedBox(height: 20),
              _SectionLabel('Review rows — correct anything before saving'),
              for (var i = 0; i < result.rows.length; i++) ...[
                _RowCard(index: i, row: result.rows[i]),
                const SizedBox(height: 10),
              ],
              const SizedBox(height: 8),
              FilledButton.icon(
                onPressed: _saveRows,
                icon: const Icon(Icons.save),
                label: const Text('Save records'),
              ),
              const SizedBox(height: 24),
            ],
          ],
        ),
      ),
    );
  }
}

// ─── Shared widgets ─────────────────────────────────────────────────────────

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

class _NetworkPreview extends StatelessWidget {
  final String label;
  final String url;
  const _NetworkPreview({required this.label, required this.url});

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label, style: Theme.of(context).textTheme.bodySmall),
        const SizedBox(height: 4),
        ClipRRect(
          borderRadius: BorderRadius.circular(4),
          child: AspectRatio(
            aspectRatio: 1,
            child: Image.network(
              url,
              fit: BoxFit.cover,
              errorBuilder: (context, error, stackTrace) => Container(
                color: Theme.of(context).colorScheme.surfaceContainerHighest,
                child: const Icon(Icons.broken_image),
              ),
            ),
          ),
        ),
      ],
    );
  }
}

class _RowCard extends StatelessWidget {
  final int index;
  final LedgerRow row;
  const _RowCard({required this.index, required this.row});

  @override
  Widget build(BuildContext context) {
    final summary = row.autoCorrectedSummary();
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Text('Row ${index + 1}', style: Theme.of(context).textTheme.labelMedium),
            const SizedBox(height: 8),
            TextField(
              controller: row.item,
              decoration: const InputDecoration(labelText: 'Item', isDense: true),
            ),
            const SizedBox(height: 8),
            Row(children: [
              Expanded(
                child: TextField(
                  controller: row.date,
                  decoration: const InputDecoration(labelText: 'Date', isDense: true),
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: TextField(
                  controller: row.qty,
                  decoration: const InputDecoration(labelText: 'Qty', isDense: true),
                  keyboardType: TextInputType.number,
                ),
              ),
            ]),
            const SizedBox(height: 8),
            Row(children: [
              Expanded(
                child: TextField(
                  controller: row.price,
                  decoration: const InputDecoration(labelText: 'Price', isDense: true),
                  keyboardType: const TextInputType.numberWithOptions(decimal: true),
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: TextField(
                  controller: row.total,
                  decoration: const InputDecoration(labelText: 'Total', isDense: true),
                  keyboardType: const TextInputType.numberWithOptions(decimal: true),
                ),
              ),
            ]),
            if (summary != null) ...[
              const SizedBox(height: 8),
              Text(
                summary,
                style: TextStyle(
                  fontSize: 12,
                  fontStyle: FontStyle.italic,
                  color: Theme.of(context).colorScheme.onSurfaceVariant,
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}

class _MessageCard extends StatelessWidget {
  final String message;
  final bool isError;
  const _MessageCard(this.message, {required this.isError});

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final bg = isError ? scheme.errorContainer : scheme.secondaryContainer;
    final fg = isError ? scheme.onErrorContainer : scheme.onSecondaryContainer;
    return Card(
      color: bg,
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Text(message, style: TextStyle(color: fg)),
      ),
    );
  }
}
