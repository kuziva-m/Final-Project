// Ledger Digitisation — Flutter Android app.
// BEFORE BUILDING: replace kApiBase below with your deployed Render URL.

import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';

// ─── Change this to your deployed API URL after pushing to Render. ───────────
const String kApiBase = 'https://YOUR-APP-NAME.onrender.com';
// ─────────────────────────────────────────────────────────────────────────────

void main() => runApp(const LedgerApp());

class LedgerApp extends StatelessWidget {
  const LedgerApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Ledger Digitisation',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF3949AB)),
        useMaterial3: true,
      ),
      home: const HomeScreen(),
    );
  }
}

// ─── Result models ────────────────────────────────────────────────────────

class ExtractedTable {
  final List<String> columns;
  final List<List<String>> rows;
  final String notes;

  const ExtractedTable({
    required this.columns,
    required this.rows,
    required this.notes,
  });

  factory ExtractedTable.fromJson(Map<String, dynamic> json) => ExtractedTable(
        columns: List<String>.from(json['columns'] as List),
        rows: (json['rows'] as List)
            .map((row) => List<String>.from(row as List))
            .toList(),
        notes: json['notes'] as String? ?? '',
      );
}

class ScanResult {
  final String scanId;
  final String cleanedImageB64;
  final List<ExtractedTable> tables;
  final double overallConfidence;

  const ScanResult({
    required this.scanId,
    required this.cleanedImageB64,
    required this.tables,
    required this.overallConfidence,
  });

  factory ScanResult.fromJson(Map<String, dynamic> json) => ScanResult(
        scanId: json['scan_id'] as String,
        cleanedImageB64: json['cleaned_image'] as String,
        tables: (json['tables'] as List)
            .map((t) => ExtractedTable.fromJson(t as Map<String, dynamic>))
            .toList(),
        overallConfidence: (json['overall_confidence'] as num).toDouble(),
      );
}

// ─── Home screen ──────────────────────────────────────────────────────────

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final _picker = ImagePicker();
  final _businessIdController = TextEditingController();

  File? _sourceFile;
  ScanResult? _result;
  String? _error;
  String? _statusMessage;

  @override
  void dispose() {
    _businessIdController.dispose();
    super.dispose();
  }

  Future<void> _pickAndProcess(ImageSource source) async {
    final businessId = _businessIdController.text.trim();
    if (businessId.isEmpty) {
      setState(() => _error = 'Enter a business ID first.');
      return;
    }

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
      // Ignore — /scan will fail with a clear error if the server is down.
    }

    setState(() => _statusMessage = 'Reading tables…');

    try {
      final request = http.MultipartRequest('POST', Uri.parse('$kApiBase/scan'));
      request.fields['business_id'] = businessId;
      request.files.add(await http.MultipartFile.fromPath('file', picked.path));

      final streamed =
          await request.send().timeout(const Duration(seconds: 120));
      final body = await streamed.stream.bytesToString();

      if (streamed.statusCode == 200) {
        final data = jsonDecode(body) as Map<String, dynamic>;
        setState(() {
          _result = ScanResult.fromJson(data);
          _statusMessage = null;
        });
      } else {
        String detail = body;
        try {
          detail = (jsonDecode(body) as Map<String, dynamic>)['detail']
                  as String? ??
              body;
        } catch (_) {
          // Body wasn't JSON — fall back to the raw text already assigned.
        }
        setState(() {
          _error = 'Server returned ${streamed.statusCode}:\n$detail';
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
            // ── Business ID ──────────────────────────────────────────────
            TextField(
              controller: _businessIdController,
              enabled: !busy,
              decoration: const InputDecoration(
                labelText: 'Business ID',
                hintText: 'e.g. my-shop-01',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 16),

            // ── Buttons ──────────────────────────────────────────────────
            Row(children: [
              Expanded(
                child: FilledButton.icon(
                  onPressed:
                      busy ? null : () => _pickAndProcess(ImageSource.camera),
                  icon: const Icon(Icons.camera_alt),
                  label: const Text('Camera'),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: FilledButton.icon(
                  onPressed: busy
                      ? null
                      : () => _pickAndProcess(ImageSource.gallery),
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
                child:
                    Image.file(_sourceFile!, height: 200, fit: BoxFit.contain),
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
              const SizedBox(height: 16),
              _ConfidenceBadge(_result!.overallConfidence),
              for (var i = 0; i < _result!.tables.length; i++) ...[
                const SizedBox(height: 20),
                _SectionLabel(_result!.tables.length > 1
                    ? 'Table ${i + 1}'
                    : 'Extracted table'),
                _TableCard(_result!.tables[i]),
              ],
              if (_result!.tables.isEmpty) ...[
                const SizedBox(height: 16),
                const Text('No table detected on this page.'),
              ],
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

class _ConfidenceBadge extends StatelessWidget {
  final double confidence;
  const _ConfidenceBadge(this.confidence);

  @override
  Widget build(BuildContext context) {
    final pct = (confidence * 100).round();
    final Color color;
    final String label;
    if (confidence >= 0.8) {
      color = Colors.green;
      label = 'High confidence';
    } else if (confidence >= 0.5) {
      color = Colors.orange;
      label = 'Review recommended';
    } else {
      color = Colors.red;
      label = 'Low confidence — please review';
    }

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: color.withOpacity(0.12),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: color),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(Icons.circle, size: 10, color: color),
          const SizedBox(width: 8),
          Text('$label ($pct%)',
              style: TextStyle(color: color, fontWeight: FontWeight.w600)),
        ],
      ),
    );
  }
}

class _TableCard extends StatelessWidget {
  final ExtractedTable table;
  const _TableCard(this.table);

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(8),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            SingleChildScrollView(
              scrollDirection: Axis.horizontal,
              child: DataTable(
                columns: table.columns
                    .map((c) => DataColumn(
                        label: Text(c,
                            style:
                                const TextStyle(fontWeight: FontWeight.bold))))
                    .toList(),
                rows: table.rows
                    .map((row) => DataRow(
                          cells:
                              row.map((cell) => DataCell(Text(cell))).toList(),
                        ))
                    .toList(),
              ),
            ),
            if (table.notes.isNotEmpty) ...[
              const Divider(),
              Padding(
                padding: const EdgeInsets.all(8),
                child: Text(
                  'Note: ${table.notes}',
                  style: TextStyle(
                    fontStyle: FontStyle.italic,
                    color: Theme.of(context).colorScheme.onSurfaceVariant,
                  ),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
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
