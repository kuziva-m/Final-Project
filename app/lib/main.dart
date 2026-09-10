// Ledger Digitisation — Flutter Android app.
// Calls the Flask backend's JSON API (web/app.py): /api/scan runs the real
// preprocessing + Tesseract/EasyOCR fusion pipeline, /api/save persists the
// (possibly corrected) rows to the same SQLite store the web UI uses.
//
// Visual design matches web/static/style.css exactly: Chinhoyi University of
// Technology blue/gold, flat institutional style — no gradients, no
// glassmorphism, no rounded "card" boxes, no drop shadows.
//
// BEFORE BUILDING: replace kApiBase below with your deployed Space URL.

import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';

// ─── Change this to your deployed API URL after pushing to Hugging Face Spaces. ───
const String kApiBase = 'https://YOUR-USERNAME-YOUR-SPACE.hf.space';
// ─────────────────────────────────────────────────────────────────────────────────

// ─── Chinhoyi University of Technology palette (matches web/static/style.css) ─────
const _kPrimary = Color(0xFF1C75BC); // CUT blue
const _kAccent = Color(0xFFC79A3B); // CUT gold — the only accent colour
const _kText = Color(0xFF333333);
const _kBg = Color(0xFFFFFFFF);
const _kBorder = Color(0xFFCCCCCC);
const _kAccentTint = Color(0xFFFBF3E1);
const _kRadius = 2.0; // --radius: 2px — flat, not rounded

void main() => runApp(const LedgerApp());

class LedgerApp extends StatelessWidget {
  const LedgerApp({super.key});

  @override
  Widget build(BuildContext context) {
    const colorScheme = ColorScheme.light(
      primary: _kPrimary,
      onPrimary: Colors.white,
      secondary: _kAccent,
      onSecondary: Colors.white,
      surface: _kBg,
      onSurface: _kText,
      outline: _kBorder,
    );
    final base = ThemeData(useMaterial3: true, colorScheme: colorScheme);

    return MaterialApp(
      title: 'Ledger Digitisation',
      debugShowCheckedModeBanner: false,
      theme: base.copyWith(
        scaffoldBackgroundColor: _kBg,
        textTheme: base.textTheme.apply(bodyColor: _kText, displayColor: _kText),
        appBarTheme: const AppBarTheme(
          backgroundColor: _kPrimary,
          foregroundColor: Colors.white,
          elevation: 0,
          scrolledUnderElevation: 0,
          centerTitle: false,
        ),
        filledButtonTheme: FilledButtonThemeData(
          style: FilledButton.styleFrom(
            backgroundColor: _kPrimary,
            foregroundColor: Colors.white,
            elevation: 0,
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(_kRadius)),
            padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
          ),
        ),
        inputDecorationTheme: InputDecorationTheme(
          isDense: true,
          filled: false,
          contentPadding: const EdgeInsets.symmetric(horizontal: 10, vertical: 10),
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(_kRadius),
            borderSide: const BorderSide(color: _kBorder),
          ),
          enabledBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(_kRadius),
            borderSide: const BorderSide(color: _kBorder),
          ),
          focusedBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(_kRadius),
            borderSide: const BorderSide(color: _kPrimary, width: 2),
          ),
        ),
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
      appBar: _BrandAppBar(),
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
              const Center(
                child: CircularProgressIndicator(color: _kPrimary),
              ),
              const SizedBox(height: 12),
              Center(child: Text(_statusMessage!, style: const TextStyle(color: _kText))),
            ],

            if (_error != null) ...[
              const SizedBox(height: 16),
              _FlashNotice(_error!),
            ],

            if (_saveConfirmation != null) ...[
              const SizedBox(height: 16),
              _FlashNotice(_saveConfirmation!),
            ],

            if (result != null && !busy) ...[
              const SizedBox(height: 20),
              _ReviewBanner(result: result),
              const SizedBox(height: 16),
              _SectionLabel('Previews'),
              Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Expanded(child: _BorderedPreview(label: 'Raw upload', url: '$kApiBase${result.rawUrl}')),
                  const SizedBox(width: 12),
                  Expanded(child: _BorderedPreview(label: 'Preprocessed', url: '$kApiBase${result.processedUrl}')),
                ],
              ),
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

// ─── Brand app bar ────────────────────────────────────────────────────────
//
// Matches web/templates/base.html's .site-header exactly: solid blue bar,
// the CUT logo, a small-caps university line over the bold app name, and a
// thin gold underline along the bottom edge.

class _BrandAppBar extends StatelessWidget implements PreferredSizeWidget {
  const _BrandAppBar();

  @override
  Widget build(BuildContext context) {
    return AppBar(
      titleSpacing: 16,
      title: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Image.asset('assets/logo.png', height: 40),
          const SizedBox(width: 12),
          const Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                'CHINHOYI UNIVERSITY OF TECHNOLOGY',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 10,
                  letterSpacing: 0.8,
                  fontWeight: FontWeight.w500,
                ),
              ),
              Text(
                'MSME Ledger Digitisation',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 17,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ),
        ],
      ),
      bottom: const PreferredSize(
        preferredSize: Size.fromHeight(3),
        child: ColoredBox(color: _kAccent),
      ),
    );
  }

  @override
  Size get preferredSize => const Size.fromHeight(kToolbarHeight + 3);
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
          style: const TextStyle(color: _kText, fontWeight: FontWeight.w600, fontSize: 16),
        ),
      );
}

/// Matches .review-banner: a thumbnail plus "Editing: <filename>" /
/// "Rows extracted from: <used_engine>", blue left border, no shadow.
class _ReviewBanner extends StatelessWidget {
  final ScanResult result;
  const _ReviewBanner({required this.result});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: const BoxDecoration(
        color: _kBg,
        border: Border(
          top: BorderSide(color: _kBorder),
          right: BorderSide(color: _kBorder),
          bottom: BorderSide(color: _kBorder),
          left: BorderSide(color: _kPrimary, width: 4),
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text.rich(
            TextSpan(
              style: const TextStyle(fontSize: 13, color: _kText),
              children: [
                const TextSpan(text: 'Editing: '),
                TextSpan(
                  text: result.sourceFile,
                  style: const TextStyle(color: _kPrimary, fontWeight: FontWeight.w600),
                ),
              ],
            ),
          ),
          const SizedBox(height: 4),
          Text(
            'Rows extracted from: ${result.usedEngine}',
            style: const TextStyle(fontSize: 13, color: _kText),
          ),
        ],
      ),
    );
  }
}

/// Matches .previews img: 1px border, 2px radius, no shadow.
class _BorderedPreview extends StatelessWidget {
  final String label;
  final String url;
  const _BorderedPreview({required this.label, required this.url});

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label, style: const TextStyle(fontSize: 12, color: _kText)),
        const SizedBox(height: 4),
        Container(
          decoration: BoxDecoration(
            border: Border.all(color: _kBorder),
            borderRadius: BorderRadius.circular(_kRadius),
          ),
          child: ClipRRect(
            borderRadius: BorderRadius.circular(_kRadius - 1),
            child: AspectRatio(
              aspectRatio: 1,
              child: Image.network(
                url,
                fit: BoxFit.cover,
                errorBuilder: (context, error, stackTrace) => const ColoredBox(
                  color: Color(0xFFF2F2F2),
                  child: Icon(Icons.broken_image, color: _kText),
                ),
              ),
            ),
          ),
        ),
      ],
    );
  }
}

/// One editable ledger row, styled to match the web table's borders/zebra
/// language: thin grey border, 2px radius, no elevation.
class _RowCard extends StatelessWidget {
  final int index;
  final LedgerRow row;
  const _RowCard({required this.index, required this.row});

  @override
  Widget build(BuildContext context) {
    final summary = row.autoCorrectedSummary();
    return Card(
      elevation: 0,
      margin: EdgeInsets.zero,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(_kRadius),
        side: const BorderSide(color: _kBorder),
      ),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Text('Row ${index + 1}', style: const TextStyle(color: _kPrimary, fontWeight: FontWeight.w600, fontSize: 13)),
            const SizedBox(height: 8),
            TextField(
              controller: row.item,
              style: const TextStyle(color: _kText),
              decoration: const InputDecoration(labelText: 'Item'),
            ),
            const SizedBox(height: 8),
            Row(children: [
              Expanded(
                child: TextField(
                  controller: row.date,
                  style: const TextStyle(color: _kText),
                  decoration: const InputDecoration(labelText: 'Date'),
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: TextField(
                  controller: row.qty,
                  style: const TextStyle(color: _kText),
                  decoration: const InputDecoration(labelText: 'Qty'),
                  keyboardType: TextInputType.number,
                ),
              ),
            ]),
            const SizedBox(height: 8),
            Row(children: [
              Expanded(
                child: TextField(
                  controller: row.price,
                  style: const TextStyle(color: _kText),
                  decoration: const InputDecoration(labelText: 'Price'),
                  keyboardType: const TextInputType.numberWithOptions(decimal: true),
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: TextField(
                  controller: row.total,
                  style: const TextStyle(color: _kText),
                  decoration: const InputDecoration(labelText: 'Total'),
                  keyboardType: const TextInputType.numberWithOptions(decimal: true),
                ),
              ),
            ]),
            if (summary != null) ...[
              const SizedBox(height: 8),
              _FlashNotice(summary, compact: true),
            ],
          ],
        ),
      ),
    );
  }
}

/// Matches .flashes exactly: accent-tint background, 1px gold border, 4px
/// gold left border. Used for every message — errors, confirmations, and
/// per-row auto-correct notes — the source design has no separate red/green
/// styling, just this one flat notice treatment.
class _FlashNotice extends StatelessWidget {
  final String message;
  final bool compact;
  const _FlashNotice(this.message, {this.compact = false});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: EdgeInsets.symmetric(horizontal: compact ? 10 : 16, vertical: compact ? 6 : 12),
      decoration: const BoxDecoration(
        color: _kAccentTint,
        border: Border(
          top: BorderSide(color: _kAccent),
          right: BorderSide(color: _kAccent),
          bottom: BorderSide(color: _kAccent),
          left: BorderSide(color: _kAccent, width: 4),
        ),
      ),
      child: Text(
        message,
        style: TextStyle(color: _kText, fontSize: compact ? 12 : 14, fontStyle: compact ? FontStyle.italic : FontStyle.normal),
      ),
    );
  }
}
