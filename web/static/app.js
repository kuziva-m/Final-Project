// Small helpers on top of server-rendered pages. Without JavaScript every
// form still works as a plain form post; this adds progress, the review
// tools and the document page's share/copy buttons.

(function () {
  "use strict";

  // --- Shared ------------------------------------------------------------------

  var I18N = window.I18N || {};

  // Text in the page's language (see web/i18n.py), with {name} placeholders filled.
  function T(text, values) {
    var result = I18N[text] || text;
    Object.keys(values || {}).forEach(function (key) {
      result = result.split("{" + key + "}").join(values[key]);
    });
    return result;
  }

  function count(n, one, many) {
    return n === 1 ? T(one) : T(many, { n: n });
  }

  // localStorage can be missing or blocked (private mode); never let that break a page.
  var store = {
    get: function (key) {
      try { return window.localStorage.getItem(key); } catch (err) { return null; }
    },
    set: function (key, value) {
      try { window.localStorage.setItem(key, value); } catch (err) { /* ignore */ }
    },
    remove: function (key) {
      try { window.localStorage.removeItem(key); } catch (err) { /* ignore */ }
    }
  };

  function getJSON(url, done) {
    var xhr = new XMLHttpRequest();
    xhr.onload = function () {
      var reply = null;
      try { reply = JSON.parse(xhr.responseText); } catch (err) { /* not JSON */ }
      done(reply, xhr.status);
    };
    xhr.onerror = function () { done(null, 0); };
    xhr.open("GET", url);
    xhr.send();
  }

  // A message at the bottom of the screen, optionally with buttons.
  var toastEl = document.getElementById("toast");
  var toastTimer = null;
  function toast(message, actions) {
    if (!toastEl) return;
    clearTimeout(toastTimer);
    toastEl.textContent = "";
    var text = document.createElement("span");
    text.textContent = message;
    toastEl.appendChild(text);
    (actions || []).forEach(function (action) {
      var button = document.createElement("button");
      button.type = "button";
      button.textContent = action.label;
      button.addEventListener("click", function () {
        hideToast();
        if (action.run) action.run();
      });
      toastEl.appendChild(button);
    });
    toastEl.hidden = false;
    toastTimer = setTimeout(hideToast, actions && actions.length ? 9000 : 3500);
  }
  function hideToast() {
    if (toastEl) toastEl.hidden = true;
  }

  function copyText(text) {
    var done = function () { toast(T("Copied. Paste it into WhatsApp, Excel or an email.")); };
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(text).then(done, function () { fallbackCopy(text, done); });
    } else {
      fallbackCopy(text, done);
    }
  }
  function fallbackCopy(text, done) {
    var area = document.createElement("textarea");
    area.value = text;
    area.setAttribute("readonly", "");
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.appendChild(area);
    area.select();
    try {
      document.execCommand("copy");
      done();
    } catch (err) {
      toast(T("Could not copy. Select the table and copy it by hand."));
    }
    document.body.removeChild(area);
  }

  // Tab-separated text pastes into Excel and WhatsApp as rows and columns.
  function tableText(title, columns, rows) {
    var clean = function (cell) { return String(cell).replace(/[\t\r\n]+/g, " "); };
    var lines = [];
    if (title) lines.push(clean(title));
    lines.push(columns.map(clean).join("\t"));
    rows.forEach(function (row) { lines.push(row.map(clean).join("\t")); });
    return lines.join("\n");
  }

  // --- Upload and waiting screen -------------------------------------------------

  function initUpload() {
    var form = document.getElementById("upload");
    var busy = document.getElementById("busy");
    if (!form || !busy || !window.XMLHttpRequest || !window.FormData) return;

    var input = form.querySelector('input[type="file"][name="image"]');
    var camera = document.getElementById("camera-input");
    var takePhoto = document.getElementById("take-photo");
    var summary = document.getElementById("file-summary");
    var nameInput = form.querySelector('input[name="name"]');
    var appendInput = form.querySelector('input[name="append_to"]');
    var maxPages = parseInt(form.getAttribute("data-max-pages"), 10) || 0;

    var titleEl = document.getElementById("busy-title");
    var fill = document.getElementById("busy-bar-fill");
    var detail = document.getElementById("busy-detail");
    var grid = document.getElementById("busy-pages");
    var note = document.getElementById("busy-note");
    var tipEl = document.getElementById("busy-tip");
    var retry = document.getElementById("busy-retry");

    var chosen = null;
    var previewUrl = null;
    var timers = [];
    var wakeLock = null;
    var audio = null;
    var finished = false;

    // Remember the last document name and suggest it next time.
    if (nameInput) {
      var lastName = store.get("ledger.lastName");
      if (lastName && !nameInput.value) nameInput.value = lastName;
      nameInput.addEventListener("focus", function () { nameInput.select(); });
    }

    // --- Choosing a file -------------------------------------------------------

    var mb = function (bytes) { return (bytes / 1048576).toFixed(1) + " MB"; };
    var isPdfFile = function (file) { return /\.pdf$/i.test(file.name) || file.type === "application/pdf"; };

    function pick(file) {
      chosen = file || null;
      if (previewUrl) { URL.revokeObjectURL(previewUrl); previewUrl = null; }
      summary.textContent = "";
      summary.hidden = !file;
      if (!file) return;

      if (/^image\//.test(file.type) && window.URL && URL.createObjectURL) {
        previewUrl = URL.createObjectURL(file);
        var img = document.createElement("img");
        img.src = previewUrl;
        img.alt = "";
        summary.appendChild(img);
      }
      var text = document.createElement("span");
      text.textContent = file.name + " · " + mb(file.size);
      summary.appendChild(text);

      if (isPdfFile(file)) {
        countPdfPages(file, function (pages) {
          if (!pages || chosen !== file) return;
          text.textContent += " · " + count(pages, "about 1 page", "about {n} pages");
          if (maxPages && pages > maxPages) {
            var warn = document.createElement("span");
            warn.className = "file-warning";
            warn.textContent = T("Only the first {max} pages will be read.", { max: maxPages });
            summary.appendChild(warn);
          }
        });
      }
    }

    // Rough page count read from the PDF itself: counts "/Type /Page" objects,
    // else the largest "/Count". Compressed PDFs may give no answer.
    function countPdfPages(file, done) {
      if (!window.FileReader || file.size > 150 * 1048576) return done(null);
      var reader = new FileReader();
      reader.onload = function () {
        var bytes = new Uint8Array(reader.result);
        var pages = 0, maxCount = 0, n = bytes.length;
        var match = function (i, word) {
          for (var k = 0; k < word.length; k++) if (bytes[i + k] !== word.charCodeAt(k)) return false;
          return true;
        };
        for (var i = 0; i < n; i++) {
          if (bytes[i] !== 47) continue; // "/"
          if (match(i, "/Type")) {
            var j = i + 5;
            while (bytes[j] === 32 || bytes[j] === 10 || bytes[j] === 13) j++;
            if (match(j, "/Page") && !match(j, "/Pages")) pages++;
          } else if (match(i, "/Count")) {
            var m = i + 6, value = 0;
            while (bytes[m] === 32) m++;
            while (bytes[m] >= 48 && bytes[m] <= 57) { value = value * 10 + bytes[m] - 48; m++; }
            if (value > maxCount) maxCount = value;
          }
        }
        done(pages || maxCount || null);
      };
      reader.onerror = function () { done(null); };
      reader.readAsArrayBuffer(file);
    }

    input.addEventListener("change", function () { pick(input.files[0]); });

    // "Take photo" opens the camera on phones; on computers it would only
    // open the same file picker, so it stays hidden there.
    if (camera && takePhoto && window.matchMedia && window.matchMedia("(pointer: coarse)").matches) {
      takePhoto.hidden = false;
      takePhoto.addEventListener("click", function () { camera.click(); });
      camera.addEventListener("change", function () {
        var file = camera.files[0];
        if (!file) return;
        try {
          var transfer = new DataTransfer();
          transfer.items.add(file);
          input.files = transfer.files;
        } catch (err) {
          input.required = false; // can't copy it over; the script sends it instead
        }
        pick(file);
      });
    }

    // --- Waiting screen ----------------------------------------------------------

    var clock = function (secs) {
      var s = Math.floor(secs);
      return Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0");
    };
    var TIPS = [
      "Tip: press Enter to move down a column.",
      "Tip: cells with a ? are amber. Check those first.",
      "Tip: fix a name once and you can fix every copy of it.",
      "Tip: after saving, download Excel or share it on WhatsApp.",
      "Tip: photographed the next page? Add it to the same document."
    ];

    function setState(state) {
      busy.classList.remove("is-uploading", "is-reading", "is-failed", "is-counting");
      busy.classList.add("is-" + state);
    }

    function stopAll() {
      timers.forEach(function (t) { clearInterval(t); });
      timers = [];
      if (wakeLock) { wakeLock.release().catch(function () {}); wakeLock = null; }
    }

    // Keep the phone screen on while waiting.
    function keepAwake() {
      if (!("wakeLock" in navigator) || wakeLock || busy.hidden || finished || busy.classList.contains("is-failed")) return;
      navigator.wakeLock.request("screen").then(function (lock) { wakeLock = lock; }, function () {});
    }
    document.addEventListener("visibilitychange", function () {
      if (document.visibilityState === "visible") { wakeLock = null; keepAwake(); }
    });

    function openBusy() {
      finished = false;
      busy.hidden = false;
      retry.hidden = true;
      note.hidden = true;
      grid.textContent = "";
      tipEl.textContent = "";
      document.body.style.overflow = "hidden";
      keepAwake();
    }

    function fail(message) {
      stopAll();
      store.remove("ledger.pending");
      setState("failed");
      titleEl.textContent = T("Something went wrong");
      detail.textContent = message;
      note.hidden = true;
      tipEl.textContent = "";
      retry.hidden = false;
      retry.focus();
    }

    retry.addEventListener("click", function () {
      busy.hidden = true;
      retry.hidden = true;
      document.body.style.overflow = "";
    });

    // A short two-note chime and a buzz when the table is ready.
    function prepareChime() {
      var Ctx = window.AudioContext || window.webkitAudioContext;
      if (!Ctx || audio) return;
      try { audio = new Ctx(); } catch (err) { audio = null; }
    }
    function notifyDone() {
      if (navigator.vibrate) navigator.vibrate([120, 80, 120]);
      if (!audio) return;
      try {
        audio.resume();
        [880, 1320].forEach(function (freq, i) {
          var osc = audio.createOscillator();
          var gain = audio.createGain();
          var at = audio.currentTime + i * 0.18;
          osc.frequency.value = freq;
          gain.gain.setValueAtTime(0.0001, at);
          gain.gain.exponentialRampToValueAtTime(0.15, at + 0.02);
          gain.gain.exponentialRampToValueAtTime(0.0001, at + 0.3);
          osc.connect(gain).connect(audio.destination);
          osc.start(at);
          osc.stop(at + 0.32);
        });
      } catch (err) { /* sound is optional */ }
    }

    function finish(url) {
      if (finished) return;
      finished = true;
      stopAll();
      store.remove("ledger.pending");
      busy.classList.add("is-counting");
      fill.style.width = "100%";
      titleEl.textContent = T("Your table is ready");
      detail.textContent = T("Opening it…");
      notifyDone();
      if (document.hidden) {
        // Show it in the tab title, and open the table on return.
        document.title = "✓ " + T("Your table is ready");
        document.addEventListener("visibilitychange", function () {
          if (!document.hidden) window.location.href = url;
        });
      } else {
        window.location.href = url;
      }
    }

    function showPages(progress) {
      var total = progress.total || 0;
      if (!total) return;
      var done = progress.done || 0;
      busy.classList.add("is-counting");
      fill.style.width = Math.round((done / total) * 100) + "%";
      if (total > 1) titleEl.textContent = T("Read {done} of {total} pages", { done: done, total: total });
      if (progress.fallback) detail.setAttribute("data-fallback", "1");

      var items = grid.children;
      (progress.pages || []).forEach(function (page, i) {
        var item = items[i];
        if (!item) {
          item = document.createElement("li");
          if (page.thumb) {
            var img = document.createElement("img");
            img.src = page.thumb;
            img.alt = "";
            item.appendChild(img);
          } else {
            item.textContent = String(i + 1);
          }
          grid.appendChild(item);
        }
        item.classList.toggle("is-done", !!page.done);
      });
    }

    // Asks how far the read has got. Also how the result is found again after
    // the connection drops, the phone sleeps, or the page is reopened.
    function poll(token, readStillRunning) {
      var url = form.getAttribute("data-progress-url").replace("__TOKEN__", encodeURIComponent(token));
      var unknown = 0, lastDone = -1, lastChange = Date.now();
      var timer = setInterval(function () {
        getJSON(url, function (progress, status) {
          if (finished || busy.classList.contains("is-failed")) return;
          if (!progress) return; // network blip: try again next time
          if (progress.stage === "done" && progress.review_url) return finish(progress.review_url);
          if (progress.stage === "error") return fail(progress.error || T("The server hit an error while reading this file."));
          if (status === 404 || progress.stage === "unknown") {
            unknown++;
            if (!readStillRunning() && unknown > 4) {
              fail(T("That scan didn't finish. Please upload the file again."));
            }
            return;
          }
          if (progress.done !== lastDone) { lastDone = progress.done; lastChange = Date.now(); }
          if (Date.now() - lastChange > 15 * 60 * 1000) {
            return fail(T("The server stopped answering. Please try again."));
          }
          showPages(progress);
        });
      }, 1500);
      timers.push(timer);
    }

    function startWaiting(isPdf) {
      setState("reading");
      fill.style.width = "";
      titleEl.textContent = isPdf ? T("Reading your pages…") : T("Reading your ledger…");
      note.hidden = false;
      var started = Date.now();
      var tick = function () {
        var hint = detail.getAttribute("data-fallback")
          ? T("Using the free reader, which is slower.") + " "
          : (isPdf ? T("Large PDFs can take a few minutes.") + " " : T("This usually takes under a minute.") + " ");
        detail.textContent = hint + T("Elapsed {time}", { time: clock((Date.now() - started) / 1000) });
      };
      detail.removeAttribute("data-fallback");
      tick();
      timers.push(setInterval(tick, 1000));
      var tip = 0;
      tipEl.textContent = T(TIPS[0]);
      timers.push(setInterval(function () {
        tip = (tip + 1) % TIPS.length;
        tipEl.textContent = T(TIPS[tip]);
      }, 7000));
    }

    // Step 2: ask the server to read the saved file, watching its progress.
    function read(token, isPdf) {
      startWaiting(isPdf);
      var running = true;
      poll(token, function () { return running; });

      var xhr = new XMLHttpRequest();
      xhr.onload = function () {
        running = false;
        var reply = null;
        try { reply = JSON.parse(xhr.responseText); } catch (err) { /* not JSON */ }
        if (xhr.status === 200 && reply && reply.review_url) return finish(reply.review_url);
        if (reply && reply.error) return fail(reply.error);
        // Anything else (a proxy timeout, say): the server may still be
        // reading, so keep asking /progress until it is done.
      };
      xhr.onerror = function () { running = false; };
      var body = new FormData();
      body.append("token", token);
      if (nameInput) body.append("name", nameInput.value.trim());
      if (appendInput) body.append("append_to", appendInput.value);
      xhr.open("POST", form.action);
      xhr.send(body);
    }

    // Step 1: send the file. /upload only saves it and answers at once, so
    // "upload finished" is exact (browsers report the end of an upload only
    // when the server starts answering).
    form.addEventListener("submit", function (event) {
      var file = chosen || (input.files && input.files[0]);
      if (!file) return; // let the browser's "required" message show

      event.preventDefault();
      if (nameInput && nameInput.value.trim()) store.set("ledger.lastName", nameInput.value.trim());
      prepareChime();
      var isPdf = isPdfFile(file);
      stopAll();
      openBusy();
      setState("uploading");
      fill.style.width = "0%";
      titleEl.textContent = T("Uploading… {pct}%", { pct: 0 });
      detail.textContent = file.name + " · " + mb(file.size);

      var xhr = new XMLHttpRequest();
      xhr.upload.onprogress = function (e) {
        if (!e.lengthComputable) return;
        var pct = Math.min(100, Math.round((e.loaded / e.total) * 100));
        fill.style.width = pct + "%";
        titleEl.textContent = T("Uploading… {pct}%", { pct: pct });
        detail.textContent = T("{sent} of {total}", { sent: mb(e.loaded), total: mb(e.total) });
      };
      xhr.onload = function () {
        var reply = null;
        try { reply = JSON.parse(xhr.responseText); } catch (err) { /* not JSON */ }
        if (xhr.status !== 200 || !reply || !reply.token) {
          fail((reply && reply.error) || T("The upload failed ({status}). Try again.", { status: xhr.status }));
          return;
        }
        store.set("ledger.pending", JSON.stringify({ token: reply.token, pdf: isPdf, started: Date.now() }));
        read(reply.token, isPdf);
      };
      xhr.onerror = function () {
        fail(T("The connection dropped while uploading. Check your connection and try again."));
      };
      xhr.open("POST", form.getAttribute("data-upload-url"));
      var data = new FormData();
      data.append("image", file);
      xhr.send(data);
    });

    // Back on the page while a scan was still being read: pick up where it was.
    var pending = null;
    try { pending = JSON.parse(store.get("ledger.pending") || "null"); } catch (err) { pending = null; }
    if (pending && pending.token && Date.now() - pending.started < 60 * 60 * 1000) {
      openBusy();
      startWaiting(!!pending.pdf);
      poll(pending.token, function () { return false; });
    } else if (pending) {
      store.remove("ledger.pending");
    }

    // Phones: a Scan button within thumb reach once the form scrolls away.
    var thumbBar = document.getElementById("thumb-bar");
    if (thumbBar && "IntersectionObserver" in window) {
      new IntersectionObserver(function (entries) {
        thumbBar.hidden = entries[0].isIntersecting;
      }).observe(form);
      thumbBar.addEventListener("click", function () {
        setTimeout(function () { (takePhoto && !takePhoto.hidden ? takePhoto : input).focus(); }, 300);
      });
    }
  }

  // --- Review page -------------------------------------------------------------------

  function initReview() {
    var form = document.getElementById("review-form");
    if (!form) return;
    var tablesMode = form.getAttribute("data-mode") === "tables";
    var summaryEl = document.getElementById("summary");
    var miniSummary = document.getElementById("actionbar-summary");
    var nextUnsure = document.getElementById("next-unsure");
    var dirty = false;

    var cells = function () { return Array.prototype.slice.call(form.querySelectorAll("tbody input")); };
    var headingInputs = function () { return Array.prototype.slice.call(form.querySelectorAll("thead input")); };
    var norm = function (text) { return String(text).trim().toLowerCase(); };
    var pages = Array.prototype.slice.call(form.querySelectorAll(".page"));

    // --- Unsure cells, summary, totals -----------------------------------------

    function markUnsure(input) {
      input.classList.toggle("unsure", input.value.indexOf("?") !== -1);
    }

    function updateSummary() {
      var rows = form.querySelectorAll("tbody tr").length;
      var unsure = form.querySelectorAll("tbody input.unsure").length;
      var text = [
        count(Math.max(pages.length, 1), "1 page", "{n} pages"),
        count(rows, "1 row", "{n} rows"),
        count(unsure, "1 cell to check", "{n} cells to check")
      ].join(" · ");
      if (summaryEl) summaryEl.textContent = text;
      if (miniSummary) miniSummary.textContent = count(unsure, "1 cell to check", "{n} cells to check");
      if (nextUnsure) nextUnsure.hidden = unsure === 0;
    }

    var parseNumber = function (text) {
      var compact = String(text).replace(/[\s,]/g, "");
      return /^-?\d+(\.\d+)?$/.test(compact) ? parseFloat(compact) : null;
    };
    var formatNumber = function (value, decimals) {
      var parts = value.toFixed(decimals).split(".");
      parts[0] = parts[0].replace(/\B(?=(\d{3})+(?!\d))/g, ",");
      return parts.join(".");
    };

    // A sum under each column of numbers, to compare with the book's own totals.
    function updateTotals(table) {
      var rows = table.tBodies[0].rows;
      var width = table.tHead.rows[0].cells.length - 1; // last is the row tools column
      var foot = table.tFoot || table.createTFoot();
      var line = foot.rows[0] || foot.insertRow();
      while (line.cells.length < width + 1) line.insertCell();
      var any = false;
      for (var c = 0; c < width; c++) {
        var filled = 0, numbers = 0, sum = 0, decimals = 0;
        for (var r = 0; r < rows.length; r++) {
          var input = rows[r].cells[c] && rows[r].cells[c].querySelector("input");
          if (!input || !input.value.trim()) continue;
          filled++;
          var value = parseNumber(input.value);
          if (value === null) continue;
          numbers++;
          sum += value;
          var dot = input.value.split(".")[1];
          if (dot) decimals = Math.max(decimals, Math.min(dot.replace(/\D/g, "").length, 4));
        }
        var isNumeric = numbers >= 2 && numbers / filled >= 0.6;
        line.cells[c].textContent = isNumeric ? formatNumber(sum, decimals) : "";
        line.cells[c].className = isNumeric ? "total" : "";
        any = any || isNumeric;
      }
      if (any && !line.cells[0].textContent) {
        line.cells[0].textContent = T("Sum of rows");
        line.cells[0].className = "total-label";
      }
      foot.hidden = !any;
    }

    function tableOf(el) { return el.closest("table"); }

    // --- Row tools: insert and delete, with undo -------------------------------

    function toolsCell() {
      var td = document.createElement("td");
      td.className = "row-tools";
      td.innerHTML =
        '<button type="button" class="row-add" title="' + T("Insert a row below") + '" aria-label="' + T("Insert a row below") + '">+</button>' +
        '<button type="button" class="row-del" title="' + T("Delete this row") + '" aria-label="' + T("Delete this row") + '">×</button>';
      return td;
    }

    function addRowAfter(row) {
      var fresh = document.createElement("tr");
      Array.prototype.forEach.call(row.cells, function (cell) {
        var source = cell.querySelector("input");
        if (!source) return;
        var td = document.createElement("td");
        var input = document.createElement("input");
        input.type = "text";
        input.setAttribute("aria-label", source.getAttribute("aria-label") || "");
        if (source.getAttribute("data-field")) input.setAttribute("data-field", source.getAttribute("data-field"));
        td.appendChild(input);
        fresh.appendChild(td);
      });
      fresh.appendChild(toolsCell());
      row.parentNode.insertBefore(fresh, row.nextSibling);
      fresh.querySelector("input").focus();
      changed(tableOf(fresh));
    }

    function deleteRow(row) {
      var body = row.parentNode, next = row.nextSibling, table = tableOf(row);
      body.removeChild(row);
      changed(table);
      toast(T("Row deleted."), [{
        label: T("Undo"),
        run: function () {
          body.insertBefore(row, next && next.parentNode === body ? next : null);
          changed(table);
        }
      }]);
    }

    function changed(table) {
      dirty = true;
      if (table) updateTotals(table);
      updateSummary();
    }

    Array.prototype.forEach.call(form.querySelectorAll(".sheet table"), function (table) {
      var head = table.tHead.rows[0];
      var th = document.createElement("th");
      th.className = "row-tools";
      th.setAttribute("aria-hidden", "true");
      head.appendChild(th);
      Array.prototype.forEach.call(table.tBodies[0].rows, function (row) { row.appendChild(toolsCell()); });
      updateTotals(table);
    });

    // --- Copy a table ------------------------------------------------------------

    Array.prototype.forEach.call(form.querySelectorAll(".table-block"), function (block) {
      var tools = block.querySelector(".table-tools");
      if (!tools) return;
      var button = document.createElement("button");
      button.type = "button";
      button.className = "link-button";
      button.textContent = T("Copy table");
      button.addEventListener("click", function () {
        var table = block.querySelector("table");
        var title = block.querySelector(".table-title");
        var columns = Array.prototype.map.call(table.tHead.rows[0].cells, function (th) {
          var input = th.querySelector("input");
          return input ? input.value : th.textContent.trim();
        }).filter(function (text, i, all) { return i < all.length - 1; });
        var rows = Array.prototype.map.call(table.tBodies[0].rows, function (row) {
          return Array.prototype.map.call(row.querySelectorAll("input"), function (input) { return input.value; });
        });
        copyText(tableText(title ? title.value : "", columns, rows));
      });
      tools.appendChild(button);
    });

    // --- Page by page -------------------------------------------------------------

    var pager = document.getElementById("pager");
    var pagerSelect = document.getElementById("pager-select");
    var current = 0;

    function showPage(index, focusTop) {
      if (!pager) return;
      current = Math.max(0, Math.min(pages.length - 1, index));
      pages.forEach(function (page, i) { page.classList.toggle("is-current", i === current); });
      pagerSelect.value = String(current);
      pager.querySelector('[data-step="-1"]').disabled = current === 0;
      pager.querySelector('[data-step="1"]').disabled = current === pages.length - 1;
      if (window.history && history.replaceState) history.replaceState(null, "", "#" + pages[current].id);
      if (focusTop) pager.scrollIntoView({ block: "start" });
    }

    if (pager && pages.length > 1) {
      form.classList.add("paged");
      pager.hidden = false;
      pager.addEventListener("click", function (e) {
        var step = e.target.closest("[data-step]");
        if (step) showPage(current + parseInt(step.getAttribute("data-step"), 10), true);
      });
      pagerSelect.addEventListener("change", function () { showPage(parseInt(pagerSelect.value, 10), true); });
      var fromHash = pages.findIndex(function (page) { return "#" + page.id === window.location.hash; });
      showPage(fromHash === -1 ? 0 : fromHash, false);

      // Swipe left/right to change page, except where the swipe scrolls a wide table or a zoomed photo.
      var startX = null, startY = 0;
      form.addEventListener("touchstart", function (e) {
        var scroller = e.target.closest(".sheet, .viewer-frame");
        startX = scroller && scroller.scrollWidth > scroller.clientWidth + 2 ? null : e.touches[0].clientX;
        startY = e.touches[0].clientY;
      }, { passive: true });
      form.addEventListener("touchend", function (e) {
        if (startX === null) return;
        var dx = e.changedTouches[0].clientX - startX, dy = e.changedTouches[0].clientY - startY;
        startX = null;
        if (Math.abs(dx) > 80 && Math.abs(dy) < 60) showPage(current + (dx < 0 ? 1 : -1), true);
      });
    }

    function revealCell(input) {
      var page = input.closest(".page");
      var index = pages.indexOf(page);
      if (pager && index !== -1 && index !== current) showPage(index, false);
    }

    if (nextUnsure) {
      nextUnsure.addEventListener("click", function () {
        var all = cells();
        var at = all.indexOf(document.activeElement);
        var next = all.slice(at + 1).concat(all.slice(0, at + 1)).filter(function (c) { return c.classList.contains("unsure"); })[0];
        if (!next) return;
        revealCell(next);
        next.focus();
        next.scrollIntoView({ block: "center" });
      });
    }

    // --- The photo beside the table ------------------------------------------------

    function setupViewer(viewer) {
      var frame = viewer.querySelector(".viewer-frame");
      var img = frame.querySelector("img");
      var approx = viewer.querySelector(".viewer-approx");
      var select = viewer.querySelector(".viewer-select");
      var zoom = 1;
      viewer.querySelector(".viewer-zoom").hidden = false;

      function setZoom(value, focusY, focusX) {
        zoom = Math.max(1, Math.min(5, value));
        img.style.width = zoom * 100 + "%";
        approx.hidden = focusY === undefined;
        var place = function () {
          if (focusY !== undefined) frame.scrollTop = focusY * img.clientHeight - frame.clientHeight / 2;
          frame.scrollLeft = (focusX === undefined ? 0.5 : focusX) * img.clientWidth - frame.clientWidth / 2;
        };
        if (img.complete) place(); else img.addEventListener("load", place, { once: true });
      }

      viewer.addEventListener("click", function (e) {
        var button = e.target.closest("[data-zoom]");
        if (!button) return;
        var kind = button.getAttribute("data-zoom");
        setZoom(kind === "in" ? zoom * 1.5 : kind === "out" ? zoom / 1.5 : 1);
      });
      if (select) {
        select.addEventListener("change", function () {
          img.src = select.value;
          setZoom(1);
        });
      }

      // Rows are assumed evenly spaced down the page, and columns evenly
      // across it, so the position is approximate.
      return function showCell(cell) {
        if (select && select.options.length > 1) return;
        var row = cell.parentNode;
        var rows = Array.prototype.slice.call(viewer.closest(".page").querySelectorAll("tbody tr"));
        var index = rows.indexOf(row);
        if (index === -1) return;
        var columns = row.querySelectorAll("input").length || 1;
        setZoom(Math.max(zoom, 2.2), 0.08 + 0.86 * (index + 0.5) / rows.length,
                0.05 + 0.9 * (cell.cellIndex + 0.5) / columns);
      };
    }

    var viewers = new Map();
    Array.prototype.forEach.call(form.querySelectorAll(".viewer"), function (viewer) {
      viewers.set(viewer.closest(".page"), setupViewer(viewer));
    });

    // --- Editing ------------------------------------------------------------------

    var before = new WeakMap();

    form.addEventListener("focusin", function (e) {
      var input = e.target;
      if (input.tagName !== "INPUT") return;
      before.set(input, input.value);
      if (input.closest("tbody")) {
        var show = viewers.get(input.closest(".page"));
        if (show) show(input.closest("td"));
      }
    });

    form.addEventListener("input", function (e) {
      if (e.target.tagName !== "INPUT") return;
      if (e.target.closest("tbody")) markUnsure(e.target);
      changed(tableOf(e.target));
    });

    function headingOf(input) {
      var cell = input.closest("td");
      var th = tableOf(input).tHead.rows[0].cells[cell.cellIndex];
      var field = th.querySelector("input");
      return norm(field ? field.value : th.textContent);
    }

    // Fixed one name? Offer to fix its other copies under the same heading.
    // Renamed a heading? Offer to rename the same heading on other pages.
    form.addEventListener("change", function (e) {
      var input = e.target;
      if (input.tagName !== "INPUT" || !before.has(input)) return;
      var old = before.get(input), now = input.value;
      before.set(input, now);
      if (!old.trim() || norm(old) === norm(now) || !now.trim()) return;

      if (input.closest("thead")) {
        var same = headingInputs().filter(function (other) { return other !== input && norm(other.value) === norm(old); });
        if (!same.length) return;
        toast(T("Rename “{old}” in {n} other tables too?", { old: old, n: same.length }), [{
          label: T("Rename all"),
          run: function () {
            same.forEach(function (other) { other.value = now; before.set(other, now); });
            dirty = true;
          }
        }]);
        return;
      }

      if (!input.closest("tbody") || !/[A-Za-z]/.test(old)) return;
      var heading = headingOf(input);
      var copies = cells().filter(function (other) {
        return other !== input && other.value === old && headingOf(other) === heading;
      });
      if (!copies.length) return;
      toast(T("Change the other {n} “{old}” to “{new}”?", { n: copies.length, old: old, "new": now }), [{
        label: T("Change all"),
        run: function () {
          var tables = new Set();
          copies.forEach(function (other) {
            other.value = now;
            markUnsure(other);
            tables.add(tableOf(other));
          });
          tables.forEach(function (table) { changed(table); });
          toast(count(copies.length, "Changed 1 cell.", "Changed {n} cells."));
        }
      }]);
    });

    form.addEventListener("click", function (e) {
      var add = e.target.closest(".row-add");
      var del = e.target.closest(".row-del");
      if (add) addRowAfter(add.closest("tr"));
      if (del) deleteRow(del.closest("tr"));
    });

    // Enter moves down a column (Shift+Enter up) instead of saving the form.
    form.addEventListener("keydown", function (e) {
      if (e.key !== "Enter" || e.target.tagName !== "INPUT" || !e.target.closest("table, .table-tools, .doc-name-field")) return;
      e.preventDefault();
      var cell = e.target.closest("td, th");
      if (!cell) return;
      var row = cell.parentNode;
      var target = e.shiftKey ? row.previousElementSibling : row.nextElementSibling;
      if (!target && !e.shiftKey && row.parentNode.tagName === "THEAD") target = tableOf(row).tBodies[0].rows[0];
      var next = target && target.cells[cell.cellIndex] && target.cells[cell.cellIndex].querySelector("input");
      if (next) next.focus();
    });

    // --- Leaving and saving -----------------------------------------------------------

    window.addEventListener("beforeunload", function (e) {
      if (!dirty) return;
      e.preventDefault();
      e.returnValue = "";
    });

    // Rows may have been added or deleted: number the fields again so the
    // server reads them in order.
    function renumber() {
      if (tablesMode) {
        Array.prototype.forEach.call(form.querySelectorAll(".table-block"), function (block) {
          var t = block.getAttribute("data-table");
          var rows = block.querySelector("tbody").rows;
          Array.prototype.forEach.call(rows, function (row, r) {
            Array.prototype.forEach.call(row.querySelectorAll("input"), function (input, c) {
              input.name = "cell_" + t + "_" + r + "_" + c;
            });
          });
          form.querySelector('input[name="num_rows_' + t + '"]').value = rows.length;
        });
      } else {
        var rows = form.querySelector("tbody").rows;
        Array.prototype.forEach.call(rows, function (row, r) {
          Array.prototype.forEach.call(row.querySelectorAll("input[data-field]"), function (input) {
            input.name = input.getAttribute("data-field") + "_" + r;
          });
        });
        form.querySelector('input[name="num_rows"]').value = rows.length;
      }
    }

    form.addEventListener("submit", function () {
      dirty = false;
      renumber();
      var button = form.querySelector('button[type="submit"]');
      // Disable after the submit has been dispatched, so the form still sends.
      setTimeout(function () {
        button.disabled = true;
        button.textContent = T("Saving…");
      }, 0);
    });

    updateSummary();
  }

  // --- Document page ----------------------------------------------------------------------

  function initDocument() {
    var share = document.getElementById("share-file");
    var copyAll = document.getElementById("copy-all");
    var data = document.getElementById("doc-tables");

    if (copyAll && data) {
      copyAll.hidden = false;
      copyAll.addEventListener("click", function () {
        var tables = JSON.parse(data.textContent);
        copyText(tables.map(function (table) {
          return tableText(table.title + (table.page ? " (" + T("Page {n}", { n: table.page }) + ")" : ""), table.columns, table.rows);
        }).join("\n\n"));
      });
    }

    // Share the Excel file itself (e.g. to WhatsApp) where the phone allows it.
    if (share && navigator.canShare && window.fetch && window.File) {
      var file = null;
      fetch(share.getAttribute("data-url")).then(function (r) { return r.blob(); }).then(function (blob) {
        var candidate = new File([blob], share.getAttribute("data-filename"), { type: blob.type });
        if (navigator.canShare({ files: [candidate] })) {
          file = candidate;
          share.hidden = false;
        }
      }).catch(function () {});
      share.addEventListener("click", function () {
        if (!file) return;
        navigator.share({ files: [file], title: share.getAttribute("data-title") }).catch(function () {});
      });
    }
  }

  initUpload();
  initReview();
  initDocument();
})();
